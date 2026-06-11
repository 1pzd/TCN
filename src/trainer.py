import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import sys


class EyeTrackingDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


def train_epoch(model, loader, optimizer, criterion, device, grad_clip, pbar):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        pbar.update(1)

    return total_loss / len(loader), correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        probs = F.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    return total_loss / len(loader), acc, all_preds, all_labels, all_probs


def train_model(model, train_loader, val_loader, cfg, device):
    train_cfg = cfg['training']
    class_weight = train_cfg.get('class_weight')
    if class_weight is not None:
        weight = torch.tensor(class_weight, dtype=torch.float).to(device)
        criterion = nn.CrossEntropyLoss(weight=weight)
    else:
        criterion = nn.CrossEntropyLoss()
    base_lr = train_cfg['learning_rate']
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=base_lr,
        weight_decay=train_cfg['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6
    )

    best_val_acc = 0
    best_model_state = None
    patience_counter = 0
    warmup_epochs = train_cfg.get('warmup_epochs', 0)

    print(f"\n{'='*50}")
    print("开始训练...")
    print(f"{'='*50}")
    header = f"{'Epoch':>5} | {'Train Loss':>10} | {'Train Acc':>9} | {'Val Loss':>9} | {'Val Acc':>8} | {'LR':>10}"
    print(header)
    print("-" * len(header))

    total_batches = len(train_loader) * train_cfg['epochs']

    with tqdm(total=total_batches, desc="Training", unit='batch',
              file=sys.stdout, dynamic_ncols=True) as pbar:
        for epoch in range(1, train_cfg['epochs'] + 1):
            if warmup_epochs > 0 and epoch <= warmup_epochs:
                warmup_lr = base_lr * epoch / warmup_epochs
                for param_group in optimizer.param_groups:
                    param_group['lr'] = warmup_lr
                current_lr = warmup_lr
            else:
                current_lr = optimizer.param_groups[0]['lr']

            train_loss, train_acc = train_epoch(
                model, train_loader, optimizer, criterion, device,
                train_cfg['grad_clip'], pbar
            )

            val_loss, val_acc, _, _, _ = evaluate(model, val_loader, criterion, device)

            if current_lr == optimizer.param_groups[0]['lr']:
                scheduler.step(val_acc)
            current_lr = optimizer.param_groups[0]['lr']

            tqdm.write(
                f"{epoch:>5} | {train_loss:>10.4f} | {train_acc:>9.4f} | "
                f"{val_loss:>9.4f} | {val_acc:>8.4f} | {current_lr:>.2e}"
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= train_cfg['patience']:
                    tqdm.write(f"\n早停于 Epoch {epoch}")
                    break

    model.load_state_dict(best_model_state)
    model = model.to(device)
    print(f"\n最佳验证准确率: {best_val_acc:.4f}")
    return model
