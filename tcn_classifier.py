import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

np.random.seed(42)
torch.manual_seed(42)

# ========== 配置参数 ==========
DATA_DIR = 'data'
BJ_PATH = os.path.join(DATA_DIR, 'dataset_BJ.csv')
ZJ_PATH = os.path.join(DATA_DIR, 'dataset_ZJ.csv')

MAX_SEQ_LEN = 180
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-3
PATIENCE = 10
TEST_SUBJECT_RATIO = 0.2
NUM_CHANNELS = [32, 64, 64, 128, 128]
KERNEL_SIZE = 3
DROPOUT = 0.2

FEATURE_COLS = ['gaze_x', 'gaze_y', 'is_fixation']


# ========== 1. 数据加载与预处理 ==========
class EyeTrackingDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


def load_and_preprocess():
    df_bj = pd.read_csv(BJ_PATH)
    df_zj = pd.read_csv(ZJ_PATH)

    df_bj['dataset'] = 'BJ'
    df_zj['dataset'] = 'ZJ'

    df = pd.concat([df_bj, df_zj], ignore_index=True)
    df = df[~((df['dataset'] == 'BJ') & (df['subject_id'] == 18))].reset_index(drop=True)
    df['label'] = (df['dataset'] == 'BJ').astype(int)

    df['unique_subject'] = df['dataset'] + '_' + df['subject_id'].astype(str)
    df['unique_clip'] = df['unique_subject'] + '_clip_' + df['clip_id'].astype(str)

    df['is_fixation'] = (df['original_type'] == 'Fixation').astype(float)
    df = df.sort_values(['unique_clip', 'timestamp']).reset_index(drop=True)

    print(f"\n{'='*50}")
    print("数据加载完成")
    print(f"  BJ 样本数: {len(df_bj)}")
    print(f"  ZJ 样本数: {len(df_zj)}")
    print(f"  总样本数: {len(df)}")
    print(f"  标签分布: 0 (ZJ)={len(df[df['label']==0])}, 1 (BJ)={len(df[df['label']==1])}")
    print(f"  唯一受试者: BJ={df_bj['subject_id'].nunique()}, ZJ={df_zj['subject_id'].nunique()}")
    print(f"  总clip数: {df['unique_clip'].nunique()}")

    return df


def create_sequences(df):
    clip_lengths = df.groupby('unique_clip').size()
    print(f"\nClip长度统计:")
    print(f"  最小值={clip_lengths.min()}, 最大值={clip_lengths.max()}, "
          f"均值={clip_lengths.mean():.1f}, 中位数={clip_lengths.median()}")

    sequences = []
    labels = []
    clip_info = []

    for (unique_clip), group in df.groupby('unique_clip'):
        group = group.sort_values('timestamp')
        values = group[FEATURE_COLS].values.astype(np.float32)
        label = group['label'].iloc[0]
        subject = group['unique_subject'].iloc[0]
        clip_id = group['clip_id'].iloc[0]

        if len(values) < 10:
            continue

        if len(values) > MAX_SEQ_LEN:
            values = values[:MAX_SEQ_LEN]
        else:
            pad_len = MAX_SEQ_LEN - len(values)
            pad = np.zeros((pad_len, values.shape[1]), dtype=np.float32)
            values = np.vstack([values, pad])

        sequences.append(values)
        labels.append(label)
        clip_info.append((subject, clip_id, unique_clip))

    sequences = np.array(sequences)
    labels = np.array(labels, dtype=np.int64)

    print(f"\n序列构建完成: {len(sequences)} 个clip序列")
    print(f"  序列形状: {sequences.shape}")
    print(f"  标签分布: 0={np.sum(labels==0)}, 1={np.sum(labels==1)}")

    return sequences, labels, clip_info


def split_by_subject(df, test_ratio, random_state=42):
    subjects = df[['unique_subject', 'label']].drop_duplicates()

    train_subjects = []
    test_subjects = []

    for label in [0, 1]:
        group = subjects[subjects['label'] == label]
        n_test = max(1, int(len(group) * test_ratio))
        sampled = group.sample(n=n_test, random_state=random_state)
        test_subjects.extend(sampled['unique_subject'].tolist())
        train = group[~group['unique_subject'].isin(sampled['unique_subject'])]
        train_subjects.extend(train['unique_subject'].tolist())

    train_df = df[df['unique_subject'].isin(train_subjects)].reset_index(drop=True)
    test_df = df[df['unique_subject'].isin(test_subjects)].reset_index(drop=True)

    train_info = train_df[['unique_subject', 'label']].drop_duplicates()
    test_info = test_df[['unique_subject', 'label']].drop_duplicates()

    print(f"按受试者拆分 (test_ratio={test_ratio}):")
    print(f"  训练集: {len(train_info)} 受试者 "
          f"(BJ={len(train_info[train_info['label']==1])}, "
          f"ZJ={len(train_info[train_info['label']==0])}), "
          f"{train_df['unique_clip'].nunique()} clips")
    print(f"  测试集: {len(test_info)} 受试者 "
          f"(BJ={len(test_info[test_info['label']==1])}, "
          f"ZJ={len(test_info[test_info['label']==0])}), "
          f"{test_df['unique_clip'].nunique()} clips")

    overlap = set(train_subjects) & set(test_subjects)
    assert len(overlap) == 0, f"Subject泄露! 重叠: {overlap}"

    return train_df, test_df, train_subjects, test_subjects


# ========== 2. TCN 模型 ==========
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout):
        super().__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(
                in_channels, out_channels, kernel_size, stride=1,
                dilation=dilation_size,
                padding=(kernel_size - 1) * dilation_size,
                dropout=dropout
            )]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TCNClassifier(nn.Module):
    def __init__(self, input_size, num_channels, num_classes=2, kernel_size=3, dropout=0.2):
        super().__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(num_channels[-1], num_classes)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.tcn(x)
        x = self.gap(x).squeeze(-1)
        x = self.fc(x)
        return x


# ========== 3. 训练流程 ==========
def train_epoch(model, loader, optimizer, criterion):
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return total_loss / len(loader), correct / total


def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    return total_loss / len(loader), acc, all_preds, all_labels


def train_model(model, train_loader, val_loader):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6
    )

    best_val_acc = 0
    best_model_state = None
    patience_counter = 0

    print(f"\n{'='*50}")
    print("开始训练...")
    print(f"{'='*50}")
    print(f"{'Epoch':>6} | {'Train Loss':>10} | {'Train Acc':>9} | {'Val Loss':>8} | {'Val Acc':>7} | {'LR':>10}")
    print("-" * 56)

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion)
        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]['lr']

        print(f"{epoch:>6} | {train_loss:>10.4f} | {train_acc:>9.4f} | {val_loss:>8.4f} | {val_acc:>7.4f} | {current_lr:>.2e}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\n早停于 epoch {epoch}")
                break

    model.load_state_dict(best_model_state)
    model = model.to(device)
    print(f"\n最佳验证准确率: {best_val_acc:.4f}")
    return model


# ========== 4. 预测与 Bagging 投票决策 ==========
def predict_clips(model, df):
    model.eval()
    sequences_list = []
    clip_info_list = []

    for (unique_clip), group in df.groupby('unique_clip'):
        group = group.sort_values('timestamp')
        values = group[FEATURE_COLS].values.astype(np.float32)
        subject = group['unique_subject'].iloc[0]
        clip_id = group['clip_id'].iloc[0]
        label = group['label'].iloc[0]

        if len(values) < 10:
            continue

        if len(values) > MAX_SEQ_LEN:
            values = values[:MAX_SEQ_LEN]
        else:
            pad_len = MAX_SEQ_LEN - len(values)
            pad = np.zeros((pad_len, values.shape[1]), dtype=np.float32)
            values = np.vstack([values, pad])

        sequences_list.append(values)
        clip_info_list.append((subject, clip_id, unique_clip, label))

    sequences = np.array(sequences_list)
    dataset = EyeTrackingDataset(
        torch.FloatTensor(sequences),
        torch.LongTensor([ci[3] for ci in clip_info_list])
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    all_probs = []
    all_preds = []
    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    results = []
    for (subject, clip_id, unique_clip, true_label), pred, prob in zip(clip_info_list, all_preds, all_probs):
        results.append({
            'unique_subject': subject,
            'clip_id': clip_id,
            'unique_clip': unique_clip,
            'true_label': true_label,
            'pred_label': pred,
            'prob_0': prob[0],
            'prob_1': prob[1]
        })

    return pd.DataFrame(results)


def subject_majority_vote(clip_results):
    subject_votes = {}
    for _, row in clip_results.iterrows():
        subject = row['unique_subject']
        pred = row['pred_label']
        if subject not in subject_votes:
            subject_votes[subject] = {'votes': [], 'true_label': row['true_label']}
        subject_votes[subject]['votes'].append(pred)

    decision_results = []
    for subject, data in subject_votes.items():
        votes = data['votes']
        true_label = data['true_label']
        vote_counts = Counter(votes)
        final_pred = 1 if vote_counts.get(1, 0) > vote_counts.get(0, 0) else 0
        tie = vote_counts.get(1, 0) == vote_counts.get(0, 0)

        decision_results.append({
            'unique_subject': subject,
            'true_label': true_label,
            'pred_label': final_pred,
            'vote_0': vote_counts.get(0, 0),
            'vote_1': vote_counts.get(1, 0),
            'total_clips': len(votes),
            'tie': tie
        })

    return pd.DataFrame(decision_results)


# ========== 5. 主流程 ==========
def main():
    df = load_and_preprocess()
    print(f"\n数据集label分布:")
    print(df['dataset'].value_counts())

    train_df, test_df, _, _ = split_by_subject(df, TEST_SUBJECT_RATIO, random_state=42)

    X_train, y_train, _ = create_sequences(train_df)
    X_test, y_test, test_clip_info = create_sequences(test_df)

    train_dataset = EyeTrackingDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    test_dataset = EyeTrackingDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    input_size = len(FEATURE_COLS)
    model = TCNClassifier(
        input_size=input_size,
        num_channels=NUM_CHANNELS,
        num_classes=2,
        kernel_size=KERNEL_SIZE,
        dropout=DROPOUT
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n模型参数量: {total_params:,}")
    print(model)

    model = train_model(model, train_loader, test_loader)

    _, test_acc, test_preds, test_labels = evaluate(model, test_loader, criterion=nn.CrossEntropyLoss())
    print(f"\n{'='*50}")
    print("Clip级测试结果:")
    print(f"  准确率: {accuracy_score(test_labels, test_preds):.4f}")
    print(f"  精确率: {precision_score(test_labels, test_preds):.4f}")
    print(f"  召回率: {recall_score(test_labels, test_preds):.4f}")
    print(f"  F1分数: {f1_score(test_labels, test_preds):.4f}")
    print(f"  混淆矩阵:")
    cm = confusion_matrix(test_labels, test_preds)
    print(f"    TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"    FN={cm[1,0]}  TP={cm[1,1]}")

    clip_results = predict_clips(model, df)
    
    print(f"\n{'='*50}")
    print("受试者级 Bagging 投票决策:")
    subject_results = subject_majority_vote(clip_results)
    
    correct_subjects = (subject_results['true_label'] == subject_results['pred_label']).sum()
    total_subjects = len(subject_results)
    subject_acc = correct_subjects / total_subjects

    for _, row in subject_results.iterrows():
        dataset_actual = 'BJ' if row['true_label'] == 1 else 'ZJ'
        dataset_pred = 'BJ' if row['pred_label'] == 1 else 'ZJ'
        correct_mark = '[OK]' if row['true_label'] == row['pred_label'] else '[NO]'
        print(f"  {row['unique_subject']:>12}: "
              f"真实={dataset_actual} 预测={dataset_pred} "
              f"投票(0:{row['vote_0']}, 1:{row['vote_1']}) "
              f"clip数={row['total_clips']} {correct_mark}")

    print(f"\n{'='*50}")
    print(f"受试者级准确率: {subject_acc:.4f} ({correct_subjects}/{total_subjects})")
    print(f"受试者级 F1分数: {f1_score(subject_results['true_label'], subject_results['pred_label']):.4f}")

    tie_count = subject_results['tie'].sum()
    if tie_count > 0:
        print(f"  注意: {tie_count} 个受试者平局 (预测为 0)")

    print(f"\n{'='*50}")
    print("分类汇总:")
    for true_label, label_name in [(0, 'ZJ(弱)'), (1, 'BJ(强)')]:
        subset = subject_results[subject_results['true_label'] == true_label]
        correct = (subset['true_label'] == subset['pred_label']).sum()
        print(f"  {label_name}: {correct}/{len(subset)} 正确 ({correct/len(subset)*100:.1f}%)")

    return model, clip_results, subject_results


if __name__ == '__main__':
    model, clip_results, subject_results = main()
