import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import load_config
from src.data_loader import load_and_preprocess, create_sequences, split_by_subject
from src.model import TCNClassifier
from src.trainer import EyeTrackingDataset, train_model, evaluate
from src.predict import predict_clips, subject_majority_vote, print_clip_results, print_subject_results


def main():
    cfg = load_config()

    np.random.seed(cfg['random_state'])
    torch.manual_seed(cfg['random_state'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"PyTorch: {torch.__version__} | Device: {device}")

    df = load_and_preprocess(cfg['data']['bj_path'], cfg['data']['zj_path'])

    model_cfg = cfg['model']
    train_cfg = cfg['training']

    train_df, test_df, _, _ = split_by_subject(
        df, train_cfg['test_subject_ratio'], cfg['random_state']
    )

    X_train, y_train, _ = create_sequences(
        train_df, model_cfg['feature_cols'],
        model_cfg['max_seq_len'], model_cfg['min_clip_len']
    )
    X_test, y_test, test_clip_info = create_sequences(
        test_df, model_cfg['feature_cols'],
        model_cfg['max_seq_len'], model_cfg['min_clip_len']
    )

    train_dataset = EyeTrackingDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    test_dataset = EyeTrackingDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))

    train_loader = DataLoader(train_dataset, batch_size=train_cfg['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=train_cfg['batch_size'], shuffle=False)

    model = TCNClassifier(
        input_size=len(model_cfg['feature_cols']),
        num_channels=model_cfg['num_channels'],
        num_classes=2,
        kernel_sizes=model_cfg.get('kernel_sizes', 3),
        dropout=model_cfg['dropout']
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n模型参数量: {total_params:,}")

    model = train_model(model, train_loader, test_loader, cfg, device)

    _, _, test_preds, test_labels = evaluate(
        model, test_loader, torch.nn.CrossEntropyLoss(), device
    )
    print_clip_results(test_labels, test_preds)

    clip_results = predict_clips(
        model, df, model_cfg['feature_cols'], model_cfg['max_seq_len'],
        model_cfg['min_clip_len'], train_cfg['batch_size'], device
    )
    vote_threshold = cfg.get('vote', {}).get('threshold', 0.5)
    subject_results = subject_majority_vote(clip_results, threshold=vote_threshold)
    print_subject_results(subject_results)

    os.makedirs('output_model', exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_size': len(model_cfg['feature_cols']),
        'num_channels': model_cfg['num_channels'],
        'num_classes': 2,
        'kernel_sizes': model_cfg.get('kernel_sizes', 3),
        'dropout': model_cfg['dropout'],
    }, 'output_model/tcn_model.pth')

    torch.save({
        'sequences': torch.FloatTensor(X_test),
        'labels': [int(l) for l in y_test],
        'clip_info': test_clip_info,
    }, 'output_model/test_clip_data.pt')

    print(f"\n{'='*50}")
    print(f"模型已保存至 output_model/tcn_model.pth")
    print(f"测试集数据已保存至 output_model/test_clip_data.pt")

    return model, clip_results, subject_results


if __name__ == '__main__':
    model, clip_results, subject_results = main()
