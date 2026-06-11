import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.config import load_config
from src.data_loader import load_and_preprocess, create_sequences, kfold_split_subjects, split_inner_val
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

    fold_info = kfold_split_subjects(df, n_splits=3, random_state=cfg['random_state'])

    all_clip_results = []
    fold_metrics = []
    heldout_subjects = []

    for fold in range(3):
        print(f"\n{'='*60}")
        print(f"折 {fold+1}/3")
        print(f"{'='*60}")

        test_subjects = fold_info[fold_info['fold'] == fold]['unique_subject'].tolist()
        train_subjects = fold_info[fold_info['fold'] != fold]['unique_subject'].tolist()
        heldout_subjects.extend(test_subjects)
        print(f"  Outer test subjects ({len(test_subjects)}): {sorted(test_subjects)}")
        print(f"  Outer train subjects ({len(train_subjects)}): {sorted(train_subjects)}")

        inner_train_subjects, inner_val_subjects = split_inner_val(
            df, train_subjects, val_ratio=0.2, random_state=cfg['random_state'] + fold
        )

        all_subjects = inner_train_subjects + inner_val_subjects + test_subjects
        assert len(set(inner_train_subjects) & set(inner_val_subjects)) == 0, "inner train/val overlap"
        assert len(set(inner_train_subjects) & set(test_subjects)) == 0, "inner train/test overlap"
        assert len(set(inner_val_subjects) & set(test_subjects)) == 0, "inner val/test overlap"
        print(f"  Disjoint check passed: {len(inner_train_subjects)} inner-train + "
              f"{len(inner_val_subjects)} inner-val + {len(test_subjects)} outer-test = "
              f"{len(set(all_subjects))} unique")

        inner_train_df = df[df['unique_subject'].isin(inner_train_subjects)].reset_index(drop=True)
        inner_val_df = df[df['unique_subject'].isin(inner_val_subjects)].reset_index(drop=True)
        test_df = df[df['unique_subject'].isin(test_subjects)].reset_index(drop=True)

        X_inner_train, y_inner_train, _ = create_sequences(
            inner_train_df, model_cfg['feature_cols'],
            model_cfg['max_seq_len'], model_cfg['min_clip_len']
        )
        X_inner_val, y_inner_val, _ = create_sequences(
            inner_val_df, model_cfg['feature_cols'],
            model_cfg['max_seq_len'], model_cfg['min_clip_len']
        )
        X_test, y_test, test_clip_info = create_sequences(
            test_df, model_cfg['feature_cols'],
            model_cfg['max_seq_len'], model_cfg['min_clip_len']
        )

        inner_train_dataset = EyeTrackingDataset(torch.FloatTensor(X_inner_train), torch.LongTensor(y_inner_train))
        inner_val_dataset = EyeTrackingDataset(torch.FloatTensor(X_inner_val), torch.LongTensor(y_inner_val))
        test_dataset = EyeTrackingDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))

        inner_train_loader = DataLoader(inner_train_dataset, batch_size=train_cfg['batch_size'], shuffle=True)
        inner_val_loader = DataLoader(inner_val_dataset, batch_size=train_cfg['batch_size'], shuffle=False)
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

        model = train_model(model, inner_train_loader, inner_val_loader, cfg, device)

        _, _, test_preds, test_labels, test_probs = evaluate(
            model, test_loader, torch.nn.CrossEntropyLoss(), device
        )
        print_clip_results(test_labels, test_preds, test_probs)

        clip_results = predict_clips(
            model, test_df, model_cfg['feature_cols'], model_cfg['max_seq_len'],
            model_cfg['min_clip_len'], train_cfg['batch_size'], device
        )
        vote_threshold = cfg.get('vote', {}).get('threshold', 0.5)
        subject_results = subject_majority_vote(clip_results, threshold=vote_threshold)
        print_subject_results(subject_results)

        fold_metrics.append({
            'fold': fold + 1,
            'test_preds': test_preds,
            'test_labels': test_labels,
            'test_probs': test_probs,
            'clip_results': clip_results,
            'subject_results': subject_results,
        })

        all_clip_results.append(clip_results)

        os.makedirs('output_model', exist_ok=True)
        torch.save({
            'model_state_dict': model.state_dict(),
            'input_size': len(model_cfg['feature_cols']),
            'num_channels': model_cfg['num_channels'],
            'num_classes': 2,
            'kernel_sizes': model_cfg.get('kernel_sizes', 3),
            'dropout': model_cfg['dropout'],
        }, f'output_model/tcn_model_fold{fold+1}.pth')

        torch.save({
            'sequences': torch.FloatTensor(X_test),
            'labels': [int(label) for label in y_test],
            'clip_info': test_clip_info,
        }, f'output_model/test_clip_data_fold{fold+1}.pt')

    all_subjects = fold_info['unique_subject'].tolist()
    assert len(heldout_subjects) == len(set(heldout_subjects)), "outer test subject repeated across folds"
    assert set(heldout_subjects) == set(all_subjects), "outer test subjects do not cover all subjects"

    all_labels = []
    all_preds = []
    all_probs = []
    for m in fold_metrics:
        all_labels.extend(m['test_labels'])
        all_preds.extend(m['test_preds'])
        all_probs.extend(m['test_probs'])

    print(f"\n\n{'='*60}")
    print("3折交叉验证 — 汇总（所有 held-out 预测合并）")
    print(f"{'='*60}")
    print_clip_results(all_labels, all_preds, all_probs)

    combined_clip_results = pd.concat(all_clip_results, ignore_index=True)
    vote_threshold = cfg.get('vote', {}).get('threshold', 0.5)
    combined_subject_results = subject_majority_vote(combined_clip_results, threshold=vote_threshold)
    print_subject_results(combined_subject_results)

    print("\n单折模型已保存至: output_model/tcn_model_fold1~3.pth")
    print("单折测试数据已保存至: output_model/test_clip_data_fold1~3.pt")

    return fold_metrics, combined_subject_results


if __name__ == '__main__':
    fold_metrics, combined_subject_results = main()
