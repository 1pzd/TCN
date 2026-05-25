import os
import sys
import pandas as pd
import torch
from sklearn.metrics import f1_score, roc_auc_score

from src.config import load_config
from src.model import TCNClassifier
from src.predict import predict_from_data, subject_majority_vote


CLIP_IDS = [1]


def main():
    cfg = load_config()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"PyTorch: {torch.__version__} | Device: {device}")

    print(f"\n{'='*60}")
    print(f"利用 clip_id ∈ {CLIP_IDS} 进行评分")
    print(f"{'='*60}")

    all_clip_results = []

    for fold in range(3):
        model_path = f'output_model/tcn_model_fold{fold+1}.pth'
        data_path = f'output_model/test_clip_data_fold{fold+1}.pt'

        if not os.path.exists(model_path) or not os.path.exists(data_path):
            print(f"[错误] 折{fold+1} 文件不完整，请先运行 python train.py")
            sys.exit(1)

        checkpoint = torch.load(model_path, map_location=device, weights_only=False)

        model = TCNClassifier(
            input_size=checkpoint['input_size'],
            num_channels=checkpoint['num_channels'],
            num_classes=checkpoint['num_classes'],
            kernel_sizes=checkpoint.get('kernel_sizes', 3),
            dropout=checkpoint['dropout']
        ).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        test_data = torch.load(data_path, map_location='cpu', weights_only=False)
        test_sequences = test_data['sequences']
        test_labels = test_data['labels']
        test_clip_info = test_data['clip_info']

        clip_info_with_labels = [
            (subject, clip_id, unique_clip, test_labels[i])
            for i, (subject, clip_id, unique_clip) in enumerate(test_clip_info)
        ]

        fold_results = predict_from_data(model, test_sequences, clip_info_with_labels,
                                         cfg['training']['batch_size'], device)

        fold_filtered = fold_results[fold_results['clip_id'].isin(CLIP_IDS)].copy()
        dropped = len(fold_results) - len(fold_filtered)
        print(f"  折{fold+1}: {len(fold_results)} clips → 过滤后 {len(fold_filtered)} clips (排除 {dropped})")
        all_clip_results.append(fold_filtered)

    print(f"\n{'='*60}")
    print(f"汇总 — 仅 clip_id ∈ {CLIP_IDS}")
    print(f"{'='*60}")

    combined_clip = pd.concat(all_clip_results, ignore_index=True)

    subject_results = subject_majority_vote(combined_clip, threshold=0.55)

    print(f"\n各受试者投票结果:")
    for _, row in subject_results.iterrows():
        dataset_actual = 'BJ' if row['true_label'] == 1 else 'ZJ'
        dataset_pred = 'BJ' if row['pred_label'] == 1 else 'ZJ'
        correct_mark = '[OK]' if row['true_label'] == row['pred_label'] else '[NO]'
        print(f"  {row['unique_subject']:>12}: "
              f"真实={dataset_actual} 预测={dataset_pred} "
              f"ZJ:BJ={row['vote_0']}:{row['vote_1']} "
              f"(BJ占比{row['bj_ratio']:.1%}) "
              f"({row['total_clips']}clips) {correct_mark}")

    correct = (subject_results['true_label'] == subject_results['pred_label']).sum()
    total = len(subject_results)
    print(f"\n受试者级准确率: {correct/total:.4f} ({correct}/{total})")
    print(f"受试者级 F1: {f1_score(subject_results['true_label'], subject_results['pred_label']):.4f}")
    try:
        subject_auc = roc_auc_score(subject_results['true_label'], subject_results['bj_ratio'])
        print(f"受试者级 AUC:  {subject_auc:.4f}")
    except ValueError:
        print(f"受试者级 AUC:  N/A")

    tie_count = subject_results['tie'].sum()
    if tie_count > 0:
        print(f"  平局受试者: {tie_count}")

    print(f"\n分类汇总:")
    for true_label, label_name in [(0, 'ZJ(弱)'), (1, 'BJ(强)')]:
        subset = subject_results[subject_results['true_label'] == true_label]
        correct_sub = (subset['true_label'] == subset['pred_label']).sum()
        print(f"  {label_name}: {correct_sub}/{len(subset)} ({correct_sub/len(subset)*100:.1f}%)")

    return combined_clip, subject_results


if __name__ == '__main__':
    combined_clip, subject_results = main()
