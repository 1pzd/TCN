import os
import sys
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from src.config import load_config
from src.model import TCNClassifier
from src.predict import predict_from_data, subject_majority_vote


CLIP_PAIRS = [list(range(1, i)) for i in range(3, 23, 2)]


def main():
    cfg = load_config()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"PyTorch: {torch.__version__} | Device: {device}")

    # ── Step 1: 3折预测（保存全部clip结果） ──
    print(f"\n{'='*60}")
    print("第1步：运行3折预测（保留全部clip）")
    print(f"{'='*60}")

    all_fold_results = []
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

        print(f"  折{fold+1}: {len(fold_results)} clips")
        all_fold_results.append(fold_results)

    # ── Step 2: 10组 clip pair 逐一评估 ──
    print(f"\n{'='*80}")
    print("第2步：clip 两两组评估（共10组）")
    print(f"{'='*80}")

    rows = []
    for clip_ids in CLIP_PAIRS:
        pair_parts = [
            df[df['clip_id'].isin(clip_ids)]
            for df in all_fold_results
        ]
        combined_clip = pd.concat(pair_parts, ignore_index=True)
        subject_results = subject_majority_vote(combined_clip, threshold=0.55)

        # Subject-level ACC / AUC
        subject_acc = (subject_results['true_label'] == subject_results['pred_label']).mean()
        try:
            subject_auc = roc_auc_score(subject_results['true_label'], subject_results['bj_ratio'])
        except ValueError:
            subject_auc = None

        # Clip-level ACC / AUC
        clip_acc = (combined_clip['true_label'] == combined_clip['pred_label']).mean()
        try:
            clip_auc = roc_auc_score(combined_clip['true_label'], combined_clip['prob_1'])
        except ValueError:
            clip_auc = None

        group_label = f"1-{clip_ids[-1]}"
        rows.append({
            'pair':      group_label,
            'subj_acc':  subject_acc,
            'subj_auc':  subject_auc,
            'clip_acc':  clip_acc,
            'clip_auc':  clip_auc,
        })

        subj_auc_s = f'{subject_auc:.4f}' if subject_auc is not None else ' N/A'
        clip_auc_s = f'{clip_auc:.4f}' if clip_auc is not None else ' N/A'
        print(f"  clip 1-{clip_ids[-1]:<2}:  subject ACC={subject_acc:.4f} AUC={subj_auc_s}  |  "
              f"clip ACC={clip_acc:.4f} AUC={clip_auc_s}")

    # ── Step 3: 汇总表格 ──
    results_df = pd.DataFrame(rows)

    print(f"\n{'='*90}")
    print("汇总表格")
    print(f"{'='*90}")
    header = f"{'pair':>10}  {'subj_acc':>9}  {'subj_auc':>9}  {'clip_acc':>9}  {'clip_auc':>9}"
    print(header)
    print('-' * len(header))
    for _, r in results_df.iterrows():
        fmt = lambda v: f'{v:.4f}' if pd.notna(v) else '   N/A'
        print(f"{r['pair']:>10}  {fmt(r['subj_acc']):>9}  {fmt(r['subj_auc']):>9}  "
              f"{fmt(r['clip_acc']):>9}  {fmt(r['clip_auc']):>9}")

    print(f"\n均值 ± 标准差:")
    for col, label in [('subj_acc', 'subject ACC'), ('subj_auc', 'subject AUC'),
                       ('clip_acc', 'clip ACC'),   ('clip_auc', 'clip AUC')]:
        mean_v = results_df[col].mean()
        std_v  = results_df[col].std()
        print(f"  {label:>15}: {mean_v:.4f} ± {std_v:.4f}")

    return results_df


if __name__ == '__main__':
    results_df = main()
