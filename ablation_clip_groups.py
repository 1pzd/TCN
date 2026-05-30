import os
import sys
import itertools
import pandas as pd
import torch
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

from src.config import load_config
from src.model import TCNClassifier
from src.predict import predict_from_data, subject_majority_vote

ALL_CLIP_IDS = list(range(1, 21))


def evaluate_combination(combined_clip, clip_ids, threshold=0.55):
    filtered = combined_clip[combined_clip['clip_id'].isin(clip_ids)].copy()
    if len(filtered) == 0:
        return None
    subject_results = subject_majority_vote(filtered, threshold=threshold)
    if len(subject_results) == 0:
        return None

    correct = (subject_results['true_label'] == subject_results['pred_label']).sum()
    total = len(subject_results)
    acc = correct / total

    try:
        f1 = f1_score(subject_results['true_label'], subject_results['pred_label'])
    except Exception:
        f1 = 0.0

    try:
        auc = roc_auc_score(subject_results['true_label'], subject_results['bj_ratio'])
    except ValueError:
        auc = float('nan')

    return {
        'clip_ids': clip_ids,
        'accuracy': acc,
        'f1': f1,
        'auc': auc,
        'correct': correct,
        'total': total,
        'n_subjects': total,
        'n_clips': len(filtered)
    }


def main():
    cfg = load_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

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
        all_fold_results.append(fold_results)
        print(f"折{fold+1}: {len(fold_results)} clips")

    combined_clip = pd.concat(all_fold_results, ignore_index=True)
    available_ids = sorted(combined_clip['clip_id'].unique().tolist())
    usable_ids = [cid for cid in ALL_CLIP_IDS if cid in available_ids]
    print(f"\n可用 clip_id: {usable_ids}")
    print(f"总计 clips: {len(combined_clip)}")

    threshold = cfg['training'].get('threshold', 0.55)

    for group_name, group_size in [("两两一组 (2 clips)", 2), ("三三一组 (3 clips)", 3)]:
        print(f"\n{'='*60}")
        print(f"消融实验: {group_name}")
        print(f"{'='*60}")

        all_combos = list(itertools.combinations(usable_ids, group_size))
        print(f"组合总数: {len(all_combos)}")

        results = []
        for i, combo in enumerate(all_combos):
            combo_ids = list(combo)
            res = evaluate_combination(combined_clip, combo_ids, threshold=threshold)
            if res is not None:
                results.append(res)
            if (i + 1) % 100 == 0:
                print(f"  进度: {i+1}/{len(all_combos)}")

        if not results:
            print("  无有效结果")
            continue

        results.sort(key=lambda x: x['accuracy'], reverse=True)
        top3 = results[:3]

        print(f"\n排名前三:")
        for rank, r in enumerate(top3, 1):
            print(f"\n  第{rank}名: clip_ids = {r['clip_ids']}")
            print(f"    受试者准确率: {r['accuracy']:.4f} ({r['correct']}/{r['total']})")
            print(f"    受试者 F1:     {r['f1']:.4f}")
            print(f"    受试者 AUC:    {r['auc']:.4f}" if not np.isnan(r['auc']) else
                  f"    受试者 AUC:    N/A")
            print(f"    受试者数: {r['n_subjects']}, clips: {r['n_clips']}")

        accs = [r['accuracy'] for r in results]
        print(f"\n  统计: 总数={len(results)}, "
              f"最高准确率={max(accs):.4f}, "
              f"中位数={np.median(accs):.4f}, "
              f"最低={min(accs):.4f}")


if __name__ == '__main__':
    main()
