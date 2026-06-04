"""
生成前端可视化所需的数据文件 (frontend_data.json)
用法: python generate_data.py
前置条件: 先运行 python train.py 训练模型
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from src.config import load_config
from src.model import TCNClassifier
from src.predict import predict_from_data, subject_majority_vote


# Clip 分组定义（与 ablation_clip_groups.py 一致）
CLIP_GROUPS = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
    [10, 11, 12],
    [13, 14, 15],
    [16, 17, 18],
    [19, 20, 1],
]


def load_fold(fold, device):
    """加载指定折的模型和测试数据"""
    model_path = f'output_model/tcn_model_fold{fold}.pth'
    data_path = f'output_model/test_clip_data_fold{fold}.pt'

    if not os.path.exists(model_path) or not os.path.exists(data_path):
        print(f"[错误] 折{fold} 文件不完整，请先运行 python train.py")
        sys.exit(1)

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = TCNClassifier(
        input_size=checkpoint['input_size'],
        num_channels=checkpoint['num_channels'],
        num_classes=checkpoint['num_classes'],
        kernel_sizes=checkpoint.get('kernel_sizes', 3),
        dropout=checkpoint['dropout'],
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    test_data = torch.load(data_path, map_location='cpu', weights_only=False)
    return model, test_data


def get_fold_predictions(model, test_data, batch_size, device):
    """获取单折的 clip 级预测结果"""
    test_sequences = test_data['sequences']
    test_labels = test_data['labels']
    test_clip_info = test_data['clip_info']

    clip_info_with_labels = [
        (subject, clip_id, unique_clip, test_labels[i])
        for i, (subject, clip_id, unique_clip) in enumerate(test_clip_info)
    ]

    return predict_from_data(
        model, test_sequences, clip_info_with_labels, batch_size, device
    )


def compute_metrics(clip_df, threshold=0.55):
    """从 clip 级 DataFrame 计算全部指标，返回 JSON 可序列化的 dict"""
    subject_df = subject_majority_vote(clip_df, threshold=threshold)

    # clip 级
    clip_acc = float((clip_df['true_label'] == clip_df['pred_label']).mean())
    try:
        clip_auc = float(roc_auc_score(clip_df['true_label'], clip_df['prob_1']))
    except ValueError:
        clip_auc = None

    # 受试者级
    subject_acc = float((subject_df['true_label'] == subject_df['pred_label']).mean())
    try:
        subject_auc = float(roc_auc_score(subject_df['true_label'], subject_df['bj_ratio']))
    except ValueError:
        subject_auc = None

    # 受试者详情列表
    subjects = []
    for _, row in subject_df.iterrows():
        subjects.append({
            'id': str(row['unique_subject']),
            'true_label': int(row['true_label']),
            'pred_label': int(row['pred_label']),
            'vote_0': int(row['vote_0']),
            'vote_1': int(row['vote_1']),
            'total_clips': int(row['total_clips']),
            'bj_ratio': round(float(row['bj_ratio']), 3),
            'result': 'correct' if row['true_label'] == row['pred_label'] else 'wrong',
        })

    # clip 详情列表
    clips = []
    for _, row in clip_df.iterrows():
        clips.append({
            'subject_id': str(row['unique_subject']),
            'clip_id': int(row['clip_id']),
            'true_label': int(row['true_label']),
            'pred_label': int(row['pred_label']),
            'prob_0': round(float(row['prob_0']), 4),
            'prob_1': round(float(row['prob_1']), 4),
            'result': 'correct' if row['true_label'] == row['pred_label'] else 'wrong',
        })

    return {
        'clip_acc': round(clip_acc, 4),
        'clip_auc': round(clip_auc, 4) if clip_auc is not None else None,
        'subject_acc': round(subject_acc, 4),
        'subject_auc': round(subject_auc, 4) if subject_auc is not None else None,
        'total_clips': int(len(clip_df)),
        'total_subjects': int(len(subject_df)),
        'bj_subjects': int((subject_df['true_label'] == 1).sum()),
        'zj_subjects': int((subject_df['true_label'] == 0).sum()),
        'subjects': subjects,
        'clips': clips,
    }


def main():
    cfg = load_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    threshold = cfg.get('vote', {}).get('threshold', 0.55)
    batch_size = cfg['training']['batch_size']

    print(f"PyTorch: {torch.__version__} | Device: {device} | Threshold: {threshold}")

    # ── Step 1: 3 折预测 ──
    print(f"\n{'='*60}")
    print("第1步：运行3折预测")
    print(f"{'='*60}")

    all_fold_results = []
    for fold in range(1, 4):
        model, test_data = load_fold(fold, device)
        fold_results = get_fold_predictions(model, test_data, batch_size, device)
        print(f"  折{fold}: {len(fold_results)} clips")
        all_fold_results.append(fold_results)

    # 合并所有折结果
    all_combined = pd.concat(all_fold_results, ignore_index=True)
    print(f"\n  合计: {len(all_combined)} clips, {all_combined['unique_subject'].nunique()} 受试者")

    # ── Step 2: 全部数据指标 ──
    print(f"\n{'='*60}")
    print("第2步：计算全部数据指标")
    print(f"{'='*60}")

    all_metrics = compute_metrics(all_combined, threshold)
    print(f"  Clip级:  ACC={all_metrics['clip_acc']:.4f}  AUC={all_metrics['clip_auc']:.4f}")
    print(f"  受试者级: ACC={all_metrics['subject_acc']:.4f}  AUC={all_metrics['subject_auc']:.4f}")

    # ── Step 3: 各分组指标 ──
    print(f"\n{'='*60}")
    print("第3步：计算各分组指标")
    print(f"{'='*60}")

    groups = {}
    for i, group_ids in enumerate(CLIP_GROUPS, 1):
        group_key = f'group{i}'
        group_clip = all_combined[all_combined['clip_id'].isin(group_ids)]

        if len(group_clip) == 0:
            print(f"  {group_key} {tuple(group_ids)}: 无数据")
            groups[group_key] = None
            continue

        group_metrics = compute_metrics(group_clip, threshold)
        groups[group_key] = group_metrics
        print(f"  {group_key} {tuple(group_ids)}: "
              f"clip_acc={group_metrics['clip_acc']:.4f}  "
              f"subj_acc={group_metrics['subject_acc']:.4f}  "
              f"({group_metrics['total_clips']} clips)")

    # ── Step 4: 保存 JSON ──
    output = {
        'threshold': threshold,
        'all': all_metrics,
        'groups': groups,
        'clip_groups': CLIP_GROUPS,
    }

    output_path = 'front/frontend_data.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    file_size = os.path.getsize(output_path)
    print(f"\n{'='*60}")
    print(f"数据已保存到 {output_path} ({file_size / 1024:.1f} KB)")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
