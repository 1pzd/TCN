import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from collections import Counter

from src.data_loader import compute_gaze_features, FEATURE_NAMES
from src.model import TCNClassifier


class GazeDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = torch.FloatTensor(sequences)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]


def validate_gaze_data(df):
    required = ['gaze_x', 'gaze_y']
    for col in required:
        if col not in df.columns:
            raise ValueError(f"缺少必要列: {col}，输入数据必须包含 gaze_x, gaze_y")
    if 'timestamp' not in df.columns:
        df['timestamp'] = np.arange(len(df)) / 30.0
        print("  [提示] 未提供 timestamp，按 30Hz 自动生成")
    if 'subject_id' not in df.columns:
        df['subject_id'] = 'unknown'
        print("  [提示] 未提供 subject_id，默认为 'unknown'")
    if 'clip_id' not in df.columns:
        df['clip_id'] = 0
        print("  [提示] 未提供 clip_id，将所有数据视为一个 clip")
    return df


def preprocess_gaze(df, feature_cols, max_seq_len=200, min_clip_len=10):
    df = df.copy()
    df['unique_subject'] = df['subject_id'].astype(str)
    df['unique_clip'] = df['unique_subject'] + '_clip_' + df['clip_id'].astype(str)
    df = df.sort_values(['unique_clip', 'timestamp']).reset_index(drop=True)

    sequences = []
    clip_info = []

    for unique_clip, group in df.groupby('unique_clip'):
        group = group.sort_values('timestamp')
        subject = group['unique_subject'].iloc[0]
        clip_id = group['clip_id'].iloc[0]

        if len(group) < min_clip_len:
            print(f"  [跳过] clip {unique_clip}: 仅{len(group)}帧，不足{min_clip_len}帧")
            continue

        all_feats = compute_gaze_features(group)
        col_indices = [FEATURE_NAMES.index(col) for col in feature_cols]
        values = all_feats[:, col_indices].astype(np.float32)

        if len(values) > max_seq_len:
            values = values[:max_seq_len]
        else:
            pad_len = max_seq_len - len(values)
            pad = np.zeros((pad_len, values.shape[1]), dtype=np.float32)
            values = np.vstack([values, pad])

        sequences.append(values)
        clip_info.append((subject, clip_id, unique_clip))

    if len(sequences) == 0:
        raise ValueError(f"没有有效的 clip（最短帧数={min_clip_len}），请提供更长的数据")

    sequences = np.array(sequences)
    print(f"  预处理完成: {len(sequences)} clips, shape={sequences.shape}")
    return sequences, clip_info


def load_model(model_path='output_model/tcn_model.pth', device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    model = TCNClassifier(
        input_size=checkpoint['input_size'],
        num_channels=checkpoint['num_channels'],
        num_classes=checkpoint['num_classes'],
        kernel_sizes=checkpoint.get('kernel_sizes', 3),
        dropout=checkpoint.get('dropout', 0.2)
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"  模型加载完成: {model_path} ({sum(p.numel() for p in model.parameters()):,} 参数)")
    return model, device


def predict_gaze(model, sequences, batch_size=32, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = GazeDataset(sequences)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_probs = []
    all_preds = []

    with torch.no_grad():
        for inputs in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    return np.array(all_preds), np.array(all_probs)


def subject_vote(clip_info, all_preds, all_probs, threshold=0.5):
    subject_clips = {}
    for (subject, clip_id, unique_clip), pred, prob in zip(clip_info, all_preds, all_probs):
        if subject not in subject_clips:
            subject_clips[subject] = {'preds': [], 'probs': [], 'clip_ids': []}
        subject_clips[subject]['preds'].append(pred)
        subject_clips[subject]['probs'].append(prob)
        subject_clips[subject]['clip_ids'].append(clip_id)

    results = []
    for subject, data in subject_clips.items():
        votes = data['preds']
        probs = np.array(data['probs'])
        clip_ids = data['clip_ids']

        vote_counts = Counter(votes)
        total = len(votes)
        bj_ratio = vote_counts.get(1, 0) / total if total > 0 else 0
        final_pred = 1 if bj_ratio >= threshold else 0

        mean_prob_1 = float(np.mean(probs[:, 1]))
        confidence = max(bj_ratio, 1 - bj_ratio)
        tie = abs(bj_ratio - 0.5) < 1e-6

        results.append({
            'subject': subject,
            'pred_group': 'BJ(强组)' if final_pred == 1 else 'ZJ(弱组)',
            'pred_label': final_pred,
            'vote_BJ(强)': vote_counts.get(1, 0),
            'vote_ZJ(弱)': vote_counts.get(0, 0),
            'total_clips': total,
            'BJ_ratio': round(bj_ratio, 3),
            'mean_confidence': round(np.mean(probs.max(axis=1)), 4),
            'mean_prob_BJ': round(mean_prob_1, 4),
            'confidence': round(confidence, 3),
            'tie': tie
        })

    return pd.DataFrame(results)


def run_gaze_pipeline(gaze_csv_path=None, gaze_df=None, feature_cols=None,
                      model_path='output_model/tcn_model.pth', batch_size=32,
                      max_seq_len=200, min_clip_len=10, threshold=0.5):
    print("=" * 50)
    print("注视点数据分析流水线")
    print("=" * 50)

    if feature_cols is None:
        feature_cols = ['gaze_x', 'gaze_y', 'gaze_vel', 'gaze_vel_x',
                        'gaze_vel_y', 'gaze_acc', 'disp_x', 'disp_y']

    if gaze_csv_path is not None:
        print(f"\n[1/4] 加载数据: {gaze_csv_path}")
        df = pd.read_csv(gaze_csv_path)
    elif gaze_df is not None:
        print(f"\n[1/4] 使用已有 DataFrame ({len(gaze_df)} 行)")
        df = gaze_df.copy()
    else:
        raise ValueError("必须提供 gaze_csv_path 或 gaze_df")

    print(f"\n[2/4] 数据验证与预处理")
    df = validate_gaze_data(df)
    sequences, clip_info = preprocess_gaze(df, feature_cols, max_seq_len, min_clip_len)

    print(f"\n[3/4] 加载模型")
    model, device = load_model(model_path)

    print(f"\n[4/4] 推理与投票")
    all_preds, all_probs = predict_gaze(model, sequences, batch_size, device)
    subject_results = subject_vote(clip_info, all_preds, all_probs, threshold)

    print_clip_details(clip_info, all_preds, all_probs)
    print_subject_results(subject_results)

    return subject_results, clip_info, all_preds, all_probs


def print_clip_details(clip_info, all_preds, all_probs):
    print(f"\n{'='*70}")
    print("Clip 级预测详情")
    print(f"{'='*70}")
    print(f"{'受试者':>12} | {'ClipID':>5} | {'预测标签':>8} | {'P(弱/0)':>8} | {'P(强/1)':>8} | {'概率差':>7}")
    print("-" * 70)
    for (subject, clip_id, _), pred, prob in zip(clip_info, all_preds, all_probs):
        label_name = 'BJ(强)' if pred == 1 else 'ZJ(弱)'
        conf_gap = abs(prob[1] - prob[0])
        print(f"{subject:>12} | {clip_id:>5} | {label_name:>8} | {prob[0]:>8.4f} | {prob[1]:>8.4f} | {conf_gap:>7.4f}")
    print("-" * 70)


def print_subject_results(subject_results):
    print(f"\n{'='*50}")
    print("受试者级投票决策结果")
    print(f"{'='*50}")
    for _, row in subject_results.iterrows():
        tie_mark = ' [平局]' if row['tie'] else ''
        print(f"  {row['subject']:>12}: {row['pred_group']}  "
              f"BJ(强):{row['vote_BJ(强)']} vs ZJ(弱):{row['vote_ZJ(弱)']}  "
              f"(BJ占比{row['BJ_ratio']:.1%})  "
              f"置信度={row['confidence']:.1%}{tie_mark}")

    if len(subject_results) > 1:
        print(f"\n  共 {len(subject_results)} 位受试者")

    return subject_results


if __name__ == '__main__':
    example_csv = sys.argv[1] if len(sys.argv) > 1 else None
    if example_csv is None:
        print("用法: python gaze_input_pipeline.py <新受试者CSV文件路径>")
        print("")
        print("CSV文件应包含以下列:")
        print("  - gaze_x: 注视点X坐标 (必需)")
        print("  - gaze_y: 注视点Y坐标 (必需)")
        print("  - timestamp: 时间戳 (可选, 自动生成)")
        print("  - subject_id: 受试者编号 (可选, 默认为 'unknown')")
        print("  - clip_id: 片段编号 (可选, 默认为 0)")
        print("")
        print("示例:")
        print("  python gaze_input_pipeline.py data/new_subject.csv")
        sys.exit(1)
    run_gaze_pipeline(gaze_csv_path=example_csv)
