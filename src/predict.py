import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from collections import Counter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from src.trainer import EyeTrackingDataset
from src.data_loader import compute_gaze_features, FEATURE_NAMES


def predict_clips(model, df, feature_cols, max_seq_len, min_clip_len, batch_size, device,
                  restrict_clips=None):
    model.eval()
    sequences_list = []
    clip_info_list = []

    for unique_clip, group in df.groupby('unique_clip'):
        if restrict_clips is not None and unique_clip not in restrict_clips:
            continue
        group = group.sort_values('timestamp')
        subject = group['unique_subject'].iloc[0]
        clip_id = group['clip_id'].iloc[0]
        label = group['label'].iloc[0]

        if len(group) < min_clip_len:
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

        sequences_list.append(values)
        clip_info_list.append((subject, clip_id, unique_clip, label))

    sequences = np.array(sequences_list)
    dataset = EyeTrackingDataset(
        torch.FloatTensor(sequences),
        torch.LongTensor([ci[3] for ci in clip_info_list])
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

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
    for (subject, clip_id, unique_clip, true_label), pred, prob in zip(
            clip_info_list, all_preds, all_probs):
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


def subject_majority_vote(clip_results, threshold=0.5):
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
        total = len(votes)
        bj_ratio = vote_counts.get(1, 0) / total if total > 0 else 0
        final_pred = 1 if bj_ratio >= threshold else 0
        tie = abs(bj_ratio - 0.5) < 1e-6

        decision_results.append({
            'unique_subject': subject,
            'true_label': true_label,
            'pred_label': final_pred,
            'vote_0': vote_counts.get(0, 0),
            'vote_1': vote_counts.get(1, 0),
            'total_clips': len(votes),
            'bj_ratio': round(bj_ratio, 3),
            'tie': tie
        })

    return pd.DataFrame(decision_results)


def predict_from_data(model, sequences, clip_info, batch_size, device):
    model.eval()
    dataset = EyeTrackingDataset(sequences, torch.LongTensor([0] * len(sequences)))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

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
    for (subject, clip_id, unique_clip, true_label), pred, prob in zip(
            clip_info, all_preds, all_probs):
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


def print_clip_details(clip_results):
    clip_results = clip_results.sort_values('clip_id').reset_index(drop=True)
    print(f"\n{'='*80}")
    print("Clip 级预测详情")
    print(f"{'='*80}")
    print(f"{'受试者':>14} | {'ClipID':>5} | {'真实标签':>8} | {'预测标签':>8} | {'P(弱/0)':>8} | {'P(强/1)':>8} | {'结果':>6}")
    print("-" * 80)
    correct = 0
    for _, row in clip_results.iterrows():
        true_label = row['true_label']
        pred_label = row['pred_label']
        true_name = 'BJ(强)' if true_label == 1 else 'ZJ(弱)'
        pred_name = 'BJ(强)' if pred_label == 1 else 'ZJ(弱)'
        correct_mark = '[OK]' if true_label == pred_label else '[NO]'
        if true_label == pred_label:
            correct += 1
        print(f"{row['unique_subject']:>14} | {row['clip_id']:>5} | {true_name:>8} | {pred_name:>8} | "
              f"{row['prob_0']:>8.4f} | {row['prob_1']:>8.4f} | {correct_mark:>6}")
    print("-" * 80)
    print(f"Clip 级准确率: {correct}/{len(clip_results)} = {correct/len(clip_results)*100:.2f}%")
    return correct / len(clip_results)


def print_subject_results(subject_results):
    print(f"\n{'='*50}")
    print("受试者级 Bagging 投票决策:")
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

    tie_count = subject_results['tie'].sum()
    if tie_count > 0:
        print(f"  平局受试者: {tie_count} (预测为 0)")

    print(f"\n分类汇总:")
    for true_label, label_name in [(0, 'ZJ(弱)'), (1, 'BJ(强)')]:
        subset = subject_results[subject_results['true_label'] == true_label]
        correct_sub = (subset['true_label'] == subset['pred_label']).sum()
        print(f"  {label_name}: {correct_sub}/{len(subset)} ({correct_sub/len(subset)*100:.1f}%)")


def print_clip_results(test_labels, test_preds):
    print(f"\n{'='*50}")
    print("Clip级测试结果:")
    print(f"  准确率:  {accuracy_score(test_labels, test_preds):.4f}")
    print(f"  精确率:  {precision_score(test_labels, test_preds):.4f}")
    print(f"  召回率:  {recall_score(test_labels, test_preds):.4f}")
    print(f"  F1分数:  {f1_score(test_labels, test_preds):.4f}")
    cm = confusion_matrix(test_labels, test_preds)
    print(f"  混淆矩阵: TN={cm[0,0]} FP={cm[0,1]} FN={cm[1,0]} TP={cm[1,1]}")
