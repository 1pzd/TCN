import os
import sys
import numpy as np
import pandas as pd
import torch

from src.config import load_config
from src.model import TCNClassifier
from src.predict import predict_from_data, subject_majority_vote, print_clip_details, print_subject_results


def main():
    cfg = load_config()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"PyTorch: {torch.__version__} | Device: {device}")

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

        test_subjects = sorted(set(s for s, _, _ in test_clip_info))
        print(f"\n  折{fold+1}/3 — 测试集受试者 ({len(test_subjects)}人): {test_subjects}")

        clip_info_with_labels = [
            (subject, clip_id, unique_clip, test_labels[i])
            for i, (subject, clip_id, unique_clip) in enumerate(test_clip_info)
        ]

        fold_results = predict_from_data(model, test_sequences, clip_info_with_labels,
                                         cfg['training']['batch_size'], device)
        all_clip_results.append(fold_results)

        n_correct = (fold_results['true_label'] == fold_results['pred_label']).sum()
        n_total = len(fold_results)
        print(f"  折{fold+1} 测试: {n_total} clips, "
              f"正确={n_correct}, 错误={n_total-n_correct}, "
              f"准确率={n_correct/n_total*100:.2f}%")

        vote_threshold = cfg.get('vote', {}).get('threshold', 0.5)
        fold_subject_results = subject_majority_vote(fold_results, threshold=vote_threshold)
        print_subject_results(fold_subject_results)

    print(f"\n{'='*60}")
    print("3折交叉验证汇总")
    print(f"{'='*60}")

    combined_clip = pd.concat(all_clip_results, ignore_index=True)
    print_clip_details(combined_clip, "汇总")

    vote_threshold = cfg.get('vote', {}).get('threshold', 0.5)
    subject_results = subject_majority_vote(combined_clip, threshold=vote_threshold)
    print_subject_results(subject_results)

    return combined_clip, subject_results


if __name__ == '__main__':
    combined_clip, subject_results = main()
