import os
import sys
import numpy as np
import torch

from src.config import load_config
from src.data_loader import load_and_preprocess
from src.model import TCNClassifier
from src.predict import predict_clips, predict_from_data, subject_majority_vote, print_clip_details, print_subject_results


def main():
    if not os.path.exists('output_model/tcn_model.pth'):
        print("[错误] 未找到模型文件，请先运行 python train.py 进行训练")
        sys.exit(1)

    cfg = load_config()

    np.random.seed(cfg['random_state'])
    torch.manual_seed(cfg['random_state'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"PyTorch: {torch.__version__} | Device: {device}")

    checkpoint = torch.load('output_model/tcn_model.pth', map_location=device, weights_only=False)

    model = TCNClassifier(
        input_size=checkpoint['input_size'],
        num_channels=checkpoint['num_channels'],
        num_classes=checkpoint['num_classes'],
        kernel_sizes=checkpoint.get('kernel_sizes', 3),
        dropout=checkpoint['dropout']
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n模型参数量: {total_params:,}")
    print(f"模型加载完成: output_model/tcn_model.pth")

    test_data = torch.load('output_model/test_clip_data.pt', map_location='cpu', weights_only=False)
    test_sequences = test_data['sequences']
    test_labels = test_data['labels']
    test_clip_info = test_data['clip_info']

    clip_info_with_labels = [
        (subject, clip_id, unique_clip, test_labels[i])
        for i, (subject, clip_id, unique_clip) in enumerate(test_clip_info)
    ]

    print(f"\n测试集: {len(test_sequences)} clips")
    print(f"  标签分布: 0={sum(1 for l in test_labels if l==0)}, 1={sum(1 for l in test_labels if l==1)}")

    test_clip_results = predict_from_data(model, test_sequences, clip_info_with_labels,
                                          cfg['training']['batch_size'], device)
    print_clip_details(test_clip_results)

    df = load_and_preprocess(cfg['data']['bj_path'], cfg['data']['zj_path'])
    all_clip_results = predict_clips(
        model, df, cfg['model']['feature_cols'], cfg['model']['max_seq_len'],
        cfg['model']['min_clip_len'], cfg['training']['batch_size'], device
    )
    vote_threshold = cfg.get('vote', {}).get('threshold', 0.5)
    subject_results = subject_majority_vote(all_clip_results, threshold=vote_threshold)
    print_subject_results(subject_results)

    return test_clip_results, subject_results


if __name__ == '__main__':
    test_clip_results, subject_results = main()
