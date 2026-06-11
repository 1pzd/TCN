import os
import numpy as np
import torch

from src.config import load_config
from src.data_loader import load_and_preprocess
from src.model import TCNClassifier
from src.predict import predict_clips, subject_majority_vote, print_subject_results


def main():
    cfg = load_config()
    np.random.seed(cfg["random_state"])
    torch.manual_seed(cfg["random_state"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    checkpoint_path = "output_model/tcn_model.pth"
    if not os.path.exists(checkpoint_path):
        print(f"[错误] 模型文件不存在: {checkpoint_path}")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = TCNClassifier(
        input_size=checkpoint["input_size"],
        num_channels=checkpoint["num_channels"],
        num_classes=checkpoint["num_classes"],
        kernel_sizes=checkpoint.get("kernel_sizes", 3),
        dropout=checkpoint["dropout"],
        pooling_mode=checkpoint.get("pooling_mode", "gap_gmp"),
        classifier_hidden_size=checkpoint.get("classifier_hidden_size"),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")

    df = load_and_preprocess(cfg["data"]["bj_path"], cfg["data"]["zj_path"])

    clip_results = predict_clips(
        model, df,
        cfg["model"]["feature_cols"], cfg["model"]["max_seq_len"],
        cfg["model"]["min_clip_len"], cfg["training"]["batch_size"], device
    )

    vote_threshold = cfg.get("vote", {}).get("threshold", 0.5)
    subject_results = subject_majority_vote(clip_results, threshold=vote_threshold)
    print_subject_results(subject_results)

    return clip_results, subject_results


if __name__ == "__main__":
    clip_results, subject_results = main()
