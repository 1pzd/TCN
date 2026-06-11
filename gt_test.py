import argparse
import os
import sys

import pandas as pd
import torch

from src.config import load_config
from src.data_loader import kfold_split_subjects, load_and_preprocess
from src.model import TCNClassifier
from src.predict import (
    predict_clips,
    print_clip_details,
    print_subject_results,
    subject_majority_vote,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate trained fold models on original GT BJ/ZJ CSVs without subject leakage."
    )
    parser.add_argument("--bj", default="data/dataset_BJ.csv", help="GT BJ CSV path")
    parser.add_argument("--zj", default="data/dataset_ZJ.csv", help="GT ZJ CSV path")
    parser.add_argument("--model-dir", default="output_model", help="Directory with fold checkpoints")
    parser.add_argument("--output-dir", default="output_model", help="Directory for GT result CSVs")
    parser.add_argument("--folds", type=int, default=3, help="Number of folds")
    parser.add_argument(
        "--split-source",
        choices=("saved", "reconstructed"),
        default="saved",
        help=(
            "saved: use held-out subject IDs saved by training; "
            "reconstructed: rebuild folds from current config random_state"
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Subject vote threshold; defaults to config vote.threshold",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only verify files and fold subjects; do not run model inference",
    )
    return parser.parse_args()


def load_model(checkpoint_path, device):
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
    return model


def get_saved_test_subjects(model_dir, fold):
    data_path = os.path.join(model_dir, f"test_clip_data_fold{fold}.pt")
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"缺少 {data_path}，无法确认该 fold 训练时的 held-out 受试者。"
        )

    test_data = torch.load(data_path, map_location="cpu", weights_only=False)
    return sorted({subject for subject, _, _ in test_data["clip_info"]})


def get_reconstructed_test_subjects(df, cfg, folds):
    fold_info = kfold_split_subjects(df, n_splits=folds, random_state=cfg["random_state"])
    return {
        fold + 1: sorted(fold_info[fold_info["fold"] == fold]["unique_subject"].tolist())
        for fold in range(folds)
    }


def assert_no_subject_leakage(all_subjects, fold_to_subjects):
    seen = set()
    for fold, subjects in fold_to_subjects.items():
        subject_set = set(subjects)
        repeated = seen & subject_set
        if repeated:
            raise AssertionError(f"Fold {fold} 测试受试者与其他 fold 重复: {sorted(repeated)}")
        seen.update(subject_set)

    missing = set(all_subjects) - seen
    extra = seen - set(all_subjects)
    if missing:
        raise AssertionError(f"GT 数据中有受试者没有被任何 fold 测试: {sorted(missing)}")
    if extra:
        raise AssertionError(f"Fold 测试受试者不在 GT 数据中: {sorted(extra)}")


def main():
    args = parse_args()
    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vote_threshold = args.threshold
    if vote_threshold is None:
        vote_threshold = cfg.get("vote", {}).get("threshold", 0.5)

    print(f"PyTorch: {torch.__version__} | Device: {device}")
    print("模式: 只测试已训练模型，不训练、不微调")
    print(f"GT 数据: BJ={args.bj}, ZJ={args.zj}")
    print(f"防泄露策略: split-source={args.split_source}")

    df = load_and_preprocess(args.bj, args.zj)
    all_subjects = sorted(df["unique_subject"].unique().tolist())

    if args.split_source == "saved":
        print("使用 output_model/test_clip_data_fold*.pt 中的 held-out 受试者 ID。")
        print("注意: 这里只用 subject ID，预测数据仍从原始 GT CSV 重新构建。")
        fold_to_subjects = {
            fold: get_saved_test_subjects(args.model_dir, fold)
            for fold in range(1, args.folds + 1)
        }
    else:
        print("按当前 config random_state 重新构建 subject-level folds。")
        fold_to_subjects = get_reconstructed_test_subjects(df, cfg, args.folds)

    assert_no_subject_leakage(all_subjects, fold_to_subjects)

    for fold in range(1, args.folds + 1):
        model_path = os.path.join(args.model_dir, f"tcn_model_fold{fold}.pth")
        if not os.path.exists(model_path):
            print(f"[错误] 模型不存在: {model_path}")
            sys.exit(1)

        subjects = fold_to_subjects[fold]
        labels = df[df["unique_subject"].isin(subjects)][["unique_subject", "label"]].drop_duplicates()
        n_bj = int((labels["label"] == 1).sum())
        n_zj = int((labels["label"] == 0).sum())
        print(f"\nFold {fold}/{args.folds} held-out GT 受试者 ({len(subjects)}人):")
        print(f"  BJ={n_bj}, ZJ={n_zj}")
        print(f"  {subjects}")

    if args.dry_run:
        print("\nDry run 完成: 文件存在、fold 受试者无重叠，未运行预测。")
        return None, None

    all_clip_results = []
    all_subject_results = []

    for fold in range(1, args.folds + 1):
        print(f"\n{'=' * 60}")
        print(f"GT Fold {fold}/{args.folds}")
        print(f"{'=' * 60}")

        model_path = os.path.join(args.model_dir, f"tcn_model_fold{fold}.pth")
        model = load_model(model_path, device)

        test_subjects = fold_to_subjects[fold]
        test_df = df[df["unique_subject"].isin(test_subjects)].reset_index(drop=True)

        clip_results = predict_clips(
            model,
            test_df,
            cfg["model"]["feature_cols"],
            cfg["model"]["max_seq_len"],
            cfg["model"]["min_clip_len"],
            cfg["training"]["batch_size"],
            device,
        )
        clip_results.insert(0, "fold", fold)
        all_clip_results.append(clip_results)
        print_clip_details(clip_results, title=f"GT Fold {fold} Clip级")

        subject_results = subject_majority_vote(clip_results, threshold=vote_threshold)
        subject_results.insert(0, "fold", fold)
        all_subject_results.append(subject_results)
        print_subject_results(subject_results)

    combined_clip = pd.concat(all_clip_results, ignore_index=True)
    combined_subject = pd.concat(all_subject_results, ignore_index=True)

    print(f"\n{'=' * 60}")
    print("GT 三折 held-out 汇总")
    print(f"{'=' * 60}")
    print_clip_details(combined_clip, title="GT 汇总 Clip级")
    print_subject_results(combined_subject)

    os.makedirs(args.output_dir, exist_ok=True)
    clip_out = os.path.join(args.output_dir, "gt_clip_results.csv")
    subject_out = os.path.join(args.output_dir, "gt_subject_results.csv")
    combined_clip.to_csv(clip_out, index=False, encoding="utf-8-sig")
    combined_subject.to_csv(subject_out, index=False, encoding="utf-8-sig")
    print("\n结果已保存:")
    print(f"  {clip_out}")
    print(f"  {subject_out}")

    return combined_clip, combined_subject


if __name__ == "__main__":
    main()
