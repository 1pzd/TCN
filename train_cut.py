import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.config import load_config
from src.data_loader import create_sequences, kfold_split_subjects, split_inner_val
from src.model import TCNClassifier
from src.predict import (
    predict_clips,
    print_clip_results,
    print_subject_results,
    subject_majority_vote,
)
from src.trainer import EyeTrackingDataset, evaluate, train_model


GROUPS = (
    ("EN_BJ", "BJ", 1),
    ("EN_ZJ", "ZJ", 0),
)


def find_column(columns, exact_names=None, contains=None, required=True):
    exact_names = exact_names or []
    normalized = {str(col).strip().lower(): col for col in columns}

    for name in exact_names:
        match = normalized.get(name.strip().lower())
        if match is not None:
            return match

    if contains:
        needles = [item.lower() for item in contains]
        for col in columns:
            lowered = str(col).strip().lower()
            if all(needle in lowered for needle in needles):
                return col

    if required:
        expected = exact_names or contains
        raise ValueError(f"Missing required column: {expected}")
    return None


def parse_en_filename(path):
    parts = path.stem.split("_")
    if len(parts) < 3:
        raise ValueError(
            f"Expected filename like subject_clip_correct_*.csv, got: {path.name}"
        )

    try:
        subject_id = int(parts[0])
        clip_id = int(parts[1])
        answer_correct = int(parts[2])
    except ValueError as exc:
        raise ValueError(f"Cannot parse subject/clip/correct flag from {path.name}") from exc

    if answer_correct not in (0, 1):
        raise ValueError(f"Correct flag must be 0 or 1 in {path.name}")

    return subject_id, clip_id, answer_correct


def downsample_clip(df, sample_rate):
    if sample_rate is None or sample_rate <= 0:
        return df

    interval = 1.0 / sample_rate
    sorted_df = df.sort_values("timestamp").reset_index(drop=True).copy()
    buckets = np.floor(sorted_df["timestamp"].to_numpy() / interval).astype(np.int64)
    return sorted_df.assign(_sample_bucket=buckets).groupby("_sample_bucket", as_index=False).first().drop(columns="_sample_bucket")


def normalize_en_file(path, dataset, dataset_label, data_root, sample_rate):
    subject_id, clip_id, answer_correct = parse_en_filename(path)
    raw = pd.read_csv(path, low_memory=False)

    recording_ts_col = find_column(raw.columns, ["Recording timestamp"])
    computer_ts_col = find_column(raw.columns, ["Computer timestamp"], required=False)
    eyetracker_ts_col = find_column(raw.columns, ["Eyetracker timestamp"], required=False)
    gaze_x_col = find_column(raw.columns, ["Gaze point X (MCSnorm)"], contains=["gaze point x", "mcsnorm"])
    gaze_y_col = find_column(raw.columns, ["Gaze point Y (MCSnorm)"], contains=["gaze point y", "mcsnorm"])
    validity_left_col = find_column(raw.columns, ["Validity left"], contains=["validity", "left"])
    validity_right_col = find_column(raw.columns, ["Validity right"], contains=["validity", "right"])
    movement_col = find_column(raw.columns, ["Eye movement type"], contains=["eye movement type"], required=False)

    recording_ts = pd.to_numeric(raw[recording_ts_col], errors="coerce")
    first_ts = recording_ts.dropna().iloc[0] if recording_ts.notna().any() else 0
    timestamp = (recording_ts - first_ts) / 1_000_000.0

    gaze_x = pd.to_numeric(raw[gaze_x_col], errors="coerce")
    gaze_y = pd.to_numeric(raw[gaze_y_col], errors="coerce")
    valid_left = raw[validity_left_col].astype(str).str.strip().str.lower().eq("valid")
    valid_right = raw[validity_right_col].astype(str).str.strip().str.lower().eq("valid")
    valid_gaze = gaze_x.between(0, 1) & gaze_y.between(0, 1) & (valid_left | valid_right)

    if movement_col is None:
        original_type = pd.Series("Unclassified", index=raw.index)
    else:
        original_type = raw[movement_col].fillna("Unclassified").astype(str).str.strip()
        original_type = original_type.replace("", "Unclassified")

    rel_path = path.relative_to(data_root).as_posix()

    out = pd.DataFrame({
        "subject_id": subject_id,
        "clip_id": clip_id,
        "answer_correct": answer_correct,
        "dataset": dataset,
        "dataset_label": dataset_label,
        "label": dataset_label,
        "frame_path": rel_path,
        "source_file": rel_path,
        "timestamp": timestamp,
        "gaze_x": gaze_x.where(valid_gaze, -1.0),
        "gaze_y": gaze_y.where(valid_gaze, -1.0),
        "validity": valid_gaze.astype(int),
        "original_type": original_type,
    })

    if computer_ts_col is not None:
        out["computer_timestamp"] = pd.to_numeric(raw[computer_ts_col], errors="coerce")
        out["eye_tracker_timestamp"] = out["computer_timestamp"]
    if eyetracker_ts_col is not None:
        out["tobii_eyetracker_timestamp"] = pd.to_numeric(raw[eyetracker_ts_col], errors="coerce")

    out = out[out["timestamp"].notna()].reset_index(drop=True)
    out = downsample_clip(out, sample_rate)
    return out


def load_en_ground_truth(data_root, sample_rate=30.0):
    data_root = Path(data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"EN data root not found: {data_root}")

    frames = []
    raw_file_count = 0
    for group_dir, dataset, dataset_label in GROUPS:
        root = data_root / group_dir
        if not root.exists():
            raise FileNotFoundError(f"Expected group folder not found: {root}")

        for path in sorted(root.glob("*/*.csv"), key=lambda p: str(p)):
            raw_file_count += 1
            try:
                frames.append(
                    normalize_en_file(
                        path=path,
                        dataset=dataset,
                        dataset_label=dataset_label,
                        data_root=data_root,
                        sample_rate=sample_rate,
                    )
                )
            except Exception as exc:
                raise RuntimeError(f"Failed to load {path}") from exc

    if not frames:
        raise ValueError(f"No EN CSV files found under {data_root}")

    df = pd.concat(frames, ignore_index=True)
    before_filter = len(df)

    df["unique_subject"] = df["dataset"] + "_" + df["subject_id"].astype(str)
    df["unique_clip"] = df["unique_subject"] + "_clip_" + df["clip_id"].astype(str)

    df = df[df["validity"] == 1].reset_index(drop=True)
    df = df[df["clip_id"] != 0].reset_index(drop=True)
    df = df[df["clip_id"] != -1].reset_index(drop=True)
    df = df[df["clip_id"] != 1].reset_index(drop=True)
    df = df[df["original_type"].isin(["Fixation", "Saccade"])].reset_index(drop=True)
    df = df.sort_values(["unique_clip", "timestamp"]).reset_index(drop=True)

    if df.empty:
        raise ValueError("All EN rows were filtered out; check validity and event columns.")

    print(f"\n{'=' * 50}")
    print("EN ground-truth data loaded")
    print(f"  Data root: {data_root}")
    print(f"  Source files: {raw_file_count}")
    print(f"  Rows before filters: {before_filter}")
    print(f"  Rows after filters: {len(df)}")
    print("  Excluded first question: clip_id == 1")
    print(f"  Sample rate: {'raw' if sample_rate <= 0 else sample_rate}")
    print("  Label source: EN_BJ/EN_ZJ ability class")
    print(f"  Label distribution: 0={len(df[df['label'] == 0])}, 1={len(df[df['label'] == 1])}")
    print(f"  Dataset rows: BJ={len(df[df['dataset'] == 'BJ'])}, ZJ={len(df[df['dataset'] == 'ZJ'])}")
    print(f"  Subjects: BJ={df[df['dataset'] == 'BJ']['subject_id'].nunique()}, ZJ={df[df['dataset'] == 'ZJ']['subject_id'].nunique()}")
    print(f"  Clips: {df['unique_clip'].nunique()}")

    return df


def assert_single_label_per_subject(df):
    label_counts = df.groupby("unique_subject")["label"].nunique()
    mixed_subjects = label_counts[label_counts > 1].index.tolist()
    assert not mixed_subjects, f"Subjects with mixed labels found: {mixed_subjects}"


def assert_no_overlap(name, left_values, right_values):
    overlap = set(left_values) & set(right_values)
    assert not overlap, f"{name} overlap: {sorted(overlap)[:20]}"


def assert_split_no_leakage(train_df, val_df, test_df):
    splits = {
        "train": train_df,
        "val": val_df,
        "test": test_df,
    }
    for left_name, right_name in (("train", "val"), ("train", "test"), ("val", "test")):
        left_df = splits[left_name]
        right_df = splits[right_name]
        pair = f"{left_name}/{right_name}"
        assert_no_overlap(f"{pair} subject", left_df["unique_subject"], right_df["unique_subject"])
        assert_no_overlap(f"{pair} clip", left_df["unique_clip"], right_df["unique_clip"])
        assert_no_overlap(f"{pair} source file", left_df["source_file"], right_df["source_file"])


def build_fold_info(df, random_state):
    assert_single_label_per_subject(df)
    return kfold_split_subjects(df, n_splits=3, random_state=random_state)


def validate_three_fold_no_leakage(df, fold_info, random_state):
    heldout_subjects = []
    for fold in range(3):
        test_subjects = fold_info[fold_info["fold"] == fold]["unique_subject"].tolist()
        train_subjects = fold_info[fold_info["fold"] != fold]["unique_subject"].tolist()
        heldout_subjects.extend(test_subjects)

        inner_train_subjects, inner_val_subjects = split_inner_val(
            df, train_subjects, val_ratio=0.2, random_state=random_state + fold
        )

        inner_train_df = df[df["unique_subject"].isin(inner_train_subjects)].reset_index(drop=True)
        inner_val_df = df[df["unique_subject"].isin(inner_val_subjects)].reset_index(drop=True)
        test_df = df[df["unique_subject"].isin(test_subjects)].reset_index(drop=True)
        assert_split_no_leakage(inner_train_df, inner_val_df, test_df)

    all_subjects = fold_info["unique_subject"].tolist()
    assert len(heldout_subjects) == len(set(heldout_subjects)), "outer test subject repeated across folds"
    assert set(heldout_subjects) == set(all_subjects), "outer test subjects do not cover all subjects"
    print("\nNo leakage detected across 3-fold subject splits.")


def attach_answer_correct(clip_results, source_df):
    answer_correct_df = source_df[["unique_clip", "answer_correct"]].drop_duplicates()
    merged = clip_results.merge(answer_correct_df, on="unique_clip", how="left", validate="one_to_one")
    if merged["answer_correct"].isna().any():
        missing = merged.loc[merged["answer_correct"].isna(), "unique_clip"].tolist()
        raise ValueError(f"Missing answer_correct for clips: {missing[:20]}")
    merged["answer_correct"] = merged["answer_correct"].astype(int)
    return merged


def print_answer_correct_analysis(clip_results):
    print(f"\n{'=' * 50}")
    print("Answer-correct analysis")
    for answer_correct, label in ((1, "answer_correct=1"), (0, "answer_correct=0")):
        subset = clip_results[clip_results["answer_correct"] == answer_correct]
        print(f"\n{label}: {len(subset)} clips")
        if subset.empty:
            continue
        print_clip_results(
            subset["true_label"].to_numpy(),
            subset["pred_label"].to_numpy(),
            subset["prob_1"].to_numpy(),
        )


def build_parser():
    parser = argparse.ArgumentParser(description="Train TCN on EN ground-truth gaze CSV files.")
    parser.add_argument("--data-root", default="data/EN", help="Root containing EN_BJ and EN_ZJ folders.")
    parser.add_argument("--output-dir", default="output_model_cut", help="Directory for saved fold artifacts.")
    parser.add_argument("--sample-rate", type=float, default=30.0, help="Downsample rate in Hz; <=0 keeps raw rows.")
    parser.add_argument("--dry-run", action="store_true", help="Load data and build sequences without training.")
    return parser


def run_dry_run(df, model_cfg, fold_info, random_state):
    validate_three_fold_no_leakage(df, fold_info, random_state)
    sequences, labels, clip_info = create_sequences(
        df,
        model_cfg["feature_cols"],
        model_cfg["max_seq_len"],
        model_cfg["min_clip_len"],
    )
    print("\nDry run complete")
    print(f"  Sequences shape: {sequences.shape}")
    print(f"  Labels shape: {labels.shape}")
    print(f"  Clip info rows: {len(clip_info)}")


def main():
    args = build_parser().parse_args()
    cfg = load_config()

    np.random.seed(cfg["random_state"])
    torch.manual_seed(cfg["random_state"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PyTorch: {torch.__version__} | Device: {device}")

    model_cfg = dict(cfg["model"])

    df = load_en_ground_truth(
        data_root=args.data_root,
        sample_rate=args.sample_rate,
    )

    train_cfg = cfg["training"]
    fold_info = build_fold_info(df, random_state=cfg["random_state"])

    if args.dry_run:
        run_dry_run(df, model_cfg, fold_info, cfg["random_state"])
        return None, None

    all_clip_results = []
    fold_metrics = []
    heldout_subjects = []
    os.makedirs(args.output_dir, exist_ok=True)

    for fold in range(3):
        print(f"\n{'=' * 60}")
        print(f"EN fold {fold + 1}/3")
        print(f"{'=' * 60}")

        test_subjects = fold_info[fold_info["fold"] == fold]["unique_subject"].tolist()
        train_subjects = fold_info[fold_info["fold"] != fold]["unique_subject"].tolist()
        heldout_subjects.extend(test_subjects)
        print(f"  Outer test subjects ({len(test_subjects)}): {sorted(test_subjects)}")
        print(f"  Outer train subjects ({len(train_subjects)}): {sorted(train_subjects)}")

        inner_train_subjects, inner_val_subjects = split_inner_val(
            df, train_subjects, val_ratio=0.2, random_state=cfg["random_state"] + fold
        )

        all_subjects = inner_train_subjects + inner_val_subjects + test_subjects
        assert len(set(inner_train_subjects) & set(inner_val_subjects)) == 0, "inner train/val overlap"
        assert len(set(inner_train_subjects) & set(test_subjects)) == 0, "inner train/test overlap"
        assert len(set(inner_val_subjects) & set(test_subjects)) == 0, "inner val/test overlap"
        print(
            f"  Disjoint check passed: {len(inner_train_subjects)} inner-train + "
            f"{len(inner_val_subjects)} inner-val + {len(test_subjects)} outer-test = "
            f"{len(set(all_subjects))} unique"
        )

        inner_train_df = df[df["unique_subject"].isin(inner_train_subjects)].reset_index(drop=True)
        inner_val_df = df[df["unique_subject"].isin(inner_val_subjects)].reset_index(drop=True)
        test_df = df[df["unique_subject"].isin(test_subjects)].reset_index(drop=True)
        assert_split_no_leakage(inner_train_df, inner_val_df, test_df)

        X_inner_train, y_inner_train, _ = create_sequences(
            inner_train_df,
            model_cfg["feature_cols"],
            model_cfg["max_seq_len"],
            model_cfg["min_clip_len"],
        )
        X_inner_val, y_inner_val, _ = create_sequences(
            inner_val_df,
            model_cfg["feature_cols"],
            model_cfg["max_seq_len"],
            model_cfg["min_clip_len"],
        )
        X_test, y_test, test_clip_info = create_sequences(
            test_df,
            model_cfg["feature_cols"],
            model_cfg["max_seq_len"],
            model_cfg["min_clip_len"],
        )

        inner_train_dataset = EyeTrackingDataset(torch.FloatTensor(X_inner_train), torch.LongTensor(y_inner_train))
        inner_val_dataset = EyeTrackingDataset(torch.FloatTensor(X_inner_val), torch.LongTensor(y_inner_val))
        test_dataset = EyeTrackingDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))

        inner_train_loader = DataLoader(inner_train_dataset, batch_size=train_cfg["batch_size"], shuffle=True)
        inner_val_loader = DataLoader(inner_val_dataset, batch_size=train_cfg["batch_size"], shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=train_cfg["batch_size"], shuffle=False)

        model = TCNClassifier(
            input_size=len(model_cfg["feature_cols"]),
            num_channels=model_cfg["num_channels"],
            num_classes=2,
            kernel_sizes=model_cfg.get("kernel_sizes", 3),
            dropout=model_cfg["dropout"],
        ).to(device)

        total_params = sum(param.numel() for param in model.parameters())
        print(f"\nModel parameters: {total_params:,}")

        model = train_model(model, inner_train_loader, inner_val_loader, cfg, device)

        _, _, test_preds, test_labels, test_probs = evaluate(
            model, test_loader, torch.nn.CrossEntropyLoss(), device
        )
        print_clip_results(test_labels, test_preds, test_probs)

        clip_results = predict_clips(
            model,
            test_df,
            model_cfg["feature_cols"],
            model_cfg["max_seq_len"],
            model_cfg["min_clip_len"],
            train_cfg["batch_size"],
            device,
        )
        clip_results = attach_answer_correct(clip_results, test_df)
        vote_threshold = cfg.get("vote", {}).get("threshold", 0.5)
        subject_results = subject_majority_vote(clip_results, threshold=vote_threshold)
        print_subject_results(subject_results)
        print_answer_correct_analysis(clip_results)

        fold_metrics.append({
            "fold": fold + 1,
            "test_preds": test_preds,
            "test_labels": test_labels,
            "test_probs": test_probs,
            "clip_results": clip_results,
            "subject_results": subject_results,
        })
        all_clip_results.append(clip_results)

        torch.save({
            "model_state_dict": model.state_dict(),
            "input_size": len(model_cfg["feature_cols"]),
            "num_channels": model_cfg["num_channels"],
            "num_classes": 2,
            "kernel_sizes": model_cfg.get("kernel_sizes", 3),
            "dropout": model_cfg["dropout"],
        }, os.path.join(args.output_dir, f"tcn_model_fold{fold + 1}.pth"))

        torch.save({
            "sequences": torch.FloatTensor(X_test),
            "labels": [int(label) for label in y_test],
            "clip_info": test_clip_info,
        }, os.path.join(args.output_dir, f"test_clip_data_fold{fold + 1}.pt"))

    all_subjects = fold_info["unique_subject"].tolist()
    assert len(heldout_subjects) == len(set(heldout_subjects)), "outer test subject repeated across folds"
    assert set(heldout_subjects) == set(all_subjects), "outer test subjects do not cover all subjects"

    all_labels = []
    all_preds = []
    all_probs = []
    for metrics in fold_metrics:
        all_labels.extend(metrics["test_labels"])
        all_preds.extend(metrics["test_preds"])
        all_probs.extend(metrics["test_probs"])

    print(f"\n\n{'=' * 60}")
    print("3-fold cross validation summary")
    print(f"{'=' * 60}")
    print_clip_results(all_labels, all_preds, all_probs)

    combined_clip_results = pd.concat(all_clip_results, ignore_index=True)
    vote_threshold = cfg.get("vote", {}).get("threshold", 0.5)
    combined_subject_results = subject_majority_vote(combined_clip_results, threshold=vote_threshold)
    print_subject_results(combined_subject_results)
    print_answer_correct_analysis(combined_clip_results)

    print(f"\nFold models saved to: {args.output_dir}/tcn_model_fold1~3.pth")
    print(f"Fold test data saved to: {args.output_dir}/test_clip_data_fold1~3.pt")

    return fold_metrics, combined_subject_results


if __name__ == "__main__":
    fold_metrics, combined_subject_results = main()
