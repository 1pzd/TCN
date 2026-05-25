import pandas as pd
import argparse
import os


def adapt_gaze_csv(gaze_csv_path, bj_path, zj_path, output_path=None):
    df_gaze = pd.read_csv(gaze_csv_path)
    df_bj = pd.read_csv(bj_path)
    df_zj = pd.read_csv(zj_path)

    df_orig = pd.concat([df_bj, df_zj], ignore_index=True)

    df_gaze["subject_id"] = df_gaze["subject_id"].astype(int)
    df_orig["subject_id"] = df_orig["subject_id"].astype(int)

    print(f"Gaze CSV: {len(df_gaze)} rows")
    print(f"Original dataset: {len(df_orig)} rows")

    # 判断 gaze CSV 中 frame_path 的格式
    sample_gaze = df_gaze["frame_path"].iloc[0]
    sample_orig = df_orig["frame_path"].iloc[0]
    print(f"\nGaze frame_path 示例: {sample_gaze}")
    print(f"Original frame_path 示例: {sample_orig}")

    need_join = False
    if sample_gaze != sample_orig:
        # 可能是纯文件名，需要拼接成完整路径
        if "\\" not in str(sample_gaze) and "/" not in str(sample_gaze):
            df_gaze["_frame_key"] = (
                df_gaze["subject_id"].astype(str)
                + "\\images\\"
                + df_gaze["subject_id"].astype(str)
                + "_"
                + df_gaze["frame_path"]
            )
            need_join = True
        else:
            df_gaze["_frame_key"] = df_gaze["frame_path"].str.replace("/", "\\")
            need_join = True
    else:
        df_gaze["_frame_key"] = df_gaze["frame_path"]

    df_orig["_frame_key"] = df_orig["frame_path"]

    # 只保留需要的字段
    df_lookup = df_orig[["_frame_key", "subject_id", "clip_id", "timestamp", "original_type", "validity"]].drop_duplicates(
        subset=["_frame_key", "subject_id"]
    )

    # join 恢复 clip_id 和 timestamp
    df_merged = df_gaze.merge(df_lookup, on=["_frame_key", "subject_id"], how="left")

    matched = df_merged["clip_id"].notna().sum()
    print(f"\n匹配到 clip_id: {matched} / {len(df_merged)} ({matched/len(df_merged)*100:.1f}%)")

    unmatched = df_merged[df_merged["clip_id"].isna()]
    if len(unmatched) > 0:
        print(f"未匹配行数: {len(unmatched)}")
        print("未匹配的 frame_path 示例:")
        print(unmatched["frame_path"].head(5).to_list())
        print("这些帧在原始数据集中不存在（可能是 Saccade 或其他类型）")

    # 丢弃未匹配的行（只保留 Fixation）
    df_merged = df_merged.dropna(subset=["clip_id"]).reset_index(drop=True)
    df_merged["clip_id"] = df_merged["clip_id"].astype(int)
    df_merged["validity"] = df_merged["validity"].astype(int)

    # 构建管线所需的输出格式
    df_out = pd.DataFrame()
    df_out["subject_id"] = df_merged["subject_id"]
    df_out["clip_id"] = df_merged["clip_id"]
    df_out["frame_path"] = df_merged["frame_path"]
    df_out["timestamp"] = df_merged["timestamp"]
    df_out["gaze_x"] = df_merged["pred_x"]
    df_out["gaze_y"] = df_merged["pred_y"]
    df_out["validity"] = df_merged["validity"]
    df_out["original_type"] = df_merged["original_type"]

    # 构建管线所需的 unique_clip
    df_out["unique_clip"] = df_out["subject_id"].astype(str) + "_clip_" + df_out["clip_id"].astype(str)
    df_out = df_out.sort_values(["unique_clip", "timestamp"])

    if output_path is None:
        base, ext = os.path.splitext(gaze_csv_path)
        output_path = f"{base}_adapted.csv"

    df_out.to_csv(output_path, index=False)
    print(f"\n输出已保存至: {output_path}")
    print(f"输出行数: {len(df_out)}")
    print(f"唯一 subject: {df_out['subject_id'].nunique()}")
    print(f"唯一 clip: {df_out['unique_clip'].nunique()}")
    print(f"\n字段: {list(df_out.columns)}")

    return df_out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将 gaze 预测输出适配为 TCN 管线输入格式")
    parser.add_argument("gaze_csv", help="gaze 生成代码输出的 CSV 文件路径")
    parser.add_argument("--bj", default="data/dataset_BJ.csv", help="原始 BJ 数据集路径")
    parser.add_argument("--zj", default="data/dataset_ZJ.csv", help="原始 ZJ 数据集路径")
    parser.add_argument("--output", default=None, help="输出路径（默认: {原文件名}_adapted.csv）")
    args = parser.parse_args()

    adapt_gaze_csv(args.gaze_csv, args.bj, args.zj, args.output)
