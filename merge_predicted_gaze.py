import pandas as pd
import os

PRED_TRAIN = "E:/code-cn/predict_outputs/predicted_gaze_train.csv"
PRED_TEST = "E:/code-cn/predict_outputs/predicted_gaze_test.csv"
DOWN_BJ = "data/dataset_BJ.csv"
DOWN_ZJ = "data/dataset_ZJ.csv"
OUT_BJ = "data/dataset_BJ_predicted.csv"
OUT_ZJ = "data/dataset_ZJ_predicted.csv"


def load_prediction_lookup():
    df_train = pd.read_csv(PRED_TRAIN)
    df_test = pd.read_csv(PRED_TEST)
    df = pd.concat([df_train, df_test], ignore_index=True)

    def make_key(row):
        sid = str(row["subject_id"])
        fp = str(row["frame_path"]).replace("\\", "/").replace("//", "/")
        return f"{sid}|{fp}"

    lookup = {}
    for _, row in df.iterrows():
        key = make_key(row)
        lookup[key] = (row["pred_x"], row["pred_y"])

    print(f"预测表: {len(lookup)} 条 (train={len(df_train)}, test={len(df_test)})")
    return lookup


def make_key(sid, fp):
    return f"{sid}|{str(fp).replace('\\', '/').replace('//', '/')}"


def merge_one(csv_path, out_path, lookup, tag):
    df = pd.read_csv(csv_path)

    replaced = 0
    kept = 0
    for i, row in df.iterrows():
        key = make_key(row["subject_id"], row["frame_path"])
        if key in lookup:
            df.at[i, "gaze_x"] = lookup[key][0]
            df.at[i, "gaze_y"] = lookup[key][1]
            replaced += 1
        else:
            kept += 1

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[{tag}] 输出: {out_path}")
    print(f"  → 预测替换: {replaced} 行, 保留原始: {kept} 行 "
          f"(保留率 {kept/(replaced+kept)*100:.1f}%)")
    return df


def main():
    lookup = load_prediction_lookup()
    merge_one(DOWN_BJ, OUT_BJ, lookup, "BJ")
    merge_one(DOWN_ZJ, OUT_ZJ, lookup, "ZJ")
    print("\n完成。接下来: 修改 config.yaml 的 bj_path / zj_path 指向以上两个文件")


if __name__ == "__main__":
    main()
