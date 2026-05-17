import pandas as pd
import os


def filter_fixation_only(input_path, output_path):
    df = pd.read_csv(input_path)
    total = len(df)

    df = df[df['original_type'] == 'Fixation'].reset_index(drop=True)
    kept = len(df)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"  {os.path.basename(input_path)}:")
    print(f"    原始行数: {total}")
    print(f"    保留行数: {kept} (Fixation占比 {kept/total*100:.1f}%)")
    print(f"    过滤行数: {total - kept} (非Fixation)")
    print(f"    受试者数: {df['subject_id'].nunique()}")
    print(f"    Clip数:   {df['clip_id'].nunique()}")

    return df


def main():
    print("=" * 50)
    print("预处理：仅保留 Fixation 数据")
    print("=" * 50)

    bj_src = "data/dataset_BJ.csv"
    bj_dst = "data/dataset_BJ_fixation_only.csv"
    zj_src = "data/dataset_ZJ.csv"
    zj_dst = "data/dataset_ZJ_fixation_only.csv"

    print("\n[1/2] 处理 BJ 数据集...")
    df_bj = filter_fixation_only(bj_src, bj_dst)

    print(f"\n[2/2] 处理 ZJ 数据集...")
    df_zj = filter_fixation_only(zj_src, zj_dst)

    total_original = pd.read_csv(bj_src).shape[0] + pd.read_csv(zj_src).shape[0]
    total_kept = df_bj.shape[0] + df_zj.shape[0]

    print(f"\n{'=' * 50}")
    print(f"汇总:")
    print(f"  原始总行数: {total_original}")
    print(f"  保留总行数: {total_kept} ({total_kept/total_original*100:.1f}%)")
    print(f"  过滤总行数: {total_original - total_kept} ({100-total_kept/total_original*100:.1f}%)")
    print(f"\n已保存至:")
    print(f"  {os.path.abspath(bj_dst)}")
    print(f"  {os.path.abspath(zj_dst)}")


if __name__ == '__main__':
    main()
