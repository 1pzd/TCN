import numpy as np
import pandas as pd


def load_and_preprocess(bj_path, zj_path):
    df_bj = pd.read_csv(bj_path)
    df_zj = pd.read_csv(zj_path)

    df_bj['dataset'] = 'BJ'
    df_zj['dataset'] = 'ZJ'

    df = pd.concat([df_bj, df_zj], ignore_index=True)
    df = df[~((df['dataset'] == 'BJ') & (df['subject_id'] == 18))].reset_index(drop=True)

    df['label'] = (df['dataset'] == 'BJ').astype(int)

    df['unique_subject'] = df['dataset'] + '_' + df['subject_id'].astype(str)
    df['unique_clip'] = df['unique_subject'] + '_clip_' + df['clip_id'].astype(str)

    df = df[df['validity'] == 1].reset_index(drop=True)
    df = df[df['clip_id'] != 0].reset_index(drop=True)
    df = df[df['clip_id'] != -1].reset_index(drop=True)
    df = df[df['original_type'] == 'Fixation'].reset_index(drop=True)
    df = df.sort_values(['unique_clip', 'timestamp']).reset_index(drop=True)

    print(f"\n{'='*50}")
    print("数据加载完成")
    print(f"  BJ 样本数: {len(df_bj)}")
    print(f"  ZJ 样本数: {len(df_zj)}")
    print(f"  总样本数: {len(df)}")
    print(f"  标签分布: 0(ZJ)={len(df[df['label']==0])}, 1(BJ)={len(df[df['label']==1])}")
    print(f"  唯一受试者: BJ={df_bj['subject_id'].nunique()}, ZJ={df_zj['subject_id'].nunique()}")
    print(f"  总clip数: {df['unique_clip'].nunique()}")

    return df


def create_sequences(df, feature_cols, max_seq_len, min_clip_len):
    clip_lengths = df.groupby('unique_clip').size()
    print(f"\nClip长度统计: min={clip_lengths.min()}, max={clip_lengths.max()}, "
          f"mean={clip_lengths.mean():.1f}, median={clip_lengths.median()}")

    sequences = []
    labels = []
    clip_info = []

    for unique_clip, group in df.groupby('unique_clip'):
        group = group.sort_values('timestamp')
        values = group[feature_cols].values.astype(np.float32)
        label = group['label'].iloc[0]
        subject = group['unique_subject'].iloc[0]
        clip_id = group['clip_id'].iloc[0]

        if len(values) < min_clip_len:
            continue

        if len(values) > max_seq_len:
            values = values[:max_seq_len]
        else:
            pad_len = max_seq_len - len(values)
            pad = np.zeros((pad_len, values.shape[1]), dtype=np.float32)
            values = np.vstack([values, pad])

        sequences.append(values)
        labels.append(label)
        clip_info.append((subject, clip_id, unique_clip))

    sequences = np.array(sequences)
    labels = np.array(labels, dtype=np.int64)

    print(f"序列构建完成: {len(sequences)} clips, shape={sequences.shape}")
    print(f"  标签分布: 0={np.sum(labels==0)}, 1={np.sum(labels==1)}")

    return sequences, labels, clip_info


def kfold_split_subjects(df, n_splits=3, random_state=42):
    subjects = df[['unique_subject', 'label']].drop_duplicates().reset_index(drop=True)
    n_subjects = len(subjects)
    fold_labels = np.full(n_subjects, -1, dtype=int)
    rng = np.random.RandomState(random_state)
    for label in [0, 1]:
        idxs = subjects[subjects['label'] == label].index.tolist()
        rng.shuffle(idxs)
        for i, idx in enumerate(idxs):
            fold_labels[idx] = i % n_splits
    subjects['fold'] = fold_labels
    print(f"\nK折划分 (n_splits={n_splits}):")
    for f in range(n_splits):
        fold_subjects = subjects[subjects['fold'] == f]
        subject_ids = sorted(fold_subjects['unique_subject'].tolist())
        print(f"  折{f+1}: {len(fold_subjects)} 受试者 "
              f"(BJ={len(fold_subjects[fold_subjects['label']==1])}, "
              f"ZJ={len(fold_subjects[fold_subjects['label']==0])})")
        print(f"    受试者ID: {subject_ids}")
    return subjects


def split_by_subject(df, test_ratio, random_state=42):
    subjects = df[['unique_subject', 'label']].drop_duplicates()

    train_subjects = []
    test_subjects = []

    for label in [0, 1]:
        group = subjects[subjects['label'] == label]
        n_test = max(1, int(len(group) * test_ratio))
        sampled = group.sample(n=n_test, random_state=random_state)
        test_subjects.extend(sampled['unique_subject'].tolist())
        train = group[~group['unique_subject'].isin(sampled['unique_subject'])]
        train_subjects.extend(train['unique_subject'].tolist())

    train_df = df[df['unique_subject'].isin(train_subjects)].reset_index(drop=True)
    test_df = df[df['unique_subject'].isin(test_subjects)].reset_index(drop=True)

    train_info = train_df[['unique_subject', 'label']].drop_duplicates()
    test_info = test_df[['unique_subject', 'label']].drop_duplicates()

    print(f"\n按受试者拆分 (test_ratio={test_ratio}):")
    print(f"  训练集: {len(train_info)} 受试者 "
          f"(BJ={len(train_info[train_info['label']==1])}, "
          f"ZJ={len(train_info[train_info['label']==0])}), "
          f"{train_df['unique_clip'].nunique()} clips")
    print(f"  测试集: {len(test_info)} 受试者 "
          f"(BJ={len(test_info[test_info['label']==1])}, "
          f"ZJ={len(test_info[test_info['label']==0])}), "
          f"{test_df['unique_clip'].nunique()} clips")

    overlap = set(train_subjects) & set(test_subjects)
    assert len(overlap) == 0, f"Subject泄露! 重叠: {overlap}"

    return train_df, test_df, train_subjects, test_subjects
