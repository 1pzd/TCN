## 项目结构

```
d:\code-en\
├── configs/
│   └── config.yaml              # 集中管理所有超参和路径配置
├── src/
│   ├── __init__.py              # 空文件，标记 src 为 Python 包
│   ├── config.py                # 加载 YAML 配置文件
│   ├── data_loader.py           # 数据加载、清洗、序列构建、K折/按受试者划分
│   ├── model.py                 # TCN 模型定义（含 SEBlock、多层级特征融合）
│   ├── trainer.py               # 训练循环、评估函数、早停机制
│   └── predict.py               # Clip 级预测 + Bagging 投票 + 结果打印
├── data/
│   ├── dataset_BJ.csv           # BJ 强组原始数据集（标签=1）
│   ├── dataset_ZJ.csv           # ZJ 弱组原始数据集（标签=0）
│   ├── dataset_BJ_predicted.csv # BJ 数据集（gaze 替换为预测值）
│   └── dataset_ZJ_predicted.csv # ZJ 数据集（gaze 替换为预测值）
├── train.py                     # 3折交叉验证训练入口
├── test.py                      # 3折交叉验证独立测试入口
├── test_nums_control.py         # 指定 clip_id 评分的测试入口
├── predict_only.py              # 纯预测入口（加载已有模型预测全部数据）
├── tcn_classifier.py            # 独立单文件版 TCN 分类器（无配置文件依赖）
├── merge_predicted_gaze.py      # 将外部 gaze 预测结果合并到原始数据集
├── adapt_gaze_to_pipeline.py    # 将外部 gaze 预测 CSV 适配为管线输入格式
├── .gitignore                   # Git 忽略规则
└── README.md                    # 本文件
```

---

## 各文件/文件夹详细说明

### `configs/` — 配置文件夹

#### `configs/config.yaml`
所有超参数集中管理的唯一配置文件。通过 [src/config.py](file:///d:/code-en/src/config.py) 加载，供所有入口脚本使用。包含以下配置块：

| 配置块 | 配置项 | 说明 |
|--------|--------|------|
| `data` | `bj_path`, `zj_path` | BJ/ZJ 数据集 CSV 路径（可切换原始数据或预测替换数据） |
| `model` | `max_seq_len` | 时序序列最大长度（超过截断，不足补零） |
| `model` | `num_channels` | TCN 各层通道数（列表，长度决定网络深度） |
| `model` | `kernel_sizes` | 各层卷积核大小（列表，与 num_channels 长度一致） |
| `model` | `dropout` | Dropout 比率 |
| `model` | `feature_cols` | 模型使用的特征列（默认 `gaze_x`, `gaze_y`） |
| `model` | `min_clip_len` | 最小有效 clip 长度，小于此值被过滤 |
| `training` | `batch_size` | 训练批次大小 |
| `training` | `epochs` | 最大训练轮数 |
| `training` | `learning_rate` | 初始学习率 |
| `training` | `weight_decay` | L2 正则化系数 |
| `training` | `patience` | 早停耐心值（验证准确率不再提升时停止） |
| `training` | `test_subject_ratio` | 按受试者划分测试集的比例（单次划分时使用） |
| `training` | `grad_clip` | 梯度裁剪阈值 |
| `training` | `warmup_epochs` | 学习率预热轮数 |
| `training` | `class_weight` | 类别权重（null 表示不使用） |
| `vote` | `threshold` | Bagging 投票阈值（BJ 比例超过此值才判为 BJ） |
| `random_state` | | 全局随机种子 |

---

### `src/` — 核心源码包

#### `src/__init__.py`
空文件，将 `src/` 标记为 Python 包，使各模块可以被 `from src.xxx import yyy` 导入。

---

#### `src/config.py`
**作用**：加载 `configs/config.yaml` 配置文件。

**函数**：
- `load_config(config_path='configs/config.yaml')` → `dict`
  - 读取 YAML 文件并解析为 Python 字典
  - 自动拼接 `data_dir` 与文件名得到完整数据路径

**使用示例**：
```python
from src.config import load_config
cfg = load_config()
print(cfg['model']['num_channels'])
```

---

#### `src/data_loader.py`
**作用**：数据加载、预处理、清洗、时序序列构建、数据集划分。

**函数**：

| 函数 | 说明 |
|------|------|
| `load_and_preprocess(bj_path, zj_path)` | 加载 BJ/ZJ 两个 CSV，合并、打标签（BJ=1, ZJ=0）、过滤无效数据（validity=1、排除 clip_id=0/-1、仅保留 Fixation 类型） |
| `create_sequences(df, feature_cols, max_seq_len, min_clip_len)` | 按 `unique_clip` 分组构建固定长度时序序列，超过截断，不足补零 |
| `kfold_split_subjects(df, n_splits, random_state)` | 按受试者 K 折分层划分（保持各类别比例），返回每个受试者所属折号 |
| `split_by_subject(df, test_ratio, random_state)` | 按受试者单次划分训练/测试集（分层采样），返回划分后的 DataFrame |

**使用示例**：
```python
from src.data_loader import load_and_preprocess, create_sequences
df = load_and_preprocess('data/dataset_BJ.csv', 'data/dataset_ZJ.csv')
X, y, info = create_sequences(df, ['gaze_x', 'gaze_y'], max_seq_len=200, min_clip_len=10)
```

---

#### `src/model.py`
**作用**：定义 TCN（时序卷积网络）模型架构，含 SE 注意力机制和多层级特征融合。

**类**：

| 类 | 说明 |
|----|------|
| `Chomp1d` | 裁剪层，去除因果卷积右侧多余填充 |
| `SEBlock` | Squeeze-and-Excitation 通道注意力模块，自动学习各通道重要性 |
| `TemporalBlock` | 单个时序残差块（2层膨胀卷积 + 残差连接 + SE 模块） |
| `TemporalConvNet` | 多层 TemporalBlock 堆叠，支持中间层特征提取 |
| `SelfAttentionPooling` | 自注意力池化层（当前未使用） |
| `TCNClassifier` | **主模型**：TCN 主干 + 多层级特征融合（GAP + GMP）+ 全连接分类头 |

模型结构特点：
- 8 层膨胀卷积，膨胀率呈 2 的指数增长
- 从第 2、4、7 层提取中间特征
- 每层特征同时使用全局平均池化（GAP）和全局最大池化（GMP）
- 融合后的特征经 LayerNorm 后送入分类头

**使用示例**：
```python
from src.model import TCNClassifier
model = TCNClassifier(
    input_size=2,
    num_channels=[64, 96, 128, 192, 256, 320, 384, 448],
    num_classes=2,
    kernel_sizes=[5,5,5,3,3,3,3,3],
    dropout=0.1
)
```

---

#### `src/trainer.py`
**作用**：训练循环、评估函数、早停机制。

**类/函数**：

| 名称 | 说明 |
|------|------|
| `EyeTrackingDataset` | PyTorch Dataset 封装，适配 DataLoader |
| `train_epoch(model, loader, optimizer, criterion, device, grad_clip, pbar)` | 单 epoch 训练，含梯度裁剪，返回平均 loss 和准确率 |
| `evaluate(model, loader, criterion, device)` | 模型评估，返回 loss、准确率、预测值、真实值、概率值 |
| `train_model(model, train_loader, test_loader, cfg, device)` | **完整训练流程**：AdamW 优化器 + ReduceLROnPlateau 调度 + 学习率预热 + 早停 + 最佳模型保存 |

**使用示例**：
```python
from src.trainer import train_model, evaluate
model = train_model(model, train_loader, test_loader, cfg, device)
```

---

#### `src/predict.py`
**作用**：Clip 级预测、受试者级 Bagging 多数投票、结果打印。

**函数**：

| 函数 | 说明 |
|------|------|
| `predict_clips(model, df, feature_cols, max_seq_len, min_clip_len, batch_size, device)` | 对 DataFrame 中所有 clip 进行预测，返回含 `unique_subject`, `pred_label`, `prob_0/1` 的 DataFrame |
| `predict_from_data(model, sequences, clip_info, batch_size, device)` | 对已构建好的序列数据直接预测（用于 test.py 加载预存数据） |
| `subject_majority_vote(clip_results, threshold)` | 按受试者汇总所有 clip 预测结果，投票阈值可调，返回受试者级决策 |
| `print_clip_details(clip_results, title)` | 打印 clip 级准确率和 AUC |
| `print_subject_results(subject_results)` | 打印每位受试者的预测详情、准确率、F1、AUC |
| `print_clip_results(test_labels, test_preds, test_probs)` | 打印 clip 级评估指标（准确率、精确率、召回率、F1、AUC、混淆矩阵） |

**投票逻辑**：
```
对每个受试者：
  统计该受试者所有 clip 的预测标签
  计算 BJ（标签=1）的比例
  BJ比例 >= threshold → 判为 BJ（强组）
  BJ比例 < threshold  → 判为 ZJ（弱组）
```

**使用示例**：
```python
from src.predict import predict_clips, subject_majority_vote
clip_results = predict_clips(model, df, ['gaze_x','gaze_y'], 200, 10, 32, device)
subject_results = subject_majority_vote(clip_results, threshold=0.55)
```

---

### `data/` — 数据文件夹

| 文件 | 说明 |
|------|------|
| `dataset_BJ.csv` | BJ（北京）强组原始眼动数据集，标签=1，28 名受试者，81,584 条记录 |
| `dataset_ZJ.csv` | ZJ（浙江）弱组原始眼动数据集，标签=0，21 名受试者，65,319 条记录 |
| `dataset_BJ_predicted.csv` | BJ 数据集的 gaze_x/gaze_y 被外部 gaze 预测模型替换后的版本 |
| `dataset_ZJ_predicted.csv` | ZJ 数据集的 gaze_x/gaze_y 被外部 gaze 预测模型替换后的版本 |

原始 CSV 包含字段：`subject_id`, `clip_id`, `frame_path`, `timestamp`, `gaze_x`, `gaze_y`, `validity`, `original_type`。

**切换数据**：编辑 `configs/config.yaml` 中的 `bj_path`/`zj_path`，注释/取消注释对应行即可。

---

### 根目录入口脚本

#### `train.py` — 3 折交叉验证训练
**作用**：完整训练主流程。
- 加载配置和数据
- 按受试者进行 3 折分层划分
- 每折独立训练一个 TCN 模型
- 输出 clip 级和受试者级评估结果
- 保存模型和测试数据到 `output_model/` 目录

**保存文件**：
- `output_model/tcn_model_fold{1,2,3}.pth` — 3 个单折模型
- `output_model/test_clip_data_fold{1,2,3}.pt` — 3 份测试集序列数据（供 test.py 使用）

**运行**：
```bash
python train.py
```

---

#### `test.py` — 3 折交叉验证独立测试
**作用**：加载 `train.py` 保存的模型和数据，重新评估。

**特点**：
- 不重新训练，只做预测和评估
- 汇总 3 折的 held-out 预测结果
- 输出 clip 级和受试者级的汇总指标

**运行**（需先执行 `train.py`）：
```bash
python test.py
```

---

#### `test_nums_control.py` — 指定 clip_id 评分测试
**作用**：仅对特定 `clip_id` 的 clip 进行评分。

**特点**：
- 通过 `CLIP_IDS` 列表（第 12 行）控制只使用哪些 clip_id
- 适合分析特定 clip 的分类效果
- 默认仅使用 `clip_id=1`

**运行**（需先执行 `train.py`）：
```bash
python test_nums_control.py
```

**自定义 clip 范围**：修改文件顶部 `CLIP_IDS = [1]` 为需要的列表。

---

#### `predict_only.py` — 纯预测入口
**作用**：加载已训练的模型，对全部数据进行预测。

**特点**：
- 使用单模型（非 3 折）对完整数据集做预测
- 适合部署场景或批量预测
- 默认加载 `output_model/tcn_model.pth`

**运行**（需先有训练好的模型文件）：
```bash
python predict_only.py
```

---

#### `tcn_classifier.py` — 独立单文件版分类器
**作用**：将整个管线（配置、数据加载、模型、训练、预测）整合到一个文件中。

**特点**：
- 不依赖 `configs/config.yaml` 和 `src/` 包
- 所有超参数在文件顶部硬编码
- 适合快速实验、调试、或迁移到其他环境
- 功能与 `train.py` + `src/` 组合等价

**运行**：
```bash
python tcn_classifier.py
```

**注意**：此文件与 `src/` 包版本功能重复，属于独立实验版本，修改参数需直接编辑文件顶部配置区。

---

#### `merge_predicted_gaze.py` — 合并外部 gaze 预测结果
**作用**：将外部 gaze 预测模型（如深度学习注视点估计模型）的输出结果，替换到原始数据集的 `gaze_x`/`gaze_y` 列中。

**流程**：
1. 加载外部预测 CSV（`predicted_gaze_train.csv` + `predicted_gaze_test.csv`）
2. 按 `subject_id` + `frame_path` 构建查找表
3. 遍历原始数据集，匹配到的行替换 gaze 坐标
4. 输出 `dataset_BJ_predicted.csv` 和 `dataset_ZJ_predicted.csv`

**运行**：
```bash
python merge_predicted_gaze.py
```

**路径配置**：编辑文件顶部 `PRED_TRAIN`, `PRED_TEST`, `DOWN_BJ`, `DOWN_ZJ`, `OUT_BJ`, `OUT_ZJ` 变量。

---

#### `adapt_gaze_to_pipeline.py` — 适配外部 gaze CSV 为管线输入
**作用**：将外部 gaze 预测代码输出的 CSV（仅含 `subject_id`, `frame_path`, `pred_x`, `pred_y`）与原始数据集匹配，生成与 `dataset_BJ.csv` 格式兼容的输入文件。

**与 `merge_predicted_gaze.py` 的区别**：
- `merge_predicted_gaze.py` 保持原始数据结构，仅替换 gaze 值
- `adapt_gaze_to_pipeline.py` 从外部 CSV 重建完整数据结构（匹配 clip_id、timestamp 等）

**运行**：
```bash
python adapt_gaze_to_pipeline.py <gaze_csv_path> [--bj <bj_path>] [--zj <zj_path>] [--output <output_path>]
```

**参数说明**：
| 参数 | 说明 |
|------|------|
| `gaze_csv_path` | gaze 预测输出的 CSV 文件路径（必填） |
| `--bj` | 原始 BJ 数据集路径（默认 `data/dataset_BJ.csv`） |
| `--zj` | 原始 ZJ 数据集路径（默认 `data/dataset_ZJ.csv`） |
| `--output` | 输出路径（默认 `{原文件名}_adapted.csv`） |

---

## 环境要求

| 组件 | 最低版本 |
|------|----------|
| Python | 3.8+ |
| PyTorch | 1.13+ |
| scikit-learn | 1.0+ |
| pandas | 1.3+ |
| numpy | 1.21+ |
| PyYAML | 6.0+ |
| tqdm | 4.60+ |

一键安装依赖：
```bash
pip install torch numpy pandas scikit-learn pyyaml tqdm
```

---

## 快速开始

```bash
# 进入项目目录
cd d:\code-en

# 完整训练 + 评估（3 折交叉验证）
python train.py

# 独立测试已训练的模型
python test.py

# 使用单文件版本（不依赖配置文件）
python tcn_classifier.py
```

---

## 典型工作流

```
1. 数据准备
   ├── data/dataset_BJ.csv + data/dataset_ZJ.csv（原始数据）
   └── 可选：python merge_predicted_gaze.py（替换为预测 gaze）

2. 训练
   └── python train.py
       ├── 加载 configs/config.yaml
       ├── 3 折交叉验证训练
       └── 输出 output_model/tcn_model_fold{1,2,3}.pth

