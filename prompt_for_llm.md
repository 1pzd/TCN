# 眼动数据TCN分类模型 — 优化提示词

## 一、项目背景

这是一个基于 **TCN（时序卷积网络）** 的眼动数据二分类项目，目标是根据受试者的注视点（gaze）数据将其分为**强组（BJ，标签=1）**和**弱组（ZJ，标签=0）**。

### 数据概况

| 类别 | 标签 | 受试者数 | Fixation样本数 |
|------|:----:|:---------:|:--------------:|
| BJ（强组） | 1 | 28人（实际27人，排除subj_18） | 74,819 |
| ZJ（弱组） | 0 | 21人 | 61,044 |

原始CSV字段：`subject_id, timestamp, gaze_x, gaze_y, validity, original_type, clip_id`

### 数据预处理
- 已过滤掉所有非 Fixation 数据
- 每个受试者有 21 个 clip（视频片段），每个 clip 包含 26~177 帧不等的连续注视点
- 序列统一 padding/截断到 200 帧

### 当前模型架构

```
输入 [batch, 200, 8]          # 8维特征
    │  transpose(1,2)
输入 [batch, 8, 200]
    │
TCN 5层膨胀卷积：
  Conv1D  8→32  dilation=1, kernel=5
  Conv1D 32→48  dilation=2, kernel=5
  Conv1D 48→64  dilation=4, kernel=3
  Conv1D 64→80  dilation=8, kernel=3
  Conv1D 80→96  dilation=16, kernel=3
    │  每层后接 SEBlock（通道注意力）
    │
多级特征融合（第2/4层 + 最后一层）
AvgPool + MaxPool 双池化
    │
分类头: LayerNorm → Linear(384→192) → ReLU → Dropout → Linear(192→2)
```

### 当前特征（8维）

| 特征 | 计算方式 | 含义 |
|:-----|:---------|:-----|
| gaze_x, gaze_y | 原始坐标 | 注视位置 |
| gaze_vel | sqrt(dx² + dy²) | 帧间速度（注视微动幅度） |
| gaze_vel_x, gaze_vel_y | dx, dy | 方向性速度 |
| gaze_acc | d(vel)/dt | 加速度（微眼跳力度） |
| disp_x, disp_y | 窗口=5 的滚动标准差 | 短期注视离散度 |

### 训练配置

```yaml
training:
  batch_size: 32
  learning_rate: 8.0e-4
  weight_decay: 1.0e-5
  patience: 50
  warmup_epochs: 10
  grad_clip: 1.0
model:
  dropout: 0.1
```

### 模型参数量

**209,418**（约21万参数）

---

## 二、当前存在的问题（需优化）

### 问题 1：准确率瓶颈 — 66.7%（32/48），且特定受试者始终被分错

**症状**：
- 无论怎么调参，受试者级准确率始终在 66.7% 附近
- 具体来说，**固定的 16 名受试者**在所有运行中一直被分错（6名BJ + 10名ZJ）
- 说明这些受试者的注视模式本身就落在两组的数据重叠区域，仅靠当前特征无法区分

**证据**：
```
BJ 组内 gaze_x 均值范围: [0.4828, 0.5367] → 跨度仅 0.054
ZJ 组内 gaze_x 均值范围: [0.4772, 0.5145] → 跨度仅 0.037
两组 gaze_x 均值差异: 0.5041 vs 0.4996 = 0.0045（几乎为零）
```

**本质**：当前特征对两组的区分度已经用尽，需要**新的信息维度**。

### 问题 2：数据中的 Saccade 信息没有被利用

虽然已经过滤掉非 Fixation 数据，但完全丢弃了扫视信息。实际上**扫视频率和幅度**可能是区分强弱组的重要指标。

**可用字段**：原始数据中有 `original_type` 字段（Fixation/Saccade），之前的预处理脚本 `preprocess.py` 中只保留了 Fixation 数据。实际上应该：
- 可以考虑**同时保留两种类型**，或者
- 计算每个 clip 中 Fixation 和 Saccade 的**转换模式**（如 Fixation 持续时间、Saccade 幅度分布）

### 问题 3：没有数据增强

当前训练数据直接输入模型，没有做任何增强。以下增强策略可以尝试：

- **高斯噪声注入**：gaze_x/y 加 N(0, 0.002) 的噪声（模拟眼动追踪精度误差）
- **时间掩码**：随机遮挡连续 5~20 帧，增强对缺失数据的鲁棒性
- **序列裁剪**：200 帧中随机裁剪 150 帧长度的子序列
- **时间扭曲**：对时间轴做轻微的伸缩

### 问题 4：模型架构可能不是最优选择

当前 TCN 的感受野约 69 帧，但数据中区分性较强的可能是**整体统计特征**而非时序局部模式。可以考虑：

- **Transformer 编码器**：原生支持序列建模，自注意力可能更好地捕捉帧间全局依赖
- **TCN + Transformer 混合**：底层 TCN 提取局部模式 + 顶层 Transformer 建模全局依赖
- **纯统计模型**：对每个 clip 计算统计量（均值、方差、偏度、峰度、百分位数）直接用 MLP 分类
- **TimesNet**：最新的时序基础模型，专为时序分类设计

### 问题 5：没有模型集成

当前是单模型 + clip 级投票。可以做模型级集成：
- 不同随机种子训练多个模型
- 不同 Train/Test 划分训练多个模型
- 在受试者级做**模型投票**（不是 clip 投票）

### 问题 6：类别不平衡无处理

| 类别 | 受试者数 |
|:-----|:--------:|
| BJ（强组） | 27 |
| ZJ（弱组） | 21 |

比例约 1.29:1，虽然不算严重，但 ZJ 识别率偏低（52.4%）。可以考虑：
- 对 ZJ 类的训练样本过采样
- 对 BJ 类欠采样
- 焦点损失（Focal Loss）

---

## 三、优化方向（请按此顺序实施）

### Phase 1：数据增强（最易实现，预期 +2~3%）

**实施**：
1. 修改 `src/trainer.py` 中的 `EyeTrackingDataset.__getitem__` 方法
2. 添加可配置的增强策略（通过 config.yaml 控制开关和强度）

```python
def __getitem__(self, idx):
    seq = self.sequences[idx].clone()
    label = self.labels[idx]
    if self.augment and self.training:
        # 高斯噪声（仅 gaze_x/y 坐标）
        noise = torch.randn_like(seq[:, :2]) * 0.002
        seq[:, :2] = seq[:, :2] + noise
        # 时间掩码
        if torch.rand(1) < 0.1:
            mask_len = torch.randint(5, 20, (1,)).item()
            mask_start = torch.randint(0, len(seq) - mask_len, (1,)).item()
            seq[mask_start:mask_start+mask_len] = 0
    return seq, label
```

### Phase 2：利用 Saccade 信息（预期 +3~5%）

**实施**：
1. 修改 `preprocess.py`，同时保留 Fixation 和 Saccade 数据
2. 在 `compute_gaze_features()` 中提取扫视相关特征
3. 思路：每个 clip 内部，Fixation 和 Saccade 交替出现，可以提取：
   - **扫视频率**：单位时间内扫视次数
   - **平均扫视幅度**：Saccade 期间的 gaze 位移
   - **Fixation/Saccade 比例**：Fixation 时长 / Saccade 时长
   - **Fixation 持续时间**：连续 Fixation 帧数的统计

### Phase 3：高级特征工程（预期 +5~10%）

**实施**：
在 `src/data_loader.py` 的 `compute_gaze_features()` 中增加：

| 新特征 | 计算方式 | 生物学意义 |
|:-------|:---------|:----------|
| 注视熵 | H = -Σ p(i) log p(i)（gaze 空间分 bin） | 注视空间分布的均匀性 |
| 注视时空密度 | 高斯核密度估计 | 注视聚集模式 |
| 注视方向熵 | 速度方向角的直方图熵 | 注视移动的方向随机性 |
| 注视变点频率 | CUSUM 检测注视位置突变 | 注视稳定性的时间模式 |
| 时序自相关 | gaze_x(t) 和 gaze_x(t+τ) 的 Pearson 相关 | 注视的时序依赖性 |

### Phase 4：模型架构升级（预期 +3~5%）

**实施**：
在 `src/model.py` 中新增 `TCNTransformerClassifier`：

```
输入 [batch, 200, 8]
    │
    ├── TCN Backbone（5层，输出 96 通道）
    │   → [batch, 96, 200] → transpose → [batch, 200, 96]
    │
    ├── Transformer Encoder（2层，d_model=96, nhead=8）
    │   → [batch, 200, 96]
    │
    ├── CLS Token 池化 或 全局平均池化
    │   → [batch, 96]
    │
    └── 分类头
         → [batch, 2]
```

### Phase 5：模型集成（预期 +3~5%）

**实施**：
1. 在 `train.py` 中使用不同的 `random_state`（42, 123, 456, 789, 1111）训练 5 个模型
2. 在 `subject_majority_vote()` 之前，先对 5 个模型的预测结果做模型级投票
3. 策略：每个模型预测 clip 的概率，5 个模型概率取平均后再做分类决策

### Phase 6：类别不平衡处理（预期 +2~3%）

**实施**：
1. 在 config.yaml 中启用 `class_weight: [1.0, 0.78]`
2. 或在 `EyeTrackingDataset` 中使用 WeightedRandomSampler 对 ZJ 类过采样

---

## 四、代码文件索引

| 文件 | 路径 | 说明 |
|:-----|:-----|:------|
| 主配置 | `configs/config.yaml` | 所有超参数集中管理 |
| 数据加载 | `src/data_loader.py` | `load_and_preprocess()` + `create_sequences()` + `compute_gaze_features()` |
| 模型定义 | `src/model.py` | TCN 各组件 + `TCNClassifier` |
| 训练循环 | `src/trainer.py` | `EyeTrackingDataset` + `train_model()` + `evaluate()` |
| 预测与投票 | `src/predict.py` | `predict_clips()` + `subject_majority_vote()` |
| 训练入口 | `train.py` | 主训练流程 |
| 测试入口 | `test.py` | 独立测试流程 |
| 预处理 | `preprocess.py` | 原始 CSV → 仅 Fixation |
| 接入接口 | `gaze_input_pipeline.py` | 新受试者数据接入 |

---

## 五、运行命令

```bash
# 训练
python train.py

# 独立测试
python test.py

# 新受试者预测
python gaze_input_pipeline.py data/new_subject.csv
```

---

## 六、评估指标

| 指标 | 当前值 | 目标值 |
|:-----|:------:|:------:|
| 受试者级准确率 | 66.67% | ≥75% |
| 受试者级 F1 | 0.7241 | ≥0.80 |
| BJ（强组）识别率 | 77.8% | ≥80% |
| ZJ（弱组）识别率 | 52.4% | ≥70% |
| Clip 级准确率 | 58.2% | ≥65% |

---

## 七、底线要求

- 保持项目结构不变，不要随意重组文件
- 不要删除已有功能，只做增量修改
- 所有新增超参数必须在 `config.yaml` 中可配置
- 训练和测试必须可复现（`random_state=42`）
- `python train.py` 和 `python test.py` 必须能正常运行
- 新增依赖必须在 README 中注明
- 保持原有的 Bagging 投票决策逻辑
- 不要对输入数据做标准化/归一化（保留原始分布差异）
