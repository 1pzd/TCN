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
AvgPool + MaxPool