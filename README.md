# 眼动数据时序分类任务

基于 **TCN（时序卷积网络）** 的眼动数据强/弱组二分类模型，使用 **Bagging 多数投票** 机制完成受试者级决策。

## 项目结构

```
d:\code-en\
├── configs/
│   └── config.yaml          # 配置文件，所有超参集中管理
├── src/
│   ├── __init__.py
│   ├── config.py            # 加载 config.yaml 配置
│   ├── data_loader.py       # 数据加载、清洗、序列构建
│   ├── model.py             # TCN 模型定义
│   ├── trainer.py           # 训练循环 + tqdm 进度条
│   └── predict.py           # clip 预测 + Bagging 投票 + 结果输出
├── data/
│   ├── dataset_BJ.csv       # BJ 数据集（强组，标签=1）
│   └── dataset_ZJ.csv       # ZJ 数据集（弱组，标签=0）
├── train.py                 # 训练入口，训练模型并保存到 output_model/
├── test.py                  # 测试入口，加载模型进行独立评估
└── README.md                # 本文件
```

## 环境要求

| 组件 | 最低版本 |
|------|---------|
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

## 数据说明

| 文件 | 标签 | 含义 | 受试者数 | 样本数 |
|------|------|------|---------|-------|
| `dataset_BJ.csv` | 1 | 强组（BJ） | 28 人 | 81,584 |
| `dataset_ZJ.csv` | 0 | 弱组（ZJ） | 21 人 | 65,319 |

关键字段：

| 字段 | 说明 |
|------|------|
| `subject_id` | 受试者编号 |
| `clip_id` | 视频片段编号（0~20） |
| `timestamp` | 时间戳 |
| `gaze_x, gaze_y` | 注视点 x/y 坐标 |
| `original_type` | 眼动类型（Fixation/Saccade） |
| `validity` | 数据有效性（全部为 1） |

## 快速开始

```bash
# 进入项目目录
cd d:\code-en

# 训练模型（保存到 output_model/）
python train.py

# 独立测试已训练好的模型
python test.py
```

输出内容：

```
train.py: 数据加载完成 → 序列构建 → 训练（tqdm 进度条）→ Clip 级测试结果 → 受试者级投票决策 → 保存模型
test.py:  加载模型 → Clip 级预测详情表格 → 受试者级投票决策（基于全部 clip）
```

## 配置文件说明

所有超参集中在 [`configs/config.yaml`](file:///d:/code-en/configs/config.yaml)：

| 配置项 | 默认值 | 说明 |
|-------|--------|------|
| `max_seq_len` | 180 | 序列最大长度，超过截断，不足补零 |
| `num_channels` | [32,64,64,128,128] | TCN 各层通道数（5 层膨胀卷积） |
| `kernel_size` | 3 | 时序卷积核大小 |
| `dropout` | 0.2 | Dropout 比率 |
| `min_clip_len` | 10 | 最小有效 clip 长度，小于此值被过滤 |
| `batch_size` | 32 | 训练批次大小 |
| `epochs` | 50 | 最大训练轮数 |
| `learning_rate` | 1e-3 | 初始学习率 |
| `weight_decay` | 1e-4 | L2 正则化系数 |
| `patience` | 10 | 早停耐心值 |
| `test_size` | 0.1 | 测试集占比 |
| `grad_clip` | 1.0 | 梯度裁剪阈值 |

## 模型架构

```
输入 (gaze_x, gaze_y, is_fixation)  [batch, 180, 3]
    │
    ▼  transpose(1, 2)
输入 [batch, 3, 180]
    │
    ▼  TemporalConvNet (5 层膨胀卷积)
Conv1D 3→32   dilation=1
Conv1D 32→64  dilation=2
Conv1D 64→64  dilation=4
Conv1D 64→128 dilation=8
Conv1D 128→128 dilation=16
    │
    ▼  AdaptiveAvgPool1d
特征向量 [batch, 128]
    │
    ▼  Linear(128 → 2)
输出分类概率 [batch, 2]
```

特点：
- **因果卷积**：通过 Chomp1d 保证只依赖历史时刻
- **残差连接**：缓解深层网络梯度消失
- **膨胀卷积**：指数级扩大感受野
- **全局平均池化**：减少参数量，防止过拟合

## 预测与决策流程

```
输入：全部受试者的所有有效 clip
    │
    ▼ Step 1：Clip 级预测
每个 clip → 模型输出 0（ZJ/弱）或 1（BJ/强）
    │
    ▼ Step 2：受试者级 Bagging 投票
统计该受试者所有 clip 的预测结果：
  vote_1 > vote_0 → 强组（BJ/1）
  vote_1 ≤ vote_0 → 弱组（ZJ/0）
    │
    ▼ 输出
每位受试者的最终分类结果
```

## 运行示例

```powershell
PS D:\code-en> python train.py
PyTorch: 2.7.1+cu118 | Device: cuda

数据加载完成
  BJ 样本数: 81584
  ZJ 样本数: 65319
  总样本数: 146903

序列构建完成: 1024 clips, shape=(1024, 180, 3)

训练集: 921 clips | 测试集: 103 clips
模型参数量: 230,050

Training: 100%|████████| 406/1450 [00:03<00:00]
   14 |     0.6285 |    0.6135 |    0.6495 |   0.6214

受试者级准确率: 0.6327 (31/49)
```

## 自定义配置

```powershell
# 直接修改 configs/config.yaml 后运行
python train.py
```

## License

MIT
