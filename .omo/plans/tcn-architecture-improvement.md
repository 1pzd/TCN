# TCN 眼动分类模型——架构改进计划

## TL;DR

> **Quick Summary**: 对 TCN 二分类模型进行 6 项结构性改进，包括激活未使用的 SelfAttentionPooling、引入双分支 SE 注意力（通道+时序）、添加正弦位置编码、实现多层级特征加权融合、构建残差分类头、保留 GAP+GMP 作为补充通道。目标是将准确率从 66.6% 提升到 72~78%。
>
> **Deliverables**:
> - 修改后的 `src/model.py`（新增 4 个模块类 + 修改 TemporalBlock + TCNClassifier）
> - 更新后的 `configs/config.yaml`（新增 5 个架构开关参数）
> - 所有修改点保留 `# ARCH_BACKUP` 注释，支持一键回滚
>
> **Estimated Effort**: Medium
> **Parallel Execution**: YES - 3 waves
> **Critical Path**: Task 1 → Task 2 → Task 3 → Task 4 → Task 5

---

## Context

### Original Request
用户要求对 TCN 眼动分类模型进行架构层面的修改（非超参数调优），在原有代码基础上改进，保留原始代码用公式注释标注以便回滚。

### 项目概况
- **任务**: 二分类（BJ 强组 label=1 vs ZJ 弱组 label=0），基于眼动凝视轨迹
- **数据**: 48 个 subject，每个 20 个 clip，每个 clip 20~200 帧（过滤后约几十万帧）
- **特征**: gaze_x, gaze_y（归一化坐标 0~1）
- **当前指标**: Accuracy=66.6%, F1=0.73, AUC=0.74
- **训练方式**: 3-fold CV（按 subject 分割，不重叠），clip 级预测 → subject 级 bagging
- **模型**: 8 层 TCN + SE 注意力 + 多层级特征融合 + MLP 分类器

### 关键文件
- `src/model.py` (154 行) — 模型定义（主要修改目标）
- `src/data_loader.py` (170 行) — 数据加载管道（不修改）
- `src/trainer.py` (261 行) — 训练逻辑（不修改）
- `src/predict.py` (148 行) — 预测与投票（不修改）
- `train.py` (71 行) — 交叉验证入口（不修改）
- `configs/config.yaml` (90 行) — 配置文件（需新增参数）
- `tcn_classifier.py` (297 行) — 独立单文件版本（不修改）

### 当前架构问题诊断
1. **SelfAttentionPooling 已定义但未使用** — 模型无法学习"哪个时间步更重要"
2. **多层级特征融合粗糙** — layer[2,4,7] 的特征直接 concat，各层等权
3. **SEBlock 只做通道注意力** — 缺少时序维度的注意力
4. **没有位置编码** — 模型不知道"当前注视在序列的哪个位置"
5. **分类头表达能力不足** — 两层 MLP 无法在最优阈值附近做出准确决策

---

## Work Objectives

### Core Objective
通过 6 项结构性改进，提升 TCN 模型的分类能力，将准确率从 66.6% 提升到 72~78%。

### Concrete Deliverables
- 修改后的 `src/model.py`：新增 DualSEBlock、SinusoidalPositionalEncoding、LevelWeightedFusion、ResidualClassifier 四个模块类
- 修改后的 `src/model.py`：TemporalBlock 使用 DualSEBlock，TCNClassifier 使用位置编码 + 自注意力池化 + 层级加权融合 + 残差分类头
- 更新后的 `configs/config.yaml`：新增 use_dual_se、se_temporal_kernel、use_positional_encoding、use_level_weighting、classifier_hidden_dim 参数

### Definition of Done
- [ ] `src/model.py` 中所有修改点都有 `# ARCH_BACKUP` 注释
- [ ] `configs/config.yaml` 新增参数可正常加载
- [ ] 模型可正常实例化和前向传播
- [ ] 训练流程可正常运行（loss 下降，指标可计算）
- [ ] 所有原始代码可通过 `# ARCH_BACKUP` 注释回滚

### Must Have
- 所有修改点保留 `# ARCH_BACKUP: 原始公式为 ...` 注释
- 新增模块类必须在 `src/model.py` 内完成，不新增文件
- 配置文件新增参数必须有默认值，不破坏现有配置
- 模型输出维度不变（2 分类）
- 位置编码使用正弦编码（非可学习），注入时序位置信息

### Must NOT Have (Guardrails)
- 不修改 `src/data_loader.py`、`src/trainer.py`、`src/predict.py`、`train.py`
- 不修改 `tcn_classifier.py`（独立单文件版本）
- 不新增文件（所有新模块在 `src/model.py` 内）
- 不改变超参数（学习率、batch_size、epoch 等）
- 不引入 Transformer 架构（只用注意力机制，不用 self-attention layer）
- 不使用可学习位置编码（用固定的正弦编码）

---

## Verification Strategy

> **ZERO HUMAN INTERVENTION** — ALL verification is agent-executed. No exceptions.

### Test Decision
- **Infrastructure exists**: NO（项目无测试框架）
- **Automated tests**: None
- **Framework**: none

### QA Policy
Every task MUST include agent-executed QA scenarios.
Evidence saved to `.omo/evidence/task-{N}-{scenario-slug}.{ext}`.

- **模型代码**: Use Bash — `python -c "..."` 验证实例化和前向传播
- **配置文件**: Use Bash — `python -c "..."` 验证配置加载
- **训练流程**: Use Bash — 运行训练命令验证 loss 下降

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Start Immediately — 新增模块定义，全部可并行):
├── Task 1: 新增 DualSEBlock 模块 [quick]
├── Task 2: 新增 SinusoidalPositionalEncoding 模块 [quick]
├── Task 3: 新增 LevelWeightedFusion 模块 [quick]
└── Task 4: 新增 ResidualClassifier 模块 [quick]

Wave 2 (After Wave 1 — 修改现有代码，依赖 Wave 1 的模块):
├── Task 5: 修改 TemporalBlock 使用 DualSEBlock [quick]
├── Task 6: 修改 TCNClassifier.__init__ 新增组件 [quick]
└── Task 7: 修改 TCNClassifier.forward() 完整前向传播 [quick]

Wave 3 (After Wave 2 — 配置与验证):
├── Task 8: 更新 configs/config.yaml 新增参数 [quick]
└── Task 9: 端到端验证（实例化 + 前向传播 + 训练 1 epoch） [unspecified-high]

Wave FINAL (After ALL tasks — 4 parallel reviews):
├── Task F1: Plan compliance audit (oracle)
├── Task F2: Code quality review (unspecified-high)
├── Task F3: Real manual QA (unspecified-high)
└── Task F4: Scope fidelity check (deep)
-> Present results -> Get explicit user okay
```

### Dependency Matrix

| Task | Depends On | Blocks |
|------|-----------|--------|
| 1 | None | 5 |
| 2 | None | 6 |
| 3 | None | 6 |
| 4 | None | 6 |
| 5 | 1 | 7 |
| 6 | 2, 3, 4 | 7 |
| 7 | 5, 6 | 9 |
| 8 | None | 9 |
| 9 | 7, 8 | F1-F4 |

### Agent Dispatch Summary

- **Wave 1**: 4 tasks → `quick` (each: 新增一个类定义)
- **Wave 2**: 3 tasks → `quick` (each: 修改现有代码的一处)
- **Wave 3**: 2 tasks → `quick` + `unspecified-high`
- **FINAL**: 4 tasks → `oracle` + `unspecified-high` + `unspecified-high` + `deep`

---

## TODOs

- [ ] 1. 新增 DualSEBlock 模块（通道+时序双分支注意力）

  **What to do**:
  - 在 `src/model.py` 中，`SEBlock` 类定义之后，新增 `DualSEBlock` 类
  - 实现通道注意力分支：AdaptiveAvgPool1d(1) → Flatten → Linear(C, C//reduction) → ReLU → Linear(C//reduction, C) → Sigmoid
  - 实现时序注意力分支：Conv1d(C, C, kernel_size, padding, groups=C) → Sigmoid（深度可分离卷积）
  - forward 中：`x * s_c * s_t`（通道注意力 × 时序注意力）
  - 保留原始 `SEBlock` 类不删除（回滚用）

  **Must NOT do**:
  - 不删除原始 `SEBlock` 类
  - 不修改 `SEBlock` 的任何代码
  - 不引入 Transformer 的 MultiheadAttention

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 单个类定义，逻辑清晰，无复杂依赖
  - **Skills**: []
    - 无需特殊技能

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 2, 3, 4)
  - **Blocks**: Task 5
  - **Blocked By**: None (can start immediately)

  **References**:

  **Pattern References**:
  - `src/model.py:18-32` — 原始 `SEBlock` 实现，DualSEBlock 的通道注意力分支应保持相同逻辑

  **API/Type References**:
  - PyTorch `nn.Conv1d` 的 `groups` 参数：当 `groups=channels` 时为深度可分离卷积，每个通道独立卷积

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: DualSEBlock 实例化和前向传播
    Tool: Bash (python -c)
    Preconditions: src/model.py 已修改
    Steps:
      1. python -c "from src.model import DualSEBlock; import torch; se = DualSEBlock(64, reduction=8, kernel_size=7); x = torch.randn(2, 64, 200); y = se(x); print(f'Input: {x.shape}, Output: {y.shape}')"
      2. 验证输出形状与输入形状一致：(2, 64, 200)
    Expected Result: Output shape = Input shape = (2, 64, 200)
    Failure Indicators: ImportError, RuntimeError, shape mismatch
    Evidence: .omo/evidence/task-1-dual-se-forward.txt

  Scenario: DualSEBlock 与原始 SEBlock 共存
    Tool: Bash (python -c)
    Preconditions: src/model.py 已修改
    Steps:
      1. python -c "from src.model import SEBlock, DualSEBlock; print('Both classes imported successfully')"
    Expected Result: 两个类都可正常导入，无冲突
    Failure Indicators: ImportError, name collision
    Evidence: .omo/evidence/task-1-coexist.txt
  ```

  **Commit**: YES
  - Message: `feat(model): add DualSEBlock with channel + temporal attention`
  - Files: `src/model.py`
  - Pre-commit: `python -c "from src.model import DualSEBlock; print('OK')"`

- [ ] 2. 新增 SinusoidalPositionalEncoding 模块

  **What to do**:
  - 在 `src/model.py` 中，`DualSEBlock` 类定义之后，新增 `SinusoidalPositionalEncoding` 类
  - 实现正弦位置编码：`PE(pos, 2i) = sin(pos / 10000^(2i/d))`，`PE(pos, 2i+1) = cos(pos / 10000^(2i/d))`
  - 使用 `register_buffer` 注册编码矩阵（不参与梯度计算）
  - forward 中：`x + pe[:, :, :T]`（将位置编码加到输入上）
  - 编码矩阵形状：`(1, D, T)`，其中 D=d_model，T=max_len

  **Must NOT do**:
  - 不使用可学习的位置编码（必须用固定的正弦编码）
  - 不修改任何现有代码

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 单个类定义，数学公式明确，无复杂依赖
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 3, 4)
  - **Blocks**: Task 6
  - **Blocked By**: None (can start immediately)

  **References**:

  **Pattern References**:
  - Transformer 论文 "Attention Is All You Need" 中的正弦位置编码公式

  **API/Type References**:
  - PyTorch `register_buffer`：注册不参与梯度计算的张量
  - `torch.arange`、`torch.exp`、`torch.sin`、`torch.cos` 的用法

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: SinusoidalPositionalEncoding 实例化和前向传播
    Tool: Bash (python -c)
    Preconditions: src/model.py 已修改
    Steps:
      1. python -c "from src.model import SinusoidalPositionalEncoding; import torch; pe = SinusoidalPositionalEncoding(2, max_len=200); x = torch.randn(2, 2, 200); y = pe(x); print(f'Input: {x.shape}, Output: {y.shape}')"
      2. 验证输出形状与输入形状一致：(2, 2, 200)
    Expected Result: Output shape = Input shape = (2, 2, 200)
    Failure Indicators: ImportError, RuntimeError, shape mismatch
    Evidence: .omo/evidence/task-2-pos-encoding-forward.txt

  Scenario: 位置编码值域检查
    Tool: Bash (python -c)
    Preconditions: src/model.py 已修改
    Steps:
      1. python -c "from src.model import SinusoidalPositionalEncoding; import torch; pe = SinusoidalPositionalEncoding(64, max_len=200); print('PE range:', pe.pe.min().item(), '~', pe.pe.max().item())"
      2. 验证编码值在 [-1, 1] 范围内
    Expected Result: min >= -1.0, max <= 1.0
    Failure Indicators: 值超出 [-1, 1] 范围
    Evidence: .omo/evidence/task-2-pos-encoding-range.txt
  ```

  **Commit**: YES (groups with Task 1)
  - Message: `feat(model): add SinusoidalPositionalEncoding`
  - Files: `src/model.py`
  - Pre-commit: `python -c "from src.model import SinusoidalPositionalEncoding; print('OK')"`

- [ ] 3. 新增 LevelWeightedFusion 模块

  **What to do**:
  - 在 `src/model.py` 中，`SinusoidalPositionalEncoding` 类定义之后，新增 `LevelWeightedFusion` 类
  - 实现门控机制：Linear(D*L, L) → Softmax(dim=-1)，其中 L=层级数，D=每层特征维度
  - forward 中：将各层特征 concat → 通过门控得到权重 → 各层特征乘以对应权重 → 再 concat
  - 输出维度：`(B, D * L)`（与原始 concat 维度相同，但各层特征已被加权）

  **Must NOT do**:
  - 不使用加权求和（必须保留 concat 结构，只调节各层贡献比例）
  - 不修改任何现有代码

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 单个类定义，逻辑清晰，无复杂依赖
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 2, 4)
  - **Blocks**: Task 6
  - **Blocked By**: None (can start immediately)

  **References**:

  **Pattern References**:
  - `src/model.py:136-139` — 原始的直接 concat 融合方式，LevelWeightedFusion 替代此处逻辑

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: LevelWeightedFusion 实例化和前向传播
    Tool: Bash (python -c)
    Preconditions: src/model.py 已修改
    Steps:
      1. python -c "from src.model import LevelWeightedFusion; import torch; fusion = LevelWeightedFusion(3, 448); feats = [torch.randn(2, 448), torch.randn(2, 448), torch.randn(2, 448)]; y = fusion(feats); print(f'Output shape: {y.shape}')"
      2. 验证输出形状：(2, 448*3) = (2, 1344)
    Expected Result: Output shape = (2, 1344)
    Failure Indicators: ImportError, RuntimeError, shape mismatch
    Evidence: .omo/evidence/task-3-level-fusion-forward.txt

  Scenario: 权重和为 1 验证
    Tool: Bash (python -c)
    Preconditions: src/model.py 已修改
    Steps:
      1. python -c "from src.model import LevelWeightedFusion; import torch; fusion = LevelWeightedFusion(3, 448); feats = [torch.randn(2, 448), torch.randn(2, 448), torch.randn(2, 448)]; concat = torch.cat(feats, dim=-1); weights = fusion.gate(concat); print('Weight sum:', weights.sum(dim=-1).mean().item())"
      2. 验证权重和 ≈ 1.0（softmax 输出）
    Expected Result: 权重和 ≈ 1.0（误差 < 0.001）
    Failure Indicators: 权重和远离 1.0
    Evidence: .omo/evidence/task-3-weight-sum.txt
  ```

  **Commit**: YES (groups with Tasks 1, 2)
  - Message: `feat(model): add LevelWeightedFusion for multi-level feature weighting`
  - Files: `src/model.py`
  - Pre-commit: `python -c "from src.model import LevelWeightedFusion; print('OK')"`

- [ ] 4. 新增 ResidualClassifier 模块

  **What to do**:
  - 在 `src/model.py` 中，`LevelWeightedFusion` 类定义之后，新增 `ResidualClassifier` 类
  - 实现主路径：LayerNorm → Linear(input_dim, hidden_dim) → GELU → Dropout → Linear(hidden_dim, hidden_dim//2) → GELU → Dropout → Linear(hidden_dim//2, num_classes)
  - 实现跳跃连接：Linear(input_dim, num_classes)
  - forward 中：`out = fc3(h) + skip(x)`（残差连接）
  - 保留原始 `self.classifier` 的 `# ARCH_BACKUP` 注释

  **Must NOT do**:
  - 不删除原始分类器代码（保留为注释）
  - 不使用 ReLU（用 GELU 避免 dead neuron）

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 单个类定义，标准 MLP 结构
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 2, 3)
  - **Blocks**: Task 6
  - **Blocked By**: None (can start immediately)

  **References**:

  **Pattern References**:
  - `src/model.py:148-153` — 原始分类器定义，ResidualClassifier 替代此处

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: ResidualClassifier 实例化和前向传播
    Tool: Bash (python -c)
    Preconditions: src/model.py 已修改
    Steps:
      1. python -c "from src.model import ResidualClassifier; import torch; clf = ResidualClassifier(2688, 1344, 2, 0.5); x = torch.randn(2, 2688); y = clf(x); print(f'Input: {x.shape}, Output: {y.shape}')"
      2. 验证输出形状：(2, 2)
    Expected Result: Output shape = (2, 2)
    Failure Indicators: ImportError, RuntimeError, shape mismatch
    Evidence: .omo/evidence/task-4-residual-clf-forward.txt

  Scenario: 残差连接梯度流验证
    Tool: Bash (python -c)
    Preconditions: src/model.py 已修改
    Steps:
      1. python -c "from src.model import ResidualClassifier; import torch; clf = ResidualClassifier(2688, 1344, 2, 0.5); x = torch.randn(2, 2688, requires_grad=True); y = clf(x); y.sum().backward(); print('Skip grad exists:', clf.skip.weight.grad is not None); print('FC3 grad exists:', clf.fc3.weight.grad is not None)"
      2. 验证跳跃连接和主路径的梯度都存在
    Expected Result: 两个梯度都存在（True, True）
    Failure Indicators: 任一梯度为 None
    Evidence: .omo/evidence/task-4-residual-grad.txt
  ```

  **Commit**: YES (groups with Tasks 1, 2, 3)
  - Message: `feat(model): add ResidualClassifier with skip connection`
  - Files: `src/model.py`
  - Pre-commit: `python -c "from src.model import ResidualClassifier; print('OK')"`

- [ ] 5. 修改 TemporalBlock 使用 DualSEBlock

  **What to do**:
  - 在 `src/model.py` 的 `TemporalBlock.__init__` 中，将 `self.se = SEBlock(n_inputs, se_reduction)` 改为 `self.dual_se = DualSEBlock(n_inputs, se_reduction, se_temporal_kernel)`
  - 在 `TemporalBlock.forward` 中，将 `out = self.se(out)` 改为 `out = self.dual_se(out)`
  - 在修改行上方添加注释：`# ARCH_BACKUP: 原始公式为 out = SEBlock(out) = out * sigmoid(FC(GAP(out)))`
  - 保留原始 `self.se` 行作为注释（不删除）

  **Must NOT do**:
  - 不删除原始 `self.se` 行（保留为注释）
  - 不修改 `TemporalBlock` 的其他部分（残差连接、chomp 等）
  - 不修改 `TemporalConvNet` 的任何代码

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 两行代码修改，逻辑清晰
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 6, 7)
  - **Blocks**: Task 7
  - **Blocked By**: Task 1 (需要 DualSEBlock 已定义)

  **References**:

  **Pattern References**:
  - `src/model.py:48-49` — 原始 `self.se` 初始化
  - `src/model.py:66` — 原始 `out = self.se(out)` 调用
  - Task 1 的 DualSEBlock 定义 — 确认构造函数参数

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: TemporalBlock 使用 DualSEBlock 前向传播
    Tool: Bash (python -c)
    Preconditions: src/model.py 已修改，Task 1 已完成
    Steps:
      1. python -c "from src.model import TemporalBlock; import torch; block = TemporalBlock(2, 64, kernel_size=5, stride=1, dilation=1, padding=4, dropout=0.2, se_reduction=8); x = torch.randn(2, 2, 200); y = block(x); print(f'Input: {x.shape}, Output: {y.shape}')"
      2. 验证输出形状：(2, 64, 200)
    Expected Result: Output shape = (2, 64, 200)
    Failure Indicators: ImportError, RuntimeError, shape mismatch
    Evidence: .omo/evidence/task-5-temporal-block-forward.txt

  Scenario: ARCH_BACKUP 注释完整性检查
    Tool: Grep
    Preconditions: src/model.py 已修改
    Steps:
      1. grep -n "ARCH_BACKUP" src/model.py | grep -i "SEBlock"
      2. 验证 TemporalBlock 部分有 ARCH_BACKUP 注释
    Expected Result: 至少 1 行包含 SEBlock 的 ARCH_BACKUP 注释
    Failure Indicators: 无 ARCH_BACKUP 注释
    Evidence: .omo/evidence/task-5-arch-backup-check.txt
  ```

  **Commit**: YES
  - Message: `refactor(model): replace SEBlock with DualSEBlock in TemporalBlock`
  - Files: `src/model.py`
  - Pre-commit: `python -c "from src.model import TemporalBlock; print('OK')"`

- [ ] 6. 修改 TCNClassifier.__init__ 新增组件

  **What to do**:
  - 在 `src/model.py` 的 `TCNClassifier.__init__` 中，新增以下组件：
    1. `self.pos_encoding = SinusoidalPositionalEncoding(input_channels, max_seq_len)` — 位置编码
    2. `self.attn_pool = SelfAttentionPooling(out_channels)` — 自注意力池化
    3. `self.level_fusion = LevelWeightedFusion(len(fusion_levels), out_channels)` — 层级加权融合
  - 将 `self.classifier` 替换为 `ResidualClassifier`：
    ```python
    self.classifier = ResidualClassifier(
        input_dim=fusion_ch,
        hidden_dim=fusion_ch // 2,
        num_classes=num_classes,
        dropout=dropout
    )
    ```
  - 在修改行上方添加注释：`# ARCH_BACKUP: 原始公式为 y = Linear(ReLU(Linear(LayerNorm(x))))`
  - 保留原始 `self.classifier` 定义作为注释

  **Must NOT do**:
  - 不删除原始 `self.classifier` 定义（保留为注释）
  - 不修改 `TCNClassifier` 的其他 `__init__` 代码（TCN 初始化等）
  - 不修改 `TemporalConvNet` 的任何代码

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 在 __init__ 中新增几行，逻辑清晰
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 5, 7)
  - **Blocks**: Task 7
  - **Blocked By**: Tasks 2, 3, 4 (需要 SinusoidalPositionalEncoding, LevelWeightedFusion, ResidualClassifier 已定义)

  **References**:

  **Pattern References**:
  - `src/model.py:108-115` — 原始 `TCNClassifier.__init__` 中的分类器定义
  - Tasks 2, 3, 4 的模块定义 — 确认构造函数参数

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: TCNClassifier 实例化（修改后的 __init__）
    Tool: Bash (python -c)
    Preconditions: src/model.py 已修改，Tasks 1-4 已完成
    Steps:
      1. python -c "from src.model import TCNClassifier; m = TCNClassifier(input_channels=2); print('Model instantiated successfully'); print(f'Has pos_encoding: {hasattr(m, \"pos_encoding\")}'); print(f'Has attn_pool: {hasattr(m, \"attn_pool\")}'); print(f'Has level_fusion: {hasattr(m, \"level_fusion\")}')"
      2. 验证所有新增组件都存在
    Expected Result: 三个组件都存在（True, True, True）
    Failure Indicators: AttributeError, ImportError
    Evidence: .omo/evidence/task-6-init-check.txt

  Scenario: ARCH_BACKUP 注释完整性检查
    Tool: Grep
    Preconditions: src/model.py 已修改
    Steps:
      1. grep -n "ARCH_BACKUP" src/model.py | grep -i "classifier"
      2. 验证 TCNClassifier.__init__ 部分有 ARCH_BACKUP 注释
    Expected Result: 至少 1 行包含 classifier 的 ARCH_BACKUP 注释
    Failure Indicators: 无 ARCH_BACKUP 注释
    Evidence: .omo/evidence/task-6-arch-backup-check.txt
  ```

  **Commit**: YES (groups with Task 5)
  - Message: `refactor(model): integrate new modules into TCNClassifier.__init__`
  - Files: `src/model.py`
  - Pre-commit: `python -c "from src.model import TCNClassifier; print('OK')"`

- [ ] 7. 修改 TCNClassifier.forward() 完整前向传播

  **What to do**:
  - 修改 `src/model.py` 的 `TCNClassifier.forward()` 方法，按以下顺序：
    1. **位置编码**：在 TCN 之前添加 `x = self.pos_encoding(x)`
       - 注释：`# ARCH_BACKUP: 原始公式为 y = TCN(x)`
    2. **自注意力池化**：将 `gap = F.adaptive_avg_pool1d(y, 1).squeeze(-1)` 替换为：
       ```python
       z_attn = self.attn_pool(y)  # 自注意力池化
       gap = F.adaptive_avg_pool1d(y, 1).squeeze(-1)  # 保留作为补充
       gmp = F.adaptive_max_pool1d(y, 1).squeeze(-1)  # 保留作为补充
       fused_outputs.append(torch.cat([z_attn, gap, gmp], dim=1))
       ```
       - 注释：`# ARCH_BACKUP: 原始公式为 z = concat(GAP(x), GMP(x))`
    3. **层级加权融合**：将 `fused = self.norm(fused)` 替换为：
       ```python
       fused = self.level_fusion(fusion_outputs)
       fused = self.norm(fused)
       ```
       - 注释：`# ARCH_BACKUP: 原始公式为 z = LayerNorm(concat(z_0, z_1, z_2))`
    4. **分类器**：`out = self.classifier(fused)` 保持不变（ResidualClassifier 已在 __init__ 中替换）

  **Must NOT do**:
  - 不修改 `TCNClassifier.forward()` 的其他部分（dropout、reshape 等）
  - 不改变输出维度（仍然是 (B, 2)）
  - 不删除原始代码行（保留为注释）

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 按照明确的修改清单逐行修改，无复杂逻辑
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 5, 6)
  - **Blocks**: Task 9
  - **Blocked By**: Tasks 5, 6 (需要 TemporalBlock 和 __init__ 已修改)

  **References**:

  **Pattern References**:
  - `src/model.py:126-153` — 原始 `TCNClassifier.forward()` 完整实现
  - Tasks 5, 6 的修改 — 确认新增组件已就位

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: TCNClassifier 完整前向传播
    Tool: Bash (python -c)
    Preconditions: src/model.py 已修改，Tasks 1-6 已完成
    Steps:
      1. python -c "from src.model import TCNClassifier; import torch; m = TCNClassifier(input_channels=2); x = torch.randn(2, 2, 200); y = m(x); print(f'Input: {x.shape}, Output: {y.shape}')"
      2. 验证输出形状：(2, 2)
    Expected Result: Output shape = (2, 2)
    Failure Indicators: RuntimeError, shape mismatch, NaN in output
    Evidence: .omo/evidence/task-7-forward-pass.txt

  Scenario: 变长序列前向传播（padding 场景）
    Tool: Bash (python -c)
    Preconditions: src/model.py 已修改
    Steps:
      1. python -c "from src.model import TCNClassifier; import torch; m = TCNClassifier(input_channels=2); x = torch.randn(2, 2, 50); y = m(x); print(f'Short sequence - Input: {x.shape}, Output: {y.shape}')"
      2. 验证短序列（50 帧）也能正常前向传播
    Expected Result: Output shape = (2, 2)，无错误
    Failure Indicators: RuntimeError, shape mismatch
    Evidence: .omo/evidence/task-7-short-seq-forward.txt

  Scenario: ARCH_BACKUP 注释完整性检查
    Tool: Grep
    Preconditions: src/model.py 已修改
    Steps:
      1. grep -c "ARCH_BACKUP" src/model.py
      2. 验证至少有 4 处 ARCH_BACKUP 注释（DualSE、PE、Fusion、Classifier）
    Expected Result: count >= 4
    Failure Indicators: count < 4
    Evidence: .omo/evidence/task-7-arch-backup-count.txt
  ```

  **Commit**: YES
  - Message: `refactor(model): update TCNClassifier.forward() with new modules`
  - Files: `src/model.py`
  - Pre-commit: `python -c "from src.model import TCNClassifier; import torch; m = TCNClassifier(input_channels=2); x = torch.randn(2, 2, 200); y = m(x); print('OK')"`

- [ ] 8. 更新 configs/config.yaml 新增架构开关参数

  **What to do**:
  - 在 `configs/config.yaml` 的 `model:` 部分末尾新增以下参数：
    ```yaml
    # 架构改进开关（新增）
    use_dual_se: true              # 是否使用双分支 SE（通道+时序）
    se_temporal_kernel: 7          # 时序注意力卷积核大小
    use_positional_encoding: true  # 是否使用正弦位置编码
    use_level_weighting: true      # 是否使用层级加权融合
    classifier_hidden_dim: null    # 分类器隐藏层维度，null 则为 fusion_ch // 2
    ```
  - 保持原有所有参数不变
  - 新增参数必须有合理默认值，不破坏现有配置加载

  **Must NOT do**:
  - 不修改任何原有参数的值
  - 不修改其他 section（training、data 等）
  - 不删除任何原有注释

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 纯文本追加，无逻辑
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Task 9)
  - **Blocks**: Task 9
  - **Blocked By**: None (can start immediately)

  **References**:

  **Pattern References**:
  - `configs/config.yaml:17-29` — 原始 `model:` section，新增参数追加到末尾

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: 配置文件加载验证
    Tool: Bash (python -c)
    Preconditions: configs/config.yaml 已修改
    Steps:
      1. python -c "import yaml; cfg = yaml.safe_load(open('configs/config.yaml')); print('use_dual_se:', cfg['model']['use_dual_se']); print('se_temporal_kernel:', cfg['model']['se_temporal_kernel']); print('use_positional_encoding:', cfg['model']['use_positional_encoding']); print('use_level_weighting:', cfg['model']['use_level_weighting']); print('classifier_hidden_dim:', cfg['model']['classifier_hidden_dim'])"
      2. 验证所有新增参数都可正常读取
    Expected Result: 5 个参数都可正常读取，值分别为 true/7/true/true/null
    Failure Indicators: KeyError, yaml.YAMLError
    Evidence: .omo/evidence/task-8-config-load.txt

  Scenario: 原有参数未被修改
    Tool: Bash (python -c)
    Preconditions: configs/config.yaml 已修改
    Steps:
      1. python -c "import yaml; cfg = yaml.safe_load(open('configs/config.yaml')); m = cfg['model']; assert m['input_channels'] == 2, 'input_channels changed'; assert m['max_seq_len'] == 200, 'max_seq_len changed'; assert m['se_reduction'] == 8, 'se_reduction changed'; print('All original params intact')"
      2. 验证关键原有参数值未变
    Expected Result: "All original params intact"
    Failure Indicators: AssertionError
    Evidence: .omo/evidence/task-8-original-params.txt
  ```

  **Commit**: YES
  - Message: `feat(config): add architecture switch parameters`
  - Files: `configs/config.yaml`
  - Pre-commit: `python -c "import yaml; yaml.safe_load(open('configs/config.yaml')); print('OK')"`

- [ ] 9. 端到端验证（实例化 + 前向传播 + 训练 1 epoch）

  **What to do**:
  - 运行端到端验证，确认整个管道正常工作：
    1. 模型实例化：`TCNClassifier(input_channels=2)` 无错误
    2. 前向传播：输入 `(2, 2, 200)` → 输出 `(2, 2)` 无错误
    3. 训练 1 epoch：运行 `python train.py --config configs/config.yaml` 的前 2 个 batch，验证 loss 下降
    4. 配置加载：验证 `configs/config.yaml` 中所有新增参数可正常加载
  - 保存所有验证输出到 `.omo/evidence/task-9-e2e/`

  **Must NOT do**:
  - 不运行完整训练（只验证前 2 个 batch）
  - 不修改任何代码

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: 需要理解整个训练管道，可能需要调试
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Task 8)
  - **Blocks**: F1-F4
  - **Blocked By**: Tasks 7, 8 (需要所有代码修改和配置更新完成)

  **References**:

  **Pattern References**:
  - `train.py` — 训练入口，了解如何启动训练
  - `src/trainer.py` — 训练逻辑，了解 loss 计算和优化器
  - `src/data_loader.py` — 数据加载，了解数据格式

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: 模型实例化 + 前向传播
    Tool: Bash (python -c)
    Preconditions: Tasks 1-8 已完成
    Steps:
      1. python -c "from src.model import TCNClassifier; import torch; m = TCNClassifier(input_channels=2); x = torch.randn(4, 2, 200); y = m(x); print(f'Output shape: {y.shape}'); print(f'Output sample: {y[0].detach().numpy()}')"
      2. 验证输出形状 (4, 2)，无 NaN/Inf
    Expected Result: Output shape = (4, 2)，值为有限数
    Failure Indicators: RuntimeError, NaN, Inf
    Evidence: .omo/evidence/task-9-e2e/forward-pass.txt

  Scenario: 训练管道验证（2 个 batch）
    Tool: Bash (python train.py)
    Preconditions: Tasks 1-8 已完成，数据文件存在
    Steps:
      1. 运行训练脚本（限制为 2 个 batch）
      2. 验证 loss 值存在且为有限数
      3. 验证无 runtime error
    Expected Result: loss 为有限数，无错误
    Failure Indicators: RuntimeError, NaN loss, 数据加载错误
    Evidence: .omo/evidence/task-9-e2e/training-log.txt

  Scenario: 配置加载 + 模型参数统计
    Tool: Bash (python -c)
    Preconditions: Tasks 1-8 已完成
    Steps:
      1. python -c "from src.model import TCNClassifier; import yaml; cfg = yaml.safe_load(open('configs/config.yaml')); m = TCNClassifier(**{k:v for k,v in cfg['model'].items() if k in ['input_channels','output_channels','kernel_size','dropout','max_seq_len','se_reduction','fusion_levels']}); total = sum(p.numel() for p in m.parameters()); trainable = sum(p.numel() for p in m.parameters() if p.requires_grad); print(f'Total params: {total:,}'); print(f'Trainable params: {trainable:,}')"
      2. 验证参数量合理（预计增加 15~20%）
    Expected Result: 参数量统计正常，无错误
    Failure Indicators: ImportError, KeyError
    Evidence: .omo/evidence/task-9-e2e/param-count.txt
  ```

  **Commit**: NO (验证任务，不产生代码变更)

---

## Final Verification Wave

- [ ] F1. **Plan Compliance Audit** — `oracle`
  Read the plan end-to-end. For each "Must Have": verify implementation exists (read file, check comments). For each "Must NOT Have": search codebase for forbidden patterns — reject with file:line if found. Check evidence files exist in .omo/evidence/. Compare deliverables against plan.
  Output: `Must Have [N/N] | Must NOT Have [N/N] | Tasks [N/N] | VERDICT: APPROVE/REJECT`

- [ ] F2. **Code Quality Review** — `unspecified-high`
  Review `src/model.py` for: syntax errors, import consistency, dimension mismatches, `# ARCH_BACKUP` comment completeness. Run `python -c "from src.model import TCNClassifier; print('OK')"` to verify importability.
  Output: `Import [PASS/FAIL] | ARCH_BACKUP [N/N complete] | VERDICT`

- [ ] F3. **Real Manual QA** — `unspecified-high`
  Run full training pipeline for 1 fold, 2 epochs. Verify: loss decreases, metrics are computed, no runtime errors. Save training log to `.omo/evidence/final-qa/training-log.txt`.
  Output: `Training [PASS/FAIL] | Loss Decreased [YES/NO] | Metrics [values] | VERDICT`

- [ ] F4. **Scope Fidelity Check** — `deep`
  For each task: read "What to do", read actual diff. Verify 1:1 — everything in spec was built (no missing), nothing beyond spec was built (no creep). Check "Must NOT do" compliance. Detect cross-task contamination. Flag unaccounted changes.
  Output: `Tasks [N/N compliant] | Contamination [CLEAN/N issues] | Unaccounted [CLEAN/N files] | VERDICT`

---

## Commit Strategy

- **Wave 1**: `feat(model): add DualSEBlock, PositionalEncoding, LevelWeightedFusion, ResidualClassifier modules` — src/model.py
- **Wave 2**: `refactor(model): integrate new modules into TemporalBlock and TCNClassifier` — src/model.py
- **Wave 3**: `feat(config): add architecture switch parameters` — configs/config.yaml
- **Final**: `chore: verify architecture improvements` — evidence files

---

## Success Criteria

### Verification Commands
```bash
python -c "from src.model import TCNClassifier; m = TCNClassifier(input_channels=2); import torch; x = torch.randn(2, 2, 200); y = m(x); print(f'Output shape: {y.shape}')"  # Expected: Output shape: torch.Size([2, 2])
```

### Final Checklist
- [ ] All "Must Have" present
- [ ] All "Must NOT Have" absent
- [ ] All `# ARCH_BACKUP` comments in place
- [ ] Model can be instantiated and forward pass works
- [ ] Training pipeline runs without errors
