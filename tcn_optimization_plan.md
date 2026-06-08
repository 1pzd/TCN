# TCN眼动分类模型优化方案

> 针对当前bagging后准确率66.6%的基线，从特征工程、模型架构、损失函数、训练策略、投票机制五个维度提出优化方向。

---

## 一、特征工程优化

### 1.1 问题诊断

当前仅使用 `gaze_x` 和 `gaze_y` 两个原始坐标特征，信息密度不足。眼动数据蕴含丰富的时序动态特征，仅用坐标会丢失大量判别信息。

### 1.2 建议新增特征

| 特征类别 | 具体特征 | 计算方式 | 判别价值 |
|---------|---------|---------|---------|
| **速度特征** | 速度 magnitude | `sqrt(dx² + dy²) / dt` | 强组可能有更快的扫视速度 |
| **加速度特征** | 加速度 | 速度的差分 | 反映眼动控制的精细程度 |
| **方向特征** | 运动方向角 | `atan2(dy, dx)` | 反注视策略差异 |
| **曲率特征** | 轨迹曲率 | 方向角的变化率 | 强组可能有更平滑的轨迹 |
| **统计特征** | 滑动窗口均值/方差 | 窗口大小=10~30 | 局部统计分布差异 |
| **频域特征** | FFT能量分布 | 对序列做FFT | 振动频率差异 |
| **离散度特征** | 位移离散度 | 窗口内标准差 | 注视稳定性指标 |

### 1.3 实现建议

在 `src/data_loader.py` 中添加特征计算函数：

```python
def compute_gaze_features(df: pd.DataFrame) -> pd.DataFrame:
    """从原始gaze坐标派生高级特征"""
    # 时间差分
    df['dt'] = df.groupby('unique_clip')['timestamp'].diff().fillna(0.01)
    
    # 速度
    dx = df.groupby('unique_clip')['gaze_x'].diff().fillna(0)
    dy = df.groupby('unique_clip')['gaze_y'].diff().fillna(0)
    df['velocity'] = np.sqrt(dx**2 + dy**2) / df['dt']
    
    # 加速度
    df['acceleration'] = df.groupby('unique_clip')['velocity'].diff().fillna(0) / df['dt']
    
    # 方向角
    df['direction'] = np.arctan2(dy, dx)
    
    # 曲率
    df['curvature'] = df.groupby('unique_clip')['direction'].diff().fillna(0)
    
    # 滑动窗口统计（窗口=15）
    for col in ['gaze_x', 'gaze_y', 'velocity']:
        df[f'{col}_rolling_mean'] = df.groupby('unique_clip')[col].transform(
            lambda x: x.rolling(15, min_periods=1).mean()
        )
        df[f'{col}_rolling_std'] = df.groupby('unique_clip')[col].transform(
            lambda x: x.rolling(15, min_periods=1).std().fillna(0)
        )
    
    return df
```

**预期收益**：特征维度从2维提升到10~15维，捕获更多判别信息，准确率提升3~5%。

---

## 二、模型架构优化

### 2.1 多尺度并行TCN

当前模型使用单一尺度的串行TCN。建议引入多尺度并行分支，捕获不同时间粒度的模式。

```python
class MultiScaleTCN(nn.Module):
    """多尺度并行TCN - 同时捕获短期和长期依赖"""
    def __init__(self, num_inputs, num_channels, kernel_sizes, dropout):
        super().__init__()
        # 小尺度分支：捕获快速眼动（saccade）
        self.small_scale = TemporalConvNet(
            num_inputs, [c//2 for c in num_channels], 
            kernel_size=3, dropout=dropout
        )
        # 大尺度分支：捕获慢速追踪（pursuit）
        self.large_scale = TemporalConvNet(
            num_inputs, [c//2 for c in num_channels], 
            kernel_size=7, dropout=dropout
        )
        # 融合层
        self.fusion = nn.Conv1d(num_channels[-1], num_channels[-1], 1)
    
    def forward(self, x):
        small = self.small_scale(x)  # [B, C//2, T]
        large = self.large_scale(x)  # [B, C//2, T]
        combined = torch.cat([small, large], dim=1)  # [B, C, T]
        return self.fusion(combined)
```

### 2.2 引入Transformer编码器

TCN擅长局部模式，但全局依赖建模能力有限。建议在TCN后接Transformer编码器：

```python
class TCNTransformerClassifier(nn.Module):
    """TCN + Transformer混合架构"""
    def __init__(self, tcn_model, d_model, nhead, num_layers):
        super().__init__()
        self.tcn = tcn_model
        self.pos_encoding = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, 
            dim_feedforward=d_model*4, dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.classifier = nn.Linear(d_model, 2)
    
    def forward(self, x):
        # TCN提取局部特征
        tcn_out = self.tcn(x)  # [B, C, T]
        # 转换为Transformer输入格式
        tcn_out = tcn_out.permute(2, 0, 1)  # [T, B, C]
        tcn_out = self.pos_encoding(tcn_out)
        # Transformer建模全局依赖
        trans_out = self.transformer(tcn_out)  # [T, B, C]
        # 池化 + 分类
        pooled = trans_out.mean(dim=0)  # [B, C]
        return self.classifier(pooled)
```

### 2.3 使用已定义但未用的SelfAttentionPooling

当前多层级特征融合使用简单的GAP+GMP，建议替换为SelfAttentionPooling，让模型自适应地关注重要时间步：

```python
# 在TCNClassifier中替换池化方式
# 原：gap = x.mean(dim=2)
# 新：
self.attn_pool = SelfAttentionPooling(hidden_dim)
attn_out = self.attn_pool(x.permute(0, 2, 1))  # [B, T, C] -> [B, C]
```

**预期收益**：多尺度TCN + Transformer混合架构可提升2~4%准确率。

---

## 三、损失函数优化

### 3.1 问题诊断

当前使用标准CrossEntropyLoss，对类别不平衡和难样本处理不足。

### 3.2 Focal Loss

针对类别不平衡（BJ:28人 vs ZJ:21人）和难样本：

```python
class FocalLoss(nn.Module):
    """Focal Loss - 降低易样本权重，聚焦难样本"""
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return focal_loss.mean()
```

### 3.3 对比学习损失

在特征空间中拉近同类样本、推远异类样本：

```python
class ContrastiveLoss(nn.Module):
    """监督对比学习损失"""
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, features, labels):
        # features: [B, D] 特征向量
        # labels: [B] 类别标签
        features = F.normalize(features, dim=1)
        similarity = torch.matmul(features, features.T) / self.temperature
        
        # 构建正负样本掩码
        labels = labels.unsqueeze(1)
        mask = torch.eq(labels, labels.T).float()
        
        # 对比损失
        exp_sim = torch.exp(similarity)
        log_prob = similarity - torch.log(exp_sim.sum(dim=1, keepdim=True))
        mean_log_prob = (mask * log_prob).sum(dim=1) / mask.sum(dim=1)
        loss = -mean_log_prob.mean()
        
        return loss
```

### 3.4 多任务学习

添加辅助任务增强特征学习：

```python
class MultiTaskLoss(nn.Module):
    """多任务损失：主分类 + 眼动模式预测"""
    def __init__(self, main_weight=1.0, aux_weight=0.3):
        super().__init__()
        self.main_weight = main_weight
        self.aux_weight = aux_weight
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
    
    def forward(self, main_pred, aux_pred, main_target, aux_target):
        main_loss = self.ce_loss(main_pred, main_target)
        aux_loss = self.mse_loss(aux_pred, aux_target)
        return self.main_weight * main_loss + self.aux_weight * aux_loss
```

**预期收益**：Focal Loss + 对比学习可提升1~3%准确率，尤其对难样本识别。

---

## 四、训练策略优化

### 4.1 数据增强

```python
class GazeAugmentation:
    """眼动数据增强策略"""
    
    @staticmethod
    def time_warp(seq, sigma=0.2):
        """时间扭曲：模拟不同阅读速度"""
        T = seq.shape[1]
        tt = torch.arange(T).float()
        warp = torch.randn(T) * sigma
        tt_warped = tt + warp
        tt_warped = torch.clamp(tt_warped, 0, T-1)
        return F.interpolate(seq.unsqueeze(0), size=T, mode='linear').squeeze(0)
    
    @staticmethod
    def add_noise(seq, noise_level=0.01):
        """添加高斯噪声：模拟采集误差"""
        noise = torch.randn_like(seq) * noise_level
        return seq + noise
    
    @staticmethod
    def random_crop(seq, crop_ratio=0.9):
        """随机裁剪：增强位置不变性"""
        T = seq.shape[1]
        crop_len = int(T * crop_ratio)
        start = torch.randint(0, T - crop_len, (1,))
        return seq[:, start:start+crop_len]
    
    @staticmethod
    def mixup(seq1, seq2, label1, label2, alpha=0.2):
        """Mixup数据增强"""
        lam = np.random.beta(alpha, alpha)
        mixed_seq = lam * seq1 + (1 - lam) * seq2
        mixed_label = lam * label1 + (1 - lam) * label2
        return mixed_seq, mixed_label
```

### 4.2 学习率调度优化

当前使用ReduceLROnPlateau，建议改用OneCycleLR或CosineAnnealingWarmRestarts：

```python
# OneCycleLR：更激进的学习率调度
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=1e-3,
    epochs=100,
    steps_per_epoch=len(train_loader),
    pct_start=0.3,  # 30%时间用于warmup
    anneal_strategy='cos'
)

# 或 CosineAnnealingWarmRestarts：周期性重启
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=10, T_mult=2, eta_min=1e-6
)
```

### 4.3 梯度累积

当batch_size受限时，使用梯度累积模拟大batch：

```python
accumulation_steps = 4  # 等效batch_size = 32 * 4 = 128

for i, (inputs, labels) in enumerate(train_loader):
    outputs = model(inputs)
    loss = criterion(outputs, labels) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### 4.4 标签平滑

```python
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
```

**预期收益**：数据增强 + 学习率优化可提升1~2%准确率。

---

## 五、投票机制优化

### 5.1 问题诊断

当前使用简单多数投票（threshold=0.55），未考虑预测置信度。

### 5.2 置信度加权投票

```python
def confidence_weighted_vote(predictions: List[Dict], threshold: float = 0.5) -> Tuple[int, float]:
    """置信度加权投票"""
    bj_weight = 0.0
    zj_weight = 0.0
    
    for pred in predictions:
        prob = pred['probability']
        # 使用概率作为权重，而非二值化
        if prob >= 0.5:
            bj_weight += prob
        else:
            zj_weight += (1 - prob)
    
    total_weight = bj_weight + zj_weight
    bj_ratio = bj_weight / total_weight if total_weight > 0 else 0.5
    
    prediction = 1 if bj_ratio >= threshold else 0
    return prediction, bj_ratio
```

### 5.3 学习投票权重

使用验证集学习每个折模型的最优权重：

```python
def learn_voting_weights(fold_predictions, labels):
    """学习最优投票权重"""
    from sklearn.linear_model import LogisticRegression
    
    # 收集各折预测概率
    X = np.column_stack([pred['probability'] for pred in fold_predictions])
    
    # 训练元学习器
    meta_model = LogisticRegression()
    meta_model.fit(X, labels)
    
    return meta_model.coef_[0]  # 返回学习到的权重
```

### 5.4 分层阈值优化

```python
def optimize_threshold(predictions, labels, n_trials=100):
    """在验证集上搜索最优阈值"""
    best_threshold = 0.5
    best_f1 = 0
    
    for threshold in np.linspace(0.3, 0.7, n_trials):
        preds = [1 if p >= threshold else 0 for p in predictions]
        f1 = f1_score(labels, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    return best_threshold
```

**预期收益**：置信度加权 + 阈值优化可提升1~2%准确率。

---

## 六、集成策略优化

### 6.1 模型多样性

当前3折模型结构相同，建议引入结构多样性：

```python
# 不同架构的模型
models = {
    'tcn_small': TCNClassifier(num_channels=[32, 48, 64, 96, 128]),
    'tcn_large': TCNClassifier(num_channels=[64, 96, 128, 192, 256, 320, 384, 448]),
    'tcn_multi_scale': MultiScaleTCNClassifier(),
    'tcn_transformer': TCNTransformerClassifier(),
}

# Stacking集成
def stacking_ensemble(model_preds, labels):
    """使用元学习器融合多模型预测"""
    from sklearn.ensemble import GradientBoostingClassifier
    
    X = np.column_stack(model_preds)
    meta_model = GradientBoostingClassifier(n_estimators=100)
    meta_model.fit(X, labels)
    return meta_model
```

### 6.2 快照集成

在训练过程中保存多个时间点的模型：

```python
class SnapshotEnsemble:
    """快照集成：在学习率周期最低点保存模型"""
    def __init__(self, model, n_snapshots=5):
        self.model = model
        self.snapshots = []
        self.n_snapshots = n_snapshots
    
    def save_snapshot(self):
        self.snapshots.append(copy.deepcopy(self.model.state_dict()))
    
    def predict(self, x):
        predictions = []
        for state_dict in self.snapshots:
            self.model.load_state_dict(state_dict)
            predictions.append(self.model(x).softmax(dim=1))
        return torch.stack(predictions).mean(dim=0)
```

**预期收益**：模型集成可提升2~4%准确率。

---

## 七、优化优先级排序

根据预期收益和实现难度，建议按以下优先级实施：

| 优先级 | 优化方向 | 预期收益 | 实施难度 | 建议顺序 |
|-------|---------|---------|---------|---------|
| **P0** | 特征工程 | +3~5% | 低 | 立即实施 |
| **P0** | Focal Loss | +1~2% | 低 | 立即实施 |
| **P1** | 多尺度TCN | +2~3% | 中 | 特征工程后 |
| **P1** | 数据增强 | +1~2% | 中 | 特征工程后 |
| **P1** | 置信度加权投票 | +1~2% | 低 | 立即实施 |
| **P2** | Transformer编码器 | +2~4% | 高 | 多尺度TCN后 |
| **P2** | 对比学习损失 | +1~3% | 中 | Focal Loss后 |
| **P3** | 模型集成 | +2~4% | 中 | 其他优化后 |
| **P3** | 学习率调度优化 | +1% | 低 | 随时可做 |

---

## 八、实验计划

### 实验1：特征工程验证（1~2天）
- 添加速度、加速度、曲率特征
- 对比2维 vs 10维特征的效果
- 消融实验：单个特征的贡献

### 实验2：损失函数对比（1天）
- CrossEntropy vs FocalLoss vs LabelSmoothing
- 搜索最优alpha和gamma参数

### 实验3：架构改进（2~3天）
- 串行TCN vs 多尺度TCN
- TCN vs TCN+Transformer
- 消融实验：SelfAttentionPooling的效果

### 实验4：集成策略（1~2天）
- 简单投票 vs 置信度加权 vs Stacking
- 阈值搜索实验

---

## 九、预期最终效果

| 优化阶段 | 准确率 | 提升幅度 |
|---------|--------|---------|
| 当前基线 | 66.6% | - |
| +特征工程 | 70~72% | +3~5% |
| +FocalLoss+数据增强 | 72~74% | +2~3% |
| +多尺度TCN | 74~76% | +2~3% |
| +集成优化 | 76~78% | +2~4% |
| **最终预期** | **75~80%** | **+8~13%** |

> 注：以上为保守估计，实际效果需通过实验验证。眼动分类任务的上限可能受数据质量和类别可分性限制。
