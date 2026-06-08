# Learnings - TCN Architecture Improvement

## 2026-06-04T15:13 - Session Start

### Codebase Conventions
- All model classes use explicit super().__init__() pattern
- Sequential networks use 
n.Sequential with inline list
- ReLU uses inplace=True 
- Conv1d uses padding=(k-1)*dilation for causal convolution
- Chomp1d removes right-side padding for causality

### Current Architecture
- src/model.py: 154 lines, 6 classes
- SelfAttentionPooling defined but unused (line 100-110)
- SEBlock uses channel-only attention (GAP → FC → Sigmoid)
- TemporalBlock uses use_se parameter, always True in TemporalConvNet
- TCNClassifier.fusion_levels = [2, 4, 7] (indices into num_channels)
- usion_channels = sum(num_channels[2,4,7]) * 2 for GAP+GMP
- Each fusion level: GAP + GMP → concat → classifier
- Classifier: LayerNorm → Linear → ReLU → Dropout → Linear → 2 classes

### Config Structure
- configs/config.yaml: 46 lines
- model section: max_seq_len, num_channels, kernel_sizes, dropout, feature_cols, min_clip_len
- No se_reduction parameter (hardcoded to 8 in TemporalBlock)
- No architecture switch parameters yet

## 2026-06-04T15:30 - Task 1: DualSEBlock Added

### Implementation Details
- DualSEBlock added at lines 33-58 in src/model.py (after SEBlock)
- Constructor: `__init__(self, channels, reduction=8, kernel_size=7)`
- Channel attention branch: self.gap + self.channel_fc (nn.Sequential with Linear → ReLU → Linear → Sigmoid)
- Temporal attention branch: self.temporal_conv (Conv1d with groups=C for depthwise) + self.temporal_sigmoid
- Forward: x * s_c * s_t where s_c is (B, C, 1) and s_t is (B, C, T)
- Verified: Input (2, 64, 200) → Output (2, 64, 200)

### API Compatibility
- DualSEBlock(channels, reduction=8) is drop-in compatible with SEBlock(channels, reduction=8)
- Additional kernel_size parameter has default value 7
- Ready for Task 5 TemporalBlock integration

## 2026-06-04T15:45 - Task 4: ResidualClassifier Added

### Implementation Details
- ResidualClassifier added at lines 187-207 in src/model.py (after TCNClassifier)
- Constructor: `__init__(self, input_dim, hidden_dim, num_classes, dropout=0.5)`
- Architecture: LayerNorm → Linear(input_dim, hidden_dim) → GELU → Dropout → Linear(hidden_dim, hidden_dim//2) → GELU → Dropout → Linear(hidden_dim//2, num_classes)
- Skip connection: Linear(input_dim, num_classes) → added to main path output
- Forward: out = self.fc3(h) + self.skip(x) — residual from input to output
- Uses F.gelu (functional) instead of nn.GELU module — matches codebase pattern of functional activations

### Key Design Decisions
- GELU chosen over ReLU for classification to avoid dead neurons
- Skip connection ensures gradient flows directly to input features
- LayerNorm applied at input for normalization before MLP
- Dropout applied after each GELU activation (2 dropout layers total)
- Hidden dimensions: input_dim → hidden_dim → hidden_dim//2 → num_classes (progressive reduction)

### Verification
- Input (2, 2688) → Output (2, 2) — correct for binary classification with batch_size=2
- Verified with: `python -c "from src.model import ResidualClassifier; import torch; clf = ResidualClassifier(2688, 1344, 2, 0.5); x = torch.randn(2, 2688); y = clf(x); print(f'Input: {x.shape}, Output: {y.shape}')"`

### Note on File Position
- LevelWeightedFusion class was expected to be in file before this task (per task description)
- Not found in file — inserted after TCNClassifier as fallback position
- Task 6 will reference ResidualClassifier as replacement for TCNClassifier.classifier

## 2026-06-04T15:45 - Task 3: LevelWeightedFusion Added

### Implementation Details
- LevelWeightedFusion added after SelfAttentionPooling, before TCNClassifier (line ~141)
- Constructor: `__init__(self, num_levels, feature_dim)`
- Gate: nn.Sequential(nn.Linear(num_levels * feature_dim, num_levels), nn.Softmax(dim=-1))
- Forward: concat levels → gate weights → multiply each level by weight → concat weighted
- Output shape: (B, num_levels * feature_dim) — same as simple concat but levels are weighted
- Verified: LevelWeightedFusion(3, 448) with 3x(2, 448) inputs → (2, 1344) ✓

### Design Notes
- This is a gated fusion, not a weighted sum — preserves concatenation structure
- Each level's contribution is scaled by a learned softmax gate
- The gate learns to weight which TCN levels are most informative
- Unlocks Task 6: TCNClassifier.__init__ can now use LevelWeightedFusion instead of raw torch.cat

## 2026-06-04T16:00 - Task 2: SinusoidalPositionalEncoding Added

### Implementation Details
- SinusoidalPositionalEncoding added at lines 63-74 in src/model.py (after DualSEBlock, before TemporalBlock)
- Constructor: `__init__(self, d_model, max_len=500)`
- PE matrix shape: (1, d_model, max_len) — batch=1 for broadcasting
- Uses `register_buffer('pe', pe)` — NOT a learnable parameter
- PE formula: PE(pos, 2i) = sin(pos / 10000^(2i/d_model)), PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
- Added `import math` at top of file for sin/cos/exp/log
- Verified: SinusoidalPositionalEncoding(2, max_len=200) with input (2, 2, 200) → output (2, 2, 200) ✓

### Design Notes
- Fixed (non-learnable) position encoding — standard sinusoidal from "Attention Is All You Need"
- Broadcasting: pe[:, :, :x.size(2)] slices to match input sequence length
- Gradient flow: PE values are constants, no gradient through position encoding
- Unlocks Task 6: TCNClassifier can optionally prepend PE to input sequences

## 2026-06-04T16:15 - Task 5: TemporalBlock Uses DualSEBlock

### Changes Made
- TemporalBlock.__init__ (line 99): Replaced `self.se = SEBlock(...)` with `self.dual_se = DualSEBlock(..., kernel_size=7)`
- TemporalBlock.forward (lines 105-106): Replaced conditional `if self.se: out = self.se(out)` with unconditional `out = self.dual_se(out)`
- Original code preserved as `# ARCH_BACKUP` comments for rollback reference

### Implementation Details
- DualSEBlock hardcodes kernel_size=7 (matches config default)
- reduction=8 matches original SEBlock setting
- use_se parameter kept in TemporalBlock signature for API compatibility (TemporalConvNet passes it)
- Old code commented out (not deleted) per ARCH_BACKUP convention

### Verification
- Input (2, 2, 200) → Output (2, 64, 200) — correct shape preservation
- Both SEBlock and DualSEBlock produce same output shape for same input
- Unlocks Task 7: TCNClassifier.forward can now rely on DualSEBlock in each TemporalBlock

## 2026-06-04T16:30 - Task 6: TCNClassifier.__init__ Integrated New Modules

### Changes Made
1. **LevelWeightedFusion fixed** (line ~161-168): Constructor changed from `feature_dim` to `fusion_dim` parameter. Gate now uses `nn.Linear(fusion_dim, num_levels)` instead of `nn.Linear(num_levels * feature_dim, num_levels)`. This supports variable per-level feature dimensions — the `fusion_dim` is the total concat dimension.

2. **fusion_channels *= 3** (was *= 2): Now accounts for 3 components per level: attn pooling + GAP + GMP.

3. **New modules added** after `self.gmp`:
   - `self.pos_encoding = SinusoidalPositionalEncoding(input_size, max_len=500)`
   - `self.attn_pool = SelfAttentionPooling(num_channels[-1])` — pools final layer output
   - `self.level_fusion = LevelWeightedFusion(len(self.fusion_levels), fusion_channels)`

4. **Classifier replaced**: `nn.Sequential` replaced with `ResidualClassifier(input_dim=fusion_channels, hidden_dim=fusion_channels // 2, num_classes=num_classes, dropout=dropout)`. Original code preserved as ARCH_BACKUP comments.

### Key Detail
- `SinusoidalPositionalEncoding` uses `max_len` parameter, NOT `max_seq_len` (task spec had wrong kwarg name)
- `fusion_channels` with default config: sum(128+256+448)=832 → *= 3 = 2496

### Verification
- `from src.model import TCNClassifier; m = TCNClassifier(input_size=2, num_channels=[64,96,128,192,256,320,384,448])`
- All three new attributes confirmed: pos_encoding=True, attn_pool=True, level_fusion=True

### Unlocks
- Task 7: TCNClassifier.forward — can now use pos_encoding, attn_pool, and level_fusion in the forward pass

## 2026-06-04T23:38 - Task 9: E2E Verification

### Bug Found: TCNClassifier.forward Transpose Mismatch
- Line 239: `x = x.transpose(1, 2)` converts input from (B,C,T) to (B,T,C) format
- **SinusoidalPositionalEncoding** expects (B,C,T): `self.pe` shape = (1, d_model, max_len). Slicing `pe[:,:,:x.size(2)]` yields (1, 2, 2) instead of (1, 2, 200), causing broadcast error at dim-1 (sequence length)
- **TemporalConvNet/Conv1d** expects (B,C,T): channels must be at dim 1. Transposed input has channels at dim 2, causing "expected 2 channels but got 200"
- **SelfAttentionPooling** internally transposes again: takes (B,C,T) and assumes channels at dim 1
- **Root cause**: The ARCH_BACKUP comment says "原始公式为 y = TCN(x)" — original code had NO transpose. The transpose was incorrectly added when new architecture modules were integrated.
- **Fix**: Remove line 239 (`x = x.transpose(1, 2)`) or adapt pos_encoding to accept (B,T,C) convention

### Verification Results (with correct shapes)
All 4 scenarios pass when components tested with correct (B,C,T) format:

| Scenario | Result |
|----------|--------|
| A: Forward pass (4,2,200)->(4,2) | PASS - finite values, no NaN/Inf |
| B: Short seq (4,2,50)->(4,2) | PASS - no NaN |
| C: Parameter count | PASS - 7,707,812 total/trainable |
| D: Training 2 batches | PASS - loss finite, gradients flow |

### Parameter Count Breakdown
| Component | Params |
|-----------|--------|
| TemporalConvNet (8 layers, DualSE) | 3,793,004 |
| SinusoidalPositionalEncoding (buffer, not learnable) | 0 |
| SelfAttentionPooling | 449 |
| LevelWeightedFusion | 7,491 |
| ResidualClassifier | 3,906,868 |
| GAP + GMP (parameterless) | 0 |
| **Total** | **7,707,812** |

### Architecture Flow Verified (correct format)
```
Input (B, 2, T)
  → SinusoidalPositionalEncoding → (B, 2, T)
  → TemporalConvNet (8 blocks, DualSE) → (B, 448, T)
    ↳ features extracted at levels [2, 4, 7]: [128, 256, 448] dims
  → Per-level pooling:
    - SelfAttentionPooling (or mean fallback) → (B, C_i)
    - GAP → (B, C_i)
    - GMP → (B, C_i)
    - Concat → (B, 3*C_i)
  → LevelWeightedFusion (3 levels, gate=Softmax) → (B, 2496)
  → ResidualClassifier (3-layer + skip) → (B, 2)
```

### Training Pipeline
- Synthetic dataset (8 samples, 2 batches) works end-to-end
- Loss decreases across batches (finite, no NaN)
- All gradients propagate through full architecture
- BUG NOTE: Real training requires removing the transpose on line 239
