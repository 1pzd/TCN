# Decisions - TCN Architecture Improvement

## 2026-06-04T15:13 - Initial Decisions

### Wave 1 File Conflict Risk
- **Decision**: Fire Tasks 1-4 sequentially to avoid write conflicts on src/model.py
- **Rationale**: All 4 tasks insert classes at different positions in the same file. Sequential ensures integrity.
- **Alternative**: Batch all 4 into one task — rejected because plan specifies separate tasks.

### DualSEBlock Parameters
- **Decision**: Keep se_reduction hardcoded to 8, se_temporal_kernel hardcoded to 7
- **Rationale**: TemporalBlock.__init__ doesn't receive these from config; YAML values are documentation defaults

### Positional Encoding Placement
- **Decision**: Apply PE after x.transpose(1,2), before TCN call
- **Rationale**: PE expects (B,D,T) input; after transpose x is (B,C,T)=(B,2,T) which matches PE(2, max_len)

### Fusion Channel Update
- **Decision**: Update fusion_channels *= 3 to account for Attn+GAP+GMP
