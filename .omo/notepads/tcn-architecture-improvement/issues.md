# Issues - TCN Architecture Improvement

## Open Issues

### ISSUE-1: ~~FALSE ALARM~~ — Transpose is correct and was in original code
- **Severity**: NONE — FALSE ALARM from Task 9 subagent
- **Resolution**: `x = x.transpose(1, 2)` at line 239 was in the ORIGINAL forward method (confirmed via `git show HEAD:src/model.py`). Data loader produces `(B, T, F)` = `(B, 200, 2)`. TCN Conv1d expects `(B, C_in, L)` = `(B, F, T)`. The transpose is correct and necessary.
- **Verified**: `(4, 200, 2)` → `(4, 2)`, no NaN, 7.7M params, training 2 batches passes
- **Closed**: 2026-06-04
