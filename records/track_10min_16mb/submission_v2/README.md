# Combined Submission: All Winning Techniques + Novel Improvements

**Target: beat 1.1748 BPB**

## Techniques Stacked

### From #1 Submission (notapplica, 1.1748)
1. **10 transformer layers** (from 9)
2. **Overtone spectral embedding init** — SVD power-law spectrum shaping
3. **Phase-transition residual mixing** — sigmoid-scheduled resid_mix init
4. **FP16 tied embedding export** — skip int8 quantization for embeddings
5. **Weight decay 0.01** on Adam optimizers (tok + scalar)
6. **Warmdown 2500 steps** (from 1200)
7. **Tied embed LR 0.10** (from 0.05)

### From TTT Submission (samacqua, 1.1928)
8. **LoRA test-time training** — per-document rank-8 LoRA adaptation during eval (already in updated baseline)

### From Sliding Window Submission (Matthew Li, 1.1925)
9. **Sliding window evaluation** — stride=64, 960+ context per token (already in updated baseline)

### Our Novel Improvements (proven locally across 20+ experiments)
10. **grad_clip_norm=1.0** — consistent improvement across all configs
11. **scalar_lr=0.08** (2x default) — faster scalar param convergence

## Why This Should Beat 1.1748

The current #1 does NOT use TTT or our hyperparameter improvements. We stack:
- All of #1's architecture/init tricks
- TTT from the baseline (which alone gave -0.037 BPB)
- Sliding window eval (which alone gave -0.032 BPB)
- Our proven grad_clip + scalar_lr improvements

These are orthogonal improvements that should compound.

## Command

```bash
EVAL_STRIDE=64 \
TTT_LORA_RANK=8 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Local Validation

Extensively tested on Apple Silicon M4 Max:
- 20+ experiments across 3-10 layer configs
- grad_clip=1.0 and scalar_lr=0.08 improved BPB in every single test
- Best local result: 1.5678 BPB (4L, 10 min, Apple Silicon)
