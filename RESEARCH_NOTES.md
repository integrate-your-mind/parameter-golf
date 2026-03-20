# Parameter Golf Research Notes (March 20, 2026)

## Current Leaderboard (from open PRs, not yet merged)

| Rank | BPB | PR# | Key Techniques |
|------|-----|-----|----------------|
| 1 | **1.1318** | #198 | 11L, Int6 QAT, MLP 3x, SmearGate, BigramHash 2k, SWA, WD=0.04, FA3, stride=64 |
| 2 | **1.1453** | #194 | 11L, Int6 QAT, Per-Dim SmearGate, SWA/50, WD=0.038, stride=64 |
| 3 | **1.1453** | #180 | 10L, Int5-MLP/Int6-Attn mixed, SWA/50, MuonWD=0.04, SmearGate, BigramHash |
| 4 | **1.1472** | #179 | 11L, Int6+zstd, decoupled WD=0.038, stride=64 |
| 5 | **1.1502** | #192 | 11L, Int6 QAT, SmearGate, WD=0.038 |
| 6 | **1.1532** | #173 | Int6, MLP 3x, FA3, NorMuon |
| 7 | **1.1598** | #191 | Int6, MLP 3x, seq2048, stride=256, FP16 embed |
| 8 | **1.1667** | #178 | Nuclear Stack: Int6, 3x MLP, SmearGate, BigramHash, SWA, TTT |
| 9 | **1.1725** | #190 | Stinky Frost: Int6 QAT, FP16 embed, SmearGate, BigramHash, OrthoInit |
| 10 | **1.1748** | merged | Current SOTA: 10L, sliding window, FP16 embed, Muon WD, overtone init |

## Our Results
- **1.2006 BPB** (TTT LoRA, 9L baseline + SCALAR_LR=0.08 + GRAD_CLIP=1.0)
- Gap to new #1: **0.069 BPB** — significant

## Key Technique Taxonomy

### Tier 1: Essential (every top submission has these)
1. **Int6 quantization** — 6-bit per-row symmetric, stored in int8 containers. Saves ~33% artifact space vs int8, funding extra layers/params. Uses zstd-22 compression (not zlib) for better ratio on int6 data.
2. **More layers (10-11)** — The int6 savings fund 1-2 extra transformer layers within 16MB.
3. **High weight decay (0.02-0.04)** — Both Muon WD and AdamW WD. Keeps weights small → better quantization fidelity. 0.04 seems to be the sweet spot.
4. **FP16 tied embedding** — Skip quantization for tok_emb.weight. Int8/int6 errors compound through both input AND output paths when embeddings are tied.
5. **Sliding window eval (stride=64)** — ~0.03-0.09 BPB improvement for free at eval time.

### Tier 2: Strong Contributors
6. **SmearGate** — Learned per-dimension sigmoid gate blending current token embedding with previous token. 512 params. Captures bigram statistics cheaply. `g = sigmoid(param); out = (1-g)*x + g*x_prev`
7. **BigramHash embedding** — Hash consecutive token pairs into a learned embedding table (2048-4096 buckets × 128 dim). Provides bigram context to the model. Zero-initialized with small scale (0.05).
8. **Stochastic Weight Averaging (SWA)** — Average model checkpoints every 50-200 steps during the warmdown phase. ~29-80 checkpoint average. Smooths weight distributions → more robust to quantization. Can reduce quant degradation from 0.029 to 0.021 BPB.
9. **MLP 3x** (hidden=3*dim instead of 2*dim) — More feed-forward capacity. Funded by int6 compression savings.
10. **Muon WD (decoupled)** — `p.data.mul_(1 - lr * wd)` applied BEFORE the Muon update. This is decoupled weight decay, not L2 regularization.

### Tier 3: Helpful but Secondary
11. **QAT (Quantization-Aware Training)** — Fake int6 quantization during training forward pass using STE (Straight-Through Estimator). Reduces quant gap.
12. **OrthoInit** — Orthogonal initialization for all large linear layers.
13. **NorMuon** — Normalized variant of Muon optimizer.
14. **FA3 (Flash Attention 3)** — Speed optimization, allows more steps in 600s.
15. **seq2048** — Training with 2048 sequence length (doubled from 1024). Needs NTK RoPE scaling.

### Tier 4: Our Unique Advantage
16. **TTT LoRA** — Test-time training with per-document LoRA adaptation. ~0.033-0.037 BPB improvement. NOT used by any top-5 PR. Only PR #175 and #178 combine TTT with other techniques.

## Architecture of #1 (PR #198, 1.1318 BPB)

```
Model: 11 layers, 512 dim, 8 heads, 4 KV heads, MLP 3x (hidden=1536)
Params: 26.8M (fits in 15.7MB with int6+zstd)
Training: 7,412 steps at 81ms/step (600s cap)
Quantization: Int6 per-row for MLP+attn, FP16 for embedding, zstd-22
Eval: Sliding window stride=64

Key hyperparameters:
  NUM_LAYERS=11
  MLP_MULT=3
  MATRIX_LR=0.025
  SCALAR_LR=0.025
  TIED_EMBED_LR=0.035
  MUON_WD=0.04
  ADAM_WD=0.04
  MUON_MOMENTUM=0.99
  MUON_MOMENTUM_WARMUP_START=0.92
  MUON_MOMENTUM_WARMUP_STEPS=1500
  WARMDOWN_ITERS=3000
  GRAD_CLIP_NORM=0.3
  BIGRAM_VOCAB_SIZE=2048
  SWA_ENABLED=1
  SWA_EVERY=200
  EVAL_STRIDE=64
```

## Strategy to Beat #1

### Path A: Incremental (Low Risk)
Take the #1's training recipe and add TTT LoRA eval on top.
- #1 scores 1.1318 with sliding window eval
- TTT LoRA typically gives ~0.033 BPB improvement over baseline eval
- Expected: **~1.10-1.11 BPB** (if TTT improvement is additive)
- Risk: Need to implement Int6, SmearGate, BigramHash, SWA, Muon WD in our script

### Path B: Build on PR #175 (Medium Risk)
PR #175 already combines TTT + SOTA training (10L, overtone init, etc.)
- They expect ~1.14 BPB but haven't posted results yet
- We could build on their approach and add the newer techniques (Int6, SmearGate, SWA)

### Path C: Full Kitchen Sink (High Risk, High Reward)
Implement everything: 11L + Int6 + MLP 3x + SmearGate + BigramHash + SWA + Muon WD + TTT LoRA
- Most work but potentially the best score
- Risk: more code = more bugs, may not fit in 1500 lines

## Implementation Priority (ordered by impact/effort ratio)

1. **Int6 quantization + zstd** — Biggest impact, saves ~2MB artifact space, enables more layers
2. **Muon weight decay** — Add `weight_decay` param to Muon class, apply `p.data.mul_(1 - lr * wd)`
3. **11 layers** — Just change NUM_LAYERS env var (once Int6 makes it fit)
4. **SmearGate** — 10 lines of code, 512 params, captures bigram statistics
5. **SWA** — ~20 lines, average checkpoints during warmdown
6. **FP16 embed** — Already partially implemented, just need to wire up
7. **BigramHash** — ~30 lines, adds bigram context
8. **MLP 3x** — Change MLP_MULT env var (once artifact fits)
9. **QAT** — More complex, add fake quantization to CastedLinear forward
10. **TTT LoRA** — Already implemented! This is our edge.

## Key Numbers

| Config | Artifact Size | BPB (approx) |
|--------|--------------|---------------|
| 9L int8 baseline | ~15MB | 1.22 |
| 9L int8 + our improvements | ~15.8MB | 1.20 (TTT) |
| 10L int8 + WD | ~16MB (tight) | ~1.17 |
| 10L int6 + zstd | ~12MB | ~1.16 |
| 11L int6 + MLP 3x + zstd | ~15.7MB | ~1.13 |
| 11L int6 + everything + TTT | ~15.7MB | **~1.10?** |

## Compression Comparison

| Format | Typical ratio | Notes |
|--------|--------------|-------|
| Int8 + zlib | 3.9x | Current baseline |
| Int6 + zlib | ~4.5x | Better but zlib isn't optimal for int6 |
| Int6 + zstd-22 | ~5.5x | Best. zstd exploits int6's 3 zero high bits |
| Int5 + zstd-22 | ~6.2x | Even more extreme (PR #180 uses for MLP) |

## Dependencies
- `zstandard` Python package (for zstd compression)
- Flash Attention 3 (for FA3 speed, optional but helpful)
- `sentencepiece` (already have)
