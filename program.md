# Parameter Golf Autoresearch Program

## Mission
You are an autonomous research agent competing in the OpenAI Parameter Golf Challenge.

**Goal:** Minimize `val_bpb` (bits per byte, lower is better) on the FineWeb validation set.
**Hard constraint:** The final compressed artifact (model weights + `train_gpt.py` code) must fit in **≤16,000,000 bytes** when compressed with zstd-22.
**Time budget:** Each training run must complete in ≤10 minutes on 8xH100s (or your current hardware equivalent).

## The Only File You Edit
**`train_gpt.py`** — contains the full model, optimizer, quantization, and training loop. Everything is fair game:
- Architecture (layers, dimensions, attention heads, MLP width)
- Quantization (int8, int6, int5, QAT — Quantization-Aware Training)
- Optimizer (Muon, AdamW, weight decay, LR schedule)
- Embeddings (weight tying, BigramHash, FP16 embeddings)
- Gates and activations (SmearGate, SwiGLU, etc.)
- Initialization (OrthoInit, spectral)
- Compression (zstd-22, how weights are serialized)
- Evaluation (sliding window eval, stride)

**Do NOT modify:** `autoresearch_prepare.py`, `program.md`, or any data pipeline files.

## How to Run One Experiment

```bash
# Full run (10 min cap, 8xH100)
RUN_ID=exp_$(date +%s) torchrun --standalone --nproc_per_node=8 train_gpt.py

# Local smoke test (1 GPU, short)
MAX_WALLCLOCK_SECONDS=60 RUN_ID=smoke torchrun --standalone --nproc_per_node=1 train_gpt.py
```

At the end of each run, the script prints:
- `val_bpb`: your score (lower is better)
- `compressed_size_bytes`: artifact size (must be <16,000,000)

## Validation (REQUIRED after every run)

After each run, validate before keeping the change:

```bash
python validate.py --script train_gpt.py --model <saved_model_path> --bpb <val_bpb> --baseline <previous_best_bpb>
```

- Exit 0 = artifact is valid (size OK) → evaluate bpb improvement
- Exit 1 = artifact is INVALID (over 16MB) → **revert train_gpt.py immediately**

**Decision rule:**
- `validate.py` exits 0 AND `val_bpb` < previous best → **keep**, log, commit
- `validate.py` exits 1 OR `val_bpb` ≥ previous best → **revert** `train_gpt.py` to last good commit, log failure

## Experiment Loop

For each experiment:
1. Read the current `train_gpt.py` and the experiment log below
2. Propose ONE focused change (not a rewrite) with a clear hypothesis
3. Edit `train_gpt.py`
4. Run the experiment
5. Record results in the log below
6. If `val_bpb` improved AND `compressed_size_bytes < 16000000`: keep the change
7. If either condition fails: revert `train_gpt.py` to the previous working version

## Current Leaderboard Context
| Technique | bpb | Notes |
|-----------|-----|-------|
| Baseline | ~1.2244 | Our starting point |
| BigramHash + Int5 + Muon | 1.1428 | Top public entry uses these |
| GPTQ-lite + EMA + QAT | 1.1228 | Current SOTA as of 2026-03-22 |

**Our target:** Beat 1.1228. Minimum viable improvement: 0.005 nats below current SOTA.

## Promising Directions (in rough priority order)
1. **QAT (int6 → int5)**: Quantization-Aware Training squeezes more params into 16MB
2. **BigramHash**: Adds cheap bigram vocab signal without extra parameters
3. **Weight tying**: Share input embedding + output projection weights
4. **Muon optimizer + weight decay**: Better gradient updates than AdamW alone
5. **OrthoInit**: Clean orthogonal weight initialization
6. **SmearGate**: Soft gating for expressive layers with low param cost
7. **EMA/SWA**: Exponential moving average of weights for better generalization
8. **Sliding window eval**: Better BPB estimation at test time
9. **zstd-22**: Max compression on saved weights
10. **Depth/width search**: More layers vs wider MLP given 16MB budget

## Experiment Log

### Exp 001 — Baseline
- **Change:** Starting point, unmodified baseline
- **val_bpb:** ~1.2244
- **compressed_size:** <16MB ✅
- **Kept:** Yes (baseline)

---
<!-- Add new experiments below this line -->
