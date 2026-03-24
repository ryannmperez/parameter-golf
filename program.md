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

## Hardware
**Current:** Single NVIDIA A6000 (`CUDA_VISIBLE_DEVICES=0`). Multi-GPU support TBD.

## How to Run One Experiment

```bash
# Standard run on A6000 (single GPU)
CUDA_VISIBLE_DEVICES=0 RUN_ID=exp_$(date +%s) torchrun --standalone --nproc_per_node=1 train_gpt.py

# Smoke test (short wallclock)
CUDA_VISIBLE_DEVICES=0 MAX_WALLCLOCK_SECONDS=60 RUN_ID=smoke torchrun --standalone --nproc_per_node=1 train_gpt.py
```

Note: Final leaderboard submissions must run in ≤10 min on 8xH100s. Local A6000 runs will be slower — use them for iteration and correctness checks. Before submitting a record, verify timing on H100s via RunPod.

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
2. Count completed experiments in the log (Exp 001, 002, etc.)
3. **Every 10th experiment:** enter Literature Review mode (see below) before proposing the next idea
4. Propose ONE focused change (not a rewrite) with a clear hypothesis
5. Edit `train_gpt.py`
6. Run the experiment
7. Validate with `validate.py`
8. Record results in the log below
9. If `val_bpb` improved AND `compressed_size_bytes < 16000000`: keep the change, commit
10. If either condition fails: revert `train_gpt.py` to last good commit, log failure

## Literature Review Mode (every 10th experiment)

When you reach experiment 10, 20, 30, etc.:

1. **Search** for a recent LLM paper (published ≥ 2025) relevant to parameter efficiency, quantization, small LMs, training dynamics, or **novel tokenization methods**. Good sources:
   - arXiv cs.LG / cs.CL: https://arxiv.org/search/?searchtype=all&query=language+model+quantization&start=0
   - Semantic Scholar: https://api.semanticscholar.org/graph/v1/paper/search?query=efficient+language+model+2025&fields=title,year,abstract
   - Papers with Code: https://paperswithcode.com/methods
   - For tokenization specifically: search "tokenization language model 2025", "byte-level tokenizer", "BPE alternatives", "tokenizer-free LM"

2. **Read** the abstract and key sections. Understand the paper's core insight — what problem it solves and *why* the approach works mechanistically.

3. **Synthesize a novel idea** inspired by (not copied from) the paper:
   - Ask: "What is the underlying principle here, and how could that principle apply differently in our constrained 16MB setting?"
   - The idea should be a **new combination, adaptation, or extrapolation** — not a direct reimplementation
   - It must be implementable in a single `train_gpt.py` edit
   - It must not have been tried in the experiment log
   - Example: paper proposes low-rank weight decomposition for fine-tuning → synthesized idea: apply low-rank factorization only to MLP layers to save bytes for more attention heads

4. **Log the paper + synthesis** in the Literature section below:
   - Title, authors, year, arXiv ID
   - Core insight from the paper
   - Your synthesized idea (distinct from the paper's direct method)
   - Why you believe it applies to the 16MB/bpb constraint

5. **Proceed** with the synthesized idea as the next experiment (or document why synthesis wasn't feasible and fall back to Promising Directions)

## Literature Log

<!-- Papers reviewed by the agent go here -->
<!-- Format:
### Paper Review — Exp XXX
- **Paper:** Title (Year) — arXiv:XXXX.XXXXX
- **Authors:** ...
- **Core insight:** ...
   - **Synthesized idea:** ... (distinct from direct implementation)
- **Feasibility:** High/Medium/Low — reason
- **Applied in:** Exp XXX (or "Skipped — reason")
-->

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
11. **Novel tokenization**: Smaller vocab, byte-level, or hybrid tokenizers — vocab size directly impacts embedding table size in the 16MB budget

## Experiment Log

### Exp 001 — Baseline
- **Change:** Starting point, unmodified baseline
- **val_bpb:** ~1.2244
- **compressed_size:** <16MB ✅
- **Kept:** Yes (baseline)

---
<!-- Add new experiments below this line -->
