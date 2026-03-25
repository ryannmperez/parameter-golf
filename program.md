# parameter-golf

This is an experiment to have the LLM do its own research, competing in the OpenAI Parameter Golf Challenge. The goal is to minimize `val_bpb` (bits per byte, lower is better) on the FineWeb validation set, subject to a hard 16MB compressed artifact constraint.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar24`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current main.
3. **Read the in-scope files**: Read these files for full context:
   - `program.md` — this file. Repository context and experiment instructions.
   - `train_gpt.py` — the file you modify. Model architecture, optimizer, quantization, training loop.
   - `validate.py` — submission validator. Checks compressed artifact size (train_gpt.py + model weights) fits within 16MB via zstd-22, and optionally checks val_bpb against a baseline. Do not modify.
4. **Verify data exists**: Check that data shards and tokenizer are available. If not, tell the human to run the data preparation step.
5. **Confirm hardware**: Current hardware is a single NVIDIA A6000 (`CUDA_VISIBLE_DEVICES=0`).
6. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
7. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single GPU. The training script runs for a **fixed time budget of 5 minutes** (wall clock training time). You launch it as:

```bash
CUDA_VISIBLE_DEVICES=0 MAX_WALLCLOCK_SECONDS=300 torchrun --standalone --nproc_per_node=1 train_gpt.py
```

**What you CAN do:**
- Modify `train_gpt.py` — this is the only file you edit. Everything is fair game: architecture (layers, dimensions, attention heads, MLP width), quantization (int8, int6, int5, QAT), optimizer (Muon, AdamW, weight decay, LR schedule), embeddings (weight tying, BigramHash, FP16), gates and activations (SmearGate, SwiGLU), initialization (OrthoInit, spectral), compression (zstd-22, serialization), evaluation (sliding window, stride).

**What you CANNOT do:**
- Modify `validate.py`. It is read-only. It validates compressed artifact size and bpb results.
- Modify `program.md`.
- Install new packages or add dependencies.
- Modify the evaluation harness.

**The goal is simple: get the lowest val_bpb.** Everything is fair game as long as the code runs without crashing, finishes within the time budget, and the final compressed artifact (model weights + `train_gpt.py` code) fits in **16,000,000 bytes** when compressed with zstd-22.


**VRAM** is a soft constraint. Some increase is acceptable for meaningful val_bpb gains, but it should not blow up dramatically.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 0.001 val_bpb improvement that adds 20 lines of hacky code? Probably not worth it. A 0.001 val_bpb improvement from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## Output format

Once the script finishes it prints a summary including:

```
val_bpb:              X.XXXXXX
compressed_size_bytes: XXXXXXX
```

You can extract the key metrics from the log file:

```
grep "^val_bpb:\|^compressed_size_bytes:" run.log
```

## Validation

After each run, validate the artifact before keeping the change:

```bash
python validate.py --script train_gpt.py --model <saved_model_path> --bpb <val_bpb> --baseline <previous_best_bpb>
```

- Exit 0 = artifact is valid (size OK)
- Exit 1 = artifact is INVALID (over 16MB) — **revert train_gpt.py immediately**

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 6 columns:

```
commit	val_bpb	compressed_mb	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. val_bpb achieved (e.g. 1.234567) — use 0.000000 for crashes
3. compressed artifact size in MB, round to .1f (e.g. 7.9 — divide compressed_size_bytes by 1048576) — use 0.0 for crashes. Must be < 15.3 (16,000,000 bytes)
4. peak memory in GB, round to .1f (e.g. 12.3 — divide peak_vram_mb by 1024) — use 0.0 for crashes
5. status: `keep`, `discard`, or `crash`
6. short text description of what this experiment tried

Example:

```
commit	val_bpb	compressed_mb	memory_gb	status	description
a1b2c3d	1.984200	7.9	44.0	keep	baseline
b2c3d4e	1.970100	8.1	44.2	keep	QAT delayed start at 15%
c3d4e5f	1.990000	7.9	44.0	discard	switch to GeLU activation
d4e5f6g	0.000000	0.0	0.0	crash	double model width (OOM)
```


## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar24`).

LOOP FOREVER:

1. Read the current `train_gpt.py` and `results.tsv`
2. Count completed experiments. **Every 10th experiment:** enter Literature Review mode (see below) before proposing the next idea
3. Propose ONE focused change (not a rewrite) with a clear hypothesis
4. Edit `train_gpt.py`
5. **Pre-flight size check**: Before spending 5 minutes on a training run, estimate whether the artifact will fit in 16MB. Count total parameters from the model config (embedding + transformer layers + output head), multiply by bytes-per-weight for the target quantization (e.g. int8 = 1 byte, int6 = 0.75, int5 = 0.625), and compare against ~15MB (leaving ~1MB headroom for code + overhead). If the estimate exceeds 15MB, adjust the architecture before running.
6. git commit
7. Run the experiment: `CUDA_VISIBLE_DEVICES=0 MAX_WALLCLOCK_SECONDS=300 torchrun --standalone --nproc_per_node=1 train_gpt.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
8. Read out the results: `grep "^val_bpb:\|^compressed_size_bytes:" run.log`
9. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the stack trace and attempt a fix
10. Validate with `validate.py`
11. Record results in `results.tsv` (NOTE: do not commit results.tsv, leave it untracked by git)
12. If val_bpb improved AND compressed_size_bytes < 16000000: keep the change, commit
13. If either condition fails: revert `train_gpt.py` to last good commit, log failure

**Timeout**: Each experiment should take ~5 minutes total (+ startup/eval overhead). If a run exceeds 10 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~5 minutes then you can run approx 12/hour, for a total of about 100 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!

**Print the results table to the screen after each round.**

## Literature Review Mode (every 10th experiment)

When you reach experiment 10, 20, 30, etc.:

1. **Search** for a recent LLM paper (published >= 2025) relevant to parameter efficiency, quantization, small LMs, training dynamics.

2. **Read** the abstract and key sections. Understand the paper's core insight — what problem it solves and *why* the approach works mechanistically.

3. **Synthesize a novel idea** inspired by (not copied from) the paper:
   - Ask: "What is the underlying principle here, and how could that principle apply differently in our constrained 16MB setting?"
   - The idea should be a **new combination, adaptation, or extrapolation** — not a direct reimplementation
   - It must be implementable in a single `train_gpt.py` edit
   - It must not have been tried in `results.tsv`
   - Example: paper proposes low-rank weight decomposition for fine-tuning -> synthesized idea: apply low-rank factorization only to MLP layers to save bytes for more attention heads

4. **Log the paper** to `literature.tsv` (see format below). Check `literature.tsv` first to avoid reviewing a paper that's already been logged.

5. **Proceed** with the synthesized idea as the next experiment (or document why synthesis wasn't feasible and fall back to Promising Directions)

## Literature logging

Log papers to `literature.tsv` (tab-separated, NOT comma-separated). Do not commit this file.

The TSV has a header row and 4 columns:

```
title	url	synthesized_idea	status
```

1. Paper title (short)
2. URL (arXiv link or similar)
3. Synthesized idea — your adaptation for the 16MB/bpb constraint (not a direct reimplementation)
4. status: `tested` (with exp number, e.g. `tested:exp012`), `pending`, or `skipped`

Example:

```
title	url	synthesized_idea	status
Low-Rank Adapters for Tiny LMs	https://arxiv.org/abs/2025.12345	factorize MLP layers to free bytes for more attention heads	tested:exp012
BitNet: 1-bit LLMs	https://arxiv.org/abs/2025.67890	binary weights for embedding table only	skipped
```

## Baseline

The first experiment must be an unmodified baseline run. Record it in `results.tsv` as the first entry. All subsequent experiments are compared against this baseline.

**A6000_BASELINE_BPB: 1.9842** <- established 2026-03-24

## Promising Directions (in rough priority order)

> First run: Before any experiments, run the unmodified baseline for 5 min to establish `A6000_BASELINE_BPB`. Then start with Exp 002 below.

1. **QAT — delayed start at 15% of training time** <- START HERE (Exp 002)
   - The baseline uses post-training int8 quantization (quantize after training). QAT simulates quantization *during* training so the model adapts to it.
   - Training is time-based (no epochs). At 5 min = 300s, 15% = ~45 seconds in.
   - Implement Straight-Through Estimator (STE) for gradients through quantized weights.
   - Start with int6 (safer). If artifact fits in 16MB with headroom, try int5.
   - SOTA entry uses `QAT @ 0.15` (15% of training elapsed) — use that as starting point.
   - Hypothesis: model learns quantization-robust representations -> better bpb at same compressed size.

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

