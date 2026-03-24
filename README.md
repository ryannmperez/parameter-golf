# 🦞 Money Lobster — Parameter Golf Entry

OpenAI [Parameter Golf Challenge](https://github.com/openai/parameter-golf) entry.

**Goal:** Train the best LM that fits in 16MB. Evaluated on FineWeb validation set by bits-per-byte (bpb). Lower is better.

## Team
- Ryann Perez

## Current Best
| Run | bpb | Notes |
|-----|-----|-------|
| Baseline | ~1.22 | Starting point |

## Setup

```bash
git clone https://github.com/ryannmperez/parameter-golf.git
cd parameter-golf
pip install -r requirements.txt
python data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1
```

## Run
```bash
RUN_ID=baseline torchrun --standalone --nproc_per_node=1 train_gpt.py
```

## Strategy
- [ ] Int6/Int5 quantization (QAT)
- [ ] Weight tying
- [ ] BigramHash embeddings
- [ ] SmearGate + OrthoInit
- [ ] Muon optimizer
- [ ] zstd-22 compression
- [ ] Automated ablation search
