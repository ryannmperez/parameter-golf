# Autoresearch Setup

This repo uses the [autoresearch](https://github.com/ryannmperez/lobster-vision) loop, adapted for the OpenAI Parameter Golf Challenge.

## How It Works

An AI agent (Claude/Codex) reads `program.md` and autonomously:
1. Proposes one focused change to `train_gpt.py`
2. Runs a training experiment (~10 min)
3. Checks `val_bpb` and compressed artifact size
4. Keeps the change if it improves bpb AND stays under 16MB
5. Logs results in `program.md` and repeats

**You wake up to a log of experiments and a better model.**

## Starting the Agent

Open this repo in Claude Code or Codex and prompt:

```
Read program.md and kick off the next autoresearch experiment.
```

The agent will handle the rest. Disable file system permissions outside this repo.

## Key Files

| File | Purpose |
|------|---------|
| `train_gpt.py` | The model — agent edits this |
| `program.md` | Agent instructions + experiment log |
| `autoresearch_prepare.py` | Data prep (do not modify) |
| `autoresearch.md` | This file |

## Tips

- Run smoke tests locally with `MAX_WALLCLOCK_SECONDS=60 nproc_per_node=1` before committing to full H100 runs
- Commit after every successful experiment so you can always revert
- The agent should make ONE change per experiment — not rewrites
- Check compressed size after every run — easy to accidentally blow the 16MB budget
