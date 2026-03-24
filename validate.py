#!/usr/bin/env python3
"""
validate.py — Parameter Golf submission validator

Checks:
  1. Compressed artifact size <= 16,000,000 bytes (train_gpt.py + model weights)
  2. val_bpb is present and numeric
  3. (Optional) val_bpb beats a provided baseline

Usage:
  python validate.py --model <path_to_model.bin> [--baseline 1.2244] [--bpb 1.1500]

Exits 0 if valid, 1 if invalid.
"""

import argparse
import os
import struct
import sys
import tempfile
import zstandard as zstd

MAX_ARTIFACT_BYTES = 16_000_000
TRAIN_SCRIPT = "train_gpt.py"


def compress_size(paths: list[str]) -> int:
    """Return compressed size of all files concatenated, using zstd level 22."""
    cctx = zstd.ZstdCompressor(level=22)
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp_path = tmp.name
        with cctx.stream_writer(tmp) as writer:
            for p in paths:
                if not os.path.exists(p):
                    print(f"[ERROR] File not found: {p}")
                    sys.exit(1)
                with open(p, "rb") as f:
                    writer.write(f.read())
    size = os.path.getsize(tmp_path)
    os.unlink(tmp_path)
    return size


def main():
    parser = argparse.ArgumentParser(description="Parameter Golf validator")
    parser.add_argument("--model", required=False, help="Path to saved model weights file")
    parser.add_argument("--bpb", type=float, required=False, help="val_bpb from training run")
    parser.add_argument("--baseline", type=float, default=None, help="bpb to beat (optional)")
    parser.add_argument("--script", default=TRAIN_SCRIPT, help="Training script path")
    args = parser.parse_args()

    ok = True

    # --- Check 1: artifact size ---
    files = [args.script]
    if args.model:
        files.append(args.model)

    size = compress_size(files)
    size_ok = size <= MAX_ARTIFACT_BYTES
    status = "✅" if size_ok else "❌"
    print(f"{status} Compressed artifact size: {size:,} / {MAX_ARTIFACT_BYTES:,} bytes ({size/MAX_ARTIFACT_BYTES*100:.1f}%)")
    if not size_ok:
        print(f"   → OVER BUDGET by {size - MAX_ARTIFACT_BYTES:,} bytes")
        ok = False

    # --- Check 2: bpb ---
    if args.bpb is not None:
        print(f"ℹ️  val_bpb: {args.bpb:.4f}")
        if args.baseline is not None:
            improvement = args.baseline - args.bpb
            beats = improvement >= 0.005
            status = "✅" if beats else "⚠️ "
            print(f"{status} vs baseline {args.baseline:.4f}: improvement = {improvement:+.4f} (need ≥0.005)")
            if not beats:
                print(f"   → Improvement below significance threshold (0.005 nats)")
                # Not a hard failure — still useful, just not a record submission
    else:
        print("⚠️  No --bpb provided, skipping bpb check")

    # --- Summary ---
    print()
    if ok:
        print("✅ VALID — artifact fits within 16MB budget")
        sys.exit(0)
    else:
        print("❌ INVALID — fix issues above before submitting")
        sys.exit(1)


if __name__ == "__main__":
    main()
