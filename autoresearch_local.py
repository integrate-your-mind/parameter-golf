#!/usr/bin/env python3
"""
Local autoresearch loop for parameter-golf on Apple Silicon (MLX).

Systematically sweeps hyperparameters using short 3-minute runs,
compares val_bpb, and logs all results. Safe for 48GB unified memory.

Usage:
    python3 autoresearch_local.py              # run all experiments
    python3 autoresearch_local.py --baseline   # run baseline only
"""
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

SCRIPT = Path(__file__).parent / "train_gpt_mlx.py"
LOG_DIR = Path(__file__).parent / "autoresearch_logs"
RESULTS_FILE = Path(__file__).parent / "autoresearch_results.jsonl"

# Safe defaults for 48GB M4 Max (12GB MLX limit)
BASE_ENV = {
    "MLX_MEM_LIMIT_GB": "12",
    "TRAIN_BATCH_TOKENS": "16384",
    "VAL_BATCH_SIZE": "32768",
    "MLX_MAX_MICROBATCH_TOKENS": "8192",
    "MLX_EAGER_EVAL": "1",
    "MAX_WALLCLOCK_SECONDS": "180",  # 3 min per experiment
    "VAL_LOSS_EVERY": "0",          # only eval at end
    "TRAIN_LOG_EVERY": "200",
    "WARMUP_STEPS": "5",            # fast warmup
    "NUM_LAYERS": "4",              # sweet spot locally
    "SCALAR_LR": "0.04",            # baseline
    "GRAD_CLIP_NORM": "0.0",        # baseline
}

# Experiments to run: name -> env overrides
EXPERIMENTS = {
    # Baseline
    "baseline_4L": {},

    # Proven winners (verify they still win)
    "scalar_lr_0.08_clip_1.0": {
        "SCALAR_LR": "0.08",
        "GRAD_CLIP_NORM": "1.0",
    },

    # Warmdown schedule (H100 #1 uses 2500)
    "warmdown_2500": {
        "SCALAR_LR": "0.08",
        "GRAD_CLIP_NORM": "1.0",
        "WARMDOWN_ITERS": "2500",
    },
    "warmdown_3000": {
        "SCALAR_LR": "0.08",
        "GRAD_CLIP_NORM": "1.0",
        "WARMDOWN_ITERS": "3000",
    },

    # Embed LR sweep
    "embed_lr_0.08": {
        "SCALAR_LR": "0.08",
        "GRAD_CLIP_NORM": "1.0",
        "TIED_EMBED_LR": "0.08",
    },
    "embed_lr_0.03": {
        "SCALAR_LR": "0.08",
        "GRAD_CLIP_NORM": "1.0",
        "TIED_EMBED_LR": "0.03",
    },

    # Matrix LR sweep
    "matrix_lr_0.05": {
        "SCALAR_LR": "0.08",
        "GRAD_CLIP_NORM": "1.0",
        "MATRIX_LR": "0.05",
    },
    "matrix_lr_0.06": {
        "SCALAR_LR": "0.08",
        "GRAD_CLIP_NORM": "1.0",
        "MATRIX_LR": "0.06",
    },

    # Beta2 sweep
    "beta2_0.98": {
        "SCALAR_LR": "0.08",
        "GRAD_CLIP_NORM": "1.0",
        "BETA2": "0.98",
    },
    "beta2_0.92": {
        "SCALAR_LR": "0.08",
        "GRAD_CLIP_NORM": "1.0",
        "BETA2": "0.92",
    },

    # Grad clip sweep
    "clip_0.5": {
        "SCALAR_LR": "0.08",
        "GRAD_CLIP_NORM": "0.5",
    },
    "clip_2.0": {
        "SCALAR_LR": "0.08",
        "GRAD_CLIP_NORM": "2.0",
    },

    # Scalar LR sweep (beyond 0.08)
    "scalar_lr_0.10": {
        "SCALAR_LR": "0.10",
        "GRAD_CLIP_NORM": "1.0",
    },
    "scalar_lr_0.12": {
        "SCALAR_LR": "0.12",
        "GRAD_CLIP_NORM": "1.0",
    },

    # Weight decay (AdamW, like #1 submission)
    "wd_0.01": {
        "SCALAR_LR": "0.08",
        "GRAD_CLIP_NORM": "1.0",
        "ADAM_WEIGHT_DECAY": "0.01",
    },
    "wd_0.02": {
        "SCALAR_LR": "0.08",
        "GRAD_CLIP_NORM": "1.0",
        "ADAM_WEIGHT_DECAY": "0.02",
    },
    "wd_0.005": {
        "SCALAR_LR": "0.08",
        "GRAD_CLIP_NORM": "1.0",
        "ADAM_WEIGHT_DECAY": "0.005",
    },

    # Combined best guesses
    "combo_warmdown_embed": {
        "SCALAR_LR": "0.08",
        "GRAD_CLIP_NORM": "1.0",
        "WARMDOWN_ITERS": "2500",
        "TIED_EMBED_LR": "0.08",
    },
    "combo_all_aggressive": {
        "SCALAR_LR": "0.10",
        "GRAD_CLIP_NORM": "1.0",
        "WARMDOWN_ITERS": "2500",
        "TIED_EMBED_LR": "0.08",
        "MATRIX_LR": "0.05",
    },
    "combo_wd_warmdown": {
        "SCALAR_LR": "0.08",
        "GRAD_CLIP_NORM": "1.0",
        "ADAM_WEIGHT_DECAY": "0.01",
        "WARMDOWN_ITERS": "2500",
    },
    "combo_full_v1": {
        "SCALAR_LR": "0.08",
        "GRAD_CLIP_NORM": "1.0",
        "ADAM_WEIGHT_DECAY": "0.01",
        "WARMDOWN_ITERS": "2500",
        "TIED_EMBED_LR": "0.08",
    },
}


def run_experiment(name: str, overrides: dict) -> dict:
    """Run a single experiment, return results dict."""
    LOG_DIR.mkdir(exist_ok=True)
    log_path = LOG_DIR / f"{name}.log"

    env = dict(os.environ)
    env.update(BASE_ENV)
    env.update(overrides)

    print(f"\n{'='*60}")
    print(f"  EXPERIMENT: {name}")
    print(f"  Overrides: {overrides or '(baseline)'}")
    print(f"{'='*60}")

    t0 = time.time()
    result = subprocess.run(
        [sys.executable, str(SCRIPT)],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,  # hard kill at 10 min
    )
    elapsed = time.time() - t0

    output = result.stdout + "\n" + result.stderr
    log_path.write_text(output)

    # Parse results
    val_bpb = None
    val_loss = None
    steps = None
    artifact_bytes = None

    for line in output.split("\n"):
        # Final int8 roundtrip is the real score
        m = re.search(r"final_int8_zlib_roundtrip val_loss:([\d.]+) val_bpb:([\d.]+)", line)
        if m:
            val_loss = float(m.group(1))
            val_bpb = float(m.group(2))

        m = re.search(r"step:(\d+)/\d+.*val_loss", line)
        if m:
            steps = int(m.group(1))

        m = re.search(r"serialized_model_int8_zlib:(\d+) bytes", line)
        if m:
            artifact_bytes = int(m.group(1))

    record = {
        "name": name,
        "overrides": overrides,
        "val_bpb": val_bpb,
        "val_loss": val_loss,
        "steps": steps,
        "artifact_bytes": artifact_bytes,
        "elapsed_s": round(elapsed, 1),
        "exit_code": result.returncode,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    if val_bpb is not None:
        print(f"  RESULT: val_bpb={val_bpb:.4f}  steps={steps}  artifact={artifact_bytes}  time={elapsed:.0f}s")
    else:
        print(f"  FAILED: exit_code={result.returncode}  time={elapsed:.0f}s")
        if result.returncode == -9 or result.returncode == 137:
            print("  *** OOM KILL detected! Reduce batch size or memory limit. ***")

    # Append to results log
    with RESULTS_FILE.open("a") as f:
        f.write(json.dumps(record) + "\n")

    return record


def print_leaderboard(results: list[dict]):
    """Print sorted results table."""
    valid = [r for r in results if r["val_bpb"] is not None]
    valid.sort(key=lambda r: r["val_bpb"])

    print(f"\n{'='*80}")
    print(f"  LOCAL AUTORESEARCH LEADERBOARD")
    print(f"{'='*80}")
    print(f"  {'Rank':<5} {'Name':<30} {'BPB':<10} {'Steps':<8} {'Time':<8} {'Artifact'}")
    print(f"  {'-'*5} {'-'*30} {'-'*10} {'-'*8} {'-'*8} {'-'*10}")
    for i, r in enumerate(valid, 1):
        art = f"{r['artifact_bytes']:,}" if r['artifact_bytes'] else "?"
        print(f"  {i:<5} {r['name']:<30} {r['val_bpb']:<10.4f} {r['steps'] or '?':<8} {r['elapsed_s']:<8.0f} {art}")

    if valid:
        best = valid[0]
        worst = valid[-1]
        delta = worst["val_bpb"] - best["val_bpb"]
        print(f"\n  Best: {best['name']} = {best['val_bpb']:.4f} BPB")
        print(f"  Spread: {delta:.4f} BPB across {len(valid)} experiments")


def main():
    if "--baseline" in sys.argv:
        experiments = {"baseline_4L": {}}
    else:
        experiments = EXPERIMENTS

    results = []
    total = len(experiments)

    for i, (name, overrides) in enumerate(experiments.items(), 1):
        print(f"\n[{i}/{total}] Starting {name}...")
        try:
            record = run_experiment(name, overrides)
            results.append(record)
        except subprocess.TimeoutExpired:
            print(f"  TIMEOUT: {name} exceeded 10 minutes, killed.")
            results.append({"name": name, "val_bpb": None, "error": "timeout"})
        except Exception as e:
            print(f"  ERROR: {name}: {e}")
            results.append({"name": name, "val_bpb": None, "error": str(e)})

    print_leaderboard(results)


if __name__ == "__main__":
    main()
