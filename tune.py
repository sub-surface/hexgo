"""
tune.py — HexGo autotune trial orchestrator.

Usage:
    python tune.py [--gens N] [--games N]

Flow per call:
  1. Backup current config.py
  2. Read pre-trial ELO for eisenstein_def from elo.json
  3. Run train.py --tune --gens N --games N --sims {CFG["SIMS"]}
  4. Read post-trial ELO, compute delta
  5. Append result to tune_log.jsonl
  6. If delta < 0: revert config.py from backup, print REVERTED
  7. Print summary line

Claude proposes config.py changes before calling this script.
Claude reads tune_log.jsonl to reason about the next proposal.
"""

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ELO_FILE      = Path("elo.json")
CONFIG_FILE   = Path("config.py")
CONFIG_BACKUP = Path("config.py.bak")
TUNE_LOG      = Path("tune_log.jsonl")
TUNE_RESULT   = Path("tune_result.json")
PYTHON        = sys.executable


def _read_elo(agent: str = "eisenstein_def") -> float | None:
    """Return current ELO rating for agent, or None if not found."""
    if not ELO_FILE.exists():
        return None
    data = json.loads(ELO_FILE.read_text())
    return data.get("ratings", {}).get(agent)


def _read_cfg() -> dict:
    """Import CFG from config.py via exec (avoids stale module cache)."""
    ns: dict = {}
    exec(CONFIG_FILE.read_text(), ns)
    return ns["CFG"]


def run_trial(gens: int = 5, games: int = 10) -> dict:
    # 1. Backup current config
    shutil.copy(CONFIG_FILE, CONFIG_BACKUP)
    cfg = _read_cfg()

    # 2. Pre-trial ELO
    elo_before = _read_elo("eisenstein_def")

    # 3. Clear previous tune_result.json
    if TUNE_RESULT.exists():
        TUNE_RESULT.unlink()

    # 4. Run training
    cmd = [
        PYTHON, "train.py",
        "--tune",
        "--gens",  str(gens),
        "--games", str(games),
        "--sims",  str(cfg["SIMS"]),
    ]
    print(f"Running: {' '.join(cmd)}", flush=True)
    t0 = time.perf_counter()
    result = subprocess.run(cmd)
    elapsed = time.perf_counter() - t0

    if result.returncode != 0:
        print("ERROR: train.py exited with non-zero status — reverting config")
        shutil.copy(CONFIG_BACKUP, CONFIG_FILE)
        return {"error": "train failed", "cfg": cfg}

    # 5. Post-trial ELO
    elo_after = _read_elo("eisenstein_def")
    elo_delta = None
    if elo_before is not None and elo_after is not None:
        elo_delta = round(elo_after - elo_before, 1)

    # 6. Read per-gen metrics
    gen_metrics: list = []
    if TUNE_RESULT.exists():
        try:
            gen_metrics = json.loads(TUNE_RESULT.read_text())
        except Exception:
            pass

    avg_eis_winrate = None
    if gen_metrics:
        rates = [g["eis_winrate"] for g in gen_metrics if g.get("eis_winrate") is not None]
        avg_eis_winrate = round(sum(rates) / len(rates), 3) if rates else None

    # 7. Build log entry
    # eisenstein_def ELO rises when the net loses more to Eisenstein (net is WORSE).
    # A good config lowers eisenstein_def ELO (net improves). Keep when delta <= 0.
    kept = elo_delta is None or elo_delta <= 0
    entry = {
        "cfg":             cfg,
        "elo_before":      elo_before,
        "elo_after":       elo_after,
        "elo_delta":       elo_delta,
        "avg_eis_winrate": avg_eis_winrate,
        "gen_metrics":     gen_metrics,
        "elapsed_s":       round(elapsed, 1),
        "kept":            kept,
    }

    # 8. Append to log
    with TUNE_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

    # 9. Revert if negative delta
    if not kept:
        shutil.copy(CONFIG_BACKUP, CONFIG_FILE)
        print(f"REVERTED  elo_delta={elo_delta:+.1f}  avg_eis_wr={avg_eis_winrate}")
    else:
        delta_str = f"{elo_delta:+.1f}" if elo_delta is not None else "n/a"
        print(f"KEPT      elo_delta={delta_str}  avg_eis_wr={avg_eis_winrate}")

    return entry


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run one autotune trial")
    parser.add_argument("--gens",  type=int, default=5,  help="Gens per trial")
    parser.add_argument("--games", type=int, default=10, help="Games per gen")
    args = parser.parse_args()
    run_trial(gens=args.gens, games=args.games)
