# HexGo Autotune — Design Spec
*2026-03-29*

## Goal

Autonomous hyperparameter tuning for HexGo's AlphaZero training loop. A Claude agent proposes configs, runs 5-generation trials, measures ELO delta vs `EisensteinGreedyAgent` as the RL reward signal, and iterates overnight without human intervention.

---

## Files

```
hexgo/
  config.py          # tunable hyperparams — the only file the agent edits per trial
  tune.py            # orchestrator: write config, run trial, measure reward, log, loop
  tune_log.jsonl     # append-only experiment log: one JSON line per trial
  train.py           # modified: imports constants from config.py instead of inline
```

---

## config.py

Flat dict of all tunable parameters. Imported by `train.py` at startup.

```python
CFG = {
    "LR":               1e-3,
    "BATCH_SIZE":       64,
    "SIMS":             50,       # default self-play sim budget
    "CAP_FULL_FRAC":    0.25,     # KataGo: fraction of games at full sim budget
    "CPUCT":            1.5,      # PUCT exploration constant
    "DIRICHLET_ALPHA":  0.3,
    "DIRICHLET_EPS":    0.25,
    "ZOI_MARGIN":       6,
}
```

`train.py` replaces its inline constants with `from config import CFG` and references like `CFG["LR"]`. No other file changes.

---

## Parameter search space

| Parameter | Range | Notes |
|-----------|-------|-------|
| `LR` | 1e-4 – 5e-3 | log scale |
| `BATCH_SIZE` | 32, 64, 128 | discrete |
| `SIMS` | 25 – 50 | floor set by SIMS_MIN=25 |
| `CAP_FULL_FRAC` | 0.1 – 0.5 | |
| `CPUCT` | 1.0 – 3.0 | research recommends 2.5 for Phase 1 |
| `DIRICHLET_ALPHA` | 0.1 – 0.5 | |
| `DIRICHLET_EPS` | 0.10 – 0.35 | |
| `ZOI_MARGIN` | 3 – 8 | 3 is correctness floor per research |

Architecture params (blocks/channels) excluded — signal too slow for 5-gen trials.

---

## Trial loop (`tune.py`)

```
for each trial:
  1. Claude proposes new config.py (one or two param changes from current best)
  2. tune.py writes config.py
  3. train.py --gens 5 runs, importing config.py
  4. elo_delta = elo_after["EisensteinGreedyAgent"] - elo_before["EisensteinGreedyAgent"]
  5. Append to tune_log.jsonl: {trial, config, elo_delta, avg_loss, avg_ent, gen_time_s}
  6. If elo_delta < 0: revert config.py to previous best
  7. Claude reads last N log entries, reasons about landscape, proposes next trial
```

### ELO tournament changes during tuning

- Drop `mcts_50` match entirely — too slow for tight trial budgets
- Keep greedy-only eval every gen (already fast, zero MCTS overhead)
- Restore full tournament (`mcts_50` + greedy) after tuning completes

### Reward signal

`elo_delta` vs `EisensteinGreedyAgent` only. Promotion threshold: `delta > 0` (5 gens is too short for 55% win-rate confidence used in full training). Any positive improvement is kept.

---

## Skill: `hexgo-autotune`

Invoked as `/hexgo-autotune [N]` where N = number of trials (default 10).

The skill instructs Claude to:
1. Read `tune_log.jsonl` for experiment history
2. Read current `config.py` for baseline
3. Read `elo.json` for current ELO snapshot
4. Propose next config (change 1–2 params, explain reasoning)
5. Signal `tune.py` to run the trial
6. After result: log reasoning + outcome, propose next

Claude reasons about the log as a trajectory: "LR 5e-3 hurt after 5 gens (probably unstable), LR 1e-3 neutral, trying 3e-4 with larger batch to stabilise."

---

## Efficiency focus

The primary efficiency levers in order of expected impact:
1. **SIMS** — directly controls self-play throughput; 25 vs 50 is ~2× gen time
2. **CPUCT** — affects search depth distribution, influences batch utilisation
3. **LR + BATCH_SIZE** — training step efficiency
4. **CAP_FULL_FRAC** — ratio of cheap vs expensive games
5. **ZOI_MARGIN** — smaller margin → fewer candidates → faster MCTS per sim

Tuning should probe these roughly in this order across the first 10 trials.

---

## Non-goals

- Architecture search (blocks/channels) — deferred to Phase 5
- Multi-objective optimisation (ELO + speed) — ELO delta is the single signal
- Parallel trials — sequential only to avoid GPU contention on RTX 2060
