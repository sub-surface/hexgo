# HexGo — Autotune System (`tune.py`, `config.py`, skill `hexgo-autotune`)

## Overview

Autonomous hyperparameter tuning following the autoresearch pattern (Karpathy, 2024):
Claude proposes a config change, runs a fixed-budget trial, measures a scalar reward, and iterates.

```
[Claude reads tune_log.jsonl]
    → proposes config.py edit (1-2 params)
    → python tune.py --gens 5 --games 10
    → tune.py measures ELO delta
    → appends to tune_log.jsonl
    → keep or revert config.py
```

---

## `tune.py` Trial Loop

1. Backup `config.py` → `config.py.bak`
2. Read pre-trial ELO from `elo.json`
3. Delete `tune_result.json`
4. Run `python train.py --tune --gens 5 --games 10 --sims {CFG["SIMS"]}`
   - `--tune` disables checkpoint tournament and `mcts_50` eval (too slow for tuning)
   - Each gen writes one entry to `tune_result.json`: `{gen, eis_winrate, avg_loss, avg_ent, gen_time_s}`
5. Read post-trial ELO, compute delta
6. Append to `tune_log.jsonl`: `{cfg, elo_before, elo_after, elo_delta, avg_eis_winrate, gen_metrics, elapsed_s, kept}`
7. If `elo_delta < 0`: revert `config.py` from backup

---

## Critical Bug: Inverted Reward Signal

`tune.py` tracks `eisenstein_def`'s ELO as the reward signal, and **keeps** configs
when `elo_delta >= 0` (eisenstein's rating went up).

But eisenstein_def is the **opponent**. When the net improves, eisenstein loses
more games and its ELO **falls**. So `elo_delta >= 0` for eisenstein means
the net got **worse** — the logic is inverted.

**Effect**: every config that helped the net has been reverted; every harmful config has been kept.

**Fix**: Track the net's ELO delta (`net_gen{last_gen}`) instead, or use
`avg_eis_winrate` from `tune_result.json` and keep when it **decreases** (net winning more against eisenstein).

---

## `tune_log.jsonl` Format

One JSON object per line:
```json
{
  "cfg": {"LR": 0.001, "SIMS": 50, ...},
  "elo_before": 1403.1,
  "elo_after": 1397.2,
  "elo_delta": -5.9,
  "avg_eis_winrate": 0.72,
  "gen_metrics": [
    {"gen": 1, "eis_winrate": 0.70, "avg_loss": 0.42, "avg_ent": 2.1, "gen_time_s": 45.0},
    ...
  ],
  "elapsed_s": 230.0,
  "kept": true
}
```

Note: `elo_delta` here is eisenstein's delta. Until the bug above is fixed, a
negative `elo_delta` in the log means the net probably **improved**.

---

## Parameter Space for Tuning

| Parameter | Range | Priority | Notes |
|-----------|-------|----------|-------|
| `SIMS` | 25–200 | 1 | Biggest throughput lever |
| `CPUCT` | 1.0–3.0 | 2 | Research recommends 2.5 → 2.0 → 1.5 across phases; currently 1.0 (too low) |
| `LR` | 1e-4–5e-3 | 3 | Log scale |
| `BATCH_SIZE` | 32, 64, 128 | 3 | Discrete |
| `SIMS_MIN` | 6–25 | 3 | Should be `SIMS//8` for real playout cap diversity |
| `CAP_FULL_FRAC` | 0.1–0.5 | 4 | |
| `ZOI_MARGIN` | 3–8 | 5 | Floor=3 for correctness |
| `TD_GAMMA` | 0.95–1.0 | 5 | Lower = faster early-position signal |
| `TEMP_HORIZON` | 20–60 | 6 | Note: floor reached at TEMP_HORIZON/2 moves |
| `WEIGHT_SYNC_BATCHES` | 5–40 | 6 | |
| `DIRICHLET_ALPHA` | 0.08–0.5 | 7 | Research: 10/\|ZoI\| ≈ 0.08–0.10; current 0.3 may be too high |
| `DIRICHLET_EPS` | 0.10–0.35 | 7 | |

### Research-Guided Targets

- **CPUCT**: Research recommends 2.5 for Phase 1 (Silver et al., 2018; Wu, 2019). Current 1.0 may under-explore.
- **DIRICHLET_ALPHA**: Research recommends `10/|ZoI|` ≈ 0.08–0.10 for ZoI~100 cells. Current 0.3 injects too much noise.
- **SIMS_MIN**: Should be `SIMS // 8` per KataGo playout cap design. Currently 25 = SIMS/2.
- **TD_GAMMA**: 0.97 for faster early-position learning (vs current 0.99).

---

## Future Experiments (not yet in `CFG`)

These require code changes beyond `config.py`:

| Experiment | Files | Expected gain | Effort |
|------------|-------|---------------|--------|
| **CA weight init** | `net.py` | Z[ω]-aligned priors; faster early convergence | 2h — add `WEIGHT_INIT: "xavier"\|"ca"` to CFG, implement `init_weights_ca(net)` |
| **Auxiliary heads** (ownership + threat) | `net.py`, `train.py` | 20–50% convergence speedup via dense gradients | 4h — add `AUX_LOSS_OWN`, `AUX_LOSS_THREAT` to CFG |
| **Recency-weighted replay** | `train.py` | Better policy tracking vs mode collapse | 1h — add `RECENCY_WEIGHT` 0.5–0.9 |
| **MuZero-style reanalysis** | `train.py` | Freshen stale policy targets | 4h |
| **Scale trunk** (4 blocks / 64ch) | `net.py` | Capacity for harder games | 2h — requires async self-play first |

---

## Skill: `/hexgo-autotune`

The `hexgo-autotune` skill at `~/.claude/skills/hexgo-autotune/SKILL.md` provides
Claude with the full parameter space, research-guided targets, and reasoning protocol
for autonomous tuning sessions.

Invoke with: `/hexgo-autotune [N]` where N is the number of trials (default 10).

**Reasoning protocol**: After each trial, reason about the ELO trajectory before
proposing the next config. Change 1 param when direction is uncertain; 2 when
direction is clear. Start with efficiency params (SIMS, CPUCT) before quality
params (LR, Dirichlet).
