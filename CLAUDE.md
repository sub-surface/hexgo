# HexGo — Claude Context

## What this project is

HexGo is an AlphaZero-style self-play system for an infinite hexagonal Connect6 variant
played on the Eisenstein integer ring Z[ω]. The win condition is 6 consecutive pieces
along any of the three Z[ω] unit axes. Turn rule: P1 places 1 stone on turn 1, both
players place 2 per turn thereafter (1-2-2).

The mathematical framing is not cosmetic — it drives three concrete engineering choices:
- `HexConv2d`: masks the two non-adjacent corners `[0,0]` and `[2,2]` of 3×3 kernels
- D6 augmentation: 12 transforms = full dihedral symmetry group of the Eisenstein lattice
- `EisensteinGreedyAgent`: Erdős-Selfridge potential maximizer; used as curriculum opponent and permanent ELO anchor

---

## Python interpreter

**Always use `C:\Program Files\Python312\python.exe`** — this is the only install with
torch+CUDA, fastapi, and uvicorn. Running `python` or `py` from PowerShell will pick up
the wrong interpreter and fail with `ModuleNotFoundError`.

- **bash/WSL**: use `bash run.sh <script.py>` (already hard-codes the path)
- **PowerShell**: use `run.bat <script.py>` or `& "C:\Program Files\Python312\python.exe" <script.py>`

---

## How to run

**Training:**
```bash
bash run.sh train.py --gens 50 --games 20 --sims 100
```

**Dashboard (recommended):**
```bash
bash run.sh app.py
# Opens http://127.0.0.1:7860 in browser
# Controls training via Start/Stop buttons, live loss/ELO charts, replay viewer
```

**Autotune:**
```bash
bash run.sh tune.py --trials 10 --gens 5
```

**Replay viewer (terminal):**
```bash
bash run.sh replay.py replays/game_first_gen0001_20260329_160347.json
```

---

## Key files

| File | Role |
|------|------|
| `game.py` | HexGame engine: make/unmake, win detection, ZOI pruning |
| `net.py` | HexNet: HexConv2d trunk, value/policy heads, D6 augmentation |
| `mcts.py` | MCTS: pure rollout + AlphaZero modes, `_backprop` with 1-2-2 sign convention |
| `inference.py` | InferenceServer: batched GPU eval with CUDA Graphs + persistent cache |
| `train.py` | Training loop: self-play workers, overlap training, TD-lambda targets, ELO eval |
| `elo.py` | ELO system: NetAgent, EisensteinGreedyAgent, run_match |
| `config.py` | Single CFG dict: all tunable hyperparameters |
| `tune.py` | Autotune: random trial orchestrator, reward = Eisenstein winrate decrease |
| `replay.py` | Terminal hex-grid replay renderer |
| `server.py` | FastAPI backend: 12 REST endpoints + SSE event stream |
| `dashboard.html` | Single-file dark-mode dashboard: Training/Replay/Config tabs |
| `app.py` | Thin launcher: starts uvicorn, opens browser |

---

## Architecture decisions

### MCTS backprop sign convention
Value is from `node.player`'s perspective (+1 = node.player wins). Sign is only negated
when `node.parent.player != node.player` — this correctly handles the 1-2-2 rule where
the same player makes two consecutive placements without a sign flip between them.

`evaluate()` (InferenceServer/net) returns value from `game.current_player`'s POV.
At leaf nodes: if `node.player != game.current_player`, negate before backprop.

### Training self-play vs. ELO evaluation
- `mcts_policy()` in train.py = primary training path; uses `InferenceServer` (batched)
- `mcts_with_net()` in mcts.py = used only in ELO `NetAgent` matches (unbatched)
- `mcts()` = pure rollout; used in old `MCTSAgent` (removed from default eval path)

### Checkpoint tournament (REMOVED 2026-03-30)
`_tourney_promote()` was removed. Loading `net_gen*.pt` files saved before
`torch.compile` into an `OptimizedModule` wrapper crashes with `RuntimeError:
Error(s) in loading state_dict`. Old checkpoints are in `checkpoints/legacy/`.

### Dashboard thread safety
SSE uses `queue.Queue` (not `asyncio.Queue`) for broadcast from the `_metrics_watcher`
background thread. The SSE endpoint polls with `q.get_nowait()` + `asyncio.sleep(0.25)`.
Never switch back to `asyncio.Queue` — it is not thread-safe from non-async threads.

---

## Config (`config.py`)

```python
CFG = {
    "LR": 1e-3,
    "WEIGHT_DECAY": 1e-4,
    "BATCH_SIZE": 64,
    "SIMS": 50,
    "SIMS_MIN": 6,              # must be << SIMS (e.g. SIMS//8) for playout cap diversity
    "CAP_FULL_FRAC": 0.25,
    "GUMBEL_SELECTION": True,   # Gumbel argmax root selection (vs softmax-temp sampling)
    "CPUCT": 2.0,               # research target 2.0–2.5 (was 1.0)
    "DIRICHLET_ALPHA": 0.09,    # 10/|ZoI| ≈ 0.09 for ZOI_MARGIN=6 (was 0.3)
    "DIRICHLET_EPS": 0.25,
    "ZOI_MARGIN": 6,
    "ZOI_LOOKBACK": 16,         # recent moves defining ZOI focus (was hardcoded 8)
    "TRUNK_BLOCKS": 4,          # residual blocks (was 2)
    "TRUNK_CHANNELS": 64,       # hidden channels (was 32) — ~480K params total
    "WEIGHT_INIT": "ca",        # "ca" = hex NCA Laplacian priors | "xavier" = standard
    "TD_GAMMA": 0.99,
    "TEMP_HORIZON": 40,         # cosine reaches floor at TEMP_HORIZON moves
    "WEIGHT_SYNC_BATCHES": 20,
    "RECENCY_WEIGHT": 0.75,     # fraction of each batch from recent half of buffer
    "AUX_LOSS_OWN": 0.1,        # ownership head loss weight
    "AUX_LOSS_THREAT": 0.1,     # threat head loss weight
    "UNC_LOSS_WEIGHT": 0.1,     # value uncertainty head (Gaussian NLL) loss weight
    "VALUE_LOSS_WEIGHT": 2.0,   # multiplier on MSE value loss — prevents policy CE dominating by ~20×
    "ENTROPY_REG": 0.01,        # policy entropy regularization bonus weight
}
```

`CPUCT` and `TRUNK_*` are loaded at module import time — process restart required to change them.

---

## Test suite

```bash
pytest tests/ -v   # 26 tests, all should pass
```

Tests cover: win detection on all 3 axes, undo correctness, D6 symmetry, EisensteinGreedyAgent,
autotune pipeline end-to-end.

---

## Current status (2026-03-30)

All confirmed correctness bugs (FIX-1 through FIX-10) are resolved. The training pipeline
is sound. Dashboard is live. Ready for sustained multi-generation training runs.

**Completed 2026-03-30 (session 2):**
- `CPUCT` raised 1.0 → 2.0 (research target); `DIRICHLET_ALPHA` reduced 0.3 → 0.09 (10/|ZoI|)
- Recency-weighted replay buffer: 75% recent half / 25% uniform per batch (`RECENCY_WEIGHT`)
- Auxiliary heads: ownership (`aux_own`) + threat (`aux_threat`) thin 1×1 convs off trunk;
  labels generated from game outcome; loss weights `AUX_LOSS_OWN=0.1`, `AUX_LOSS_THREAT=0.1`
- History planes verified already correct (uses `player_history`, not board dict)
- `VALUE_LOSS_WEIGHT=2.0`: fixes value head collapse caused by CE policy loss (~3.8) drowning MSE value loss (~0.1)
- Per-component loss tracking (`loss_v`, `loss_p`, `loss_aux`) logged per gen and written to `metrics.jsonl`
- Move accuracy metric (`move_acc`): top-1 policy agreement with `EisensteinGreedyAgent(defensive=True)`,
  sampled from 40 buffer positions per gen; baseline ~17%; written to `metrics.jsonl` and dashboard MOVE ACC chart

**Completed 2026-03-31 (session 3):**
- Board window now centers on centroid of last 20 moves (`N_RECENT=20`) — fixes silent piece clipping for spread games
- `ZOI_LOOKBACK=16` config key (was hardcoded 8) — wider threat coverage, autotune-able
- Trunk scaled to 4 blocks × 64 channels (`TRUNK_BLOCKS`, `TRUNK_CHANNELS` in CFG) — ~480K params
- CA weight init (`WEIGHT_INIT="ca"`): HexConv2d kernels initialized with hex-Laplacian NCA priors
- Value uncertainty head (`value_var`): predicts σ² via Softplus; trained with Gaussian NLL (`UNC_LOSS_WEIGHT=0.1`); `avg_sigma` logged to metrics and dashboard
- Gumbel root selection (`GUMBEL_SELECTION=True`): Gumbel argmax replaces softmax-temp sampling at root; better exploitation at low sim counts
- Policy entropy regularization (`ENTROPY_REG=0.01`): `-β·H(π)` bonus in `loss_p` prevents premature policy collapse
- All old checkpoints moved to `checkpoints/legacy/` — fresh start

**Remaining open items:**
- C++/Rust MCTS for true inference batching (avg_batch_size > 2.0)
- G-CNN full D6 equivariance (deferred — major rewrite)
- MuZero reanalysis (deferred — compute cost)

See `docs/ASSESSMENT.md` and `docs/ROADMAP.md` for full details.
