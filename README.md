# HexGo

An AlphaZero-style self-play system for an infinite hexagonal Connect6 variant played on the Eisenstein integer ring **Z[ω]**. The goal is 6 consecutive pieces along any of the three hex axes. Turn rule: P1 places 1 stone on turn 1, both players place 2 per turn thereafter (1-2-2).

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Training Loop (train.py)                │
│                                                                 │
│  ┌──────────────┐     positions     ┌───────────────────────┐  │
│  │  Self-Play   │ ────────────────► │   Replay Buffer       │  │
│  │  Workers     │                   │   (FIFO, 50k cap)     │  │
│  │  (parallel)  │                   └──────────┬────────────┘  │
│  └──────┬───────┘                              │ sample        │
│         │ evaluate(game)                        ▼               │
│         ▼                             ┌─────────────────────┐  │
│  ┌──────────────┐  batched GPU   ┌───►│  Training Step      │  │
│  │  Inference   │ ◄──────────────┤   │  policy + value +   │  │
│  │  Server      │                │   │  aux + unc loss     │  │
│  │  (batched)   │                │   └──────────┬──────────┘  │
│  └──────────────┘                │              │ weights      │
│         │                        │              ▼               │
│         │                   ┌────┘   ┌─────────────────────┐  │
│         └───────────────────┘        │  HexNet (~480K par) │  │
│                  weights sync        └─────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                         HexNet (net.py)                         │
│                                                                 │
│  Input [11, 18, 18]                                             │
│    → HexConv2d(11→64) + BN + ReLU         [hex-masked stem]    │
│    → 4× ResBlock(64ch, HexConv2d)         [trunk]              │
│    → GlobalPoolBranch                     [board context]       │
│    ├→ value head          → scalar ∈ [-1,1]                    │
│    ├→ uncertainty head    → σ² (Gaussian NLL)                  │
│    ├→ ownership aux head  → [18,18] ∈ (-1,1)                   │
│    ├→ threat aux head     → [18,18] ∈ (0,1)                    │
│    └→ policy head         → logit per candidate move           │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                         MCTS (mcts.py)                          │
│                                                                 │
│  Root node                                                      │
│    → ZOI pruning (~80-90% branch reduction)                     │
│    → Dirichlet noise on priors                                  │
│    → PUCT selection (CPUCT=2.0)                                 │
│    → Leaf: InferenceServer.evaluate()                           │
│    → Backprop with 1-2-2 sign convention                        │
│    → Gumbel argmax root selection                               │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                      Dashboard (app.py)                         │
│                                                                 │
│  Browser UI (dashboard.html)                                    │
│    ↕ REST + SSE                                                 │
│  FastAPI backend (server.py)                                    │
│    → start/stop training subprocess                             │
│    → stream metrics.jsonl via SSE                               │
│    → read/write config.py safely (ast, type-validated)          │
│    → serve replays                                              │
└─────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

**Requirements:** Python 3.12, PyTorch with CUDA, FastAPI, uvicorn.

```bash
# Install dependencies
pip install torch torchvision fastapi uvicorn numpy

# Run the dashboard (recommended)
bash run.sh app.py
# Opens http://127.0.0.1:7860 — use Start/Stop buttons to control training

# Or run training directly
bash run.sh train.py --gens 50 --games 20 --sims 100

# Run autotune
bash run.sh tune.py --trials 10 --gens 5

# Run tests
pytest tests/ -v
```

> **Windows note:** Always use `bash run.sh <script>` or `& "C:\Program Files\Python312\python.exe" <script>`. The bare `python` command will pick up the wrong interpreter.

---

## The Game

**AP-6 Maker-Maker on Z[ω]** — Connect6 on an infinite hexagonal grid.

- **Win condition:** 6 consecutive pieces along any of the 3 Z[ω] unit axes (q-axis, r-axis, diagonal)
- **Turn rule:** P1 places 1 stone on turn 1; both players place 2 per turn thereafter
- **Board:** Infinite grid; the 18×18 window is centered on the centroid of the last 20 moves
- **Coordinate system:** Axial (q, r) — isomorphic to the Eisenstein integers Z[ω]

The Z[ω] framing is not cosmetic — it drives three concrete engineering choices:

| Choice | What it does |
|--------|-------------|
| `HexConv2d` | Masks the 2 non-adjacent corners of 3×3 kernels — Z[ω]-faithful receptive field |
| D6 augmentation | 12 symmetry transforms × every training sample = 12× free data diversity |
| `EisensteinGreedyAgent` | Erdős-Selfridge potential maximizer; permanent ELO anchor and curriculum opponent |

---

## Key Files

| File | Role |
|------|------|
| `game.py` | HexGame engine: make/unmake, win detection, ZOI pruning, candidates set |
| `net.py` | HexNet: HexConv2d trunk, 4 heads, D6 augmentation, CA weight init |
| `mcts.py` | MCTS: PUCT selection, 1-2-2 backprop sign convention |
| `inference.py` | InferenceServer: dynamic batching, transposition cache, persistent cross-gen cache |
| `train.py` | Training loop: self-play workers, overlapped training, TD-lambda, ELO eval |
| `elo.py` | ELO system: NetAgent, EisensteinGreedyAgent, run_match |
| `config.py` | All tunable hyperparameters — edit this to change anything |
| `server.py` | FastAPI backend: 12 REST endpoints + SSE stream |
| `dashboard.html` | Single-file dark-mode dashboard: Training / Replay / Config tabs |
| `app.py` | Launcher: starts uvicorn, opens browser |
| `tune.py` | Autotune: random trial orchestrator, reward = Eisenstein winrate decrease |
| `replay.py` | Terminal hex-grid replay renderer |

---

## Configuration

All hyperparameters live in `config.py`. Key ones:

| Key | Default | Notes |
|-----|---------|-------|
| `SIMS` | 50 | Full MCTS sim budget (25% of games) |
| `SIMS_MIN` | 15 | Reduced budget floor (75% of games) |
| `CPUCT` | 2.0 | PUCT exploration constant |
| `TRUNK_BLOCKS` | 4 | Residual blocks — requires process restart to change |
| `TRUNK_CHANNELS` | 64 | Hidden channels — requires process restart to change |
| `LR` | 1e-3 | Learning rate (decays via CosineAnnealingLR) |
| `ENTROPY_REG` | 0.01 | Policy entropy bonus weight |
| `DIRICHLET_ALPHA` | 0.09 | Root noise concentration (10/\|ZoI\|) |

You can edit `config.py` directly or use the Config tab in the dashboard. Changes take effect on the next training run (architecture params require a restart).

---

## Training Signal

Each generation:

1. **Self-play:** `games_per_gen` games run in parallel threads, guided by MCTS + HexNet
2. **Replay buffer:** positions stored as `(board_tensor, move_plane, z)` with TD-lambda value targets
3. **Training:** batches sampled with recency weighting (75% recent half); D6 augmented on the fly
4. **ELO eval:** net vs `EisensteinGreedyAgent` every generation; ELO written to `elo.json`
5. **Metrics:** loss components, `move_acc`, `avg_sigma`, `eis_winrate` appended to `metrics.jsonl`

---

## Docs

| Doc | Contents |
|-----|----------|
| [DESIGN.md](docs/DESIGN.md) | Mathematical framework, Z[ω] axioms, component index |
| [NET.md](docs/NET.md) | Network architecture, input encoding, HexConv2d, D6 group |
| [MCTS.md](docs/MCTS.md) | Search algorithm, backprop sign convention, known issues |
| [TRAINING.md](docs/TRAINING.md) | Self-play loop, loss functions, TD-lambda targets |
| [INFERENCE.md](docs/INFERENCE.md) | Batching server, caching, performance notes |
| [ELO.md](docs/ELO.md) | Rating system, agents, evaluation mechanics |
| [AUTOTUNE.md](docs/AUTOTUNE.md) | Hyperparameter search, tune.py workflow |
| [ROADMAP.md](docs/ROADMAP.md) | Feature status and open items |
| [ASSESSMENT.md](docs/ASSESSMENT.md) | Honest code review and known issues |

---

## Tests

```bash
pytest tests/ -v   # 26 tests
```

Covers: win detection on all 3 axes, undo correctness, D6 symmetry, EisensteinGreedyAgent, autotune pipeline.
