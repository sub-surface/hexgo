# HexGo — Training Pipeline (`train.py`, `config.py`)

## Pipeline Overview

Each generation:
1. **Self-play** — 8 parallel workers generate games using the current net via `InferenceServer`
2. **Overlap training** — batches sampled from the replay buffer run concurrently with self-play
3. **Post-gen training** — additional batches to reach `max(10, positions//BATCH_SIZE)` total
4. **ELO evaluation** — new net vs `EisensteinGreedyAgent(defensive=True)` (and `mcts_50` when not in tune mode)
5. **Checkpoint tournament** — new net must beat ≥55% of top-K saved checkpoints (not in tune mode)
6. **Heatmap** — policy distribution over a fixed 10-move test position saved as PNG
7. **Checkpoint** — `net_gen{N:04d}.pt` + `net_latest.pt`

---

## Hyperparameters (`config.py`)

All tunable params live in one dict. Edited by the autotune agent; read by `train.py` and `mcts.py` at startup.

| Param | Default | Range | Notes |
|-------|---------|-------|-------|
| `LR` | 1e-3 | 1e-4–5e-3 | Adam learning rate |
| `WEIGHT_DECAY` | 1e-4 | — | L2 regularization |
| `BATCH_SIZE` | 64 | 32, 64, 128 | Gradient batch size |
| `SIMS` | 50 | 25–200 | Full simulation budget (25% of games) |
| `SIMS_MIN` | 25 | 6–25 | Reduced budget floor (75% of games) — should be `SIMS // 8` for real diversity |
| `CAP_FULL_FRAC` | 0.25 | 0.1–0.5 | Fraction of games using full SIMS |
| `CPUCT` | 1.0 | 1.0–3.0 | PUCT exploration constant (loaded at module import — process restart required to change) |
| `DIRICHLET_ALPHA` | 0.3 | 0.1–0.5 | Root noise concentration (research: 10/\|ZoI\| ≈ 0.08–0.10) |
| `DIRICHLET_EPS` | 0.25 | 0.10–0.35 | Root noise weight |
| `ZOI_MARGIN` | 6 | 3–8 | Hex-distance ZOI pruning radius |
| `TD_GAMMA` | 0.99 | 0.95–1.0 | TD-lambda discount for value targets |
| `TEMP_HORIZON` | 40 | 20–60 | Cosine temp annealing parameter (floor reached at `TEMP_HORIZON/2` moves due to cosine shape) |
| `WEIGHT_SYNC_BATCHES` | 20 | 5–40 | Batches between weight sync to inference server |

---

## Self-Play Episode (`self_play_episode`)

Each worker generates one game:
- Uses `InferenceServer.evaluate()` for batched net evaluation
- Calls `game.zoi_moves(ZOI_MARGIN)` to restrict MCTS candidates
- **KataGo playout cap randomization**: 25% of games use full `SIMS`, 75% use `max(SIMS_MIN, SIMS//8)`
- Every 5th game uses `EisensteinGreedyAgent` as adversary (curriculum)
- Dirichlet noise applied at root: `α=DIRICHLET_ALPHA`, `ε=DIRICHLET_EPS`
- Temperature schedule: `temp = max(0.05, cos(π × move / TEMP_HORIZON))` — hits floor at `TEMP_HORIZON/2` moves

### Tree Reuse
`mcts_policy()` returns `new_root`; the caller passes `prev_root` on the next call. If a matching child exists, the subtree is recycled with fresh Dirichlet noise. Saves ~`SIMS / branching_factor` simulations per move.

### Value Targets (TD-lambda)
`z_t = TD_GAMMA^(T-1-t) × z_final` — later positions receive stronger signal than early positions. `z_final = +1` if the player at position `t` eventually wins, `-1` if they lose.

### Replay Buffer
- FIFO `deque(maxlen=50000)` positions
- Each entry: `{board, oq, or_, moves, probs, z}`
- Zobrist hash deduplication per generation (prevents near-duplicate positions from same game)

---

## D6 Augmentation at Train Time

`d6_augment_sample(item, tf_idx)` applies one of 12 D6 transforms to both the board array and move coordinates. The policy probability vector is permuted to match the transformed board. Applied at batch time (`tf_idx = random.randrange(12)`).

Moves that fall outside the 18×18 window after transformation are dropped and the policy renormalized. The effective augmentation ratio is slightly below 12× for edge positions.

---

## Training Loss

```
L = MSE(z, v) + CE(π, p) + c‖θ‖²
```

- Value loss: MSE against TD-lambda target
- Policy loss: cross-entropy over all legal in-window moves, normalized by items that had at least one in-window move
- Weight decay: L2 regularization (WEIGHT_DECAY, applied via Adam)
- FP16 AMP via `torch.amp.GradScaler`

---

## Checkpoint Tournament (`_tourney_promote`)

Maintains a pool of the top-K (`CKPT_POOL_K=5`) saved nets. New net must win ≥55% of games against the pool to become the training policy. Prevents catastrophic forgetting. Skipped in `--tune` mode.

---

## Performance Tracking (`PerfTracker`)

Per-generation timing for: `self_play`, `overlap_train`, `post_train`, `checkpoint`, `tournament`, `eval`, `heatmap`. Bottleneck warnings:
- `avg_batch_size < 2.0` → inference not batching effectively
- Self-play < 30% of gen time → training dominates, data pipeline under-utilized
- Dedup rate > 30% → positions too similar, explore more broadly

---

## Known Issues

1. **SIMS_MIN=25 too high**: With SIMS=50, `SIMS//8=6`, so `max(25, 6)=25`. The "reduced" games use half the full budget — this defeats the purpose of playout cap randomization. Should be ~6–10 for real diversity.

2. **Cosine temp semantics**: The formula `cos(π × move / TEMP_HORIZON)` goes negative at `move > TEMP_HORIZON/2` and clamps to 0.05, so the floor is reached at `TEMP_HORIZON/2` moves, not `TEMP_HORIZON`. The parameter should be renamed `TEMP_HALF_LIFE` or the formula adjusted.

3. **Overlap training can overtrain**: The overlap loop calls `train_batch` on every 50ms timeout regardless of new data arriving. On slow self-play generations this can over-fit to the current buffer before new games arrive. Should cap overlap batches.

4. **`d6_augment_sample` probs copy**: `probs` is passed by reference. Downstream in-place modification would corrupt buffer entries. Should use `.copy()`.

5. **Policy loss normalization**: Normalized by item count (positions with ≥1 in-window move), not move count. Loss magnitude varies with window clip rate, which correlates with board geometry and D6 augmentation angle.
