# HexGo — Training Pipeline (`train.py`, `config.py`)

## Pipeline Overview

Each generation (as of 2026-03-30 simplification):
1. **Self-play** — 8 parallel workers generate games using the current net via `InferenceServer`
2. **Overlap training** — batches sampled from the replay buffer run concurrently with self-play (capped at `WEIGHT_SYNC_BATCHES` to prevent overfit)
3. **Post-gen training** — additional batches to reach `max(10, positions//BATCH_SIZE)` total
4. **ELO evaluation** — new net vs `EisensteinGreedyAgent(defensive=True)` only (~5s/gen)
5. **Checkpoint** — `net_gen{N:04d}.pt` + `net_latest.pt`
6. **Metrics** — appended to `metrics.jsonl` for dashboard live charts

*Removed: checkpoint tournament (crash source), MCTSAgent eval (100–177s/gen), per-gen heatmap server. `save_heatmap()` still available manually.*

---

## Hyperparameters (`config.py`)

All tunable params live in one dict. Edited by the autotune agent; read by `train.py` and `mcts.py` at startup.

| Param | Value | Range | Notes |
|-------|-------|-------|-------|
| `LR` | 1e-3 | 1e-4–5e-3 | Adam learning rate |
| `WEIGHT_DECAY` | 1e-4 | — | L2 regularization |
| `BATCH_SIZE` | 64 | 32, 64, 128 | Gradient batch size |
| `SIMS` | 50 | 25–200 | Full simulation budget (25% of games) |
| `SIMS_MIN` | 6 | 6–25 | Reduced budget floor (75% of games) — `SIMS // 8` for diversity |
| `CAP_FULL_FRAC` | 0.25 | 0.1–0.5 | Fraction of games using full SIMS |
| `CPUCT` | **2.0** | 1.0–3.0 | PUCT exploration constant — research target 2.0–2.5; loaded at module import (restart to change) |
| `DIRICHLET_ALPHA` | **0.09** | 0.05–0.3 | Root noise concentration — `10/\|ZoI\| ≈ 0.09` for ZOI_MARGIN=6 |
| `DIRICHLET_EPS` | 0.25 | 0.10–0.35 | Root noise weight |
| `ZOI_MARGIN` | 6 | 3–8 | Hex-distance ZOI pruning radius |
| `TD_GAMMA` | 0.99 | 0.95–1.0 | TD-lambda discount for value targets |
| `TEMP_HORIZON` | 40 | 20–60 | Cosine temp annealing parameter (floor reached at `TEMP_HORIZON` moves) |
| `WEIGHT_SYNC_BATCHES` | 20 | 5–40 | Batches between weight sync to inference server |
| `RECENCY_WEIGHT` | **0.75** | 0.5–1.0 | Fraction of each batch drawn from recent half of buffer |
| `AUX_LOSS_OWN` | **0.1** | 0.0–0.5 | Ownership head loss weight (0 = disabled) |
| `AUX_LOSS_THREAT` | **0.1** | 0.0–0.5 | Threat head loss weight (0 = disabled) |
| `VALUE_LOSS_WEIGHT` | **2.0** | 1.0–5.0 | Multiplier on MSE value loss — prevents policy CE (~3.8) drowning value MSE (~0.1) by ~20× without it |

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
- Each entry: `{board, oq, or_, moves, probs, z, own_label, threat_label, greedy_move}`
- Zobrist hash deduplication per generation (prevents near-duplicate positions from same game)
- **Recency-weighted sampling**: each batch draws `RECENCY_WEIGHT` fraction from the most-recent half of the buffer, remainder uniform. Prevents anchoring to early-training incompetent play.

---

## D6 Augmentation at Train Time

`d6_augment_sample(item, tf_idx)` applies one of 12 D6 transforms to the board array, move coordinates, and aux label arrays (`own_label`, `threat_label`). Aux arrays use `_transform_aux()` — a dedicated single-channel spatial remap that avoids the `to-move` channel special-case in `_transform_board()`. The policy probability vector is permuted to match the transformed board. Applied at batch time (`tf_idx = random.randrange(12)`).

Moves that fall outside the 18×18 window after transformation are dropped and the policy renormalized. The effective augmentation ratio is slightly below 12× for edge positions.

---

## Training Loss

```
L = VALUE_LOSS_WEIGHT·MSE(z, v) + CE(π, p) + AUX_OWN·MSE(own, own_pred) + AUX_THREAT·BCE(threat, threat_pred) + c‖θ‖²
```

- Value loss: MSE against TD-lambda target, scaled by `VALUE_LOSS_WEIGHT=2.0` to counter the ~20× scale gap between CE policy loss (~3.8) and MSE value loss (~0.1)
- Policy loss: cross-entropy over all legal in-window moves, normalized by items that had at least one in-window move
- Ownership aux loss: MSE on `[S, S]` map (+1=P1, -1=P2, 0=empty at game end); weight `AUX_LOSS_OWN=0.1`
- Threat aux loss: BCE-with-logits on `[S, S]` binary map (1=cell on winning 6-in-a-row); weight `AUX_LOSS_THREAT=0.1`; AMP-safe (uses logits, not post-sigmoid)
- Weight decay: L2 regularization (WEIGHT_DECAY, applied via Adam)
- FP16 AMP via `torch.amp.GradScaler`

Aux weights are kept small per the bitter-lesson principle: they guide trunk representation without dominating the main value+policy signal.

### Per-Component Loss Logging

`train_batch()` returns `{loss, loss_v, loss_p, loss_aux, entropy}`. The gen loop tracks these separately and logs:
```
Train: N batches  loss=X  loss_v=X  loss_p=X  aux=X  ent=X
```
All components are written to `metrics.jsonl` as `avg_loss`, `avg_loss_v`, `avg_loss_p`, `avg_aux`.

---

## Move Accuracy Metric (`compute_move_acc`)

After ELO evaluation each generation, `compute_move_acc(net, buffer, n_samples=40)` samples 40 positions from the buffer and computes the fraction where the net's top-1 policy move (argmax over ZOI logits) matches `EisensteinGreedyAgent(defensive=True)`'s chosen move.

The greedy agent's answer is captured at self-play time (stored as `greedy_move` in each buffer item) so the metric doesn't require reconstructing game state. Logged as `move_acc` in `metrics.jsonl` and shown on the dashboard MOVE ACC chart.

Baseline: ~17% at gen 0 (random policy). Target: >40% after sustained training indicates the net has learned basic chain extension / threat blocking.

---

## Checkpoint Tournament (REMOVED)

`_tourney_promote()` was removed in 2026-03-30. Root cause: `torch.load` of old
`net_gen*.pt` files into a `torch.compile`d (`OptimizedModule`) wrapper raised
`RuntimeError: Error(s) in loading state_dict`. The crash happened silently after
checkpoint save, making it appear to be an ELO eval crash. Old `net_gen*.pt` files
are now treated as legacy and quarantined in `checkpoints/legacy/`.

---

## Performance Tracking (`PerfTracker`)

Per-generation timing for: `self_play`, `overlap_train`, `post_train`, `checkpoint`, `tournament`, `eval`, `heatmap`. Bottleneck warnings:
- `avg_batch_size < 2.0` → inference not batching effectively
- Self-play < 30% of gen time → training dominates, data pipeline under-utilized
- Dedup rate > 30% → positions too similar, explore more broadly

---

## Known Issues

1. **Policy loss normalization**: Normalized by item count (positions with ≥1 in-window move), not move count. Loss magnitude varies with window clip rate, which correlates with board geometry and D6 augmentation angle. (Low priority — consistent across gens.)

Previously listed issues (1–4 above) have been resolved as of 2026-03-30:
- `SIMS_MIN` set to `6`
- Cosine temp formula fixed to `cos(π/2 × move / TEMP_HORIZON)`
- Overlap training capped by `WEIGHT_SYNC_BATCHES`
- D6 augmentation uses `probs.copy()`
