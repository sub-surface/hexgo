# HexGo — Design Document

## Game: Hexagonal Connect6

Infinite hexagonal grid, axial (q, r) coordinates, flat-top orientation.
Six-in-a-row wins along any of the three hex axes.
Board stored as a sparse dict — no pre-allocated grid, no size limit.

### Complexity estimate
Go 19×19 has ~2×10^170 states. Our game is infinite but practical games
run 60–250 moves. At each move ~20–60 candidates exist (neighbors of
existing pieces). Branching factor ≈ 30, depth ≈ 150 → game tree ≈ 30^150,
vastly larger than Go in principle but in practice heavily pruned by the
candidate restriction. Effective complexity is closer to 9×9 Go.

### Turn Logic: The "1-2-2" Rule
To balance the first-mover advantage, HexGo follows Connect6 turn logic:
- **Turn 1:** Black (Player 1) places **one** tile.
- **Turn 2+:** Each player places **two** tiles on their turn.
- **Win Detection:** A win is checked after *every* placement. If a player
  completes 6-in-a-row with their first tile of a 2-tile turn, the game ends
  immediately — the second tile is never placed.

Implementation: `game.py` tracks `placements_in_turn` (int). The limit is
`1` when `len(move_history) == 1` (the first move ever placed), else `2`.
Player flips when `placements_in_turn >= limit`. The undo stack saves
`prev_placements` and `prev_player` to restore state exactly.

---

## Engine (`game.py`)

- `make(q, r)` / `unmake()` — zero-copy move/undo via undo stack.
  Undo entry: `(q, r, removed_candidates, added_candidates, prev_placements, prev_winner, prev_player)`
- `candidates` set maintained incrementally — `legal_moves()` is O(1).
- Win check only scans the 3 axes through the last piece — O(WIN_LENGTH).
- `clone()` deep-copies board, candidates, `placements_in_turn` (used by ELO evaluation).
- `play()` is an alias for `make()` for backward compatibility.

---

## MCTS (`mcts.py`)

### Node
`Node.__slots__` = `(move, parent, children, visits, value, prior, player)`

`player` records which player is to move *at that node*. This is set to
`game.current_player` at the moment the node is created.

### Multi-Placement Backpropagation
Standard MCTS negates the backpropagated value at every node (assuming the
player alternates). In HexGo, value is only negated when `node.parent.player
!= node.player`. This correctly models two consecutive placements by the same
player without sign flip.

### Two modes
- `mcts(game, sims)` — pure rollout baseline (used by `MCTSAgent` in ELO matches).
- `mcts_with_net(game, net, sims)` — AlphaZero style. Net value replaces
  rollout; policy logits set node priors; Dirichlet noise (α=0.3, ε=0.25)
  added to root for exploration.

**Known gap:** In `mcts_with_net`, leaf children are created without
`player=game.current_player` (uses default `player=1`). This does not affect
`train.py` which uses the inline `mcts_policy()` function instead, but it
means `mcts_with_net` in standalone mode has subtly incorrect backprop for
second-placements. Low priority — ELO uses pure `mcts()` for the baseline.

---

## Neural Net (`net.py`)

### Sizing & Performance

| Factor          | Value        | Reasoning                                              |
|-----------------|--------------|--------------------------------------------------------|
| Board window    | 18×18        | Centered on centroid; wider than 15×15 for long games  |
| Hidden channels | 32           | Optimized for RTX 2000-series throughput               |
| Residual blocks | 2            | Deep enough for strategy, shallow for <5ms GPU call    |
| Precision       | **FP16 AMP** | `torch.amp.autocast` doubles memory bandwidth          |
| Total params    | ~121K        | Down from 355K (64ch/4 blocks) — 3× faster inference  |

### Board Encoding
`encode_board(game)` returns `float32 [3, 18, 18]` centered on the centroid
of all pieces. Channel 0 = player 1 pieces, channel 1 = player 2 pieces,
channel 2 = to-move plane (0.0 or 1.0).

Returns `(arr, (oq, or_))` — the origin offset is needed to map candidate
moves into the window.

### Action Representation
`encode_move(q, r, oq, or_)` returns a `[1, 18, 18]` one-hot plane, or
`None` if the move is outside the 18×18 window. Callers must handle `None`.

Policy head: `(trunk_features, move_plane) → scalar logit`.
Training: **Cross-Entropy** over all legal moves in a position (more stable
than single-move BCE; avoids the index-0 bug of earlier versions).

### Architecture
```
Input [3,18,18]
  → Conv2d(3→32, 3×3) + BN + ReLU          [stem]
  → 2× ResBlock(32ch)                       [trunk]
  ┌→ Conv2d(32→1, 1×1) + BN + ReLU → FC → Tanh  [value head → scalar]
  └→ Conv2d(32→2, 1×1) + BN + ReLU → cat(move_plane) → FC  [policy → scalar]
```

### Checkpoint compatibility
Old checkpoints trained with BOARD_SIZE=15 or HIDDEN=64/N_BLOCKS=4 are
**incompatible** with the current architecture. Stale files are in
`checkpoints/legacy/`. Always use `checkpoints/net_latest.pt`.

---

## Inference Server (`inference.py`)

### Dynamic Batching
Parallel self-play threads each call `server.evaluate(game)` which blocks on
a `queue.Queue`. The server loop collects up to `batch_size` (default 4)
requests within `timeout_ms` (default 5ms), then fires a single GPU call.

Throughput scales with `num_workers` until the GPU saturates. On RTX 2060
with our 121K-param net, useful gains up to ~8–16 workers.

**Known issue:** `avg_batch_size` frequently reads ~1.0 in practice. This
indicates the inference timeout is smaller than the MCTS think-time per
position, so requests rarely arrive together. Increase `INF_TIMEOUT` or
`NUM_WORKERS` to improve batching.

### Evaluation Cache
A transposition table keyed on `frozenset(game.board.items())`. Typically
achieves 20–40% hit rate during MCTS. Thread-safe via `_cache_lock`.

The cache is scoped to a single generation (cleared when `InferenceServer` is
re-created each gen) to avoid staleness across weight updates.

---

## Training (`train.py`)

### Pipeline
1. **Self-play:** 4 parallel workers generate games using the current net
   via `InferenceServer`.
2. **Buffering:** Positions stored in a 50K-position FIFO `deque`.
3. **Training:**
   - `torch.amp.GradScaler` for FP16 numerical stability.
   - Value loss: MSE against `z ∈ {+1, -1, 0}`.
   - Policy loss: Cross-entropy over all legal moves (visit-count distribution
     from MCTS root as target).
4. **ELO Evaluation:** New net vs `mcts_50` baseline (`run_match()`).
5. **Checkpoint:** `net_genXXXX.pt` + `net_latest.pt` after each generation.

### Buffer Format
Each buffer entry is a dict:
```python
{
    "board":  np.ndarray [3,18,18],
    "oq":     int,        # centroid origin q
    "or_":    int,        # centroid origin r
    "moves":  list[tuple],# all legal moves within the 18×18 window
    "probs":  np.ndarray, # normalized MCTS visit distribution
    "z":      float,      # +1 winner, -1 loser, 0 draw
}
```

### Metrics
- **avg_loss** — combined value + policy loss
- **avg_ent** — policy entropy (lower = more confident)
- **hits** — inference cache hits during self-play

### Known Issues
- `avg_loss=nan` can occur if all moves in a position are clipped outside the
  18×18 window (zero-sum visit distribution). Guard: `if s > 0: v_dist /= s
  else: uniform`. This fix is in place in the current code.
- `avg_batch_size ≈ 1.0` — see Inference Server section.

---

## GUI (`app.py`)

A `tkinter` monitor for real-time visualization:
- **Pause/Resume:** `threading.Event` halts the self-play worker, freeing GPU.
- **Reset Wins:** Clears win/loss counters.
- **Training Feed:** Streams real-time INFO logs into the UI.
- **Board canvas:** Debounced `<Configure>` handler prevents crash on resize.

---

## ELO (`elo.py`)

- `ELO` class: tracks ratings per agent name, updates via standard ELO formula.
- `NetAgent(net, sims)`: uses `mcts_with_net` (or `mcts_policy` via
  `InferenceServer` — check usage in `train.py`).
- `MCTSAgent(sims)`: uses pure rollout `mcts()`.
- `run_match(a, b, n_games)`: plays N games alternating colors, returns win counts.

---

## File Map

```
hexgo/
  game.py       Engine — 1-2-2 turn logic, incremental candidates, make/unmake
  mcts.py       MCTS — player-aware backprop, rollout + net modes
  net.py        HexNet — ResNet 18x18/32ch/2blk + FP16 AMP, encode_board/move
  inference.py  InferenceServer — dynamic batching + transposition cache
  train.py      Training loop — AMP, FIFO buffer, CE policy loss, ELO eval
  elo.py        ELO rating — NetAgent, MCTSAgent, run_match
  app.py        GUI monitor — pause/resume, win counter, log stream
  replay.py     Terminal replay — 1-2-2 aware, colored last-move bracket
  test_game.py  Unit tests — all pass (8 tests)
  render.py     Hex board renderer (shared by app + replay)
  checkpoints/  net_gen*.pt, net_latest.pt
  checkpoints/legacy/  old 15×15 / 64ch checkpoints (incompatible)
  replays/      game_first_genXXXX_*.json, game_last_genXXXX_*.json
  docs/         DESIGN.md (this file), ROADMAP.md
```

---

## Running

```bash
# Live self-play monitor
python app.py

# Train (50 generations, 100 sims, 20 games/gen)
python train.py --gens 50 --sims 100 --games 20

# Terminal Replay
python replay.py replays/game_XXXX.json --delay 0.1

# Smoke tests
python net.py
python test_game.py
```

Always use Python 3.12: `"C:\Program Files\Python312\python.exe"`.
Python 3.14 has no CUDA PyTorch wheels.
