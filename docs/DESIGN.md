# HexGo — Design Document

## Mathematical Framework: AP-6 Maker-Maker on Z[ω]

### Eisenstein Integer Isomorphism

The hex grid with axial coordinates (q, r) is isomorphic to the Eisenstein
integer ring **Z[ω]** where ω = e^(2πi/3). Each cell maps to q + r·ω ∈ Z[ω].

The three win axes correspond exactly to the three unit directions:
- u1 = 1       (q-axis, direction (1,0))
- u2 = ω       (r-axis, direction (0,1))
- u3 = ω² = −1−ω  (diagonal, direction (1,−1))

A win is an **arithmetic progression of length 6** in Z[ω] with a unit step —
i.e., a set {z, z+u, z+2u, z+3u, z+4u, z+5u} for some z ∈ Z[ω] and u ∈ {u1,u2,u3}.

### Symmetry Group

The lattice Z[ω] has symmetry group **D6** (order 12): 6 rotations at 60°
steps, 6 reflections. In axial coordinates, all 12 transforms are linear
(integer 2×2 matrix multiply on (q,r)). This means:

1. **HexConv2d** — the geometrically correct spatial kernel is the 7-cell
   Z[ω] neighbourhood (standard 3×3 with 2 non-hex corners masked).
2. **D6 augmentation** — every training position can be augmented 12×
   for free, all by linear transforms on (q,r) with no interpolation.

### Connection to Combinatorics

- **Van der Waerden W(6;2) = 1132**: any 2-coloring of {1…1132} contains a
  monochromatic AP-6. The grid version (Hales-Jewett) implies similar bounds.
- **Erdős-Selfridge potential**: ∑ 2^(−|L|) < 1 over all incomplete lines L
  guarantees a draw strategy for the second player. The EisensteinGreedyAgent
  approximates this potential on the Z[ω] lattice as a curriculum adversary.
- **Game value**: HexGo is likely a first-player win (as in all Connect-k
  games with k ≤ board diameter), but the exact proof is open for infinite hex.

### Sutton's Bitter Lesson — the Right Resolution

The concern: does encoding Z[ω] structure into the architecture violate the
"general methods + compute" principle?

Resolution: **geometry (structural substrate) ≠ game knowledge**.

- HexConv2d and D6 augmentation fix the *coordinate system* — they ensure the
  net operates in the correct lattice, not that it is told how to play.
- The net must still discover that six-in-a-row on any axis wins, that
  blocking matters, that forks are dangerous — none of this is encoded.
- AlphaGo Zero similarly used a board-aware architecture (19×19 conv stack,
  not a flat MLP); the geometry was fixed, the strategy was learned.

The `save_heatmap` scientific instrument tests this: policy mass should
spontaneously concentrate on the three Z[ω] win axes as training progresses,
without any explicit encoding of that structure.

---

## Game: Hexagonal Connect6

Infinite hexagonal grid, axial (q, r) coordinates, flat-top orientation.
Six-in-a-row wins along any of the three hex axes (Z[ω] unit directions).
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
| Total params    | ~113K        | Parameter-golfed policy head; HexConv2d masked kernels |

### Board Encoding
`encode_board(game)` returns `float32 [11, 18, 18]` centered on the centroid
of all pieces:
- Channels 0–3: P1 piece positions for t, t−1, t−2, t−3 (history planes)
- Channels 4–7: P2 piece positions for t, t−1, t−2, t−3 (history planes)
- Channel 8:    player 1 current pieces (same as ch 0, for compatibility)
- Channel 9:    player 2 current pieces (same as ch 4, for compatibility)
- Channel 10:   to-move plane (0.0 = P1 to move, 1.0 = P2 to move)

`N_HISTORY=4`, `IN_CH=11`. Returns `(arr, (oq, or_))`.

### HexConv2d
`HexConv2d(nn.Conv2d)` registers a `hex_mask` buffer that zeros weight
positions `[*, *, 0, 0]` and `[*, *, 2, 2]` — the two non-adjacent corners of
a 3×3 kernel that have no corresponding Z[ω] neighbour. Applied in all
ResBlocks. The stem uses standard `Conv2d`.

The 7 non-zero kernel positions match the Z[ω] 7-cell neighbourhood exactly:
```
  . X X
  X X X
  X X .
```
(top-left and bottom-right corners are masked)

### D6 Augmentation
`D6_MATRICES` — 12×2×2 int32 numpy array of all D6 linear transforms on
axial (q,r) coordinates. `d6_augment_sample(sample, tf_idx)` transforms both
the board array and move coordinates consistently. Applied at `train_batch`
time with `tf_idx = random.randrange(12)`.

### Action Representation
`encode_move(q, r, oq, or_)` returns a `[1, 18, 18]` one-hot plane, or
`None` if the move is outside the 18×18 window. Callers must handle `None`.

Policy head: `(trunk_features, move_plane) → scalar logit`.
Training: **Cross-Entropy** over all legal moves in a position.

### Architecture
```
Input [11,18,18]
  → Conv2d(11→32, 3×3) + BN + ReLU               [stem, standard conv]
  → 2× ResBlock(32ch, HexConv2d)                  [trunk, Z[omega] kernels]
  +→ Conv2d(32→1, 1×1) + BN + ReLU → FC → Tanh   [value head → scalar]
  +→ HexConv2d(32→4, 1×1) + BN + ReLU →
       cat(move_plane) → Linear(4*S*S+S*S, 32)    [policy → scalar]
```

### Checkpoint compatibility
Breaking architecture changes (IN_CH 3→11, policy head size) are handled by
`load_latest()`: on `RuntimeError` it moves all `net_*.pt` to
`checkpoints/legacy/` and starts fresh. Always use `checkpoints/net_latest.pt`.

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
1. **Self-play:** 8 parallel workers generate games using the current net
   via `InferenceServer`. Every 5th game uses `EisensteinGreedyAgent` as P2
   (adversarial curriculum).
2. **Buffering:** Positions stored in a 50K-position FIFO `deque`.
3. **D6 augmentation:** Each training sample augmented with a random D6
   transform at batch time (`d6_augment_sample(item, random.randrange(12))`).
4. **Training:**
   - `torch.amp.GradScaler` for FP16 numerical stability.
   - Value loss: MSE against TD-lambda target `z_t = 0.99^(T−t) × z_final`.
   - Policy loss: Cross-entropy over all legal moves (MCTS visit distribution).
5. **ELO Evaluation:** New net vs `mcts_50` and `EisensteinGreedyAgent(defensive=True)`.
6. **Heatmap:** `save_heatmap(server, gen)` saves `heatmaps/gen_XXXX.png` —
   policy distribution over a fixed 10-move test position. Confirms the net
   discovers Z[ω] axis structure without being told.
7. **Checkpoint:** `net_genXXXX.pt` + `net_latest.pt` after each generation.

### mcts_policy (tree reuse)
Returns `(chosen_move, visit_distribution, legal_moves, new_root)`. The caller
passes `prev_root` on the next call; if it has children matching the game state,
the subtree is recycled with fresh Dirichlet noise. Saves ~N_SIMS/branching_factor
simulations per move.

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

- `ELO` class: tracks ratings per agent name, updates via standard ELO formula (K=32).
- `NetAgent(net, sims)`: uses `mcts_with_net` / `mcts_policy` via `InferenceServer`.
- `MCTSAgent(sims)`: pure rollout `mcts()` baseline.
- `RandomAgent`: uniform random legal move.
- `EisensteinGreedyAgent`: scores each candidate by max chain length along any
  Z[ω] axis it would create (or block). `defensive=True` variant considers both
  own extension and blocking the opponent. Zero parameters, zero learned weights —
  pure Z[ω] arithmetic. Used as curriculum adversary and permanent ELO baseline.
- `run_match(a, b, n_games)`: N games alternating colors, returns win counts.

---

## File Map

```
hexgo/
  game.py       Engine — 1-2-2 turn logic, incremental candidates, make/unmake
  mcts.py       MCTS — player-aware backprop, rollout + net modes
  net.py        HexNet — HexConv2d/D6/ResNet 18x18/32ch/11ch/2blk + FP16 AMP
  inference.py  InferenceServer — dynamic batching + transposition cache
  train.py      Training loop — D6 aug, Eisenstein curriculum, heatmap, ELO
  elo.py        ELO rating — NetAgent, MCTSAgent, EisensteinGreedyAgent
  app.py        GUI monitor — pause/resume, win counter, log stream
  replay.py     Terminal replay — 1-2-2 aware, colored last-move bracket
  test_game.py  Unit tests — all 26 pass (game/encoding/D6/agent)
  render.py     Hex board renderer (shared by app + replay)
  checkpoints/  net_gen*.pt, net_latest.pt
  checkpoints/legacy/  incompatible checkpoints (auto-quarantined)
  heatmaps/     gen_XXXX.png — policy heatmap per generation
  replays/      game_first_genXXXX_*.json, game_last_genXXXX_*.json
  docs/         DESIGN.md (this file), ROADMAP.md, NOTES.md
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
