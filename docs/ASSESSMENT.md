# HexGo — Honest Project Assessment

*Code review conducted 2026-03-29/30 by 5 independent subagents across all source files.*

---

## Summary Verdict

The architecture and mathematical foundation are **genuinely impressive** for a solo project.
The Z[ω] isomorphism, HexConv2d, D6 augmentation, GlobalPoolBranch, and the overall AlphaZero
pipeline are implemented correctly in their essentials and show real depth of understanding.

However, the project has **multiple confirmed correctness bugs** that mean training is currently
not working as intended. The most damaging ones are:

1. **Autotune is anti-optimizing** — the reward signal is inverted, keeping bad configs and reverting good ones.
2. **CUDA Graphs are broken** — silently returning zeros; the expected 30–50% speedup isn't happening.
3. **Cache collides on turn state** — mid-turn positions served stale values from different-player states.
4. **NetAgent backprop corrupted** — `mcts_with_net` leaf children use `player=1` default, corrupting ~50% of P2 leaf expansions.

None of these are design failures — they're all fixable with small code changes. But they compound:
you can't trust the autotune results so far, the net evaluations are partially wrong, and a claimed
infrastructure optimization (CUDA Graphs) is silently disabled.

---

## What Works Well

### Mathematical Foundation
The Z[ω] Eisenstein integer framing is correct and well-chosen. It isn't just aesthetic — it
directly enables three concrete engineering wins: `HexConv2d` (the right kernel for hex grids),
D6 augmentation (12× free data efficiency), and `EisensteinGreedyAgent` (a principled curriculum
opponent grounded in Erdős-Selfridge potential theory). This is better-motivated than most hobby
game-playing projects.

### Game Engine (`game.py`)
Solid. Make/unmake with full undo stack is correctly implemented including `winner`, `current_player`,
and `placements_in_turn`. The incremental `candidates` set gives O(1) legal moves. Win detection
walks only the 3 Z[ω] axes through the last piece — O(WIN_LENGTH). The 1-2-2 turn logic is correct.
All 26 unit tests pass, covering win detection on all three axes, undo correctness, D6 symmetry,
and the EisensteinGreedyAgent.

### Neural Network Architecture (`net.py`)
HexConv2d correctly masks the two non-adjacent corners `[0,0]` and `[2,2]` of 3×3 kernels.
D6_MATRICES implements the correct dihedral group D6 for Z[ω] (determinant-1 rotations, det=-1
reflections, group closed under composition). GlobalPoolBranch shapes are correct. The 11-channel
input encoding (4 P1-history + 4 P2-history + 2 current + to-move) matches AlphaZero's temporal
encoding design. At ~121K params on an RTX 2060, the capacity/speed tradeoff is appropriate for
Phase 1.

### MCTS Backpropagation Convention
The multi-placement backprop rule (`negate only when node.parent.player != node.player`) correctly
handles the 1-2-2 turn structure. This is non-trivial to get right and is implemented correctly
in the pure-rollout path.

### Infrastructure
The `InferenceServer` batching design is correct in principle. Persistent cross-gen cache,
`pin_memory` async transfers, overlapped training, and `PerfTracker` with bottleneck warnings
are all production-quality additions. `load_latest()` quarantine for incompatible checkpoints
prevents silent data loss.

### EisensteinGreedyAgent
The `_chain_if_placed` implementation is correct and efficient — no board mutations, no off-by-one
in the bidirectional axis walk. The `defensive=True` variant correctly takes `max(own, block)`.
This is a genuinely useful curriculum opponent: it plays structured, principled moves without
requiring any learned weights.

---

## What's Broken or Misleading

### Critical Bugs (confirmed, must fix before trusting results)

**Autotune reward inverted** (`tune.py:57,82,101`)
Every autotune trial since the system was built has been optimizing in the wrong direction.
`eisenstein_def`'s ELO goes UP when the net gets worse. `kept = elo_delta >= 0` keeps configs
that hurt the net. If any tune_log.jsonl entries exist, they should be treated as invalid.
Fix: track net agent ELO delta or use `avg_eis_winrate` with inverted threshold.

**CUDA Graph rebinding** (`inference.py:133-136`)
The `self._graph_X = ...` assignments inside `with torch.cuda.graph(g):` rebind Python
names rather than writing into pre-allocated buffers. After `g.replay()`, reading
`self._graph_val` returns zeros. The exception handler silently falls back to eager mode
so no errors surface. The roadmap item "CUDA Graphs ✅ done" should be marked incomplete.
Additionally, `_graph_val` is pre-allocated as `[B,1]` but `value()` returns `[B]`, making
the `[val_idxs, 0]` indexing in the graph path an IndexError. Both bugs prevent the graph
path from ever running correctly.

**Cache key ignores turn state** (`inference.py:151`)
`frozenset(board.items())` is identical for two positions with the same pieces but
different `current_player` or `placements_in_turn`. Under the 1-2-2 rule, after P1's
first of two stones the board is the same as after P2's second stone if stone counts match —
but the `to-move` channel (ch 10) differs, so the net returns different values.
The wrong cached value gets served silently.

**`mcts_with_net` leaf children `player=1`** (`mcts.py:191`)
All leaf node children in `mcts_with_net` use the default `player=1`. Compare to `_expand`
which correctly passes `player=game.current_player`. This corrupts the parent-child
player comparison in `_backprop`, causing value sign errors for ~50% of P2 leaf expansions.
Every `NetAgent` ELO rating is tainted by this. The pure-rollout `mcts()` path (used by
`MCTSAgent`) is not affected. One-line fix at `mcts.py:191`.

### Important Bugs (affect results, should fix soon)

**SIMS_MIN defeats playout cap randomization** (`config.py:13`)
`SIMS_MIN=25` with `SIMS=50` gives `max(25, 50//8)=max(25,6)=25`. The "reduced budget"
games use half the full budget — no meaningful diversity gain. The KataGo design requires
`SIMS_MIN ≈ SIMS // 8` (e.g., 6 for SIMS=50) for the cheap/expensive bimodal signal.

**Cosine temp semantics** (`train.py:282`)
`cos(π × move / T)` goes negative at `move > T/2` and clamps to 0.05, so exploration
collapses at move `T/2 = 20` when `TEMP_HORIZON=40`. The parameter name implies the
floor at move 40. This halves the effective exploration window silently.

**History planes cross-reference board dict** (`net.py:168-169`)
`p1_hist = [m for m in move_history if game.board.get(m)==1]` — during MCTS tree traversal
after `unmake()`, pieces removed from `board` but still in `move_history` are silently
excluded. The history channels become incorrect during deep MCTS search. Fix: store player
alongside coordinates in `move_history`.

**ZOI lookback blind spot** (`game.py:108-141`)
The `lookback=8` window means early threats from pieces placed >8 moves ago fall outside
the ZOI computation. In games where a player builds 5-of-6 on an axis early then plays
elsewhere, the completing 6th move may be invisible to MCTS. This is game-correctness-affecting
(the net can be steered away from its own winning move).

**Terminal expansion sign** (`mcts.py:117-127`)
When expansion itself discovers a terminal (winning move is selected and played), `v=-1.0`
is backpropagated into the winning player's node rather than the losing player's. The sign
convention is correct on the selection path but wrong on the expansion path.

**Overlap training overfit** (`train.py:630-636`)
The overlap loop calls `train_batch` on every 50ms timeout regardless of new data. On
slow-self-play generations (which is most generations), this repeatedly trains the same
buffer content, potentially overfitting before new games add diversity.

### Misleading Documentation

- `inference.py` module docstring: "355K-param net" — actual net is ~121K. Stale from old architecture.
- `net.py:339`: `forward()` docstring says `[B, 3, S, S]` input; actual is `[B, 11, S, S]`.
- `net.py:331`: comment says `[B, 2*S*S]`; actual shape is `[B, 4*S*S]` (p_conv outputs 4 channels).
- `game.py:72`: comment says "len=0"; code correctly checks `len==1` (after `move_history.append`).
- `mcts.py:127`: dead code `v = 0.0 if game.winner is not None` immediately overwritten.
- `render.py/replay.py`: `abs(r)` indent formula visually wrong for negative-r positions.

---

## Training Signal Quality Assessment

Given the bugs above, what is the current training actually doing?

The **pure self-play path** (not mcts_with_net, not autotune) is largely intact:
- `mcts_policy` in `train.py` uses `_expand` which correctly sets `player=game.current_player`
- Value targets (TD-lambda), D6 augmentation, and replay buffer are correct
- Policy loss and value loss formulas are correct

The training loop does produce a learning signal. The ELO of `eisenstein_def` at ~1403 vs
new net gens at ~1200 suggests the net isn't yet beating the greedy opponent reliably —
but given the 10-game match size and K=32 rating volatility, the net may be more competitive
than the ratings suggest.

The **autotune history** (if any) should be discarded — it was running in reverse.

The **inference quality** is impacted by the cache key bug and by `mcts_with_net` player
default, but the training self-play path doesn't use `mcts_with_net`, so the core training
loop avoids the worst of these.

---

## Comparison to Research Targets

| Feature | Research Target | Current | Gap |
|---------|----------------|---------|-----|
| SIMS | 200-600 (Phase 1) | 50 (25% games), 25 (75%) | Large — but RTX 2060 constraint is real |
| CPUCT | 2.5 (Phase 1) | 1.0 | Under-exploring; try 2.0-2.5 |
| Board size | 18×18 | 18×18 | ✓ |
| Network depth | 8-15 blocks | 2 blocks | Underpowered for strong play; sufficient for Phase 1 validation |
| Dirichlet alpha | 0.08-0.10 (10/|ZoI|) | 0.3 | Too noisy at root |
| ZoI margin | 3 minimum | 6 | Conservative but correct |
| Auxiliary heads | KataGo ownership+threat | Not yet | 20-50% gain expected |
| MCTS | PUCT standard | PUCT (Python) | No Gumbel search, no C++ — acceptable for Phase 1 |

---

## Priority Fix List

Fix these in order to get a trustworthy training run:

1. **`tune.py`**: invert the kept/reverted logic — keep when `avg_eis_winrate` decreases
2. **`mcts.py:191`**: add `player=game.current_player` to leaf node creation in `mcts_with_net`
3. **`inference.py:133-136`**: fix CUDA Graph to use in-place ops (or disable the graph path entirely until proper implementation)
4. **`inference.py:151`**: add `current_player` and `placements_in_turn` to cache key
5. **`config.py:13`**: set `SIMS_MIN = max(6, SIMS // 8)` = 6 for SIMS=50
6. **`net.py:168-169`**: store player in `move_history` entries; rebuild history without board cross-reference

After these fixes, a 50-gen run should show meaningful ELO improvement vs `eisenstein_def`.
