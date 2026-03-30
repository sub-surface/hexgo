# HexGo — Honest Project Assessment

*Code review conducted 2026-03-29/30 by 5 independent subagents across all source files.*

---

## Summary Verdict

The architecture and mathematical foundation are **genuinely impressive** for a solo project.
The Z[ω] isomorphism, HexConv2d, D6 augmentation, GlobalPoolBranch, and the overall AlphaZero
pipeline are implemented correctly in their essentials and show real depth of understanding.

**As of 2026-03-30, all critical and important correctness bugs have been fixed.** The training
pipeline is now sound and ready for sustained runs. A FastAPI dashboard (server.py + dashboard.html)
provides live monitoring, training controls, and a replay viewer. The checkpoint tournament system
was removed to eliminate a crash source; evaluation is now Eisenstein-only (~5s/gen vs. ~177s/gen).

Previously identified critical bugs (now fixed):
1. ~~**Autotune anti-optimizing**~~ — reward signal inverted; fixed in `tune.py`.
2. ~~**CUDA Graphs broken**~~ — tensor rebinding bug; fixed with in-place `.copy_()`.
3. ~~**Cache collides on turn state**~~ — key now includes `current_player` + `placements_in_turn`.
4. ~~**NetAgent backprop corrupted**~~ — leaf children now use `player=game.current_player`.

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

## What's Fixed (as of 2026-03-30)

### Previously Critical Bugs — All Resolved

**Autotune reward inverted** → Fixed: `kept = elo_delta is None or elo_delta <= 0`.

**CUDA Graph rebinding** → Fixed: in-place `.copy_()` inside capture; `.detach()` before numpy; `_graph_val` shape `[B]` not `[B,1]`; removed `[val_idxs, 0]` indexing.

**Cache key ignores turn state** → Fixed: key is `(frozenset(board.items()), current_player, placements_in_turn)`.

**`mcts_with_net` leaf children `player=1`** → Fixed: `player=game.current_player` in leaf node creation.

**Terminal expansion sign** → Fixed: `v = 1.0 if game.winner == node.player else -1.0` (was always `1.0`).

**SIMS_MIN too high** → Fixed: `SIMS_MIN = 6` in `config.py`.

**Cosine temp semantics** → Fixed: `cos(π/2 × move / TEMP_HORIZON)` — now reaches floor at `TEMP_HORIZON` moves.

**Replay hex offset** → Fixed: `indent = r - r_min`.

**Overlap training overfit** → Fixed: capped by `batches_since_sync < WEIGHT_SYNC_BATCHES`.

**D6 augmentation probs corruption** → Fixed: `sample['probs'].copy()` in `d6_augment_sample`.

### Checkpoint Tournament Removed

`_tourney_promote()` deleted entirely. Loading old checkpoints into a `torch.compile`d
(`OptimizedModule`) wrapper caused `RuntimeError: Error(s) in loading state_dict`. The
tournament was also consuming 100–177s/gen. Replaced by Eisenstein-only eval (~5s/gen).

## What's Still Open

### Deferred — Lower Risk

**History planes cross-reference board dict** (`net.py`)
During deep MCTS after `unmake()`, pieces removed from `board` but still in `move_history`
cause incorrect history channel encoding. Fix: store `(q, r, player)` in `move_history`
instead of `(q, r)`. Training path is unaffected (no MCTS unmake during training data gen).

**ZOI lookback blind spot** (`game.py`)
`lookback=8` can miss early threats in long games. Conservative; increase to 16 or add
a separate threat-line set if ELO growth stalls at mid-game complexity.

### Misleading Documentation

- `inference.py` module docstring: "355K-param net" — actual net is ~121K. Stale from old architecture.
- `net.py:339`: `forward()` docstring says `[B, 3, S, S]` input; actual is `[B, 11, S, S]`.
- `net.py:331`: comment says `[B, 2*S*S]`; actual shape is `[B, 4*S*S]` (p_conv outputs 4 channels).
- `game.py:72`: comment says "len=0"; code correctly checks `len==1` (after `move_history.append`).
- `mcts.py:127`: dead code `v = 0.0 if game.winner is not None` immediately overwritten.
- `render.py/replay.py`: `abs(r)` indent formula visually wrong for negative-r positions.

---

## Training Signal Quality Assessment

With all bugs fixed, the training pipeline is now sound end-to-end:

- `mcts_policy` in `train.py` uses correct `player=game.current_player` at every node
- Terminal and leaf value signs are correct (`v = winner == node.player` + alignment flip)
- Tree reuse prunes stale children before recycling subtrees
- CUDA Graph path is functional; cache key includes turn state
- D6 augmentation no longer corrupts replay buffer entries
- Overlap training capped to prevent overfitting stale buffer
- MAX_MOVES=300 prevents runaway games that could starve the buffer

Any **tune_log.jsonl** entries from before 2026-03-30 should be discarded — they were
produced with the inverted reward signal and are not meaningful.

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

## Next Steps

All critical fixes are done. To continue improving:

1. **Run 50+ gen baseline** — verify ELO vs `eisenstein_def` improves monotonically with all fixes in.
2. **Fix history planes** (`net.py`) — store `(q, r, player)` in `move_history` to correct history channels during deep MCTS.
3. **Add auxiliary heads** (3b-vii) — ownership + threat prediction for 20–50% convergence speedup.
4. **Tune CPUCT** — current value `1.0` is lower than research target `2.5`; test `2.0` first.
5. **Reduce DIRICHLET_ALPHA** — current `0.3`; research suggests `10/|ZoI| ≈ 0.08–0.10` for stronger play.

A 50-gen run with the fixed pipeline should show meaningful ELO improvement vs `eisenstein_def`.
