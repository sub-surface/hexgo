# HexGo — Honest Project Assessment

*Code review conducted 2026-03-29/30 by 5 independent subagents across all source files.*

---

## Summary Verdict

The architecture and mathematical foundation are **genuinely impressive** for a solo project.
The Z[ω] isomorphism, HexConv2d, D6 augmentation, GlobalPoolBranch, and the overall AlphaZero
pipeline are implemented correctly in their essentials and show real depth of understanding.

**As of 2026-03-30, all critical and important correctness bugs have been fixed, and the training
pipeline has been further improved with hyperparameter tuning, recency-weighted sampling, and
auxiliary heads.** A FastAPI dashboard (server.py + dashboard.html) provides live monitoring,
training controls, and a replay viewer. The checkpoint tournament system was removed to eliminate
a crash source; evaluation is now Eisenstein-only (~5s/gen vs. ~177s/gen).

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

**ZOI lookback blind spot** (`game.py`)
`lookback=8` can miss early threats in long games. Conservative; increase to 16 or add
a separate threat-line set if ELO growth stalls at mid-game complexity.

### Fixed in session 2 (2026-03-30)

- ~~History planes cross-reference board dict~~ — verified correct: `player_history` already used.
- ~~`net.py:339` comment `[B, 2*S*S]`~~ — fixed to `[B, 4*S*S]`.
- ~~`net.py` forward docstring `[B, 3, S, S]`~~ — fixed to `[B, 11, S, S]`.
- ~~`inference.py` comment `[3, S, S]`~~ — fixed to `[11, S, S]`.

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
| SIMS | 200-600 (Phase 1) | 50 (25% games), 6–25 (75%) | Large — RTX 2060 constraint |
| CPUCT | 2.5 (Phase 1) | **2.0** ✓ | Minor; raise to 2.5 if ELO stalls |
| Board size | 18×18 | 18×18 | ✓ |
| Network depth | 8-15 blocks | 2 blocks | Sufficient for Phase 1 validation |
| Dirichlet alpha | 0.08-0.10 (10/\|ZoI\|) | **0.09** ✓ | ✓ |
| ZoI margin | 3 minimum | 6 | Conservative but correct |
| Auxiliary heads | KataGo ownership+threat | **Done** ✓ | Thin 1×1 conv heads, AUX_LOSS=0.1 each |
| Replay sampling | Recent-biased | **75/25** ✓ | ✓ |
| MCTS | PUCT standard | PUCT (Python) | No Gumbel search, no C++ — Phase 1 acceptable |

---

## Next Steps

All improvements for Phase 1 are complete. Ready to run sustained baseline:

1. **Run 50+ gen baseline** — verify ELO vs `eisenstein_def` improves monotonically.
2. **ZOI lookback** — increase from 8 to 16 if ELO stalls at mid-game complexity.
3. **Scale trunk** (5a) — 4 blocks / 64 channels (~480K params) only after ELO plateau.

A 50-gen run with the improved pipeline should show clear ELO progress vs `eisenstein_def`.
