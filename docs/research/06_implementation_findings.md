# Implementation Findings — HexGo Phase 1

*What we actually built vs. what the research recommended. Updated 2026-03-30.*

---

## 1. What Research Predicted vs. What We Implemented

### 1.1 Architecture (Doc 01, 03)

| Research recommendation | What we built | Notes |
|------------------------|--------------|-------|
| 8–15 residual blocks, 128–256 channels | 2 blocks, 32 channels (~121K params) | Intentional: RTX 2060 constraint; Phase 1 validation only |
| 12 input channels (inc. tactical hint planes) | 11 channels (4 P1-hist + 4 P2-hist + 2 current + to-move) | No threat planes yet; reasonable Phase 1 |
| Auxiliary heads (ownership, threat) | Not yet | Expected 20-50% convergence speedup; deferred to 3b-vii |
| Softmax policy over H×W positions | `(board, move_plane) → scalar` per-move logit | More memory efficient; avoids fixed output size |
| Standard 3×3 conv in backbone | HexConv2d (masked corners) in ResBlocks | Better than recommendation: geometry-faithful |
| D6 augmentation (12-fold) | Implemented; applied at train_batch time | ✓ matches recommendation |
| Global pooling branch (KataGo) | GlobalPoolBranch after trunk | ✓ matches recommendation |
| Separate policy heads for 1st/2nd sub-move | Single head with `to-move` channel | Simplification; adequate for Phase 1 |
| 19×19 window | 18×18 window | Near-identical; centroid-centered |

### 1.2 MCTS (Doc 04)

| Research recommendation | What we built | Notes |
|------------------------|--------------|-------|
| Sequential sub-move interleaving | ✓ via 1-2-2 turn logic in game engine | Matches recommendation |
| ZoI margin=3 minimum | margin=6 (configurable) | Conservative; research says 3 is minimum for correctness |
| Lazy Zobrist hashing for transpositions | Frozenset board key (not Zobrist) | Weaker: hash(bytes) used for dedup, not Zobrist |
| Virtual loss for parallel MCTS | Not implemented | Single-threaded tree per worker |
| Gumbel search (sequential halving) | Not implemented | Standard PUCT only; Gumbel is significant gain |
| 400-800 sims/move | 25-50 sims (75% reduced) | Severely under-budgeted; GIL limits parallelism |
| PUCT cPUCT=2.5 (Phase 1) | CPUCT=1.0 | Too low; under-exploring |
| τ=1.0 for first 20 half-moves | Cosine annealing from move 0 | Different design; floor reached at TEMP_HORIZON/2 |

### 1.3 Training (Doc 05)

| Research recommendation | What we built | Notes |
|------------------------|--------------|-------|
| SGD + momentum 0.9 | Adam (LR=1e-3) | Common difference; Adam faster to converge |
| LR cosine schedule | Constant LR (no schedule yet) | Should add; converge faster |
| Playout cap randomization (KataGo) | ✓ 25%/75% split | Matches recommendation |
| Recency-weighted replay (75/25) | FIFO deque (uniform) | Recommendation not yet implemented (3b-viii) |
| Curriculum (4-in-a-row → 6-in-a-row) | Skipped; direct 6-in-a-row | Research said curriculum benefit unclear; reasonable skip |
| 55% win threshold for checkpoint promotion | ✓ implemented | Matches recommendation |
| Reanalysis (MuZero-style) | Not implemented | 3b-ix; deferred |

---

## 2. New Findings Not in Research Docs

### 2.1 Eisenstein Integer Curriculum Opponent

The `EisensteinGreedyAgent` based on the Z[ω] unit direction axes proved to be
a strong and natural curriculum opponent — not just a random baseline. It approximates
the Erdős-Selfridge potential function on the lattice. Running 1-in-5 self-play games
against it as adversary provides structured training signal that pure self-play
(random initialization) lacks in early generations.

This wasn't in the research docs; it emerged from the mathematical framing.

### 2.2 1-2-2 Turn Backpropagation

Standard AlphaZero MCTS negates value at every node boundary. The 1-2-2 rule
(same player makes two consecutive placements) requires negating only at player
transitions: `negate iff node.parent.player != node.player`.

The research docs (Doc 04) recommended this approach ("negate at player transitions,
not within a player's sub-moves") and it is implemented correctly in the pure-rollout
path. However it is implemented **incorrectly** in `mcts_with_net` (leaf children
default to `player=1`), which affects NetAgent ELO.

### 2.3 Persistent Cross-Gen Cache Tradeoffs

The 5-generation persistent cache achieves ~20-40% hit rate on opening positions
but introduces a correctness tradeoff: positions evaluated by a weaker 5-gen-old
net are served as authoritative evaluations. For rapid early training this hurts more
than it helps. Recommendation: reduce `CACHE_MAX_AGE` to 2 after generation 5.

Also found: the cache key `frozenset(board.items())` collides on mid-turn states
under the 1-2-2 rule — same pieces, different `current_player`. The key must
include turn state.

### 2.4 Playout Cap SIMS_MIN Too High

The KataGo playout cap design requires a genuinely cheap reduced budget (`SIMS//8`)
to create bimodal exploration: expensive games for quality targets, cheap games for
positional diversity. With `SIMS_MIN=25` and `SIMS=50`, the "reduced" games cost
half the full budget — no meaningful diversity. Should be set to `max(6, SIMS//8)`.

### 2.5 CUDA Graphs on Windows/PyTorch: Silent Failure

CUDA Graph capture with variable reassignment (Python name rebinding) inside
`with torch.cuda.graph(g):` silently fails — the graph captures intermediate
allocations, not the output variables. The fallback to eager mode is silent.
On Windows, `torch.compile` also silently fails via Triton. Neither speedup
from the roadmap items 1b or 3c is currently active.

### 2.6 D6 Augmentation on 18×18 Window

For positions near the window edge, D6 rotations map significant portions of
the board outside the 18×18 clip. The effective augmentation ratio for edge positions
is substantially less than 12×. The policy probability mass over clipped moves is
dropped and renormalized — the training signal for those augmented samples is biased
toward in-window moves. This is expected and acceptable but means the 12× efficiency
claim is an upper bound, not a guaranteed ratio.

---

## 3. Open Questions for Phase 2

1. **Optimal CPUCT for this game**: Research recommends 2.5 for Phase 1. Current 1.0
   produces under-exploration. Does higher CPUCT help with the 3-axis threat structure
   of hex (more diverse threat directions → more exploration needed)?

2. **Value signal bootstrap**: With SIMS=50 and ZOI margin=6, the early net produces
   near-random policy. Does the value signal bootstrap correctly from random play, or
   does the Eisenstein curriculum help more than expected?

3. **Heatmap axis concentration**: At what generation does the policy heatmap begin
   showing concentration along the three Z[ω] axes? This is the key scientific
   measurement — confirming emergent structure without encoding it.

4. **ZOI margin empirics**: Does margin=6 vs margin=3 vs margin=5 make a measurable
   difference in game quality? The theoretical minimum (margin=5 for 6-in-a-row)
   vs the conservative current setting (6) could be benchmarked.

5. **LR schedule**: Constant LR=1e-3 may be too high after initial convergence.
   A cosine decay or step decay after gen 20 may improve stability.

---

## 4. Calibrated Status vs. Research Phase Expectations

From Doc 05, Phase 1 milestones:

| Week | Research Milestone | Current Status |
|------|--------------------|---------------|
| 1 | Game engine, MCTS skeleton, NN forward pass | ✅ Done |
| 2 | Self-play generates games, loss decreases, beats random | ✅ Done |
| 3 | Beats greedy heuristic >80%, 500-800 ELO | ❓ Unknown — autotune signal inverted; ELO unreliable until bugs fixed |

The infrastructure is Phase 1-complete. The correctness bugs mean the training
quality is unknown. After the 6 priority fixes in ASSESSMENT.md, a clean 50-gen
run will establish the actual Phase 1 benchmark position.

---

## 5. References Not in Original Research Docs

- Mordvintsev, A., et al. (2020). Growing neural cellular automata. *Distill*. — CA weight init for HexConv2d kernels (Phase 4b)
- Wu, I.-C., & Huang, D.-Y. (2006). A new family of k-in-a-row games. *ICGA Journal*, 29(1), 26-34. — Connect6 baseline; 1-2-2 rule design rationale
- Karpathy, A. (2024). autoresearch. *GitHub* (jsegov fork for Windows RTX). — Autotune design: agent as RL actor, scalar reward signal, 5-gen trials
- Erdős, P., & Selfridge, J. L. (1973). On a combinatorial game. *Journal of Combinatorial Theory, Series A*, 14(3), 298-301. — Theoretical basis for EisensteinGreedyAgent potential function
