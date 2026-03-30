# HexGo — Roadmap

Z[ω] self-play ladder: arithmetic progressions, Eisenstein symmetry, emergent structure.
Current baseline: ~121K params, 8 workers, RTX 2060, ~30–60s/gen.

---

## Mathematical Foundation (established)

HexGo = **AP-6 Maker-Maker on Z[ω]**.

The hex grid with axial (q, r) coordinates is isomorphic to the Eisenstein
integer ring Z[ω] where ω = e^(2πi/3). The three win axes correspond to the
unit directions {1, ω, ω²}. A win is exactly an arithmetic progression of
length 6 in Z[ω] with a unit step.

Practical consequences:
1. **The right convolution kernel is the 7-cell Z[ω] neighbourhood** — standard 3×3 with 2 non-adjacent corners masked. `HexConv2d` enforces this.
2. **The symmetry group of the lattice is D6 (order 12)** — every training position can be augmented 12× for free via linear transforms on axial (q, r) coordinates.

Connection to combinatorics: W(6;2) = 1132 (van der Waerden); the Erdős-Selfridge potential ∑ 2^(−|L|) < 1 gives a theoretically safe second-player draw strategy; `EisensteinGreedyAgent` approximates this.

---

## Phase 0 — Stability

- [x] 1-2-2 Connect6 rule in `game.py`
- [x] Player-aware MCTS backprop in `mcts.py`
- [x] FP16 AMP in training and inference
- [x] Cross-entropy policy loss over all legal moves
- [x] Divide-by-zero guard in visit distribution normalization
- [x] Transposition cache in `InferenceServer`
- [x] `checkpoints/legacy/` quarantine for incompatible weights
- [x] 26 unit tests pass

---

## Phase 1 — Geometry-Faithful Architecture

- [x] **1a. Inference batching** — `NUM_WORKERS=8`, `INF_BATCH=8`, `INF_TIMEOUT=30ms`
- [x] **1b. `torch.compile`** — applied at `InferenceServer.start()` (silently falls back on Windows)
- [x] **1c. Tree reuse** — `mcts_policy()` returns `new_root`; subtree recycled with fresh Dirichlet
- [x] **1d. Cosine temperature annealing** — `temp = max(0.05, cos(π × move / T))`
- [x] **1e. TD-lambda value targets** — `z_t = 0.99^(T−t) × z_final`
- [x] **1f. History planes** — 11-channel input (4 P1-history + 4 P2-history + current × 2 + to-move)
- [x] **1g. Parameter golf policy head** — `p_conv` 4ch; `p_fc` 32 hidden; ~121K total params
- [x] **1h. INT8 quantization utility** — `quantize_for_inference(net)` via `torch.ao.quantization`
- [x] **1i. HexConv2d** — masks non-hex corners `[0,0]` and `[2,2]` in all ResBlock kernels
- [x] **1j. D6 data augmentation** — 12 transforms applied at `train_batch` time
- [x] **1k. EisensteinGreedyAgent curriculum** — 1-in-5 self-play games; permanent ELO anchor
- [x] **1l. Policy heatmap** — `heatmaps/gen_XXXX.png` per generation

---

## Phase 2 — Training Quality

- [x] **2a. Zobrist-keyed buffer deduplication** — per-gen hash set prevents near-duplicate positions
- [x] **2b. Self-play curriculum** — playout cap randomization (25% full / 75% reduced)
- [x] **2c. Checkpoint tournament** — new net must beat ≥55% of top-K pool to be promoted

---

## Phase 3 — Infrastructure

- [x] **3a. Overlapped async self-play** — training runs concurrently with self-play via `concurrent.futures`
- [x] **3b. Recency / diversity sampling** — buffer FIFO with cap; Zobrist dedup
- [x] **3c. CUDA Graphs hot path** — code present; currently broken (tensor rebinding bug; falls back to eager)

---

## Phase 3b — Collaborator Integrations

- [x] **3b-i. KataGo playout cap randomization** — `_cap_sims(target)` in `train.py`
- [x] **3b-ii. Global pooling branch (KataGo)** — `GlobalPoolBranch(32ch)` after trunk
- [x] **3b-iii. ZOI pruning** — `zoi_moves(margin=6)` restricts MCTS to active area
- [x] **3b-iv. Latency / perf tracking** — `PerfTracker` with bottleneck warnings
- [x] **3b-v. CPU offloading / pin_memory** — `non_blocking=True` async host→GPU transfers
- [x] **3b-vi. Persistent cross-gen cache** — `_persistent_cache` with `CACHE_MAX_AGE=5` eviction
- [ ] **3b-vii. Ownership + threat auxiliary heads** — 20-50% convergence speedup (4h effort; needs new CFG keys `AUX_LOSS_OWN`, `AUX_LOSS_THREAT`)
- [ ] **3b-viii. Recency-weighted replay buffer** — 75% recent / 25% uniform (1h effort; add `RECENCY_WEIGHT` to CFG)
- [ ] **3b-ix. MuZero-style reanalysis** — re-search buffered positions with updated net (4h effort)

---

## Bug Fixes Required (pre-training correctness)

These are confirmed correctness bugs found in code review. They should be fixed
before running long training runs — they directly corrupt learning.

- [ ] **FIX-1: CRITICAL — CUDA Graph tensor rebinding** — `inference.py:133-136`: assignments inside `with torch.cuda.graph(g):` rebind Python names instead of writing in-place; graph replay returns stale zeros. Fix: use `self._graph_feat.copy_(...)` or pre-allocate and use in-place ops.
- [ ] **FIX-2: CRITICAL — Cache key ignores turn state** — `inference.py:151`: `frozenset(board.items())` key doesn't include `current_player` or `placements_in_turn`; mid-turn positions collide with different-player positions. Fix: add both to key.
- [ ] **FIX-3: CRITICAL — Autotune reward signal inverted** — `tune.py:57,82,101`: tracks `eisenstein_def` ELO (goes up when net worsens); `kept = delta >= 0` keeps bad configs. Fix: track net agent ELO or use `avg_eis_winrate` with inverted threshold.
- [ ] **FIX-4: IMPORTANT — `mcts_with_net` leaf children missing `player=`** — `mcts.py:191`: leaf children created with default `player=1`; corrupts backprop sign for ~50% of P2 leaf expansions. Fix: add `player=game.current_player`.
- [ ] **FIX-5: IMPORTANT — Terminal expansion sign** — `mcts.py:117-127`: when a winning move is made during expansion, `v=-1.0` is attributed to the winning player's node. Fix: verify sign convention; `v` at a terminal child should be from the next player's perspective.
- [ ] **FIX-6: IMPORTANT — History planes filter via board dict** — `net.py:168-169`: `p1_hist/p2_hist` built by cross-referencing `game.board`; stale during MCTS unmake paths. Fix: store `(q, r, player)` triples in `move_history`.
- [ ] **FIX-7: IMPORTANT — ZOI long-range threat blindness** — `game.py:108-141`: `lookback=8` window can miss early threats in games >50 moves/player; completing 6th piece may be >6 hex-steps from any of the last 8 pieces. Consider increasing lookback or adding a separate threat-line set.
- [ ] **FIX-8: IMPORTANT — Autotune `SIMS_MIN` too high** — `config.py:13`: `SIMS_MIN=25` with `SIMS=50` means reduced games use 50% of full budget; defeats playout cap diversity. Fix: set `SIMS_MIN` to `max(6, SIMS // 8)`.
- [ ] **FIX-9: IMPORTANT — Cosine temp semantics** — `train.py:282`: `cos(π × move / TEMP_HORIZON)` reaches floor at `TEMP_HORIZON/2` moves, not `TEMP_HORIZON`. Rename to `TEMP_HALF_LIFE` or fix formula.
- [ ] **FIX-10: MODERATE — Replay hex offset** — `replay.py:27`: `abs(r)` indent wrong for negative r; fix to `r - r_min` offset.

---

## Phase 4 — Equivariance (research)

- [ ] **4a. G-CNN (full D6 equivariance)** — replace augmentation with group-equivariant layers. Cohen & Welling (2016); Bekkers (2020). Principled Sutton-compatible treatment: geometry as structural substrate, not game knowledge.
- [ ] **4b. CA weight initialization** — initialize `HexConv2d` kernels from cellular automata patterns. The hex-7 neighbourhood IS the standard NCA update kernel. Mordvintsev et al. (2020). Requires `WEIGHT_INIT: "xavier"|"ca"` in CFG and `init_weights_ca(net)` in `net.py`.

---

## Phase 5 — Model Scale

- [ ] **5a. Scale trunk 4blk/64ch** — ~480K params. Only after Phase 3 infrastructure (async self-play essential for this to be efficient). 2h implementation effort.
- [ ] **5b. Activation sparsity / early exit** — profile `v_fc` first layer; if >60% sparsity implement early exit from value head. Low priority until scale experiments run.
- [ ] **5c. C++/Rust MCTS** — Python GIL serializes tree traversal and prevents true batching. Required for `avg_batch_size > 2.0`. Phase 2+ feature.

---

## Priority Order (RTX 2060, solo researcher)

| Priority | Item | Status | Expected gain |
|----------|------|--------|---------------|
| 🔴 1 | Fix autotune reward signal (FIX-3) | **todo** | Autotune currently anti-optimizes |
| 🔴 2 | Fix mcts_with_net player= (FIX-4) | **todo** | Corrupts NetAgent ELO |
| 🔴 3 | Fix CUDA graph rebinding (FIX-1) | **todo** | 30-50% inference speedup currently broken |
| 🔴 4 | Fix cache key turn state (FIX-2) | **todo** | Stale cache for mid-turn positions |
| 🟡 5 | Fix SIMS_MIN (FIX-8) | **todo** | Real playout cap diversity |
| 🟡 6 | Fix terminal expansion sign (FIX-5) | **todo** | Correct backprop at terminal nodes |
| 🟡 7 | Fix history planes (FIX-6) | **todo** | Correct input during MCTS |
| 🟡 8 | Aux heads (3b-vii) | **todo** | 20-50% convergence speedup |
| 🟡 9 | Recency replay (3b-viii) | **todo** | Better policy tracking |
| ✅ — | All Phase 0–3b-vi items | done | See sections above |
| ⬜ 10 | CA weight init (4b) | future | Z[ω]-aligned priors |
| ⬜ 11 | MuZero reanalysis (3b-ix) | future | Freshen stale targets |
| ⬜ 12 | G-CNN equivariance (4a) | future | Principled symmetry |
| ⬜ 13 | Scale trunk (5a) | future | Capacity |
| ⬜ 14 | C++/Rust MCTS (5c) | future | True inference batching |
