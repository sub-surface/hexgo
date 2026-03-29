# HexGo — Roadmap

The Z[ω] self-play ladder: arithmetic progressions, Eisenstein symmetry, and
emergent structure. Each phase is independent and additive.

Current baseline: ~113K params, 8 workers, RTX 2060, ~30–60s/gen.

---

## Mathematical Foundation (established)

HexGo = **AP-6 Maker-Maker on Z[ω]**.

The hex grid with axial (q, r) coordinates is isomorphic to the Eisenstein
integer ring Z[ω] where ω = e^(2πi/3). The three win axes correspond to the
unit directions {1, ω, ω²}. A win is exactly an arithmetic progression of
length 6 in Z[ω] with a unit step.

This framing has two practical consequences:

1. **The right convolution kernel is the 7-cell Z[ω] neighbourhood** — a
   standard 3×3 grid with 2 non-adjacent corners masked. `HexConv2d` enforces
   this directly.

2. **The symmetry group of the lattice is D6 (order 12)** — 6 rotations at 60°
   steps, 6 reflections. Every training position can be augmented 12× for free,
   all produced by linear transforms on axial (q, r) coordinates.

Connection to combinatorics: W(6;2) = 1132 (van der Waerden), meaning any
2-coloring of {1…1132} contains a monochromatic AP-6. Hales-Jewett implies the
same for the grid version. The Erdős-Selfridge potential sum ∑ 2^(−|L|) for
incomplete lines < 1 gives a theoretically safe second-player draw strategy;
our Eisenstein greedy bot approximates this on the Z[ω] lattice.

---

## Phase 0 — Stability (done)

- [x] 1-2-2 Connect6 rule in `game.py`
- [x] Player-aware MCTS backprop in `mcts.py`
- [x] FP16 AMP in training and inference
- [x] Cross-entropy policy loss over all legal moves
- [x] Divide-by-zero guard in visit distribution normalization
- [x] Transposition cache in `InferenceServer`
- [x] `checkpoints/legacy/` quarantine for incompatible weights
- [x] 8 baseline unit tests pass

---

## Phase 1 — Geometry-Faithful Architecture (done)

All implemented and verified: 26/26 unit tests pass.

### 1a. Inference batching
`NUM_WORKERS=8`, `INF_BATCH=8`, `INF_TIMEOUT=30ms`.

### 1b. `torch.compile` on inference
Applied in `InferenceServer.start()` when CUDA available.

### 1c. Tree reuse
`mcts_policy()` returns `new_root`; the next call re-uses the subtree with
re-applied Dirichlet noise.

### 1d. Cosine temperature annealing (was 3a)
`temp = max(0.05, cos(π × move / T))` — no hard cliff at move 20.

### 1e. TD-lambda value targets (was 3b)
`z_t = 0.99^(T−t) × z_final` — early positions receive discounted signal.

### 1f. History planes (was 4c)
Input channels: 3 → 11 (4 P1-history + 4 P2-history + to-move). Standard
AlphaZero temporal encoding.

### 1g. Parameter golf policy head (was 2c)
`p_conv` 2→4 channels; `p_fc` 64→32 hidden. Better capacity/param ratio.
Total: ~113K params.

### 1h. INT8 quantization utility (was 2a)
`quantize_for_inference(net)` via `torch.ao.quantization.quantize_dynamic`.

### 1i. HexConv2d (new)
`nn.Conv2d` subclass that registers a `hex_mask` buffer zeroing the two
non-hex corners `[0,0,0,0]` and `[0,0,2,2]`. Applied in all ResBlocks.
Enforces Z[ω] 7-cell neighbourhood in every spatial convolution.

### 1j. D6 data augmentation (new)
12 linear transforms (6 rotations × 2 reflections) on axial (q, r) coords.
Applied at `train_batch` time: `d6_augment_sample(item, random.randrange(12))`.
Up to 12× sample efficiency with zero extra self-play cost.

### 1k. EisensteinGreedyAgent curriculum (new)
Zero-parameter bot scoring each candidate by the maximum chain length it would
create or block along any Z[ω] axis. `defensive=True` variant considers both
own extension and opponent blocking. Used as 20% adversarial training partner
(every 5th game) and permanent ELO baseline. Approximates Erdős-Selfridge
potential on the lattice.

### 1l. Policy heatmap (new)
`save_heatmap(server, gen)` evaluates a fixed 10-move canonical test position
each generation and saves a `heatmaps/gen_XXXX.png` scatter plot of policy
mass over hex coordinates. Scientific instrument: confirms the net discovers
Z[ω] structure (policy mass should concentrate on the three win axes) without
any explicit encoding of that structure.

---

## Phase 2 — Training Quality

### 2a. Zobrist-keyed buffer deduplication (was 3c)
Connect6 on a hex grid produces near-duplicate positions from different
2-placement orderings. A Zobrist hash can deduplicate the replay buffer,
increasing effective diversity per training step.

### 2b. Self-play curriculum (was 3d)
Start `sims=25`, ramp to `sims=200` as ELO stabilizes. Noisy early games
explore policy space faster; high-sim late games refine tactics.

### 2c. Checkpoint tournament (was 5c)
New net must beat ≥55% of the top-K checkpoint pool to become training policy.
Prevents catastrophic forgetting.

---

## Phase 3 — Infrastructure

### 3a. Async training / inference separation (was 5a)
Decouple self-play from training: N self-play workers write to a shared buffer;
a separate process reads from it. Removes per-generation pause.

### 3b. Replay diversity sampling (was 5b)
Weight samples by recency or policy loss magnitude (prioritized experience
replay). Keeps the model learning from its most informative positions.

### 3c. CUDA Graphs hot path
Capture static computation graph for fixed-shape inference. Expected 30–50%
latency reduction. Requires fixed batch size — pair with Phase 3a async
separation.

---

## Phase 4 — Equivariance (research)

### 4a. G-CNN (full D6 equivariance)
Replace standard convolutions with group-equivariant layers that maintain
explicit D6 symmetry throughout the trunk, not just via augmentation. This is
the *correct* Sutton-compatible treatment: geometry (structural substrate) ≠
game knowledge. The network learns game strategy; the substrate enforces
the symmetry.

References: Cohen & Welling, "Group Equivariant Convolutional Networks" (2016);
Bekkers, "B-Spline CNNs on Lie Groups" (2020).

### 4b. CA weight initialization (deferred — see NOTES.md)
Investigate initializing HexConv2d weights using cellular automata patterns
rather than Xavier/He. The 7-cell Z[ω] neighbourhood IS the standard NCA
update kernel; a CA-derived prior might align weights with local hex dynamics
before any training. See: Mordvintsev et al., "Growing Neural Cellular
Automata" (2020).

---

## Phase 5 — Model Scale

### 5a. Scale trunk depth/width
Current: 2 ResBlocks, 32 channels. Next: 4 blocks, 64 channels (~480K params).
Only after Phase 3 infrastructure — wasted without async self-play.

### 5b. Activation sparsity / early exit
Profile `v_fc` first layer activation density. If >60% sparsity, implement
early exit from value head. Low priority until scale experiments run.

---

## Priority Order (RTX 2060, solo researcher)

| Priority | Item                             | Effort | Expected gain                  |
|----------|----------------------------------|--------|--------------------------------|
| ✅ 1     | Inference batching (1a)          | done   | 3-5× throughput                |
| ✅ 2     | Tree reuse (1c)                  | done   | ~5% free sims                  |
| ✅ 3     | TD-lambda value targets (1e)     | done   | Faster value convergence       |
| ✅ 4     | PUCT temp annealing (1d)         | done   | Better early exploration       |
| ✅ 5     | torch.compile (1b)               | done   | 30-50% inference speedup       |
| ✅ 6     | INT8 quantization utility (1h)   | done   | Reduced bandwidth              |
| ✅ 7     | History planes (1f)              | done   | Better temporal rep            |
| ✅ 8     | Parameter golf policy head (1g)  | done   | Better capacity/param          |
| ✅ 9     | HexConv2d Z[ω] kernel (1i)       | done   | Geometry-faithful conv         |
| ✅ 10    | D6 augmentation (1j)             | done   | Up to 12× sample efficiency    |
| ✅ 11    | EisensteinGreedyAgent (1k)       | done   | Structured curriculum adversary|
| ✅ 12    | Policy heatmap (1l)              | done   | Scientific instrument          |
| ✅ 13    | Zobrist buffer dedup (2a)        | done   | Better replay diversity        |
| ✅ 14    | Checkpoint tournament (2c)       | done   | Stable training policy         |
| ✅ 15    | Async self-play (3a)             | done   | Overlapped SP+train, weight sync|
| ✅ 16    | CUDA Graphs (3c)                 | done   | 30-50% inference speedup       |
| ✅ 17    | Self-play curriculum (2b)        | done   | Faster early exploration       |
| 18       | G-CNN full equivariance (4a)     | 1wk    | Principled symmetry            |
| 19       | CA weight init (4b)              | 2hr    | Z[ω]-aligned priors            |
| 20       | Scale trunk 4blk/64ch (5a)       | 2hr    | Capacity for harder games      |
