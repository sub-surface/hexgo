# HexGo — Roadmap

Phased optimization plan. Each phase is independent and additive.
Current baseline: ~121K params, 4 workers, RTX 2060, ~30–60s/gen.

---

## Phase 0 — Stability (now)

All items already addressed or confirmed working:

- [x] 1-2-2 Connect6 rule in `game.py`
- [x] Player-aware MCTS backprop in `mcts.py`
- [x] FP16 AMP in training and inference
- [x] Cross-entropy policy loss over all legal moves
- [x] Divide-by-zero guard in visit distribution normalization
- [x] Transposition cache in `InferenceServer`
- [x] `checkpoints/legacy/` quarantine for incompatible weights
- [x] All 8 unit tests pass

**Remaining gap:** `mcts_with_net()` in `mcts.py` creates leaf children
without `player=game.current_player`. Only affects standalone `mcts_with_net`
calls; `train.py` uses its own `mcts_policy()` which is correct. Fix when
`mcts_with_net` is used outside training.

---

## Phase 1 — Throughput (high ROI, low risk)

### 1a. Fix inference batching (avg_batch ≈ 1.0 → >3.0)

The inference server timeout (5ms) fires before parallel workers have all
submitted their requests. Options:
- Increase `INF_TIMEOUT` from 5ms → 20–50ms (simplest fix, adds latency)
- Increase `NUM_WORKERS` from 4 → 8–16 (more concurrent games fill the batch)
- **Best:** raise both; profile `avg_batch_size` to find the sweet spot for
  RTX 2060 (expect diminishing returns past batch=8 for a 121K-param net)

Expected gain: 3–5× throughput (from amortizing kernel launch overhead).

### 1b. CUDA Graphs for the inference hot path

PyTorch CUDA Graphs capture a static computation graph and replay it without
Python overhead. Requires fixed batch size and static tensor shapes.

```python
# Capture once after first batch
import torch
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    out = net.trunk(boards_static)
    ...
# Replay:
boards_static.copy_(new_boards)
g.replay()
```

Works well with our fixed 18×18 board encoding and batched inference server.
Expected: 30–50% reduction in inference latency on GPU.

### 1c. Tree reuse (subtree recycling)

After each move, re-root the MCTS tree at the child node matching the chosen
move and retain its subtree. AlphaZero reuses ~N_SIMS/branching_factor
simulations "for free".

```python
# After choosing move m:
new_root = next(c for c in root.children if c.move == m)
new_root.parent = None  # detach
```

Savings scale with tree depth. At 100 sims/move with branching ~20, this
saves ~5 sims worth of work per move (modest but free).

---

## Phase 2 — Model Compression (research-informed)

### 2a. INT4 / FP8 Weight Quantization (inference only)

Motivated by **OpenAI Parameter Golf** (openai/parameter-golf) and
**TurboQuant** (Google Research, 2024).

For inference-only quantization (weights quantized, activations FP16):

```python
# Using bitsandbytes or torch.ao.quantization
import bitsandbytes as bnb
# Replace nn.Linear with bnb.nn.Linear4bit for the FC layers
```

Our net is already tiny (121K params, ~240KB FP16). INT4 would reduce to
~60KB. The value is in reduced memory bandwidth pressure during batched GPU
calls, not storage.

**TurboQuant concepts applicable here:**
- **PolarQuant**: represent weights as (magnitude, angle) in 2D. For 2D conv
  weight pairs, quantize the angle to a grid (e.g., 4 bits → 16 directions).
  Preserves relative direction better than uniform quantization.
- **QJL (Quantized JL Transform)**: apply a random rotation before 1-bit
  quantization. The rotation evenly distributes weight variance across
  dimensions, making 1-bit closer to optimal. Most relevant for the FC layers
  in our value/policy heads.

Practical recommendation: apply `torch.ao.quantization.quantize_dynamic` to
the FC layers as a low-effort first step. Full INT4 for the conv trunk is a
larger investment.

### 2b. Activation Sparsity / Early Exit

The value head (`v_fc`) is a 2-layer MLP. If the first layer output is mostly
zero (after ReLU), skip the second layer and return an interpolated estimate.
Profile activation density first — only worth it if >60% sparsity observed.

### 2c. Parameter Golf: what can we remove?

Inspired by openai/parameter-golf's principle that parameter count is the
wrong objective — *effective capacity per parameter* matters.

Current architecture audit:
- The **policy FC** layer (`p_fc`) takes `2*18*18 + 18*18 = 972` inputs into
  64 hidden units. This is the largest single weight tensor (~62K params,
  51% of the model). The trunk features passed to it are a 2-channel
  compressed map — this compression is doing a lot of work.
- Try: replace `Conv2d(hidden, 2, 1)` with `Conv2d(hidden, 4, 1)` and reduce
  `p_fc` hidden size from 64 → 32. Should maintain capacity with fewer params.
- Alternatively: use a dot-product policy head (trunk_feature · move_feature)
  rather than concatenation. This is parameter-free and may generalize better.

---

## Phase 3 — Training Quality

### 3a. PUCT temperature annealing

Current: temp=1.0 for first 20 moves, then temp=0 (argmax). This creates a
hard cliff. Try cosine annealing: `temp = max(0.05, cos(pi * move / T))`.

### 3b. Value target improvement

Current: `z = +1 (winner) / -1 (loser)` for all positions in a game.
Problem: early game positions of the winner are not actually winning — they
get a misleading +1 signal.

Better: discount factor or TD-lambda style target:
```python
z_t = gamma^(T - t) * z_final  # gamma ≈ 0.99
```
This makes early positions less certain and speeds up value head convergence.

### 3c. Zobrist-keyed replay buffer deduplication

Connect6 games on a hex grid produce many near-duplicate positions (same board,
different move orderings for the 2-placement turns). A Zobrist hash can
deduplicate the buffer, increasing effective diversity per training step.

```python
import random
ZOBRIST = {(q_bits, r_bits, player): random.getrandbits(64)
           for q_bits in range(-20, 21)
           for r_bits in range(-20, 21)
           for player in (1, 2)}
```

### 3d. Self-play curriculum

Start training with `sims=25` (fast, noisy games) and ramp to `sims=200` as
ELO stabilizes. Noisy early games explore the policy space faster; high-sim
late games refine tactics.

---

## Phase 4 — Representation

### 4a. Relative coordinate encoding

Current centroid-centered window loses awareness of board orientation.
Alternative: encode position relative to the last move (or center of the last
2 moves). This makes the representation **translation-invariant** and may
reduce the window size needed.

### 4b. Hex-aware convolution (optional)

Standard square conv doesn't respect hex geometry. A hex-aware kernel mask
(zero out the two non-adjacent grid corners per 3×3 kernel) could improve
spatial learning. Modest impact for a shallow net.

### 4c. History planes

Add N=4 history planes to the input (previous moves for both players).
Changes input channels from 3 → 3 + 2N = 11. This is standard in AlphaZero
and gives the net awareness of move order within the 2-placement turns.

---

## Phase 5 — Infrastructure

### 5a. Async training / inference separation

Decouple self-play from training: run N self-play workers writing to a shared
buffer, and a separate training process reading from it. This removes the
per-generation pause and allows continuous training.

### 5b. Replay diversity sampling

Instead of uniform random buffer sampling, weight samples by recency
(prioritized experience replay, PER) or by policy loss magnitude. Keeps the
model learning from its most informative positions.

### 5c. Checkpoint tournament

Keep top-K checkpoints. New net must beat ≥55% of the tournament pool (not
just the latest baseline) to become the new training policy. Prevents
catastrophic forgetting.

---

## Priority Order (RTX 2060, solo researcher)

| Priority | Item                          | Effort | Expected gain            |
|----------|-------------------------------|--------|--------------------------|
| 1        | Fix inference batching (1a)   | 30min  | 3-5× throughput          |
| 2        | Tree reuse (1c)               | 1hr    | ~5% free sims            |
| 3        | TD-lambda value targets (3b)  | 2hr    | Faster value convergence |
| 4        | PUCT temp annealing (3a)      | 30min  | Better early exploration |
| 5        | CUDA Graphs (1b)              | 3hr    | 30-50% inference speedup |
| 6        | INT4 FC quantization (2a)     | 2hr    | Reduced bandwidth        |
| 7        | History planes (4c)           | 2hr    | Better temporal rep      |
| 8        | Parameter golf policy head (2c)| 1hr   | Better capacity/param    |
