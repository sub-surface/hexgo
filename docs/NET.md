# HexGo — Neural Network (`net.py`)

## Architecture

```
Input [11, 18, 18]
  → HexConv2d(11→64, 3×3) + BN + ReLU           [stem: hex-masked conv]
  → 4× ResBlock(64ch, HexConv2d)                 [trunk: Z[ω]-faithful kernels]
  → GlobalPoolBranch(64ch)                        [KataGo global context]
  ├→ Conv2d(64→1, 1×1) + BN + ReLU → FC → Tanh  [value head → scalar ∈ [-1,1]]
  ├→ Softplus variance head                       [value uncertainty → σ²]
  ├→ Conv2d(64→1, 1×1) + Tanh                    [ownership aux → [S,S]]
  ├→ Conv2d(64→1, 1×1)                           [threat aux → [S,S]]
  └→ Conv2d(64→4, 1×1) + BN + ReLU →
       cat(move_plane) → Linear(5·S², 64) → out  [policy → scalar logit per move]
```

| Param | Value | Rationale |
|-------|-------|-----------|
| Board window | 18×18 | Centered on recent-move centroid; covers >95% of game extents |
| Hidden channels | 64 | Configurable via CFG["TRUNK_CHANNELS"] |
| Residual blocks | 4 | Configurable via CFG["TRUNK_BLOCKS"] |
| Precision | FP16 AMP | `torch.amp.autocast` doubles memory bandwidth |
| Weight init | Hex-Laplacian CA | `init_weights_ca()` — Z[ω]-aligned diffusion prior |

---

## Input Encoding (`encode_board`)

Returns `float32 [11, 18, 18]` centered on the centroid of the last N_RECENT (20) moves.

| Channel(s) | Contents |
|-----------|----------|
| 0–3 | P1 piece positions at t, t−1, t−2, t−3 (4-step history) |
| 4–7 | P2 piece positions at t, t−1, t−2, t−3 (4-step history) |
| 8 | P1 current pieces (same as ch 0) |
| 9 | P2 current pieces (same as ch 4) |
| 10 | To-move plane: 0.0 = P1 to move, 1.0 = P2 to move |

`N_HISTORY = 4`, `IN_CH = 3 + 2×4 = 11`.

Also returns `(oq, or_)` — the integer centroid offset used to encode moves.

---

## HexConv2d

`HexConv2d(nn.Conv2d)` enforces the Z[ω] 7-cell neighbourhood by zeroing
two corners of every 3×3 kernel via a registered `hex_mask` buffer:

```
  Mask (✓ = active, ✗ = zeroed):
    ✗ ✓ ✓
    ✓ ✓ ✓
    ✓ ✓ ✗
```

Masked positions: `[*, *, 0, 0]` (top-left) and `[*, *, 2, 2]` (bottom-right).

The `forward()` hook applies `weight * hex_mask` before every pass, ensuring
weight updates cannot restore the masked corners during training.

Used in: all ResBlock convolutions. The stem uses standard `Conv2d` (intent:
allow the stem to learn the full encoding before the hex-faithful layers).

---

## D6 Data Augmentation

The symmetry group of Z[ω] is D6 (order 12): 6 rotations at 60° + 6 reflections.
In axial (q, r) coordinates, all 12 transforms are 2×2 integer matrices.

`D6_MATRICES` — shape `[12, 2, 2]` int32 array:

```python
# Rotations (counterclockwise, 60° steps)
R0:  [[ 1, 0],[ 0, 1]]   identity
R60: [[ 0,-1],[ 1, 1]]
R120:[[-1,-1],[ 1, 0]]
R180:[[-1, 0],[ 0,-1]]
R240:[[ 0, 1],[-1,-1]]
R300:[[ 1, 1],[-1, 0]]
# Reflections
S0:  [[ 0, 1],[ 1, 0]]   swap q,r
S60: [[-1, 0],[ 1, 1]]
...
```

`d6_augment_sample(sample, tf_idx)` transforms both the board array and move
coordinates consistently. Applied at `train_batch` time: `tf_idx = random.randrange(12)`.

Moves outside the 18×18 window after transform are set to `None` (handled by
caller). Up to 12× sample efficiency at zero self-play cost.

---

## GlobalPoolBranch (KataGo-style)

Inserted after `self.blocks` in `HexNet.trunk()`:

```
trunk features [B, 32, H, W]
  → avg_pool → [B, 32]
  → max_pool → [B, 32]
  → cat      → [B, 64]
  → FC(64, 32) + ReLU
  → reshape  → [B, 32, 1, 1]
  → broadcast_add to trunk features
```

Gives every spatial cell awareness of global game state (threat density,
material balance) at ~2K extra parameters.

---

## Policy Head

`(trunk_features, move_plane) → scalar logit`

`move_plane` is a `[1, 18, 18]` one-hot plane for the candidate move.
`encode_move(q, r, oq, or_)` returns `None` if the move is outside the window.

Training loss: **Cross-entropy** over all legal moves in a position using the
MCTS visit count distribution as targets.

---

## Checkpoint Compatibility

`load_latest()` handles breaking architecture changes (e.g., IN_CH 3→11):
on `RuntimeError`, all `net_*.pt` files are moved to `checkpoints/legacy/`
and training starts fresh with a clean net.

---

## Parameter Count Note

`inference.py` docstring says "355K-param net" — this is outdated. `net.py`
comment and `param_count()` report ~121K. The discrepancy is from an earlier,
larger architecture that was replaced. The inference code is correct; only
the comment is stale.
