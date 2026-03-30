"""
HexNet — small ResNet policy+value network for hexagonal 6-in-a-row.

Architecture rationale (see docs/DESIGN.md):
  - Input: 11 × 18 × 18 axial grid centered on board centroid (3 state + 8 history)
  - 2 residual blocks, 32 channels — ~121K params
  - Value head: board → scalar win probability ∈ [-1, 1]
  - Policy head: (board, move_plane) → scalar logit
    Move plane is a 1-hot 18×18 map of the candidate move.
    This avoids a fixed output size and works for any board extent.

Device: CUDA if available, else CPU.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from game import HexGame

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BOARD_SIZE = 18          # increased from 15 (sees more of the infinite grid)
N_HISTORY = 4            # 4c: history planes per player (last N moves)
IN_CH = 3 + 2 * N_HISTORY  # p1, p2, to_move + 4 p1-history + 4 p2-history = 11

# Architecture sizes — swap by changing these two constants
HIDDEN   = 32   # was 64 (4x fewer params → 4x faster inference)
N_BLOCKS = 2    # was 4

# ── D6 symmetry group of the hexagonal lattice Z[omega] ──────────────────────
#
# The hex grid is isomorphic to the Eisenstein integers Z[omega], omega=e^(2pi*i/3).
# Its symmetry group D6 (order 12) consists of 6 rotations (60° steps) and 6
# reflections.  In axial (q,r) coordinates each element is a 2x2 integer matrix:
#
#   new_q = M[0,0]*q + M[0,1]*r
#   new_r = M[1,0]*q + M[1,1]*r
#
# Applying all 12 transforms to a training sample gives 12 equivalent positions
# for free — up to 12x sample efficiency without encoding any game knowledge.

D6_MATRICES = np.array([
    # ── 6 rotations (counterclockwise, 60° steps) ──
    [[ 1,  0], [ 0,  1]],   # R0   identity
    [[ 0, -1], [ 1,  1]],   # R60
    [[-1, -1], [ 1,  0]],   # R120
    [[-1,  0], [ 0, -1]],   # R180
    [[ 0,  1], [-1, -1]],   # R240
    [[ 1,  1], [-1,  0]],   # R300
    # ── 6 reflections ──
    [[ 0,  1], [ 1,  0]],   # S0  (swap q,r)
    [[-1,  0], [ 1,  1]],   # S60
    [[-1, -1], [ 0,  1]],   # S120
    [[ 0, -1], [-1,  0]],   # S180
    [[ 1,  0], [-1, -1]],   # S240
    [[ 1,  1], [ 0, -1]],   # S300
], dtype=np.int32)           # shape [12, 2, 2]


def _transform_board(board_arr: np.ndarray, tf_idx: int,
                     size: int = BOARD_SIZE) -> np.ndarray:
    """
    Apply one of the 12 D6 symmetry transforms to a board encoding array.
    Channel 2 (to-move plane) is rotation-invariant and copied unchanged.
    Pixels that transform outside the window are dropped (zeroed in dest).
    """
    half = size // 2
    M = D6_MATRICES[tf_idx]          # [2,2]

    # Relative coordinates for every (row, col) in the source array
    # q_grid[r,q] = q_rel,  r_grid[r,q] = r_rel
    qs = np.arange(size) - half
    rs = np.arange(size) - half
    q_grid, r_grid = np.meshgrid(qs, rs)   # both [size, size]

    # Apply linear transform to all source positions
    q_dst = M[0, 0] * q_grid + M[0, 1] * r_grid   # [size, size]
    r_dst = M[1, 0] * q_grid + M[1, 1] * r_grid

    col_dst = (q_dst + half).astype(np.int32)
    row_dst = (r_dst + half).astype(np.int32)

    # Mask: only source pixels whose dest falls within the window
    valid = (col_dst >= 0) & (col_dst < size) & (row_dst >= 0) & (row_dst < size)

    # Flat source indices for valid pixels
    src_rows_v = np.where(valid, np.indices((size, size))[0], 0)[valid]
    src_cols_v = np.where(valid, np.indices((size, size))[1], 0)[valid]
    row_dst_v  = row_dst[valid]
    col_dst_v  = col_dst[valid]

    new_arr = np.zeros_like(board_arr)
    new_arr[2] = board_arr[2]          # to-move: rotation-invariant
    for ch in range(IN_CH):
        if ch == 2:
            continue
        new_arr[ch, row_dst_v, col_dst_v] = board_arr[ch, src_rows_v, src_cols_v]

    return new_arr


def d6_augment_sample(sample: dict, tf_idx: int) -> dict:
    """
    Return one D6-equivalent version of a training buffer sample.
    Board array and move coordinates are transformed consistently so the
    visit-probability distribution stays correctly assigned to moves.

    After transformation oq=or_=0 because moves are stored as relative
    coordinates; encode_move(nq, nr, 0, 0) maps them correctly.
    """
    M   = D6_MATRICES[tf_idx]
    oq, or_ = sample['oq'], sample['or_']

    new_board = _transform_board(sample['board'], tf_idx)

    new_moves = []
    for q, r in sample['moves']:
        q_rel, r_rel = q - oq, r - or_
        nq = int(M[0, 0] * q_rel + M[0, 1] * r_rel)
        nr = int(M[1, 0] * q_rel + M[1, 1] * r_rel)
        new_moves.append((nq, nr))

    return {
        'board': new_board,
        'oq': 0, 'or_': 0,
        'moves': new_moves,
        'probs': sample['probs'].copy(),   # copy to prevent in-place normalization corrupting buffer
        'z':     sample['z'],
    }


# ── Board encoding ────────────────────────────────────────────────────────────

def encode_board(game: HexGame, size: int = BOARD_SIZE) -> np.ndarray:
    """
    Returns float32 array [IN_CH, size, size] centered on centroid of all pieces.
    If board empty, centers at (0,0).

    Channel layout (IN_CH = 11):
      0   — player 1 current pieces
      1   — player 2 current pieces
      2   — to-move plane (0.0=p1, 1.0=p2)
      3-6 — player 1 last N_HISTORY moves (most recent = ch 3), one-hot each
      7-10— player 2 last N_HISTORY moves (most recent = ch 7), one-hot each
    """
    half = size // 2
    if game.board:
        cq = sum(q for q, r in game.board) / len(game.board)
        cr = sum(r for q, r in game.board) / len(game.board)
        oq, or_ = round(cq), round(cr)
    else:
        oq, or_ = 0, 0

    arr = np.zeros((IN_CH, size, size), dtype=np.float32)
    cp = game.current_player - 1   # 0 or 1
    arr[2, :, :] = cp

    # Channels 0-1: current board state
    for (q, r), p in game.board.items():
        qi = q - oq + half
        ri = r - or_ + half
        if 0 <= qi < size and 0 <= ri < size:
            arr[p - 1, ri, qi] = 1.0

    # 4c: history planes — last N_HISTORY placements per player, most recent first.
    # Use player_history (parallel to move_history) so the split is correct even
    # during MCTS tree traversal after unmake() has removed pieces from the board.
    p1_hist, p2_hist = [], []
    for m, mp in zip(reversed(game.move_history), reversed(game.player_history)):
        if mp == 1 and len(p1_hist) < N_HISTORY:
            p1_hist.append(m)
        elif mp == 2 and len(p2_hist) < N_HISTORY:
            p2_hist.append(m)
        if len(p1_hist) >= N_HISTORY and len(p2_hist) >= N_HISTORY:
            break
    for i, (q, r) in enumerate(p1_hist):
        qi = q - oq + half
        ri = r - or_ + half
        if 0 <= qi < size and 0 <= ri < size:
            arr[3 + i, ri, qi] = 1.0
    for i, (q, r) in enumerate(p2_hist):
        qi = q - oq + half
        ri = r - or_ + half
        if 0 <= qi < size and 0 <= ri < size:
            arr[7 + i, ri, qi] = 1.0

    return arr, (oq, or_)


def encode_move(q: int, r: int, oq: int, or_: int,
                size: int = BOARD_SIZE) -> np.ndarray | None:
    """1-hot [1, size, size] plane for a candidate move. Returns None if out of window."""
    half = size // 2
    qi = q - oq + half
    ri = r - or_ + half
    if 0 <= qi < size and 0 <= ri < size:
        plane = np.zeros((1, size, size), dtype=np.float32)
        plane[0, ri, qi] = 1.0
        return plane
    return None


# ── Network ───────────────────────────────────────────────────────────────────

class HexConv2d(nn.Conv2d):
    """
    3x3 convolution with a fixed hex-7 kernel mask.

    The hex grid has 6 neighbours per cell: (+-1,0), (0,+-1), (+1,-1), (-1,+1).
    In axial coordinates mapped to a 2D array (col=q, row=r), these correspond
    to all 3x3 neighbours EXCEPT the two corners at (Dq,Dr)=(-1,-1) and (+1,+1)
    — i.e., kernel positions [0,0] and [2,2].  Zeroing those weights enforces
    that the network only attends to the true Z[omega] neighbourhood rather than
    the geometrically incorrect square neighbourhood.

    Faithful to Sutton: this is a structural prior about geometry, not game knowledge.
    The network still discovers what matters within the correct substrate.
    """
    def __init__(self, in_channels: int, out_channels: int, **kwargs):
        kwargs.setdefault('padding', 1)
        super().__init__(in_channels, out_channels, kernel_size=3, **kwargs)
        mask = torch.ones(1, 1, 3, 3)
        mask[0, 0, 0, 0] = 0.0   # (-1,-1) direction: not a hex neighbour
        mask[0, 0, 2, 2] = 0.0   # (+1,+1) direction: not a hex neighbour
        self.register_buffer('hex_mask', mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.weight * self.hex_mask,
                        self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv1 = HexConv2d(ch, ch, bias=False)
        self.bn1   = nn.BatchNorm2d(ch)
        self.conv2 = HexConv2d(ch, ch, bias=False)
        self.bn2   = nn.BatchNorm2d(ch)

    def forward(self, x):
        r = F.relu(self.bn1(self.conv1(x)))
        r = self.bn2(self.conv2(r))
        return F.relu(x + r)


class GlobalPoolBranch(nn.Module):
    """
    KataGo-style global pooling branch.

    After the residual trunk, concatenate board-wide average and max pooling
    features and project them back to the spatial feature map. This gives
    every cell awareness of the global game state (total material, threat density,
    board extent) at essentially zero extra compute.

    Architecture:
        x [B, C, H, W] → avg_pool → [B, C]  ─┐
                        → max_pool → [B, C]  ─┼→ FC(2C→C) → broadcast → x + g
    """
    def __init__(self, ch: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(ch * 2, ch),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=(2, 3))                          # [B, C]
        mx  = x.amax(dim=(2, 3))                          # [B, C]
        g   = self.fc(torch.cat([avg, mx], dim=1))        # [B, C]
        g   = g.unsqueeze(-1).unsqueeze(-1).expand_as(x)  # [B, C, H, W]
        return x + g                                       # residual broadcast


class HexNet(nn.Module):
    """
    Shared trunk: IN_CH → hidden conv → n_blocks residual blocks → global pool
    Value head:   trunk → conv 1×1 → flatten → FC → tanh scalar
    Policy head:  (trunk_features, move_plane) → conv 1×1 → flatten → FC → scalar
    """
    def __init__(self, hidden: int = HIDDEN, n_blocks: int = N_BLOCKS):
        super().__init__()
        self.hidden = hidden
        self.n_blocks = n_blocks

        # Trunk
        self.stem = nn.Sequential(
            nn.Conv2d(IN_CH, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(*[ResBlock(hidden) for _ in range(n_blocks)])
        # KataGo global pool: gives each cell global board awareness (threat density etc.)
        self.global_pool = GlobalPoolBranch(hidden)

        # Value head
        self.v_conv = nn.Sequential(
            nn.Conv2d(hidden, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(),
        )
        self.v_fc = nn.Sequential(
            nn.Linear(BOARD_SIZE * BOARD_SIZE, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

        # Policy head — takes trunk features + 1-channel move plane
        # 2c (parameter golf): 4-channel compressed map (was 2) + move plane → 32 hidden (was 64)
        # More expressive trunk compression with fewer FC params overall
        self.p_conv = nn.Sequential(
            nn.Conv2d(hidden, 4, 1, bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU(),
        )
        self.p_fc = nn.Sequential(
            nn.Linear(4 * BOARD_SIZE * BOARD_SIZE + BOARD_SIZE * BOARD_SIZE, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def trunk(self, x: torch.Tensor) -> torch.Tensor:
        return self.global_pool(self.blocks(self.stem(x)))

    def value(self, features: torch.Tensor) -> torch.Tensor:
        v = self.v_conv(features).flatten(1)
        return self.v_fc(v).squeeze(-1)          # [B]

    def policy_logit(self, features: torch.Tensor,
                     move_planes: torch.Tensor) -> torch.Tensor:
        """
        features:    [B, hidden, S, S]
        move_planes: [B, 1, S, S]  — 1-hot move location
        returns:     [B] scalar logits
        """
        p = self.p_conv(features).flatten(1)     # [B, 2*S*S]
        m = move_planes.flatten(1)               # [B, S*S]
        return self.p_fc(torch.cat([p, m], dim=1)).squeeze(-1)  # [B]

    def forward(self, board_tensor: torch.Tensor,
                move_planes: torch.Tensor | None = None):
        """
        board_tensor: [B, 3, S, S]
        move_planes:  [B, 1, S, S] or None (value-only mode)
        Returns: (value [B], policy_logit [B]) or (value [B], None)
        """
        f = self.trunk(board_tensor)
        v = self.value(f)
        p = self.policy_logit(f, move_planes) if move_planes is not None else None
        return v, p


# ── Inference helpers ─────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(net: HexNet, game: HexGame) -> tuple[float, dict]:
    """
    Returns (value, {move: logit}) for all legal moves.
    value is from current player's perspective: +1 = winning, -1 = losing.
    """
    net.eval()
    board_arr, (oq, or_) = encode_board(game)
    moves = game.legal_moves()
    if not moves:
        return 0.0, {}

    # Filter moves within the net's window
    valid_moves = []
    planes = []
    for m in moves:
        p = encode_move(m[0], m[1], oq, or_)
        if p is not None:
            valid_moves.append(m)
            planes.append(p)

    device = next(net.parameters()).device

    if not valid_moves:
        # Fallback: evaluate value head only if all moves clipped
        feat = net.trunk(torch.tensor(board_arr, device=device).unsqueeze(0))
        value = net.value(feat).item()
        return value, {}

    B = len(valid_moves)
    board_t = torch.tensor(board_arr, device=device).unsqueeze(0)   # [1,3,S,S]
    boards  = board_t.expand(B, -1, -1, -1)                         # [B,3,S,S]
    move_t  = torch.tensor(np.stack(planes), device=device)         # [B,1,S,S]

    features = net.trunk(boards)
    value = net.value(features[:1]).item()
    logits = net.policy_logit(features, move_t).cpu().numpy()

    policy = {m: float(l) for m, l in zip(valid_moves, logits)}
    return value, policy


def param_count(net: HexNet) -> int:
    return sum(p.numel() for p in net.parameters())


def quantize_for_inference(net: HexNet) -> HexNet:
    """
    2a: Apply dynamic INT8 quantization to FC layers for reduced memory bandwidth
    during batched GPU inference. Call this after loading a trained checkpoint,
    before running inference-only evaluation (not during training).

    Typical gain: 20-30% reduction in FC layer compute on CPU; smaller benefit
    on GPU due to Tensor Cores already handling FP16 efficiently.
    """
    net.eval()
    return torch.ao.quantization.quantize_dynamic(
        net, {nn.Linear}, dtype=torch.qint8
    )


if __name__ == "__main__":
    import time

    net = HexNet().to(DEVICE)
    total = param_count(net)

    print(f"{'='*54}")
    print(f"  HexNet smoke test")
    print(f"  Device : {DEVICE}")
    print(f"  IN_CH  : {IN_CH}  (3 state + {2*N_HISTORY} history)")
    print(f"  Hidden : {HIDDEN}  Blocks: {N_BLOCKS}")
    print(f"{'='*54}")

    # ── param breakdown ───────────────────────────────────────
    sections = {
        "stem":          sum(p.numel() for p in net.stem.parameters()),
        "blocks":        sum(p.numel() for p in net.blocks.parameters()),
        "value_head":    sum(p.numel() for p in net.v_conv.parameters()) +
                         sum(p.numel() for p in net.v_fc.parameters()),
        "policy_head":   sum(p.numel() for p in net.p_conv.parameters()) +
                         sum(p.numel() for p in net.p_fc.parameters()),
    }
    print(f"\n  Param breakdown  (total {total:,}):")
    for name, count in sections.items():
        bar = "#" * int(count / total * 30)
        print(f"  {name:<14} {count:>7,}  {count/total*100:4.1f}%  {bar}")

    # ── board encoding & history planes ───────────────────────
    game = HexGame()
    for coord in [(0,0),(1,0),(0,1),(1,-1),(0,-1),(1,1)]:
        game.make(*coord)

    arr, (oq, or_) = encode_board(game)
    print(f"\n  encode_board shape: {arr.shape}  origin=({oq},{or_})")
    print(f"  Ch 0  (P1 pieces) : {arr[0].sum():.0f} cells occupied")
    print(f"  Ch 1  (P2 pieces) : {arr[1].sum():.0f} cells occupied")
    print(f"  Ch 2  (to-move)   : {arr[2,0,0]:.0f}  (0=P1, 1=P2)")
    for i in range(N_HISTORY):
        p1_occ = arr[3 + i].sum()
        p2_occ = arr[7 + i].sum()
        print(f"  Ch {3+i} / Ch {7+i}   (hist {i}): P1={p1_occ:.0f}  P2={p2_occ:.0f}")

    # ── policy & value quality ────────────────────────────────
    v, policy = evaluate(net, game)
    logits = np.array(list(policy.values()), dtype=np.float32)
    logits -= logits.max()
    probs = np.exp(logits); probs /= probs.sum()
    entropy = float(-(probs * np.log(probs + 1e-8)).sum())
    max_ent  = float(np.log(max(len(policy), 1)))
    print(f"\n  Value            : {v:.4f}")
    print(f"  Policy entries   : {len(policy)}")
    print(f"  Policy entropy   : {entropy:.3f}  (uniform={max_ent:.3f}, "
          f"ratio={entropy/max(max_ent,1e-8)*100:.0f}%)")

    # ── inference latency ─────────────────────────────────────
    net.eval()
    N_WARMUP, N_RUNS = 20, 200
    for B in (1, 4, 8, 16):
        x = torch.randn(B, IN_CH, BOARD_SIZE, BOARD_SIZE, device=DEVICE)
        m = torch.zeros(B, 1, BOARD_SIZE, BOARD_SIZE, device=DEVICE)
        m[:, 0, BOARD_SIZE//2, BOARD_SIZE//2] = 1.0
        # Warmup
        with torch.no_grad(), torch.amp.autocast(
                device_type="cuda" if "cuda" in str(DEVICE) else "cpu"):
            for _ in range(N_WARMUP):
                net(x, m)
        if "cuda" in str(DEVICE):
            torch.cuda.synchronize()
        # Timed
        t0 = time.perf_counter()
        with torch.no_grad(), torch.amp.autocast(
                device_type="cuda" if "cuda" in str(DEVICE) else "cpu"):
            for _ in range(N_RUNS):
                net(x, m)
        if "cuda" in str(DEVICE):
            torch.cuda.synchronize()
        ms_per_call = (time.perf_counter() - t0) / N_RUNS * 1000
        pos_per_sec  = B / ms_per_call * 1000
        print(f"  Batch={B:>2}  {ms_per_call:6.2f}ms/call  "
              f"{pos_per_sec:>8,.0f} pos/s")

    # ── shape check ───────────────────────────────────────────
    B = 8
    x = torch.randn(B, IN_CH, BOARD_SIZE, BOARD_SIZE, device=DEVICE)
    m = torch.zeros(B, 1, BOARD_SIZE, BOARD_SIZE, device=DEVICE)
    m[:, 0, BOARD_SIZE//2, BOARD_SIZE//2] = 1.0
    with torch.no_grad():
        val, pol = net(x, m)
    assert val.shape == (B,),    f"value shape wrong: {val.shape}"
    assert pol.shape == (B,),    f"policy shape wrong: {pol.shape}"
    print(f"\n  Batch shapes OK: value={tuple(val.shape)}  policy={tuple(pol.shape)}")

    # ── matplotlib param chart ────────────────────────────────
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        fig.suptitle(
            f"HexNet  {total:,} params  |  {IN_CH}ch × {BOARD_SIZE}² input  |  "
            f"{HIDDEN}ch × {N_BLOCKS} blocks  ({str(DEVICE).upper()})",
            fontsize=11,
        )

        # Left: param share by section
        colors = ["#4C9BE8", "#E87C4C", "#5DBD6C", "#B06DE8"]
        wedges, texts, autotexts = axes[0].pie(
            list(sections.values()),
            labels=list(sections.keys()),
            autopct="%1.1f%%",
            colors=colors,
            startangle=140,
        )
        axes[0].set_title("Parameter share")

        # Right: inference latency by batch size
        batch_sizes  = [1, 4, 8, 16]
        latencies_ms = []
        for B in batch_sizes:
            xb = torch.randn(B, IN_CH, BOARD_SIZE, BOARD_SIZE, device=DEVICE)
            mb = torch.zeros(B, 1, BOARD_SIZE, BOARD_SIZE, device=DEVICE)
            mb[:, 0, BOARD_SIZE//2, BOARD_SIZE//2] = 1.0
            with torch.no_grad(), torch.amp.autocast(
                    device_type="cuda" if "cuda" in str(DEVICE) else "cpu"):
                for _ in range(10):
                    net(xb, mb)
            if "cuda" in str(DEVICE):
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad(), torch.amp.autocast(
                    device_type="cuda" if "cuda" in str(DEVICE) else "cpu"):
                for _ in range(100):
                    net(xb, mb)
            if "cuda" in str(DEVICE):
                torch.cuda.synchronize()
            latencies_ms.append((time.perf_counter() - t0) / 100 * 1000)

        axes[1].bar([str(b) for b in batch_sizes], latencies_ms, color="#4C9BE8", width=0.5)
        axes[1].set_xlabel("Batch size")
        axes[1].set_ylabel("Latency (ms)")
        axes[1].set_title("Inference latency")
        for i, v in enumerate(latencies_ms):
            axes[1].text(i, v + 0.02, f"{v:.2f}ms", ha="center", fontsize=9)

        plt.tight_layout()
        out = "net_profile.png"
        plt.savefig(out, dpi=130, bbox_inches="tight")
        plt.close()
        print(f"\n  Chart saved: {out}")
    except ImportError:
        print("\n  (matplotlib not available - skipping chart)")

    print(f"\n{'='*54}")
    print("  net.py OK")
