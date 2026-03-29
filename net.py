"""
HexNet — small ResNet policy+value network for hexagonal 6-in-a-row.

Architecture rationale (see docs/DESIGN.md):
  - Input: 3 × 18 × 18 axial grid centered on board centroid
  - 4 residual blocks, 64 channels — ~170K params
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
IN_CH = 3                # p1, p2, to_move

# Architecture sizes — swap by changing these two constants
HIDDEN   = 32   # was 64 (4x fewer params → 4x faster inference)
N_BLOCKS = 2    # was 4


# ── Board encoding ────────────────────────────────────────────────────────────

def encode_board(game: HexGame, size: int = BOARD_SIZE) -> np.ndarray:
    """
    Returns float32 array [3, size, size] centered on the centroid of all pieces.
    If board empty, centers at (0,0).
    """
    half = size // 2
    if game.board:
        cq = sum(q for q, r in game.board) / len(game.board)
        cr = sum(r for q, r in game.board) / len(game.board)
        oq, or_ = round(cq), round(cr)
    else:
        oq, or_ = 0, 0

    arr = np.zeros((3, size, size), dtype=np.float32)
    cp = game.current_player - 1   # 0 or 1
    arr[2, :, :] = cp

    for (q, r), p in game.board.items():
        qi = q - oq + half
        ri = r - or_ + half
        if 0 <= qi < size and 0 <= ri < size:
            arr[p - 1, ri, qi] = 1.0

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

class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(ch)

    def forward(self, x):
        r = F.relu(self.bn1(self.conv1(x)))
        r = self.bn2(self.conv2(r))
        return F.relu(x + r)


class HexNet(nn.Module):
    """
    Shared trunk: IN_CH → hidden conv → n_blocks residual blocks
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
        # We concatenate a compressed trunk map (2ch) with the move plane (1ch)
        self.p_conv = nn.Sequential(
            nn.Conv2d(hidden, 2, 1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(),
        )
        self.p_fc = nn.Sequential(
            nn.Linear(2 * BOARD_SIZE * BOARD_SIZE + BOARD_SIZE * BOARD_SIZE, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def trunk(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.stem(x))

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


if __name__ == "__main__":
    net = HexNet().to(DEVICE)
    print(f"Device: {DEVICE}")
    print(f"Params: {param_count(net):,}")

    # Smoke test
    game = HexGame()
    for coord in [(0,0),(1,0),(0,1),(1,-1),(0,-1),(1,1)]:
        game.make(*coord)

    v, policy = evaluate(net, game)
    print(f"Value: {v:.4f}")
    print(f"Policy entries: {len(policy)}")
    print(f"Sample logits: {list(policy.items())[:3]}")

    # Shape test
    B = 4
    x = torch.randn(B, IN_CH, BOARD_SIZE, BOARD_SIZE, device=DEVICE)
    m = torch.zeros(B, 1, BOARD_SIZE, BOARD_SIZE, device=DEVICE)
    m[:, 0, 7, 7] = 1.0
    val, pol = net(x, m)
    print(f"Batch value shape: {val.shape}  policy shape: {pol.shape}")
    print("net.py OK")
