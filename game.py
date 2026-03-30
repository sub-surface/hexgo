"""
Hexagonal Connect6 — 6-in-a-row wins on an infinite hex grid.

Coordinate system: axial (q, r) — flat-top hexagons.
Neighbors of (q,r): (±1,0), (0,±1), (1,-1), (-1,1)

Optimisations:
- `candidates` set maintained incrementally — no full-board scan for legal_moves()
- `make()`/`unmake()` for zero-copy tree traversal in MCTS
- win check only walks the 3 axes through the last-placed piece
- `move_history` stores (q, r); `player_history` stores the player who made each move
"""

WIN_LENGTH = 6

# Six axial directions; first 3 are the unique axes (each covers both directions)
DIRS = ((1, 0), (0, 1), (1, -1), (-1, 0), (0, -1), (-1, 1))
AXES = DIRS[:3]


def _hex_dist(q1: int, r1: int, q2: int, r2: int) -> int:
    """Hex grid distance in axial coordinates."""
    return (abs(q1 - q2) + abs(r1 - r2) + abs((q1 + r1) - (q2 + r2))) // 2


class HexGame:
    def __init__(self):
        self.board: dict[tuple[int, int], int] = {}
        self.candidates: set[tuple[int, int]] = {(0, 0)}  # empty cells adj to pieces
        self.current_player: int = 1
        self.placements_in_turn: int = 0  # how many tiles placed in current turn
        self.winner: int | None = None
        self.move_history: list[tuple[int, int]] = []
        self.player_history: list[int] = []  # player who made each move in move_history
        # undo stack: each entry = (move, removed_candidates, added_candidates, prev_placements, prev_winner, prev_player)
        self._undo: list[tuple] = []

    # ------------------------------------------------------------------
    # Core move interface
    # ------------------------------------------------------------------

    def make(self, q: int, r: int) -> bool:
        """Place current player's piece. Returns True if legal."""
        if q is None or r is None:
            return False
        if self.winner or (q, r) in self.board:
            return False

        self.board[(q, r)] = self.current_player
        self.move_history.append((q, r))
        self.player_history.append(self.current_player)

        # Update candidates incrementally
        removed = set()
        added = set()
        if (q, r) in self.candidates:
            self.candidates.discard((q, r))
            removed.add((q, r))
        for dq, dr in DIRS:
            nb = (q + dq, r + dr)
            if nb not in self.board and nb not in self.candidates:
                self.candidates.add(nb)
                added.add(nb)

        won = self._check_win(q, r)
        prev_winner = self.winner
        if won:
            self.winner = self.current_player

        prev_placements = self.placements_in_turn
        prev_player = self.current_player
        
        self.placements_in_turn += 1
        if not won:
            # First move of game (len=0): player 1 gets 1 placement
            # Subsequent moves: 2 placements
            is_first_move = (len(self.move_history) == 1)
            limit = 1 if is_first_move else 2
            
            if self.placements_in_turn >= limit:
                self.current_player = 3 - self.current_player
                self.placements_in_turn = 0

        self._undo.append((q, r, removed, added, prev_placements, prev_winner, prev_player))
        return True

    def unmake(self):
        """Undo the last placement."""
        if not self._undo:
            return
        q, r, removed, added, prev_placements, prev_winner, prev_player = self._undo.pop()
        del self.board[(q, r)]
        self.move_history.pop()
        self.player_history.pop()
        self.candidates -= added
        self.candidates |= removed
        self.winner = prev_winner
        self.current_player = prev_player
        self.placements_in_turn = prev_placements

    # keep .play() as alias so existing tests/code still works
    def play(self, q: int, r: int) -> bool:
        return self.make(q, r)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def legal_moves(self) -> list[tuple[int, int]]:
        return list(self.candidates)

    def zoi_moves(self, margin: int = 6) -> list[tuple[int, int]]:
        """
        Zone-of-Interest pruning: return only candidates within `margin` hex
        steps of the last `lookback` placed pieces. Keeps MCTS focused on the
        current play area rather than stale candidates from earlier turns.

        Note: HexGame.candidates is already within 1 step of all pieces, so
        global-ZOI (relative to all pieces) equals candidates. This method
        uses RECENT pieces as the focus — a tighter, tactically motivated
        restriction as games grow long and candidates accumulate.

        Default margin=6 is conservative (covers the WIN_LENGTH); reduce to
        4-5 for faster play once the net is trained enough to not miss threats.
        """
        lookback = 8
        if len(self.move_history) < lookback:
            return list(self.candidates)

        # Focus: cells within margin of the most recently played pieces
        recent = self.move_history[-lookback:]
        within: set[tuple[int, int]] = set()
        for q0, r0 in recent:
            for cq, cr in self.candidates:
                if _hex_dist(q0, r0, cq, cr) <= margin:
                    within.add((cq, cr))

        # Safety: always include candidates adjacent to the very last move
        lq, lr = self.move_history[-1]
        for dq, dr in DIRS:
            nb = (lq + dq, lr + dr)
            if nb in self.candidates:
                within.add(nb)

        return list(within) if len(within) >= 3 else list(self.candidates)

    def _check_win(self, q: int, r: int) -> bool:
        assert WIN_LENGTH == 6, f"WIN_LENGTH corrupted: {WIN_LENGTH}"
        player = self.board[(q, r)]
        for dq, dr in AXES:
            count = 1
            for sign in (1, -1):
                nq, nr = q + sign * dq, r + sign * dr
                while self.board.get((nq, nr)) == player:
                    count += 1
                    nq += sign * dq
                    nr += sign * dr
            if count >= WIN_LENGTH:
                return True
        return False

    def to_tensor(self, size: int = 11) -> list[list[list[int]]]:
        """3 x dim x dim array: [p1 pieces, p2 pieces, current_player plane]."""
        dim = 2 * size + 1
        c0 = [[0] * dim for _ in range(dim)]
        c1 = [[0] * dim for _ in range(dim)]
        cp = self.current_player - 1
        c2 = [[cp] * dim for _ in range(dim)]
        for (q, r), p in self.board.items():
            qi, ri = q + size, r + size
            if 0 <= qi < dim and 0 <= ri < dim:
                (c0 if p == 1 else c1)[ri][qi] = 1
        return [c0, c1, c2]

    def clone(self) -> "HexGame":
        g = HexGame.__new__(HexGame)
        g.board = dict(self.board)
        g.candidates = set(self.candidates)
        g.current_player = self.current_player
        g.placements_in_turn = self.placements_in_turn
        g.winner = self.winner
        g.move_history = list(self.move_history)
        g.player_history = list(self.player_history)
        g._undo = []
        return g

    def __repr__(self) -> str:
        if not self.board:
            return "HexGame(empty)"
        qs = [q for q, r in self.board]
        rs = [r for q, r in self.board]
        return (f"HexGame(moves={len(self.move_history)}, winner={self.winner}, "
                f"q=[{min(qs)},{max(qs)}] r=[{min(rs)},{max(rs)}])")
