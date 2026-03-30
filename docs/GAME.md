# HexGo — Game Engine (`game.py`)

## Mathematical Substrate

HexGo is **AP-6 Maker-Maker on Z[ω]** — the first player to form an
arithmetic progression of length 6 in the Eisenstein integer ring wins.

The hex grid with axial coordinates (q, r) is isomorphic to Z[ω] where
ω = e^(2πi/3). Each cell maps to q + r·ω ∈ Z[ω]. The three win axes are
the unit directions of Z[ω]:

| Axis | Direction | Z[ω] unit |
|------|-----------|-----------|
| u1   | (1, 0)    | 1         |
| u2   | (0, 1)    | ω         |
| u3   | (1, -1)   | ω² = −1−ω |

A win is {z, z+u, z+2u, z+3u, z+4u, z+5u} for z ∈ Z[ω], u ∈ {u1, u2, u3}.

### Combinatoric connections

- **Van der Waerden W(6;2) = 1132**: any 2-colouring of {1…1132} contains
  a monochromatic AP-6. Hales-Jewett extends this to the grid.
- **Erdős-Selfridge potential**: ∑ 2^(−|L|) < 1 over all incomplete lines L
  guarantees a draw strategy for the second player. `EisensteinGreedyAgent`
  approximates this potential on the Z[ω] lattice.
- **Strategy-stealing**: the game cannot be a second-player win.
- **First-mover balance**: the 1-stone first move (Connect6 rule) removes the
  obvious first-player advantage.

---

## Rules: The 1-2-2 Turn Mechanic

Six-in-a-row on an infinite hexagonal grid using Connect6 turns:

- **Turn 1:** Player 1 places **one** stone.
- **Turn 2+:** Each player places **two** stones per turn.
- Win is checked after **every** individual stone placement. If a win occurs
  on the first of a player's two stones, the turn ends immediately.

Implementation in `game.py`:
- `placements_in_turn` tracks how many stones the current player has placed.
- `limit = 1` when `len(move_history) == 1` (first move ever), else `limit = 2`.
- Player flips when `placements_in_turn >= limit`.

### Complexity

Practical games run 60–250 moves. Branching factor ≈ 30 (ZOI-pruned candidates),
depth ≈ 150 → effective complexity closer to 9×9 Go.

---

## API

### `HexGame`

```python
game = HexGame()
game.make(q, r)     # place current player's stone; returns False if illegal
game.unmake()       # undo last placement (MCTS tree traversal)
game.legal_moves()  # O(1): returns list of candidate cells
game.zoi_moves(margin=6)  # ZOI-pruned candidates within margin hex-steps of last 8 pieces
game.clone()        # deep copy for ELO match isolation
game.current_player # 1 or 2
game.winner         # 1, 2, or None
game.move_history   # all placements in order (both players)
```

### Internal state

| Attribute | Description |
|-----------|-------------|
| `board` | `dict[(q,r) → player]` sparse stone map |
| `candidates` | `set[(q,r)]` legal moves, updated incrementally |
| `placements_in_turn` | stones placed so far in current player's turn |
| `_undo` | stack of `(move, removed_cands, added_cands, prev_placements, prev_winner, prev_player)` |

### Make/Unmake

`make()` pushes a full undo entry before mutating state.
`unmake()` restores all five fields atomically — safe for MCTS tree traversal.

### Candidates

Maintained incrementally: when a stone is placed at (q,r), the cell is
removed from candidates and its 6 hex neighbours are added if unoccupied.
`legal_moves()` is O(1) with no board scan.

### Win detection

`_check_win(q, r)` scans each of the three axes through the just-placed piece,
counting consecutive same-player stones. O(WIN_LENGTH=6) per call.

### ZOI pruning

`zoi_moves(margin)` restricts candidates to cells within `margin` hex steps of
the last `min(8, len(move_history))` placed pieces. Falls back to
`legal_moves()` when ZOI covers all candidates. Prevents exploring stale
candidates from early play in long games.

---

## Known Issues

- **POTENTIAL**: ZOI `lookback=8` window can miss early threats in very long
  games (>50 moves per player). The first pieces placed may fall outside the
  ZOI and still contain live open-ended threats. `margin=6` mitigates this;
  research recommends minimum `margin=3` for correctness, but `margin=6` is
  safer.
- **DESIGN NOTE**: `assert WIN_LENGTH == 6` in `_check_win` ensures the
  constant is not accidentally changed without updating the win check logic.
