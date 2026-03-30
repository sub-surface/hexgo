# HexGo — MCTS (`mcts.py`)

## Overview

Two search modes:

| Mode | Function | Value source | Used by |
|------|----------|-------------|---------|
| Pure rollout | `mcts(game, sims)` | Random playout | `MCTSAgent` in ELO |
| AlphaZero | `mcts_with_net(game, net, sims)` | Net value + policy priors | ELO `NetAgent` only |
| Inline policy | `mcts_policy()` in train.py | Net via `InferenceServer` | Training self-play |

`mcts_policy()` in `train.py` is the primary training path and uses the
`InferenceServer` for batched evaluation. `mcts_with_net` is used only in ELO
evaluation matches.

---

## Node

```python
class Node:
    __slots__ = ("move", "parent", "children", "visits", "value", "prior", "player")
```

`player` stores which player is to move **at** this node (set at creation).

UCB score: `Q + C_PUCT × prior × √(parent.visits) / (1 + visits)`

`C_PUCT` is loaded from `config.CFG["CPUCT"]` at module import time (frozen
for the process lifetime — changes to CFG during autotune require restart).

---

## Multi-Placement Backpropagation

Standard MCTS negates value at every node. HexGo's 1-2-2 turn rule means the
same player moves twice consecutively — negating between their own sub-moves
would be wrong.

**Fix**: value is only negated when `node.parent.player != node.player`.

```python
def _backprop(node, value):
    while node is not None:
        node.visits += 1
        node.value += value
        if node.parent and node.parent.player != node.player:
            value = -value
        node = node.parent
```

---

## Pure Rollout (`mcts`)

1. Expand root with uniform priors.
2. Selection: walk tree via UCB until leaf or terminal.
3. Expand leaf: add children with uniform priors.
4. Simulation: random playout from leaf (max 150 moves).
5. Backprop result via `_backprop`.

Terminal value: `v = -1.0` (whoever is to move lost — previous player won).

---

## AlphaZero (`mcts_with_net`)

Used only by `NetAgent` during ELO evaluation:

1. Evaluate root with net → value + policy logits.
2. Apply Dirichlet noise at root: `prior = (1-ε)·net + ε·Dir(α)`.
3. Selection + leaf expansion with net evaluation.
4. No rollout — net value used directly at leaf.

**Known bug**: leaf children created without `player=game.current_player`
(defaults to `player=1`). This corrupts backpropagation sign for ~50% of
Player 2 leaf expansions. Low priority — ELO uses pure `mcts()` for the
`MCTSAgent` baseline; `mcts_with_net` affects only `NetAgent` ELO matches.

**Terminal sign**: `v = -1.0` at terminal nodes during selection means the
value is attributed from the perspective of the **node that is about to move**
(who lost), not the node that just won. This may cause incorrect backprop sign
for the winning player's node. Needs verification.

---

## Tree Reuse (`mcts_policy` in train.py)

`mcts_policy()` returns `(chosen_move, visit_distribution, legal_moves, new_root)`.
On the next call, if `prev_root` has a child matching the game state, that
subtree is reused with fresh Dirichlet noise applied. Saves approximately
`N_SIMS / branching_factor` simulations per move.

---

## ZOI Integration

`mcts_policy` calls `game.zoi_moves(CFG["ZOI_MARGIN"])` instead of
`game.legal_moves()` to restrict expansion to the active play area.
The fallback to full `legal_moves()` activates automatically when ZOI
coverage equals full candidate set.

---

## Known Issues

1. **BUG (mcts_with_net)**: Leaf children use `player=1` default instead of
   `player=game.current_player`. Fix: add `player=game.current_player` to leaf
   `Node(...)` creation at ~line 191.

2. **POTENTIAL**: Terminal win detection during expansion assigns `v=-1.0` to
   the node about to move. The backprop sign through the winning node should be
   verified — there may be a one-level sign error.

3. **DEAD CODE**: In `mcts()`, `v = 0.0 if game.winner is not None` at line
   ~124 is immediately overwritten by the terminal check below it.
