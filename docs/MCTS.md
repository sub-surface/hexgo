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

**Fixed (2026-03-30)**: leaf children now use `player=game.current_player`.

**Fixed (2026-03-30)**: terminal sign — `v = 1.0 if game.winner == node.player else -1.0`.

**Fixed (2026-03-30)**: leaf value perspective — `evaluate()` returns value from
`game.current_player`'s POV; if `node.player != game.current_player`, value is negated
to align with the backprop convention (`value` from `node.player`'s perspective).

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

1. **DEAD CODE**: In `mcts()`, `v = 0.0 if game.winner is not None` at line
   ~124 is immediately overwritten by the terminal check below it. (Low priority.)

Previously listed bugs FIX-4 and FIX-5 have been resolved as of 2026-03-30.
