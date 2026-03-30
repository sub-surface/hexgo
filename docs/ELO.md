# HexGo — ELO System (`elo.py`)

## Agents

| Agent | Description | Used for |
|-------|-------------|----------|
| `EisensteinGreedyAgent` | Greedy Z[ω] chain maximizer. Zero params, zero learning. `defensive=True` also blocks opponent's best chain. | Curriculum adversary (1 in 5 self-play games) + permanent ELO baseline |
| `MCTSAgent(sims)` | Pure rollout MCTS, no net | ELO baseline (`mcts_50`) |
| `NetAgent(net, sims)` | `mcts_with_net` wrapper | ELO evaluation of trained net |
| `RandomAgent` | Uniform random legal move | Sanity baseline |

### EisensteinGreedyAgent

Scores each candidate move by the maximum consecutive chain it would create (or block) along any of the three Z[ω] axes. `defensive=True` takes `max(own_chain, block_chain)`.

Approximates the Erdős-Selfridge potential: ∑ 2^(−|L|) for incomplete lines. A bot that always extends its longest chain (or blocks the opponent's) follows the spirit of the optimal draw strategy for the second player.

Used as both:
1. **Curriculum adversary**: forces the net to beat structured play, not just random
2. **Permanent ELO anchor**: named `eisenstein_def` in `elo.json`, rating persists across training runs

---

## ELO Mechanics

- Standard formula, K=32
- `DEFAULT_RATING = 1200`
- Updated after every game, saved to `elo.json` after every game
- `run_match(a, b, n_games)` alternates colors (even games: a=P1, odd: a=P2)

### Issues

**K=32 is too high** for established agents. At K=32 with N=10 games, a single match can swing ratings by up to 160 points. Standard deviation of win rate at N=10 is ~16%. Ratings random-walk rather than converge. Once an agent has played >20 games, K should drop to 16 or lower.

**N=10 games per match** is statistically insufficient. A 6-4 result has a 95% CI of [0.26, 0.74] on true win rate. At minimum N=30 is needed for <10% uncertainty.

**max_moves=300 timeouts** are silently recorded as draws in `elo.json`. Many early games reach 300 moves with no winner because MCTS at 25-50 sims can't find wins quickly enough. These fake draws pollute the rating history.

---

## `NetAgent` Known Bug

`NetAgent.choose_move` calls `mcts_with_net(game, self.net, self.sims)` directly
(no `InferenceServer`, no ZOI pruning). The training self-play path uses `mcts_policy`
via `InferenceServer` with ZOI pruning. This mismatch means:

1. ELO matches are slower (synchronous single-threaded forward passes)
2. The net evaluates the full candidate set during ELO but was trained on ZOI-pruned candidates — ELO systematically underestimates the net's training-distribution quality

Additionally, `mcts_with_net` creates leaf children without `player=game.current_player`
(defaults to `player=1`), corrupting backpropagation for Player 2 leaf expansions.
All NetAgent ELO ratings are affected by this bug.

---

## `elo.json` Format

```json
{
  "ratings": {
    "eisenstein_def": 1403.1,
    "net_gen0001": 1187.2,
    "mcts_50": 1200.0
  },
  "history": [
    {
      "a": "net_gen0001", "b": "eisenstein_def",
      "winner": "eisenstein_def",
      "moves": 73, "duration": 4.2,
      "ratings": {"net_gen0001": 1187.2, "eisenstein_def": 1403.1}
    }
  ]
}
```

Net agents are named `net_gen{N:04d}` — each generation gets a fresh rating entry starting at 1200. The `eisenstein_def` entry accumulates across all training runs.

---

## `run_match` Return Value

```python
{
    f"wins_{agent_a.name}": int,
    f"wins_{agent_b.name}": int,
    "draws": int,
    "ratings": [(name, rating), ...]  # sorted descending
}
```

Dynamic keys — callers must know agent names to read win counts.

---

## Known Issues

| Issue | Severity |
|-------|----------|
| `mcts_with_net` leaf children `player=1` default — corrupts NetAgent ELO | Critical |
| K=32 too high for established agents — ratings random-walk | Important |
| Timeout games (max_moves=300) recorded as draws | Important |
| N=10 games — statistically insufficient | Important |
| Training (ZOI) vs eval (full legal moves) mismatch — ELO understates net quality | Important |
| Dynamic return dict keys break typed callers | Moderate |
