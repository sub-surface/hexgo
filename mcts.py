"""
MCTS for self-play — optimised for speed.

Two modes:
  mcts(game, sims)              — pure rollout (no net)
  mcts_with_net(game, net, sims) — AlphaZero style: net value replaces rollout,
                                   net policy logits set node priors,
                                   Dirichlet noise added to root for exploration
"""

import math
import random
import numpy as np
from game import HexGame

C_PUCT = 1.4


class Node:
    __slots__ = ("move", "parent", "children", "visits", "value", "prior", "player")

    def __init__(self, move=None, parent=None, prior: float = 1.0, player: int = 1):
        self.move = move        # (q, r) that led here; None for root
        self.parent: "Node | None" = parent
        self.children: list["Node"] = []
        self.visits: int = 0
        self.value: float = 0.0
        self.prior: float = prior
        self.player: int = player

    def ucb(self) -> float:
        if self.visits == 0:
            return float("inf")
        q = self.value / self.visits
        u = C_PUCT * self.prior * math.sqrt(self.parent.visits) / (1 + self.visits)
        return q + u

    def best_child(self) -> "Node":
        return max(self.children, key=lambda c: c.ucb())

    def best_move(self) -> tuple[int, int]:
        return max(self.children, key=lambda c: c.visits).move


def _rollout(game: HexGame) -> float:
    """
    Random playout in-place using make/unmake.
    Returns +1 if the player-to-move at rollout start wins, -1 if they lose.
    """
    start_player = game.current_player
    depth = 0
    max_moves = 150

    while game.winner is None and depth < max_moves:
        moves = game.legal_moves()
        if not moves:
            break
        game.make(*random.choice(moves))
        depth += 1

    result = 0.0 if game.winner is None else (1.0 if game.winner == start_player else -1.0)

    # Unwind
    for _ in range(depth):
        game.unmake()

    return result


def _expand(node: Node, game: HexGame):
    moves = game.legal_moves()
    prior = 1.0 / max(len(moves), 1)
    # Each child node reflects the player who is about to move from THAT state
    node.children = [Node(move=m, parent=node, prior=prior, player=game.current_player)
                     for m in moves]


def _backprop(node: Node, value: float):
    """
    Backpropagate value: +1 for player who just moved if they win.
    value is from perspective of the player to move at the terminal/leaf state.
    """
    while node is not None:
        node.visits += 1
        node.value += value
        
        if node.parent:
            # If parent has a different player, negate value for parent's perspective
            if node.parent.player != node.player:
                value = -value
        node = node.parent


def mcts(game: HexGame, num_simulations: int = 200) -> tuple[int, int]:
    """
    Run MCTS on `game` (mutates and restores it via make/unmake).
    Returns the best move as (q, r).
    """
    root = Node(player=game.current_player)
    _expand(root, game)

    for _ in range(num_simulations):
        node = root
        depth = 0

        # Selection — walk down tree, making moves on the shared game
        while node.children and game.winner is None:
            node = node.best_child()
            game.make(*node.move)
            depth += 1

        # Expansion + simulation
        if game.winner is None:
            _expand(node, game)
            if node.children:
                node = random.choice(node.children)
                game.make(*node.move)
                depth += 1

        v = 0.0 if game.winner is not None else _rollout(game)
        if game.winner is not None:
            # Terminal: value from the perspective of the node that just moved
            v = -1.0  # whoever is to move now, the previous player won

        # Restore game state
        for _ in range(depth):
            game.unmake()

        _backprop(node, v)

    return root.best_move()


def mcts_with_net(game: HexGame, net, num_simulations: int = 100,
                  dirichlet_alpha: float = 0.3, dirichlet_eps: float = 0.25
                  ) -> tuple[int, int]:
    """
    AlphaZero-style MCTS using HexNet for value + policy priors.
    No rollout — net value is used directly at leaf nodes.
    Dirichlet noise added to root priors for exploration.
    """
    from net import evaluate   # late import to avoid circular at module level

    root = Node(player=game.current_player)

    # Expand root with net priors
    value, policy = evaluate(net, game)
    moves = game.legal_moves()
    if not moves:
        raise RuntimeError("mcts_with_net called on terminal/empty game")

    # Softmax priors from logits
    logits = np.array([policy.get(m, 0.0) for m in moves], dtype=np.float32)
    logits -= logits.max()
    priors = np.exp(logits)
    priors /= priors.sum()

    # Dirichlet noise at root
    noise = np.random.dirichlet([dirichlet_alpha] * len(moves))
    priors = (1 - dirichlet_eps) * priors + dirichlet_eps * noise

    root.children = [Node(move=m, parent=root, prior=float(p))
                     for m, p in zip(moves, priors)]

    for _ in range(num_simulations):
        node = root
        depth = 0

        # Selection
        while node.children and game.winner is None:
            node = node.best_child()
            game.make(*node.move)
            depth += 1

        # Leaf evaluation
        if game.winner is not None:
            v = -1.0   # current player lost (previous player just won)
        else:
            # Expand with net
            v, leaf_policy = evaluate(net, game)
            leaf_moves = game.legal_moves()
            if leaf_moves:
                llogits = np.array([leaf_policy.get(m, 0.0) for m in leaf_moves], dtype=np.float32)
                llogits -= llogits.max()
                lpriors = np.exp(llogits)
                lpriors /= lpriors.sum()
                node.children = [Node(move=m, parent=node, prior=float(p))
                                  for m, p in zip(leaf_moves, lpriors)]

        for _ in range(depth):
            game.unmake()

        _backprop(node, v)

    return root.best_move()


def self_play_game(num_simulations: int = 100, callback=None) -> dict:
    """
    Play a complete game via MCTS self-play.
    callback(game, move) is called after each move if provided — use for UI updates.
    Returns {winner, num_moves, moves}.
    """
    game = HexGame()
    moves = []

    while game.winner is None:
        if not game.legal_moves():
            break
        move = mcts(game, num_simulations)
        game.make(*move)
        moves.append(move)
        if callback:
            callback(game, move)

    return {"winner": game.winner, "num_moves": len(moves), "moves": moves, "game": game}
