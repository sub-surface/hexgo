"""
ELO rating system for hexgo agents.

- K=32, standard chess formula
- Agents registered by name+config string
- Results persisted to elo.json after every game
- run_match() plays N games between two agents, alternating colour
"""

import json
import math
import random
import time
from pathlib import Path
from typing import Protocol

from game import HexGame

ELO_FILE = Path(__file__).parent / "elo.json"
K = 32
DEFAULT_RATING = 1200.0


# ── Agent protocol ────────────────────────────────────────────────────────────

class Agent(Protocol):
    name: str
    def choose_move(self, game: HexGame) -> tuple[int, int]: ...


# ── Concrete agents ───────────────────────────────────────────────────────────

class RandomAgent:
    def __init__(self):
        self.name = "random"

    def choose_move(self, game: HexGame) -> tuple[int, int]:
        return random.choice(game.legal_moves())


class EisensteinGreedyAgent:
    """
    Algorithmic agent grounded in the Eisenstein integer (Z[omega]) ontology.

    Scores each candidate move by the maximum chain it would create or block
    along any of the three unit-direction axes of Z[omega]:
        u1 = (1,0)   q-axis
        u2 = (0,1)   r-axis
        u3 = (1,-1)  diagonal axis

    No lookahead, no learned weights — pure greedy chain extension.
    Used as a curriculum adversary (forces the net to beat structured play)
    and as a permanent ELO baseline to measure progress against a known ontology.

    defensive=False : maximise own chain length only
    defensive=True  : max(own extension, blocking opponent's best chain)
    """
    _AXES = ((1, 0), (0, 1), (1, -1))

    def __init__(self, name: str = "eisenstein_greedy", defensive: bool = False):
        self.name = name
        self.defensive = defensive

    def choose_move(self, game: HexGame) -> tuple[int, int]:
        player   = game.current_player
        opponent = 3 - player
        best_move, best_score = None, -1

        for q, r in game.legal_moves():
            own   = self._chain_if_placed(game, q, r, player)
            block = self._chain_if_placed(game, q, r, opponent) if self.defensive else 0
            score = max(own, block)
            if score > best_score:
                best_score, best_move = score, (q, r)

        return best_move or random.choice(game.legal_moves())

    def _chain_if_placed(self, game: HexGame, q: int, r: int, player: int) -> int:
        """Longest run player would have along any Z[omega] axis if placed at (q,r)."""
        best = 1
        for dq, dr in self._AXES:
            count = 1
            for sign in (1, -1):
                nq, nr = q + sign * dq, r + sign * dr
                while game.board.get((nq, nr)) == player:
                    count += 1
                    nq += sign * dq
                    nr += sign * dr
            best = max(best, count)
        return best


class MCTSAgent:
    def __init__(self, sims: int = 100):
        self.sims = sims
        self.name = f"mcts_{sims}"

    def choose_move(self, game: HexGame) -> tuple[int, int]:
        from mcts import mcts
        return mcts(game, self.sims)


class NetAgent:
    """MCTS agent using a trained neural net for policy+value."""
    def __init__(self, net, sims: int = 100, name: str = "net"):
        self.net = net
        self.sims = sims
        self.name = name

    def choose_move(self, game: HexGame) -> tuple[int, int]:
        from mcts import mcts_with_net
        return mcts_with_net(game, self.net, self.sims)


# ── ELO tracker ───────────────────────────────────────────────────────────────

class ELO:
    def __init__(self):
        self.ratings: dict[str, float] = {}
        self.history: list[dict] = []
        self._load()

    def _load(self):
        if ELO_FILE.exists():
            data = json.loads(ELO_FILE.read_text())
            self.ratings = data.get("ratings", {})
            self.history = data.get("history", [])

    def save(self):
        ELO_FILE.write_text(json.dumps(
            {"ratings": self.ratings, "history": self.history}, indent=2))

    def rating(self, name: str) -> float:
        return self.ratings.get(name, DEFAULT_RATING)

    def expected(self, a: str, b: str) -> float:
        return 1 / (1 + 10 ** ((self.rating(b) - self.rating(a)) / 400))

    def update(self, winner: str, loser: str, draw: bool = False):
        ra, rb = self.rating(winner), self.rating(loser)
        ea = self.expected(winner, loser)
        sa = 0.5 if draw else 1.0
        self.ratings[winner] = ra + K * (sa - ea)
        self.ratings[loser]  = rb + K * ((1 - sa) - (1 - ea))

    def record(self, a: str, b: str, winner_name: str | None, moves: int, duration: float):
        draw = winner_name is None
        if not draw:
            loser = b if winner_name == a else a
            self.update(winner_name, loser, draw=False)
        else:
            self.update(a, b, draw=True)
        self.history.append({
            "a": a, "b": b, "winner": winner_name,
            "moves": moves, "duration": round(duration, 2),
            "ratings": {a: round(self.rating(a), 1), b: round(self.rating(b), 1)},
        })
        self.save()

    def leaderboard(self) -> list[tuple[str, float]]:
        return sorted(self.ratings.items(), key=lambda x: -x[1])


# ── Match runner ──────────────────────────────────────────────────────────────

def play_game(agent1: Agent, agent2: Agent, max_moves: int = 300) -> dict:
    """
    Play one game. agent1=player1(X), agent2=player2(O).
    Returns {winner_name, moves, duration}.
    """
    game = HexGame()
    agents = {1: agent1, 2: agent2}
    t0 = time.perf_counter()
    move_count = 0

    while game.winner is None and move_count < max_moves:
        legal = game.legal_moves()
        if not legal:
            break
        agent = agents[game.current_player]
        move = agent.choose_move(game)
        game.make(*move)
        move_count += 1

    dur = time.perf_counter() - t0
    winner_name = None
    if game.winner == 1:
        winner_name = agent1.name
    elif game.winner == 2:
        winner_name = agent2.name

    return {"winner_name": winner_name, "moves": move_count, "duration": dur}


def run_match(agent_a: Agent, agent_b: Agent, n_games: int = 10,
              elo: ELO | None = None, verbose: bool = True) -> dict:
    """
    Play n_games between agent_a and agent_b, alternating colours.
    Returns summary {wins_a, wins_b, draws, ratings}.
    """
    if elo is None:
        elo = ELO()
    wins = {agent_a.name: 0, agent_b.name: 0, None: 0}

    for i in range(n_games):
        # Alternate who plays X
        if i % 2 == 0:
            p1, p2 = agent_a, agent_b
        else:
            p1, p2 = agent_b, agent_a

        result = play_game(p1, p2)
        wn = result["winner_name"]
        wins[wn] += 1
        elo.record(agent_a.name, agent_b.name, wn, result["moves"], result["duration"])

        if verbose:
            print(f"  G{i+1:02d} winner={wn or 'draw':12s} "
                  f"moves={result['moves']:3d} {result['duration']:.1f}s  "
                  f"ELO {agent_a.name}={elo.rating(agent_a.name):.0f} "
                  f"{agent_b.name}={elo.rating(agent_b.name):.0f}")

    return {
        f"wins_{agent_a.name}": wins[agent_a.name],
        f"wins_{agent_b.name}": wins[agent_b.name],
        "draws": wins[None],
        "ratings": elo.leaderboard(),
    }


if __name__ == "__main__":
    print("Running: random vs mcts_50 (4 games)\n")
    result = run_match(RandomAgent(), MCTSAgent(50), n_games=4)
    print("\nLeaderboard:")
    for name, r in result["ratings"]:
        print(f"  {name:20s} {r:.0f}")
