"""
ASCII render of a HexGame board for debugging.
Uses offset coordinates to print a readable grid.
"""

from game import HexGame


SYMBOLS = {1: "X", 2: "O", None: "."}


def render(game: HexGame, pad: int = 2):
    if not game.board:
        print("(empty board)")
        return

    qs = [q for q, r in game.board]
    rs = [r for q, r in game.board]
    q_min, q_max = min(qs) - pad, max(qs) + pad
    r_min, r_max = min(rs) - pad, max(rs) + pad

    for r in range(r_min, r_max + 1):
        indent = " " * abs(r)  # hex offset
        row = []
        for q in range(q_min, q_max + 1):
            p = game.board.get((q, r))
            row.append(SYMBOLS.get(p, "?"))
        print(indent + " ".join(row))

    status = f"Winner: {game.winner}" if game.winner else f"Player {game.current_player} to move"
    print(f"Moves: {len(game.move_history)}  |  {status}\n")


if __name__ == "__main__":
    from mcts import self_play_game
    import random

    random.seed(42)
    print("Running a self-play game (100 sims/move)...\n")
    result = self_play_game(num_simulations=100)
    print(f"Game over — Winner: {result['winner']}  Moves: {result['num_moves']}")
    print(f"Move sequence: {result['moves'][:10]}{'...' if result['num_moves'] > 10 else ''}")
