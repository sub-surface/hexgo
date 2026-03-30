"""
Replay a saved game from replays/*.json in the terminal.
Usage: python replay.py replays/game_first_gen0001_*.json [--delay 0.3]
"""

import argparse
import json
import math
import time
from pathlib import Path

SYMBOLS = {1: "X", 2: "O"}
DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1))


def render(board: dict, last_move: tuple | None, move_num: int, total: int):
    if not board:
        print("(empty)")
        return
    qs = [q for q, r in board]
    rs = [r for q, r in board]
    q_min, q_max = min(qs) - 1, max(qs) + 1
    r_min, r_max = min(rs) - 1, max(rs) + 1

    print(f"\n── Move {move_num}/{total} ──")
    for r in range(r_min, r_max + 1):
        indent = " " * (r - r_min)
        row = []
        for q in range(q_min, q_max + 1):
            p = board.get((q, r))
            sym = SYMBOLS.get(p, "·")
            if (q, r) == last_move:
                sym = f"[{sym}]"
            else:
                sym = f" {sym} "
            row.append(sym)
        print(indent + "".join(row))


def replay(path: str, delay: float = 0.4):
    data = json.loads(Path(path).read_text())
    moves = [(m[0], m[1]) for m in data["moves"]]
    winner = data["winner"]
    gen = data.get("gen", "?")
    label = data.get("label", "")

    print(f"Replaying: {Path(path).name}")
    print(f"Gen {gen} | {label} | {len(moves)} moves | winner: {'X' if winner==1 else 'O' if winner==2 else 'draw'}")

    board: dict[tuple, int] = {}
    current = 1
    placements = 0

    for i, (q, r) in enumerate(moves):
        board[(q, r)] = current
        render(board, (q, r), i + 1, len(moves))
        
        placements += 1
        is_first_turn = (i < 1) # first move only
        limit = 1 if is_first_turn else 2
        
        if placements >= limit:
            current = 3 - current
            placements = 0
            
        if delay > 0:
            time.sleep(delay)

    print(f"\nGame over — Winner: {'X' if winner==1 else 'O' if winner==2 else 'Draw'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="Path to replay JSON")
    parser.add_argument("--delay", type=float, default=0.3, help="Seconds between moves")
    args = parser.parse_args()
    replay(args.file, args.delay)
