"""
play.py — Play against a HexGo checkpoint in the terminal.

Usage:
    python play.py                          # latest checkpoint, 100 sims
    python play.py --model net_gen0100.pt   # specific checkpoint
    python play.py --sims 200              # stronger bot (slower)
    python play.py --player 2              # play as O (bot goes first)

Controls:
    Enter moves as: q r  (e.g. "0 0" or "3 -1")
    Type "undo" to take back your last move + bot's response
    Type "quit" to exit
"""

import argparse
import sys
from pathlib import Path

import torch
from game import HexGame
from net import HexNet, DEVICE
from mcts import mcts_with_net
from elo import EisensteinGreedyAgent

CHECKPOINT_DIR = Path("checkpoints")

# ANSI colors
RESET = "\033[0m"
RED   = "\033[91m"
BLUE  = "\033[94m"
DIM   = "\033[90m"
BOLD  = "\033[1m"
WHITE = "\033[97m"
CYAN  = "\033[96m"
YELLOW = "\033[93m"


def render_board(game, last_move=None):
    """Render the hex board to terminal with colors."""
    if not game.board and not game.move_history:
        print(f"\n  {DIM}Empty board. First move must be (0, 0).{RESET}\n")
        return

    all_coords = set(game.board.keys())
    # Add neighbors for context
    for q, r in list(all_coords):
        for dq, dr in [(1,0),(0,1),(1,-1),(-1,0),(0,-1),(-1,1)]:
            all_coords.add((q+dq, r+dr))

    if not all_coords:
        return

    qs = [q for q, r in all_coords]
    rs = [r for q, r in all_coords]
    q_min, q_max = min(qs), max(qs)
    r_min, r_max = min(rs), max(rs)

    print()
    # Header
    print(f"  {DIM}{'q →':>6}", end="")
    for q in range(q_min, q_max + 1):
        print(f"{q:>4}", end="")
    print(RESET)

    for r in range(r_min, r_max + 1):
        # Offset for hex stagger
        indent = "  " * (r - r_min)
        print(f"  {DIM}r={r:<3}{RESET} {indent}", end="")
        for q in range(q_min, q_max + 1):
            p = game.board.get((q, r))
            is_last = last_move and last_move == (q, r)
            if p == 1:
                marker = f"{RED}{'[X]' if is_last else ' X '}{RESET}"
            elif p == 2:
                marker = f"{BLUE}{'[O]' if is_last else ' O '}{RESET}"
            else:
                marker = f"{DIM} . {RESET}"
            print(marker, end=" ")
        print()
    print()


def get_human_move(game):
    """Get a valid move from the human player."""
    legal = set(game.legal_moves())
    while True:
        try:
            raw = input(f"  {CYAN}Your move (q r): {RESET}").strip().lower()
            if raw in ("quit", "exit", "q"):
                return "quit"
            if raw == "undo":
                return "undo"
            parts = raw.replace(",", " ").split()
            if len(parts) != 2:
                print(f"  {YELLOW}Enter two numbers: q r{RESET}")
                continue
            q, r = int(parts[0]), int(parts[1])
            if (q, r) not in legal:
                print(f"  {YELLOW}({q}, {r}) is not a legal move.{RESET}")
                # Show nearby legal moves
                nearby = [(lq, lr) for lq, lr in legal
                          if abs(lq - q) <= 2 and abs(lr - r) <= 2]
                if nearby:
                    moves_str = ", ".join(f"({lq},{lr})" for lq, lr in sorted(nearby)[:8])
                    print(f"  {DIM}Nearby legal: {moves_str}{RESET}")
                continue
            return (q, r)
        except (ValueError, EOFError):
            print(f"  {YELLOW}Enter two integers: q r{RESET}")


def main():
    parser = argparse.ArgumentParser(description="Play against HexGo bot")
    parser.add_argument("--model", type=str, default=None,
                        help="Checkpoint filename (default: net_latest.pt)")
    parser.add_argument("--sims", type=int, default=100,
                        help="MCTS sims per move (higher=stronger, slower)")
    parser.add_argument("--player", type=int, default=1, choices=[1, 2],
                        help="Play as player 1 (X, first) or 2 (O, second)")
    args = parser.parse_args()

    # Load model
    if args.model:
        path = CHECKPOINT_DIR / args.model
    else:
        path = CHECKPOINT_DIR / "net_latest.pt"

    if not path.exists():
        print(f"Checkpoint not found: {path}")
        sys.exit(1)

    net = HexNet().to(DEVICE)
    net.load_state_dict(torch.load(path, map_location=DEVICE))
    net.eval()
    print(f"\n  {BOLD}{'='*50}{RESET}")
    print(f"  {BOLD}  HexGo — Hexagonal Connect-6{RESET}")
    print(f"  {BOLD}{'='*50}{RESET}")
    print(f"  {DIM}Model: {path.name}{RESET}")
    print(f"  {DIM}Sims:  {args.sims}{RESET}")
    print(f"  {DIM}Device: {DEVICE}{RESET}")
    print(f"  {DIM}You are: {'X (Player 1)' if args.player == 1 else 'O (Player 2)'}{RESET}")
    print(f"  {DIM}Win: 6 in a row along any hex axis{RESET}")
    print(f"  {DIM}Turn rule: P1 places 1, then both place 2 per turn{RESET}")
    print(f"  {DIM}Commands: q r | undo | quit{RESET}")
    print()

    game = HexGame()
    human_player = args.player
    bot_player = 3 - human_player
    last_move = None
    undo_stack = []  # (n_moves_to_undo,) for each "turn"

    while game.winner is None:
        legal = game.legal_moves()
        if not legal:
            print(f"  {DIM}No legal moves — game over.{RESET}")
            break

        current = game.current_player
        render_board(game, last_move)

        if current == human_player:
            # Human's turn
            is_first_move = len(game.move_history) == 0
            placements = 1 if is_first_move else 2
            remaining = placements - game.placements_in_turn

            print(f"  {WHITE}Your turn ({RED if human_player==1 else BLUE}"
                  f"{'X' if human_player==1 else 'O'}{WHITE}) — "
                  f"{remaining} placement{'s' if remaining > 1 else ''} left{RESET}")

            moves_this_turn = 0
            while remaining > 0 and game.winner is None:
                move = get_human_move(game)
                if move == "quit":
                    print(f"\n  {DIM}Thanks for playing!{RESET}\n")
                    return
                if move == "undo":
                    if not undo_stack:
                        print(f"  {YELLOW}Nothing to undo.{RESET}")
                        continue
                    n_undo = undo_stack.pop()
                    for _ in range(n_undo):
                        game.unmake()
                    last_move = game.move_history[-1] if game.move_history else None
                    print(f"  {DIM}Undid {n_undo} moves.{RESET}")
                    render_board(game, last_move)
                    break
                game.make(*move)
                last_move = move
                moves_this_turn += 1
                remaining -= 1
                if game.winner is not None:
                    break
                if remaining > 0:
                    render_board(game, last_move)
                    print(f"  {WHITE}Place again — {remaining} left{RESET}")

            if move == "undo":
                continue
            undo_stack.append(moves_this_turn)

        else:
            # Bot's turn
            is_first_move = len(game.move_history) == 0
            placements = 1 if is_first_move else 2
            remaining = placements - game.placements_in_turn

            print(f"  {WHITE}Bot thinking ({RED if bot_player==1 else BLUE}"
                  f"{'X' if bot_player==1 else 'O'}{WHITE})...{RESET}", end="", flush=True)

            bot_moves = 0
            while remaining > 0 and game.winner is None:
                move = mcts_with_net(game, net, args.sims)
                game.make(*move)
                last_move = move
                bot_moves += 1
                remaining -= 1
                if bot_moves == 1:
                    print(f" {move}", end="", flush=True)
                else:
                    print(f", {move}", end="", flush=True)

            print()
            undo_stack.append(bot_moves)

    # Game over
    render_board(game, last_move)
    if game.winner == human_player:
        print(f"  {BOLD}{CYAN}You win!{RESET} \n")
    elif game.winner == bot_player:
        print(f"  {BOLD}{RED}Bot wins.{RESET} Better luck next time!\n")
    else:
        print(f"  {BOLD}{YELLOW}Draw.{RESET}\n")

    print(f"  {DIM}Moves played: {len(game.move_history)}{RESET}\n")


if __name__ == "__main__":
    main()
