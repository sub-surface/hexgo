"""
Self-play training loop — AlphaZero style.

Pipeline:
  1. Self-play: net-guided MCTS generates games → replay buffer
  2. Training: sample batches from buffer, minimise policy + value loss
  3. Evaluation: new net vs old net (N games), update if significantly better
  4. Checkpoint: save net + ELO after each generation

Run: python train.py [--gens N] [--sims N] [--games N]

Key design choices:
- Replay buffer capped at BUFFER_CAP positions (FIFO) to avoid stale data
- Each game contributes (board_tensor, move_plane, z) triples where
  z = +1 for winner's moves, -1 for loser's moves (propagated back)
- Policy target = visit count distribution from MCTS root (not argmax)
- No tree reuse between moves (simpler, still trains well)
- ELO updated every generation via run_match()
"""

import argparse
import concurrent.futures
import json
import logging
import random
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from game import HexGame
from inference import InferenceServer
from mcts import Node, _backprop
from net import HexNet, encode_board, encode_move, DEVICE, param_count
from elo import ELO, NetAgent, MCTSAgent, run_match

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler("train.log", mode="a", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("train")

CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)
REPLAY_DIR = Path("replays")
REPLAY_DIR.mkdir(exist_ok=True)

BUFFER_CAP    = 50_000
BATCH_SIZE    = 64      # Reduced because each batch item now includes all legal moves
LR            = 1e-3
WEIGHT_DECAY  = 1e-4
EVAL_GAMES    = 10
NUM_WORKERS   = 4       # Reduced to 4 to save resources
INF_BATCH     = 4       # inference server max batch size (match NUM_WORKERS)
INF_TIMEOUT   = 5       # ms to wait for a full batch before flushing


# ── Game recording ───────────────────────────────────────────────────────────

def save_replay(moves: list, winner, gen: int, label: str):
    """
    Save a game replay as JSON.
    Format: {gen, label, winner, timestamp, moves: [[q,r], ...]}
    Replay with: replay.py <file>
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = REPLAY_DIR / f"game_{label}_gen{gen:04d}_{ts}.json"
    data = {
        "gen": gen, "label": label, "winner": winner,
        "timestamp": ts, "moves": [[q, r] for q, r in moves],
    }
    path.write_text(json.dumps(data, indent=2))
    log.info("Saved replay %s (%d moves, winner=%s)", path.name, len(moves), winner)


# ── MCTS with batched inference ───────────────────────────────────────────────

def mcts_policy(game: HexGame, server: InferenceServer, sims: int,
                temp: float = 1.0) -> tuple[tuple, np.ndarray, list]:
    """
    MCTS guided by the inference server (batched GPU calls).
    Returns (chosen_move, visit_dist, all_moves).
    """
    root = Node(player=game.current_player)
    value, policy = server.evaluate(game)
    moves = game.legal_moves()
    if not moves:
        return None, None, []

    logits = np.array([policy.get(m, 0.0) for m in moves], dtype=np.float32)
    logits -= logits.max()
    priors = np.exp(logits); priors /= priors.sum()

    noise = np.random.dirichlet([0.3] * len(moves))
    priors = 0.75 * priors + 0.25 * noise
    root.children = [Node(move=m, parent=root, prior=float(p), player=game.current_player)
                     for m, p in zip(moves, priors)]

    for _ in range(sims):
        node = root
        depth = 0
        while node.children and game.winner is None:
            node = node.best_child()
            game.make(*node.move)
            depth += 1
        if game.winner is not None:
            v = -1.0
        else:
            v, lp = server.evaluate(game)
            lmoves = game.legal_moves()
            if lmoves:
                ll = np.array([lp.get(m, 0.0) for m in lmoves], dtype=np.float32)
                ll -= ll.max(); lpr = np.exp(ll); lpr /= lpr.sum()
                node.children = [Node(move=m, parent=node, prior=float(p), player=game.current_player)
                                  for m, p in zip(lmoves, lpr)]
        for _ in range(depth):
            game.unmake()
        _backprop(node, v)

    visits = np.array([c.visits for c in root.children], dtype=np.float32)
    if temp == 0:
        dist = np.zeros_like(visits)
        dist[visits.argmax()] = 1.0
    else:
        visits = visits ** (1.0 / temp)
        dist = visits / visits.sum()

    chosen = moves[np.random.choice(len(moves), p=dist)]
    return chosen, dist, moves


def self_play_episode(server: InferenceServer, sims: int, temp_threshold: int = 20
                      ) -> list[tuple]:
    """
    Play one full game, return list of (board_arr, move_plane, outcome_placeholder).
    outcome_placeholder is filled in after game ends.
    """
    game = HexGame()
    positions = []

    move_num = 0
    while game.winner is None:
        legal = game.legal_moves()
        if not legal:
            break
        temp = 1.0 if move_num < temp_threshold else 0.0
        board_arr, (oq, or_) = encode_board(game)
        chosen, dist, moves = mcts_policy(game, server, sims, temp)
        if chosen is None:
            break
        chosen_idx = moves.index(chosen)
        positions.append((board_arr, oq, or_, chosen, chosen_idx, dist, moves,
                          game.current_player))
        game.make(*chosen)
        move_num += 1

    # Assign outcomes: +1 for winner's positions, -1 for loser's
    winner = game.winner
    training_data = []
    for board_arr, oq, or_, move, chosen_idx, dist, moves, player in positions:
        if winner is None:
            z = 0.0
        else:
            z = 1.0 if player == winner else -1.0

        # Filter and normalize visit distribution for moves within window
        valid_moves = []
        valid_dist = []
        for m, d in zip(moves, dist):
            if encode_move(m[0], m[1], oq, or_) is not None:
                valid_moves.append(m)
                valid_dist.append(d)
        
        if not valid_moves:
            continue
            
        v_dist = np.array(valid_dist, dtype=np.float32)
        s = v_dist.sum()
        if s > 0:
            v_dist /= s
        else:
            v_dist = np.ones_like(v_dist) / len(v_dist)

        training_data.append({
            "board": board_arr,
            "oq": oq, "or_": or_,
            "moves": valid_moves,
            "probs": v_dist,
            "z": z,
        })

    all_moves = [pos[3] for pos in positions]  # chosen move per turn
    return training_data, winner, all_moves


# ── Training step ─────────────────────────────────────────────────────────────

def train_batch(net: HexNet, optimizer, scaler, buffer: deque) -> dict:
    if len(buffer) < BATCH_SIZE:
        return {}

    batch = random.sample(buffer, BATCH_SIZE)
    net.train()

    # Value loss: standard MSE on outcome
    boards      = torch.tensor(np.stack([b["board"] for b in batch]),
                                device=DEVICE)                    # [B,3,S,S]
    z_targets   = torch.tensor([b["z"] for b in batch],
                                dtype=torch.float32, device=DEVICE)  # [B]

    optimizer.zero_grad()

    # Automatic Mixed Precision context
    with torch.amp.autocast(device_type="cuda" if "cuda" in str(DEVICE) else "cpu"):
        f = net.trunk(boards)
        val = net.value(f)
        loss_v = F.mse_loss(val, z_targets)

        # Policy loss: cross-entropy against visit distribution for ALL legal moves
        loss_p = torch.tensor(0.0, device=DEVICE)
        for i, item in enumerate(batch):
            oq, or_ = item["oq"], item["or_"]
            moves = item["moves"]
            probs = torch.tensor(item["probs"], dtype=torch.float32, device=DEVICE)
            
            planes = [encode_move(q, r, oq, or_) for q, r in moves]
            move_t = torch.tensor(np.stack(planes), device=DEVICE)       # [N, 1, S, S]
            
            feat_i = f[i:i+1].expand(len(moves), -1, -1, -1)
            logits = net.policy_logit(feat_i, move_t)                    # [N]
            
            log_preds = F.log_softmax(logits, dim=0)
            loss_p += -(probs * log_preds).sum()
            
            with torch.no_grad():
                preds = F.softmax(logits, dim=0)
                item["entropy"] = -(preds * log_preds).sum().item()

        loss_p /= BATCH_SIZE
        loss = loss_v + loss_p

    # Scaled backpropagation
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()

    avg_ent = sum(b.get("entropy", 0) for b in batch) / BATCH_SIZE
    return {"loss": loss.item(), "loss_v": loss_v.item(), "loss_p": loss_p.item(), "entropy": avg_ent}


# ── Checkpoint ────────────────────────────────────────────────────────────────
def save(net: HexNet, gen: int):
    """Save model checkpoint."""
    path = CHECKPOINT_DIR / f"net_gen{gen:04d}.pt"
    # Use _orig_mod.state_dict() if it exists to avoid torch.compile prefix
    sd = net._orig_mod.state_dict() if hasattr(net, "_orig_mod") else net.state_dict()
    torch.save(sd, path)
    # Also save as latest
    torch.save(sd, CHECKPOINT_DIR / "net_latest.pt")
    log.info("Saved checkpoint %s", path)
    return path


def load_latest(net: HexNet) -> int:
    path = CHECKPOINT_DIR / "net_latest.pt"
    if path.exists():
        net.load_state_dict(torch.load(path, map_location=DEVICE))
        log.info("Loaded %s", path)
        # Infer generation from numbered files
        nums = [int(p.stem.split("gen")[1]) for p in CHECKPOINT_DIR.glob("net_gen*.pt")]
        return max(nums) if nums else 0
    return 0


# ── Main training loop ────────────────────────────────────────────────────────

def train(n_gens: int = 50, sims: int = 100, games_per_gen: int = 20):
    log.info("=== HexGo Training ===")
    log.info("Device=%s  Params=%s  SIMS=%d  GAMES/GEN=%d",
             DEVICE, f"{param_count(HexNet()):,}", sims, games_per_gen)

    net = HexNet().to(DEVICE)
    start_gen = load_latest(net)

    optimizer = optim.Adam(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler(enabled="cuda" in str(DEVICE))
    elo = ELO()
    buffer: deque = deque(maxlen=BUFFER_CAP)

    for gen in range(start_gen + 1, start_gen + n_gens + 1):
        log.info("--- Generation %d ---", gen)
        t_gen = time.perf_counter()

        # Start inference server for this generation
        server = InferenceServer(net, batch_size=INF_BATCH, timeout_ms=INF_TIMEOUT)
        server.start()

        # Parallel self-play — NUM_WORKERS games run concurrently
        game_wins = {1: 0, 2: 0, None: 0}
        total_positions = 0
        last_moves, last_winner = [], None
        first_saved = False
        results_lock = __import__("threading").Lock()

        def run_one_game(g_idx):
            t0 = time.perf_counter()
            data, winner, moves = self_play_episode(server, sims)
            dur = time.perf_counter() - t0
            return g_idx, data, winner, moves, dur

        workers = min(NUM_WORKERS, games_per_gen)
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futs = [pool.submit(run_one_game, g) for g in range(games_per_gen)]
            for fut in concurrent.futures.as_completed(futs):
                g_idx, data, winner, moves, dur = fut.result()
                buffer.extend(data)
                total_positions += len(data)
                game_wins[winner] += 1
                last_moves, last_winner = moves, winner
                log.info("  SP G%02d winner=%s moves=%d dur=%.1fs buffer=%d "
                         "avg_batch=%.1f",
                         g_idx + 1, winner, len(moves), dur, len(buffer),
                         server.avg_batch_size)
                if not first_saved and gen == start_gen + 1:
                    save_replay(moves, winner, gen, "first")
                    first_saved = True

        server.stop()
        save_replay(last_moves, last_winner, gen, "last")

        log.info("  Self-play done: X=%d O=%d draw=%d  total_pos=%d  hits=%d",
                 game_wins[1], game_wins[2], game_wins[None], total_positions,
                 server.cache_hits)

        # Training
        if len(buffer) >= BATCH_SIZE:
            losses = []
            entropies = []
            n_batches = max(10, total_positions // BATCH_SIZE)
            for _ in range(n_batches):
                result = train_batch(net, optimizer, scaler, buffer)
                if result:
                    losses.append(result["loss"])
                    entropies.append(result.get("entropy", 0))
            log.info("  Train: %d batches  avg_loss=%.4f  avg_ent=%.4f", 
                     n_batches, sum(losses) / max(len(losses), 1),
                     sum(entropies) / max(len(entropies), 1))
        else:
            log.info("  Buffer too small to train (%d < %d)", len(buffer), BATCH_SIZE)

        # Checkpoint
        save(net, gen)

        # ELO evaluation vs mcts_50
        baseline = MCTSAgent(sims=50)
        net_agent = NetAgent(net, sims=sims // 2, name=f"net_gen{gen:04d}")
        match = run_match(net_agent, baseline, n_games=EVAL_GAMES, elo=elo, verbose=False)
        net_wins = match.get(f"wins_{net_agent.name}", 0)
        win_rate = net_wins / EVAL_GAMES
        log.info("  ELO eval vs mcts_50: win_rate=%.2f  ELO=%s",
                 win_rate, elo.leaderboard()[:3])

        log.info("  Generation %d done in %.1fs", gen, time.perf_counter() - t_gen)

    log.info("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gens",  type=int, default=50,  help="Generations to train")
    parser.add_argument("--sims",  type=int, default=100, help="MCTS sims per move")
    parser.add_argument("--games", type=int, default=20,  help="Self-play games per gen")
    args = parser.parse_args()
    train(n_gens=args.gens, sims=args.sims, games_per_gen=args.games)
