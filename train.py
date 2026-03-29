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
- Tree reuse between moves (1c): chosen child subtree recycled as next root
- ELO updated every generation via run_match()
"""

import argparse
import concurrent.futures
import json
import logging
import math
import random
import shutil
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
from net import HexNet, encode_board, encode_move, DEVICE, param_count, d6_augment_sample
from elo import ELO, NetAgent, MCTSAgent, EisensteinGreedyAgent, run_match

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
NUM_WORKERS   = 8       # 1a: increased from 4 — more concurrent games fill the batch
INF_BATCH     = 8       # 1a: inference server max batch size (match NUM_WORKERS)
INF_TIMEOUT   = 30      # 1a: ms to wait for a full batch (was 5ms — too short for MCTS think time)

# 2b: Self-play curriculum — ramp sims from SIMS_MIN to target over SIMS_RAMP_GENS
SIMS_MIN      = 25
SIMS_RAMP     = 20      # generations to reach full sims

# 2c: Checkpoint tournament — new net must win this fraction of pool games to become training policy
TOURNEY_THRESHOLD = 0.55
TOURNEY_POOL_K    = 3   # how many recent checkpoints to include in pool
TOURNEY_GAMES_K   = 4   # games per pool opponent


def _curriculum_sims(gen: int, target: int) -> int:
    """2b: linearly ramp from SIMS_MIN to target over SIMS_RAMP generations."""
    if gen >= SIMS_RAMP:
        return target
    return max(SIMS_MIN, int(SIMS_MIN + (target - SIMS_MIN) * gen / SIMS_RAMP))


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
                temp: float = 1.0,
                prev_root: "Node | None" = None) -> tuple[tuple, np.ndarray, list, "Node | None"]:
    """
    MCTS guided by the inference server (batched GPU calls).
    Returns (chosen_move, visit_dist, all_moves, new_root).

    1c (tree reuse): if prev_root is provided and has children, reuse its subtree
    rather than building a new tree from scratch. Pass the returned new_root as
    prev_root on the next call to recycle ~N/branching_factor simulations for free.
    """
    # 1c: Tree reuse — use existing subtree if available
    if prev_root is not None and prev_root.children:
        root = prev_root
        root.parent = None  # detach from old tree
        moves = [c.move for c in root.children]
        # Re-apply Dirichlet noise at new root for continued exploration
        noise = np.random.dirichlet([0.3] * len(moves))
        for c, n in zip(root.children, noise):
            c.prior = 0.75 * c.prior + 0.25 * float(n)
    else:
        root = Node(player=game.current_player)
        value, policy = server.evaluate(game)
        moves = game.legal_moves()
        if not moves:
            return None, None, [], None

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
        visits_t = visits ** (1.0 / temp)
        dist = visits_t / visits_t.sum()

    chosen = moves[np.random.choice(len(moves), p=dist)]
    # 1c: return chosen child as new root for next call's tree reuse
    new_root = next((c for c in root.children if c.move == chosen), None)
    return chosen, dist, moves, new_root


def self_play_episode(server: InferenceServer, sims: int, temp_horizon: int = 40,
                      adversary=None, adversary_player: int = 2) -> tuple:
    """
    Play one full game, return (training_data, winner, all_moves).

    3a (cosine temp annealing): temperature decays smoothly from 1→0.05 over
        temp_horizon moves via max(0.05, cos(π·move/T)), replacing the hard cliff.
    1c (tree reuse): chosen child node is recycled as the root for the next call,
        inheriting its subtree and saving ~sims/branching_factor sims per move.
    3b (TD-lambda targets): z_t = gamma^(T-1-t) * z_final instead of uniform ±1,
        making early-game positions less certain and speeding value head convergence.
    adversary: optional Agent (e.g. EisensteinGreedyAgent) that controls
        adversary_player instead of the net — curriculum training partner.
    """
    game = HexGame()
    positions = []
    prev_root = None
    move_num = 0

    while game.winner is None:
        legal = game.legal_moves()
        if not legal:
            break

        # Adversary's turn: let it play without collecting training data
        if adversary is not None and game.current_player == adversary_player:
            chosen = adversary.choose_move(game)
            game.make(*chosen)
            prev_root = None   # can't reuse tree across an adversary move
            move_num += 1
            continue

        # 3a: cosine temperature annealing
        temp = max(0.05, math.cos(math.pi * move_num / temp_horizon))
        board_arr, (oq, or_) = encode_board(game)
        # 1c: pass prev_root for tree reuse
        chosen, dist, moves, new_root = mcts_policy(game, server, sims, temp,
                                                     prev_root=prev_root)
        if chosen is None:
            break
        chosen_idx = moves.index(chosen)
        positions.append((board_arr, oq, or_, chosen, chosen_idx, dist, moves,
                          game.current_player))
        game.make(*chosen)
        prev_root = new_root   # 1c: carry subtree to next move
        move_num += 1

    # 3b: TD-lambda value targets: z_t = gamma^(T-1-t) * z_final
    gamma = 0.99
    winner = game.winner
    T = len(positions)
    training_data = []
    for t, (board_arr, oq, or_, move, chosen_idx, dist, moves, player) in enumerate(positions):
        if winner is None:
            z = 0.0
        else:
            z_final = 1.0 if player == winner else -1.0
            z = (gamma ** (T - 1 - t)) * z_final   # discount early positions

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
    # D6 augmentation: apply a random symmetry transform to each sample.
    # All 12 D6 elements are equally likely; identity (tf=0) is one of them.
    # Board array and move coordinates are transformed consistently, so the
    # policy target remains correctly aligned. ~12x effective data diversity.
    batch = [d6_augment_sample(item, random.randrange(12)) for item in batch]
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


# ── Policy heatmap ───────────────────────────────────────────────────────────

# Fixed canonical test position used every generation to track what the network
# discovers. After many generations the policy mass should cluster around cells
# that extend the longest existing chains — the Eisenstein axes — without us
# ever encoding that knowledge directly.
_HEATMAP_MOVES = [(0,0),(1,0),(1,1),(0,1),(0,2),(2,0),(1,-1),(0,-1),(-1,1),(-1,0)]

def save_heatmap(server: InferenceServer, gen: int):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
    except ImportError:
        return

    game = HexGame()
    for q, r in _HEATMAP_MOVES:
        if game.winner is None:
            game.make(q, r)

    value, policy = server.evaluate(game)
    if not policy:
        return

    logits = np.array(list(policy.values()), dtype=np.float32)
    logits -= logits.max()
    probs  = np.exp(logits); probs /= probs.sum()
    moves  = list(policy.keys())

    sqrt3_2 = 3 ** 0.5 / 2
    fig, ax = plt.subplots(figsize=(8, 7))

    # Board pieces
    for (q, r), player in game.board.items():
        x, y = q + r * 0.5, r * sqrt3_2
        ax.scatter(x, y, s=220, c='#4C9BE8' if player == 1 else '#E87C4C',
                   zorder=3, edgecolors='white', linewidths=0.8)
        ax.annotate('X' if player == 1 else 'O', (x, y),
                    ha='center', va='center', fontsize=7, color='white', zorder=4)

    # Policy distribution on candidate moves
    norm = mcolors.PowerNorm(gamma=0.5, vmin=0, vmax=probs.max())
    for (q, r), p in zip(moves, probs):
        x, y = q + r * 0.5, r * sqrt3_2
        ax.scatter(x, y, s=350, c=[[0.36, 0.74, 0.42, float(norm(p))]],
                   zorder=2, marker='h')
        if p > 0.04:
            ax.annotate(f'{p:.2f}', (x, y), ha='center', va='center',
                        fontsize=6.5, color='black', zorder=5)

    ax.set_title(f'Policy heatmap — gen {gen:04d}   value={value:+.3f}   '
                 f'(green=high policy mass)', fontsize=10)
    ax.set_aspect('equal')
    ax.axis('off')

    out = REPLAY_DIR / f'heatmap_gen{gen:04d}.png'
    plt.savefig(out, dpi=130, bbox_inches='tight')
    plt.close()
    log.info("  Heatmap saved: %s", out.name)


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
        try:
            net.load_state_dict(torch.load(path, map_location=DEVICE))
            log.info("Loaded %s", path)
            nums = [int(p.stem.split("gen")[1]) for p in CHECKPOINT_DIR.glob("net_gen*.pt")]
            return max(nums) if nums else 0
        except (RuntimeError, KeyError) as e:
            # Architecture changed — quarantine incompatible checkpoints and start fresh
            log.warning("Checkpoint incompatible (%s) — moving to legacy/", e)
            legacy_dir = CHECKPOINT_DIR / "legacy"
            legacy_dir.mkdir(exist_ok=True)
            for p in CHECKPOINT_DIR.glob("net_*.pt"):
                shutil.move(str(p), str(legacy_dir / p.name))
            log.info("Quarantined incompatible checkpoints, starting fresh")
    return 0


# ── Checkpoint tournament ────────────────────────────────────────────────────

def _tourney_promote(net: HexNet, gen: int, sims: int, elo: ELO) -> bool:
    """
    2c: Run new net against top-K recent checkpoints.
    Returns True (and saves net_latest.pt) if win_rate >= TOURNEY_THRESHOLD.
    On gen<=1 or no pool available, always promotes.
    """
    pool_paths = sorted(CHECKPOINT_DIR.glob("net_gen*.pt"))
    # Exclude the checkpoint we just saved (current gen)
    pool_paths = [p for p in pool_paths if f"gen{gen:04d}" not in p.stem]
    pool_paths = pool_paths[-TOURNEY_POOL_K:]  # most recent K

    if not pool_paths:
        return True  # no pool yet → auto-promote

    wins, total = 0, 0
    net_agent = NetAgent(net, sims=max(25, sims // 2), name=f"net_gen{gen:04d}")

    for p in pool_paths:
        old_net = HexNet().to(DEVICE)
        try:
            old_net.load_state_dict(torch.load(p, map_location=DEVICE))
        except (RuntimeError, KeyError):
            continue  # skip incompatible checkpoint silently
        old_agent = NetAgent(old_net, sims=max(25, sims // 2),
                             name=f"pool_{p.stem}")
        result = run_match(net_agent, old_agent, n_games=TOURNEY_GAMES_K,
                           elo=elo, verbose=False)
        wins  += result.get(f"wins_{net_agent.name}", 0)
        total += TOURNEY_GAMES_K

    if total == 0:
        return True

    win_rate = wins / total
    log.info("  Tournament: %d/%d (%.0f%%) vs pool of %d — %s",
             wins, total, 100 * win_rate, len(pool_paths),
             "PROMOTED" if win_rate >= TOURNEY_THRESHOLD else "held")
    return win_rate >= TOURNEY_THRESHOLD


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

    # 2b: per-gen position hash set for Zobrist-style dedup (cleared each gen)
    gen_hashes: set[int] = set()

    for gen in range(start_gen + 1, start_gen + n_gens + 1):
        log.info("--- Generation %d ---", gen)
        t_gen = time.perf_counter()

        # 2b: curriculum sims — ramp from SIMS_MIN to target over first SIMS_RAMP gens
        cur_sims = _curriculum_sims(gen - start_gen, sims)

        # Start inference server for this generation
        server = InferenceServer(net, batch_size=INF_BATCH, timeout_ms=INF_TIMEOUT)
        server.start()

        # Parallel self-play — NUM_WORKERS games run concurrently.
        # 20% of games use EisensteinGreedyAgent (defensive) as P2 — curriculum
        # adversary that forces the net to beat structured chain-building play.
        eisenstein_adv = EisensteinGreedyAgent(name="eisenstein_def", defensive=True)
        game_wins = {1: 0, 2: 0, None: 0}
        total_positions = 0
        deduped = 0
        last_moves, last_winner = [], None
        first_saved = False
        gen_hashes.clear()  # 2a: reset per-gen dedup set

        def run_one_game(g_idx):
            t0 = time.perf_counter()
            adv = eisenstein_adv if (g_idx % 5 == 0) else None
            data, winner, moves = self_play_episode(server, cur_sims, adversary=adv)
            dur = time.perf_counter() - t0
            return g_idx, data, winner, moves, dur

        workers = min(NUM_WORKERS, games_per_gen)
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futs = [pool.submit(run_one_game, g) for g in range(games_per_gen)]
            for fut in concurrent.futures.as_completed(futs):
                g_idx, data, winner, moves, dur = fut.result()
                # 2a: Zobrist-style dedup — skip positions already seen this gen
                for item in data:
                    h = hash(item["board"].tobytes())
                    if h not in gen_hashes:
                        gen_hashes.add(h)
                        buffer.append(item)
                        total_positions += 1
                    else:
                        deduped += 1
                game_wins[winner] += 1
                last_moves, last_winner = moves, winner
                log.info("  SP G%02d winner=%s moves=%d dur=%.1fs buffer=%d "
                         "avg_batch=%.1f sims=%d",
                         g_idx + 1, winner, len(moves), dur, len(buffer),
                         server.avg_batch_size, cur_sims)
                if not first_saved and gen == start_gen + 1:
                    save_replay(moves, winner, gen, "first")
                    first_saved = True

        server.stop()
        save_replay(last_moves, last_winner, gen, "last")
        log.info("  Dedup: skipped %d duplicate positions", deduped)

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

        # Checkpoint — save generation file unconditionally
        save(net, gen)

        # 2c: Checkpoint tournament — only update net_latest.pt if new net wins pool
        if not _tourney_promote(net, gen, cur_sims, elo):
            # Reload the previous best net as training policy for next gen
            prev_best = sorted(CHECKPOINT_DIR.glob("net_gen*.pt"))
            prev_best = [p for p in prev_best if f"gen{gen:04d}" not in p.stem]
            if prev_best:
                try:
                    net.load_state_dict(torch.load(prev_best[-1], map_location=DEVICE))
                    log.info("  Reverted to %s as training policy", prev_best[-1].name)
                except (RuntimeError, KeyError):
                    pass  # keep current net if load fails

        # ELO evaluation vs mcts_50
        baseline  = MCTSAgent(sims=50)
        net_agent = NetAgent(net, sims=sims // 2, name=f"net_gen{gen:04d}")
        match = run_match(net_agent, baseline, n_games=EVAL_GAMES, elo=elo, verbose=False)
        net_wins = match.get(f"wins_{net_agent.name}", 0)
        win_rate = net_wins / EVAL_GAMES
        log.info("  ELO eval vs mcts_50: win_rate=%.2f  ELO=%s",
                 win_rate, elo.leaderboard()[:3])

        # ELO evaluation vs EisensteinGreedy (defensive) — curriculum baseline
        eis_agent = EisensteinGreedyAgent(name="eisenstein_def", defensive=True)
        eis_match = run_match(net_agent, eis_agent, n_games=EVAL_GAMES // 2,
                              elo=elo, verbose=False)
        eis_wins  = eis_match.get(f"wins_{net_agent.name}", 0)
        log.info("  ELO eval vs eisenstein_def: win_rate=%.2f",
                 eis_wins / (EVAL_GAMES // 2))

        # Policy heatmap — create a fresh one-shot server (net is already trained this gen)
        heatmap_server = InferenceServer(net, batch_size=1, timeout_ms=100)
        heatmap_server.start()
        save_heatmap(heatmap_server, gen)
        heatmap_server.stop()

        log.info("  Generation %d done in %.1fs", gen, time.perf_counter() - t_gen)

    log.info("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gens",  type=int, default=50,  help="Generations to train")
    parser.add_argument("--sims",  type=int, default=100, help="MCTS sims per move")
    parser.add_argument("--games", type=int, default=20,  help="Self-play games per gen")
    args = parser.parse_args()
    train(n_gens=args.gens, sims=args.sims, games_per_gen=args.games)
