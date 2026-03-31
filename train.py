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
- Tree reuse disabled (see mcts_policy TODO): fresh root built each move
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
from inference import InferenceServer, evict_stale_cache
from mcts import Node, _backprop
from net import HexNet, encode_board, encode_move, DEVICE, param_count, d6_augment_sample, make_aux_labels, init_weights_ca
from elo import ELO, NetAgent, EisensteinGreedyAgent, run_match
from config import CFG

_CUDA = "cuda" in str(DEVICE)


# ── Latency / performance tracker ────────────────────────────────────────────

class PerfTracker:
    """
    Lightweight per-generation timing tracker.

    Usage:
        pt = PerfTracker()
        pt.start("self_play")
        ...
        pt.stop("self_play")
        log.info(pt.summary(total_time))
        pt.reset()
    """
    def __init__(self):
        self._times:  dict[str, float] = {}
        self._counts: dict[str, int]   = {}
        self._t0:     dict[str, float] = {}

    def start(self, name: str):
        self._t0[name] = time.perf_counter()

    def stop(self, name: str):
        elapsed = time.perf_counter() - self._t0.get(name, time.perf_counter())
        self._times[name]  = self._times.get(name, 0.0) + elapsed
        self._counts[name] = self._counts.get(name, 0) + 1

    def get(self, name: str) -> float:
        return self._times.get(name, 0.0)

    def summary(self, total: float) -> str:
        if total <= 0:
            return ""
        parts = []
        for name, t in sorted(self._times.items(), key=lambda x: -x[1]):
            pct = 100.0 * t / total
            n   = self._counts[name]
            avg = 1000.0 * t / n if n else 0.0
            parts.append(f"{name}={t:.1f}s({pct:.0f}%,n={n},avg={avg:.0f}ms)")
        return " | ".join(parts)

    def warnings(self, total: float, avg_batch: float, deduped: int,
                 new_pos: int) -> list[str]:
        """Return human-readable bottleneck warnings for logging."""
        w = []
        sp_frac = self._times.get("self_play", 0) / max(total, 1)
        if sp_frac < 0.3 and total > 10:
            w.append("Training dominates (>70% time) — consider fewer train batches or more workers")
        if avg_batch < 2.0:
            w.append(f"Low inference batching (avg={avg_batch:.1f}) — raise INF_TIMEOUT or NUM_WORKERS")
        dup_rate = deduped / max(deduped + new_pos, 1)
        if dup_rate > 0.3:
            w.append(f"High dedup rate ({100*dup_rate:.0f}%) — try larger margin or more diverse exploration")
        return w

    def reset(self):
        self._times.clear(); self._counts.clear(); self._t0.clear()

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
EVAL_GAMES    = 20
NUM_WORKERS   = 16      # more concurrent games keeps batch full as short games finish early
INF_BATCH     = 16      # match NUM_WORKERS
INF_TIMEOUT   = 15      # ms — inference is now ~2ms so 30ms was over-waiting; 10ms balances latency vs batching

# Tunable via config.py (autotune)
BATCH_SIZE    = CFG["BATCH_SIZE"]
LR            = CFG["LR"]
WEIGHT_DECAY  = CFG["WEIGHT_DECAY"]
SIMS_MIN      = CFG["SIMS_MIN"]
ZOI_MARGIN    = CFG["ZOI_MARGIN"]
ZOI_LOOKBACK  = CFG["ZOI_LOOKBACK"]
CAP_FULL_FRAC = CFG["CAP_FULL_FRAC"]


def _cap_sims(target: int) -> int:
    """2b: KataGo playout cap randomization — 25% full, 75% reduced budget."""
    if random.random() < CAP_FULL_FRAC:
        return target
    # 75% use reduced budget: max(SIMS_MIN, target // 8) but never more than target
    reduced = max(SIMS_MIN, target // 8)
    return min(target, reduced)


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
    # Tree reuse disabled: naive filtering of prev_root.children by board occupancy
    # retains stale siblings with visit counts from a different game state, corrupting search.
    # Correct reuse requires descending through all intermediate moves (opponent turns,
    # second placements under 1-2-2) to find the correct subtree.
    # TODO: implement correct tree descent through intermediate moves.
    root = Node(player=game.current_player)
    value, policy = server.evaluate(game)
    moves = game.zoi_moves(ZOI_MARGIN, ZOI_LOOKBACK)  # ZOI pruning: ~80-90% branch reduction
    if not moves:
        return None, None, [], None

    logits = np.array([policy.get(m, 0.0) for m in moves], dtype=np.float32)
    logits -= logits.max()
    priors = np.exp(logits); priors /= priors.sum()

    _da, _de = CFG["DIRICHLET_ALPHA"], CFG["DIRICHLET_EPS"]
    noise = np.random.dirichlet([_da] * len(moves))
    priors = (1 - _de) * priors + _de * noise
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
            v = 1.0 if game.winner == node.player else -1.0
        else:
            v, lp = server.evaluate(game)
            if node.player != game.current_player:
                v = -v   # server.evaluate returns from game.current_player's POV
            lmoves = game.zoi_moves(ZOI_MARGIN, ZOI_LOOKBACK)
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
        chosen = moves[visits.argmax()]
    elif CFG.get("GUMBEL_SELECTION", False):
        # Gumbel argmax: log(visits + ε) + Gumbel(0,1)/temp, then argmax.
        # At temp=1 equivalent to sampling proportional to visits;
        # at temp→0 converges to pure argmax — naturally anneals with cosine schedule.
        eps = 1e-8
        gumbel_noise = -np.log(-np.log(np.random.uniform(size=len(visits)) + eps) + eps)
        logits = np.log(visits + eps) + gumbel_noise / max(temp, 1e-6)
        best = logits.argmax()
        dist = visits / visits.sum()          # policy target stays as visit proportions
        chosen = moves[best]
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
        temp_horizon moves via max(0.05, cos(π/2·move/T)). Reaches floor at
        move=T (half-cosine, T is the full-life not half-life).
    Tree reuse is disabled (prev_root ignored); fresh root built each move.
    3b (TD-lambda targets): z_t = gamma^(T-1-t) * z_final instead of uniform ±1,
        making early-game positions less certain and speeding value head convergence.
    adversary: optional Agent (e.g. EisensteinGreedyAgent) that controls
        adversary_player instead of the net — curriculum training partner.
    """
    MAX_MOVES = 150   # placements; beyond this declare draw — 300 was allowing random-net games to drag for minutes
    game = HexGame()
    positions = []
    prev_root = None
    move_num = 0
    _acc_agent = EisensteinGreedyAgent(name="_acc", defensive=True)  # for move_acc labels

    while game.winner is None and len(game.move_history) < MAX_MOVES:
        legal = game.legal_moves()
        if not legal:
            break

        # Adversary's turn: let it play without collecting training data
        if adversary is not None and game.current_player == adversary_player:
            chosen = adversary.choose_move(game)
            if chosen is None or not game.make(*chosen):
                log.warning("Adversary %s returned illegal move %s", adversary.name, chosen)
                break
            prev_root = None   # can't reuse tree across an adversary move
            move_num += 1
            continue

        # 3a: cosine temperature annealing: 1→floor over temp_horizon moves.
        # Using half-cosine: cos(π/2 * move/T) → 1.0 at move=0, 0.0 at move=T.
        temp = max(0.05, math.cos(math.pi / 2 * move_num / max(temp_horizon, 1)))
        board_arr, (oq, or_) = encode_board(game)
        # Capture EisensteinGreedyAgent's preferred move at this position for move_acc metric
        greedy_move = _acc_agent.choose_move(game)
        # 1c: pass prev_root for tree reuse
        chosen, dist, moves, new_root = mcts_policy(game, server, sims, temp,
                                                     prev_root=prev_root)
        if chosen is None:
            break
        chosen_idx = moves.index(chosen)
        positions.append((board_arr, oq, or_, chosen, chosen_idx, dist, moves,
                          game.current_player, greedy_move))
        if not game.make(*chosen):
            log.warning("MCTS returned illegal move %s", chosen)
            break
        prev_root = new_root   # 1c: carry subtree to next move
        move_num += 1

    # 3b: TD-lambda value targets: z_t = gamma^(T-1-t) * z_final
    gamma = CFG["TD_GAMMA"]
    winner = game.winner
    T = len(positions)
    training_data = []
    for t, (board_arr, oq, or_, move, chosen_idx, dist, moves, player, greedy_move) in enumerate(positions):
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

        # Aux labels: ownership + threat relative to this position's window origin
        own_lbl, threat_lbl = make_aux_labels(game, winner, oq, or_)

        training_data.append({
            "board": board_arr,
            "oq": oq, "or_": or_,
            "moves": valid_moves,
            "probs": v_dist,
            "z": z,
            "own_label":    own_lbl,
            "threat_label": threat_lbl,
            "greedy_move":  greedy_move,
        })

    all_moves = list(game.move_history)  # ALL placements (net + adversary)
    return training_data, winner, all_moves


# ── Training step ─────────────────────────────────────────────────────────────

def train_batch(net: HexNet, optimizer, scaler, buffer: deque) -> dict:
    if len(buffer) < BATCH_SIZE:
        return {}

    # 3b-viii: Recency-weighted sampling — 75% from recent half, 25% uniform.
    # Prevents the net from being anchored to its own early, incompetent play.
    rw = CFG["RECENCY_WEIGHT"]
    buf_list = list(buffer)
    n_recent = max(1, len(buf_list) // 2)
    recent_half = buf_list[-n_recent:]
    n_from_recent = int(BATCH_SIZE * rw)
    n_from_all    = BATCH_SIZE - n_from_recent
    batch = (random.sample(recent_half, min(n_from_recent, len(recent_half))) +
             random.sample(buf_list,    min(n_from_all,    len(buf_list))))
    random.shuffle(batch)
    # D6 augmentation: apply a random symmetry transform to each sample.
    # All 12 D6 elements are equally likely; identity (tf=0) is one of them.
    # Board array and move coordinates are transformed consistently, so the
    # policy target remains correctly aligned. ~12x effective data diversity.
    batch = [d6_augment_sample(item, random.randrange(12)) for item in batch]
    net.train()

    # Value loss: standard MSE on outcome
    # pin_memory + non_blocking: async host→GPU transfer, overlaps with CPU work
    boards_np = np.stack([b["board"] for b in batch])
    boards = (torch.from_numpy(boards_np).pin_memory().to(DEVICE, non_blocking=True)
              if _CUDA else torch.tensor(boards_np, device=DEVICE))
    z_np = np.array([b["z"] for b in batch], dtype=np.float32)
    z_targets = (torch.from_numpy(z_np).pin_memory().to(DEVICE, non_blocking=True)
                 if _CUDA else torch.tensor(z_np, device=DEVICE))

    optimizer.zero_grad()

    # Automatic Mixed Precision context
    with torch.amp.autocast(device_type="cuda" if "cuda" in str(DEVICE) else "cpu"):
        f = net.trunk(boards)
        val = net.value(f)
        loss_v = F.mse_loss(val, z_targets)

        # Policy loss: batched cross-entropy over all legal moves across the full batch.
        # Build flat lists of (feature_row, move_plane, prob) then run policy_logit once.
        # Per-item softmax applied via segment offsets to handle variable move counts.
        all_feat_rows = []   # board feature index into f for each (item, move) pair
        all_planes    = []   # [1, S, S] move plane per (item, move) pair
        all_probs     = []   # scalar probability per (item, move) pair
        item_counts   = []   # number of valid moves per batch item (0 = skip)

        for i, item in enumerate(batch):
            oq, or_ = item["oq"], item["or_"]
            valid_indices, planes = [], []
            for j, (q, r) in enumerate(item["moves"]):
                p = encode_move(q, r, oq, or_)
                if p is not None:
                    valid_indices.append(j)
                    planes.append(p)
            if not planes:
                item_counts.append(0)
                continue
            probs_np = item["probs"][valid_indices].astype(np.float32)
            s = probs_np.sum()
            probs_np = probs_np / s if s > 1e-6 else np.ones_like(probs_np) / len(probs_np)
            n = len(planes)
            all_feat_rows.extend([i] * n)
            all_planes.extend(planes)
            all_probs.extend(probs_np.tolist())
            item_counts.append(n)

        loss_p = torch.tensor(0.0, device=DEVICE)
        n_p = sum(1 for c in item_counts if c > 0)
        if all_planes:
            feat_rows = torch.tensor(all_feat_rows, dtype=torch.long, device=DEVICE)
            move_t    = torch.tensor(np.stack(all_planes), device=DEVICE)   # [M, 1, S, S]
            probs_t   = torch.tensor(all_probs, dtype=torch.float32, device=DEVICE)  # [M]
            feats_exp = f[feat_rows]                                         # [M, C, S, S]
            all_logits = net.policy_logit(feats_exp, move_t)                # [M]

            # Per-item log_softmax via segment offsets
            offset = 0
            entropies_batch = []
            loss_ent = torch.tensor(0.0, device=DEVICE)  # differentiable entropy accumulator
            for i, cnt in enumerate(item_counts):
                if cnt == 0:
                    continue
                seg_logits = all_logits[offset:offset + cnt]
                seg_probs  = probs_t[offset:offset + cnt]
                log_preds  = F.log_softmax(seg_logits, dim=0)
                loss_p    += -(seg_probs * log_preds).sum()
                preds = F.softmax(seg_logits, dim=0)
                ent_tensor = -(preds * log_preds).sum()   # kept in graph for gradient flow
                loss_ent  += ent_tensor
                with torch.no_grad():
                    ent = ent_tensor.item()
                    batch[i]["entropy"] = ent
                    entropies_batch.append(ent)
                offset += cnt

        if n_p > 0:
            loss_p /= n_p

        # Entropy regularization — subtract differentiable loss_ent so gradients flow
        # to the policy head. Using detached scalars (old code) produced zero gradient.
        ent_reg = CFG.get("ENTROPY_REG", 0.0)
        if ent_reg > 0 and n_p > 0:
            loss_p = loss_p - ent_reg * (loss_ent / n_p)

        # Auxiliary losses — ownership (MSE) + threat (BCE), if labels present
        loss_aux = torch.tensor(0.0, device=DEVICE)
        aux_w_own    = CFG["AUX_LOSS_OWN"]
        aux_w_threat = CFG["AUX_LOSS_THREAT"]
        if aux_w_own > 0 or aux_w_threat > 0:
            aux_indices = [i for i, b in enumerate(batch) if "own_label" in b]
            if aux_indices:
                has_aux   = [batch[i] for i in aux_indices]
                own_np    = np.stack([b["own_label"]    for b in has_aux])
                threat_np = np.stack([b["threat_label"] for b in has_aux])
                own_t    = (torch.from_numpy(own_np).pin_memory().to(DEVICE, non_blocking=True)
                            if _CUDA else torch.tensor(own_np, device=DEVICE))
                threat_t = (torch.from_numpy(threat_np).pin_memory().to(DEVICE, non_blocking=True)
                            if _CUDA else torch.tensor(threat_np, device=DEVICE))
                # Reuse trunk features already computed for this batch — no second forward pass
                f_aux = f[aux_indices]
                if aux_w_own > 0:
                    own_pred = net.ownership(f_aux)           # [N, S, S]
                    loss_aux = loss_aux + aux_w_own * F.mse_loss(own_pred, own_t)
                if aux_w_threat > 0:
                    thr_logits = net.threat_logits(f_aux)     # [N, S, S] raw logits
                    loss_aux = loss_aux + aux_w_threat * F.binary_cross_entropy_with_logits(
                        thr_logits, threat_t)

        # Value uncertainty loss — Gaussian NLL: 0.5*(log(σ²) + (z-v)²/σ²)
        loss_unc = torch.tensor(0.0, device=DEVICE)
        unc_w = CFG.get("UNC_LOSS_WEIGHT", 0.0)
        if unc_w > 0:
            sigma2 = net.variance(f)                        # [B] > 0
            loss_unc = 0.5 * (sigma2.log() + (z_targets - val) ** 2 / sigma2).mean()

        loss = CFG["VALUE_LOSS_WEIGHT"] * loss_v + loss_p + loss_aux + unc_w * loss_unc
    # Scaled backpropagation
    if torch.isnan(loss):
        log.warning("NaN loss detected! Skipping batch.")
        return None

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()

    avg_ent   = sum(b.get("entropy", 0) for b in batch) / BATCH_SIZE
    avg_sigma = sigma2.detach().sqrt().mean().item() if unc_w > 0 else 0.0
    return {"loss": loss.item(), "loss_v": loss_v.item(), "loss_p": loss_p.item(),
            "loss_aux": loss_aux.item(), "loss_unc": loss_unc.item(),
            "entropy": avg_ent, "avg_sigma": avg_sigma}


# ── Move accuracy metric ─────────────────────────────────────────────────────

def compute_move_acc(net: HexNet, buffer: deque, n_samples: int = 40) -> float:
    """
    Top-1 policy agreement rate between the net and EisensteinGreedyAgent (defensive).

    NOTE: This metric is an early-training sanity check only. Once the network
    surpasses the greedy agent (~gen 20+), decreasing agreement is expected and
    healthy — it means the network learned non-trivial strategy. Do not use as
    a quality signal in later training.

    For each sampled position we encode the board, run the net's policy head,
    pick argmax move among legal ZOI moves, and check whether it matches the
    greedy agent's choice stored at collection time.  Returns fraction in [0,1].
    Items without a 'greedy_move' key (old buffer entries) are skipped.
    """
    eligible = [b for b in buffer if b.get("greedy_move") is not None]
    if not eligible:
        return 0.0
    sample = random.sample(eligible, min(n_samples, len(eligible)))

    net.eval()
    correct = 0
    with torch.no_grad():
        for item in sample:
            greedy = item["greedy_move"]
            moves  = item["moves"]
            oq, or_ = item["oq"], item["or_"]

            # Build encoded move planes for legal moves within the ZOI window
            planes, valid_moves = [], []
            for q, r in moves:
                p = encode_move(q, r, oq, or_)
                if p is not None:
                    planes.append(p)
                    valid_moves.append((q, r))

            if not planes:
                continue

            board_t = torch.tensor(item["board"][None], device=DEVICE)
            f = net.trunk(board_t)
            move_t = torch.tensor(np.stack(planes), device=DEVICE)          # [N,1,S,S]
            feat_e = f.expand(len(planes), -1, -1, -1)
            logits = net.policy_logit(feat_e, move_t)                       # [N]
            best_idx = logits.argmax().item()
            if valid_moves[best_idx] == greedy:
                correct += 1

    return correct / len(sample) if sample else 0.0


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
            target = net._orig_mod if hasattr(net, "_orig_mod") else net
            target.load_state_dict(torch.load(path, map_location=DEVICE))
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


# ── Main training loop ────────────────────────────────────────────────────────

def train(n_gens: int = 50, sims: int = CFG["SIMS"], games_per_gen: int = 20, tune_mode: bool = False):
    log.info("=== HexGo Training ===")
    log.info("Device=%s  Params=%s  SIMS=%d  GAMES/GEN=%d",
             DEVICE, f"{param_count(HexNet()):,}", sims, games_per_gen)

    net = HexNet().to(DEVICE)
    start_gen = load_latest(net)
    if start_gen == 0 and CFG.get("WEIGHT_INIT", "xavier") == "ca":
        init_weights_ca(net)
        log.info("Initialized HexConv2d kernels with hex-Laplacian CA priors.")

    optimizer = optim.Adam(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    # Warmup for first 5 gens: LR scales from LR/10 → LR, then cosine decay to LR*0.01.
    # Prevents the gen-1 loss spike caused by a high LR hammering a tiny buffer.
    WARMUP_GENS = 5
    def lr_lambda(gen_idx):  # gen_idx is 0-based steps from scheduler
        if gen_idx < WARMUP_GENS:
            return 0.1 + 0.9 * gen_idx / WARMUP_GENS
        cosine_progress = (gen_idx - WARMUP_GENS) / max(n_gens - WARMUP_GENS, 1)
        cosine_val = 0.5 * (1 + math.cos(math.pi * cosine_progress))
        return 0.01 + 0.99 * cosine_val  # decays from 1.0 → 0.01
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler(enabled="cuda" in str(DEVICE))
    elo = ELO()
    buffer: deque = deque(maxlen=BUFFER_CAP)

    # 2a: per-gen position hash set for dedup (cleared each gen)
    gen_hashes: set[int] = set()
    perf = PerfTracker()

    for gen in range(start_gen + 1, start_gen + n_gens + 1):
        log.info("--- Generation %d ---", gen)
        t_gen = time.perf_counter()
        perf.reset()
        # cur_sims is per-game (cap randomization applied inside run_one_game)
        cur_sims = sims

        # Evict stale entries from the persistent cross-gen cache
        evict_stale_cache(gen)

        # Start inference server for this generation
        server = InferenceServer(net, batch_size=INF_BATCH, timeout_ms=INF_TIMEOUT, gen=gen)
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
            adv = eisenstein_adv if (g_idx % 5 != 0) else None  # 80% vs Eisenstein, 20% self-play
            # 2b: KataGo playout cap — per-game sims budget
            game_sims = _cap_sims(cur_sims)
            data, winner, moves = self_play_episode(server, game_sims,
                                                     temp_horizon=CFG["TEMP_HORIZON"],
                                                     adversary=adv)
            dur = time.perf_counter() - t0
            return g_idx, data, winner, moves, dur, game_sims

        # 3a: Overlapped self-play + training
        WEIGHT_SYNC_BATCHES = CFG["WEIGHT_SYNC_BATCHES"]
        losses, loss_vs, loss_ps, entropies, aux_losses, sigmas = [], [], [], [], [], []
        batches_since_sync = 0
        workers = min(NUM_WORKERS, games_per_gen)

        perf.start("self_play")
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            pending = {pool.submit(run_one_game, g): g for g in range(games_per_gen)}

            while pending:
                done, pending = concurrent.futures.wait(
                    pending, timeout=0.05,
                    return_when=concurrent.futures.FIRST_COMPLETED)

                for fut in done:
                    g_idx, data, winner, moves, dur, gsims = fut.result()
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
                             server.avg_batch_size, gsims)
                    if not first_saved and gen == start_gen + 1:
                        save_replay(moves, winner, gen, "first")
                        first_saved = True

                # Train overlapped with remaining self-play.
                # Cap total overlap batches to 1 pass over current buffer size to prevent
                # overfitting when self-play is fast and buffer is small (gen 1 spike).
                max_overlap = max(1, len(buffer) // BATCH_SIZE)
                if len(buffer) >= BATCH_SIZE and batches_since_sync < WEIGHT_SYNC_BATCHES and len(losses) < max_overlap:
                    perf.start("overlap_train")
                    result = train_batch(net, optimizer, scaler, buffer)
                    perf.stop("overlap_train")
                    if result:
                        losses.append(result["loss"])
                        loss_vs.append(result.get("loss_v", 0))
                        loss_ps.append(result.get("loss_p", 0))
                        entropies.append(result.get("entropy", 0))
                        aux_losses.append(result.get("loss_aux", 0))
                        sigmas.append(result.get("avg_sigma", 0))
                        batches_since_sync += 1

                    if batches_since_sync >= WEIGHT_SYNC_BATCHES:
                        sd = (net._orig_mod.state_dict()
                              if hasattr(net, "_orig_mod") else net.state_dict())
                        # Synchronize CUDA before and after load_state_dict to prevent
                        # racing with inference server GPU forward passes (CUDA kernels
                        # release the GIL, so load_state_dict mid-forward can corrupt).
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        target = server.net._orig_mod if hasattr(server.net, "_orig_mod") else server.net
                        target.load_state_dict(sd)
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        batches_since_sync = 0

        perf.stop("self_play")
        server.stop()
        save_replay(last_moves, last_winner, gen, "last")
        log.info("  Dedup: skipped %d duplicate positions", deduped)
        log.info("  Self-play done: X=%d O=%d draw=%d  total_pos=%d  "
                 "hits=%d persistent_hits=%d",
                 game_wins[1], game_wins[2], game_wins[None], total_positions,
                 server.cache_hits, server.persistent_hits)
        log.info("  Inference: %s", server.latency_summary())

        # Post-game training: top up to 1 pass over the buffer, no more.
        perf.start("post_train")
        n_extra = max(0, len(buffer) // BATCH_SIZE - len(losses))
        for _ in range(n_extra):
            result = train_batch(net, optimizer, scaler, buffer)
            if result:
                losses.append(result["loss"])
                loss_vs.append(result.get("loss_v", 0))
                loss_ps.append(result.get("loss_p", 0))
                entropies.append(result.get("entropy", 0))
                aux_losses.append(result.get("loss_aux", 0))
                sigmas.append(result.get("avg_sigma", 0))
        perf.stop("post_train")

        if losses:
            n = len(losses)
            log.info("  Train: %d batches  loss=%.4f  loss_v=%.4f  loss_p=%.4f  aux=%.4f  sigma=%.4f  ent=%.4f",
                     n, sum(losses)/n, sum(loss_vs)/n, sum(loss_ps)/n,
                     sum(aux_losses)/n if aux_losses else 0.0,
                     sum(sigmas)/n if sigmas else 0.0,
                     sum(entropies)/n)
        else:
            log.info("  Buffer too small to train (%d < %d)", len(buffer), BATCH_SIZE)

        # Checkpoint — save generation file unconditionally
        perf.start("checkpoint")
        save(net, gen)
        perf.stop("checkpoint")

        # ELO evaluation — Eisenstein only (fast greedy, no MCTS overhead)
        perf.start("eval")
        net_agent = NetAgent(net, sims=max(25, sims // 2), name=f"net_gen{gen:04d}")
        eis_agent = EisensteinGreedyAgent(name="eisenstein_def", defensive=True)
        eis_n     = EVAL_GAMES
        eis_match = run_match(net_agent, eis_agent, n_games=eis_n, elo=elo, verbose=False)
        eis_wins  = eis_match.get(f"wins_{net_agent.name}", 0)
        move_acc_eval = compute_move_acc(net, buffer)
        log.info("  ELO eval vs eisenstein_def: win_rate=%.2f  ELO=%s  move_acc=%.3f",
                 eis_wins / eis_n, elo.leaderboard()[:3], move_acc_eval)
        perf.stop("eval")

        # Write per-gen result for tune.py to consume
        if tune_mode:
            tune_result_path = Path("tune_result.json")
            existing = []
            if tune_result_path.exists():
                try:
                    existing = json.loads(tune_result_path.read_text())
                except Exception:
                    existing = []
            avg_loss = sum(losses) / len(losses) if losses else None
            avg_ent  = sum(entropies) / len(entropies) if entropies else None
            existing.append({
                "gen":         gen,
                "eis_winrate": round(eis_wins / eis_n, 3),
                "avg_loss":    round(avg_loss, 4) if avg_loss is not None else None,
                "avg_ent":     round(avg_ent,  4) if avg_ent  is not None else None,
                "gen_time_s":  round(time.perf_counter() - t_gen, 1),
            })
            tune_result_path.write_text(json.dumps(existing, indent=2))

        # Dashboard metrics hook — append one line per gen to metrics.jsonl.
        # Written unconditionally so the dashboard always has data.
        avg_loss   = sum(losses)    / len(losses)    if losses    else None
        avg_ent    = sum(entropies) / len(entropies) if entropies else None
        avg_loss_v = sum(loss_vs)   / len(loss_vs)   if loss_vs   else None
        avg_loss_p = sum(loss_ps)   / len(loss_ps)   if loss_ps   else None
        avg_aux    = sum(aux_losses) / len(aux_losses) if aux_losses else None
        avg_sigma  = sum(sigmas)    / len(sigmas)    if sigmas    else None
        move_acc   = move_acc_eval
        _metrics_line = {
            "gen":          gen,
            "avg_loss":     round(avg_loss,   4) if avg_loss   is not None else None,
            "avg_loss_v":   round(avg_loss_v, 4) if avg_loss_v is not None else None,
            "avg_loss_p":   round(avg_loss_p, 4) if avg_loss_p is not None else None,
            "avg_aux":      round(avg_aux,    4) if avg_aux    is not None else None,
            "avg_sigma":    round(avg_sigma,  4) if avg_sigma  is not None else None,
            "avg_ent":      round(avg_ent,    4) if avg_ent    is not None else None,
            "move_acc":     round(move_acc,   3),
            "eis_winrate":  round(eis_wins / eis_n, 3),
            "gen_time_s":   round(time.perf_counter() - t_gen, 1),
            "buffer_size":  len(buffer),
            "positions":    total_positions,
            "lr":           optimizer.param_groups[0]["lr"],
        }
        with open("metrics.jsonl", "a", encoding="utf-8") as _mf:
            _mf.write(json.dumps(_metrics_line) + "\n")

        # Latency summary + bottleneck warnings
        t_total = time.perf_counter() - t_gen
        log.info("  Perf: %s", perf.summary(t_total))
        for w in perf.warnings(t_total, server.avg_batch_size, deduped, total_positions):
            log.warning("  BOTTLENECK: %s", w)
        scheduler.step()
        log.info("  Generation %d done in %.1fs  lr=%.2e", gen, t_total,
                 optimizer.param_groups[0]["lr"])

    log.info("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gens",  type=int, default=50,  help="Generations to train")
    parser.add_argument("--sims",  type=int, default=CFG["SIMS"], help="MCTS sims per move")
    parser.add_argument("--games", type=int, default=20,  help="Self-play games per gen")
    parser.add_argument("--tune",  action="store_true",   help="Tune mode: greedy-only eval, no tournament, writes tune_result.json")
    args = parser.parse_args()
    train(n_gens=args.gens, sims=args.sims, games_per_gen=args.games, tune_mode=args.tune)
