"""
Batched inference server for net-guided MCTS.

Problem: sequential GPU forward passes have ~6ms kernel launch overhead each,
regardless of batch size. 60 sims/move × 6ms = 360ms/move minimum.

Solution: run N games in parallel threads. Each game blocks waiting for a net
evaluation. The server collects requests from all games, batches them into one
GPU call, and returns results. GPU utilisation goes from ~1% to ~N%.

Throughput scales roughly linearly with num_workers until GPU is saturated.
RTX 2060 with our ~121K-param net: expect useful gains up to ~16 workers.

Usage:
    server = InferenceServer(net, batch_size=8, timeout_ms=5)
    server.start()
    value, policy = server.evaluate(game)   # blocks, thread-safe
    server.stop()
"""

import queue
import threading
import time
import numpy as np
import torch

from net import HexNet, encode_board, encode_move, DEVICE, BOARD_SIZE

_SENTINEL = object()


_CUDA = torch.cuda.is_available()

# Persistent cross-generation cache. Entries are (value, policy, gen_added).
# Pass this dict to InferenceServer to keep evaluations across weight updates.
# Eviction: entries older than CACHE_MAX_AGE generations are removed at gen start.
CACHE_MAX_AGE = 2
_persistent_cache: dict = {}   # key → (value, policy, gen)
_persistent_cache_lock = threading.Lock()


def evict_stale_cache(current_gen: int):
    """Remove entries older than CACHE_MAX_AGE from the persistent cache."""
    with _persistent_cache_lock:
        stale = [k for k, (v, p, g) in _persistent_cache.items()
                 if current_gen - g > CACHE_MAX_AGE]
        for k in stale:
            del _persistent_cache[k]


class InferenceServer:
    def __init__(self, net: HexNet, batch_size: int = 8, timeout_ms: float = 5.0,
                 gen: int = 0):
        self.net = net
        self.batch_size = batch_size
        self.timeout = timeout_ms / 1000.0
        self.gen = gen
        self._req_queue: queue.Queue = queue.Queue()
        self._thread = threading.Thread(target=self._serve, daemon=True,
                                        name="inference")
        self._running = False
        # Per-server cache (fast path); persistent cache shared across servers
        # Key: (frozenset(board.items()), current_player, placements_in_turn)
        self.cache: dict[tuple, tuple[float, dict]] = {}
        self._cache_lock = threading.Lock()
        # Latency tracking: list of (batch_size, latency_ms) per batch
        self._batch_latencies: list[tuple[int, float]] = []
        # Stats
        self.total_calls = 0
        self.total_batches = 0
        self.avg_batch_size = 0.0
        self.cache_hits = 0
        self.persistent_hits = 0

    def latency_summary(self) -> str:
        """Return min/avg/max batch latency string for logging."""
        if not self._batch_latencies:
            return "no batches"
        lats = [ms for _, ms in self._batch_latencies]
        sizes = [n for n, _ in self._batch_latencies]
        return (f"lat_ms min={min(lats):.1f} avg={sum(lats)/len(lats):.1f} "
                f"max={max(lats):.1f} | batch_sz avg={sum(sizes)/len(sizes):.1f}")

    def start(self):
        self._running = True
        self._thread.start()

    def stop(self):
        self._running = False
        self._req_queue.put(_SENTINEL)

    def evaluate(self, game) -> tuple[float, dict]:
        """
        Thread-safe. Encodes the game state, submits to batch queue, blocks
        until inference completes. Returns (value, {move: logit}).
        """
        # Include turn state: positions with same pieces but different current_player
        # or placements_in_turn (mid-turn under 1-2-2 rule) are distinct game states.
        key = (frozenset(game.board.items()), game.current_player, game.placements_in_turn)
        # 1. Per-server cache (fastest)
        with self._cache_lock:
            if key in self.cache:
                self.cache_hits += 1
                return self.cache[key]
        # 2. Persistent cross-gen cache
        with _persistent_cache_lock:
            if key in _persistent_cache:
                v, p, _ = _persistent_cache[key]
                self.persistent_hits += 1
                with self._cache_lock:
                    self.cache[key] = (v, p)
                return v, p

        board_arr, (oq, or_) = encode_board(game)
        moves = game.legal_moves()
        if not moves:
            return 0.0, {}

        resp: queue.Queue = queue.Queue(1)
        self._req_queue.put((board_arr, moves, oq, or_, key, resp))
        return resp.get()

    def _serve(self):
        while self._running:
            # Block until at least one request arrives
            try:
                first = self._req_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if first is _SENTINEL:
                break

            batch = [first]
            deadline = time.perf_counter() + self.timeout

            # Collect more requests up to batch_size or timeout
            while len(batch) < self.batch_size:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                try:
                    item = self._req_queue.get(timeout=remaining)
                    if item is _SENTINEL:
                        self._running = False
                        break
                    batch.append(item)
                except queue.Empty:
                    break

            t_batch = time.perf_counter()
            self._process_batch(batch)
            lat_ms = (time.perf_counter() - t_batch) * 1000.0

            # Update stats
            self.total_batches += 1
            self.total_calls += len(batch)
            self.avg_batch_size = self.total_calls / self.total_batches
            self._batch_latencies.append((len(batch), lat_ms))

    def _process_batch(self, batch: list):
        """Run one batched forward pass for all requests.

        Key optimisation: trunk runs once per board (batch of B boards), not once
        per (board × move) pair. The policy head is then applied with features
        broadcast per-request, keeping the expensive trunk pass at B=8 instead
        of B=8*~100_moves=800. This is the dominant factor in inference latency.
        """
        request_slices = []  # (board_idx, resp, valid_moves, key)
        board_arrs = []
        all_planes = []      # [1, S, S] per move, all requests concatenated
        feat_row_idx = []    # which board (0..B-1) each move plane belongs to

        for i, (board_arr, moves, oq, or_, key, resp) in enumerate(batch):
            valid_moves = []
            planes = []
            for m in moves:
                p = encode_move(m[0], m[1], oq, or_)
                if p is not None:
                    valid_moves.append(m)
                    planes.append(p)
            board_arrs.append(board_arr)
            request_slices.append((i, resp, valid_moves, key))
            all_planes.extend(planes)
            feat_row_idx.extend([i] * len(planes))

        # Trunk pass: B boards (not B*N_moves)
        boards_np = np.stack(board_arrs)
        self.net.eval()
        if _CUDA:
            boards_t = (torch.from_numpy(boards_np).pin_memory()
                        .to(DEVICE, non_blocking=True))
        else:
            boards_t = torch.tensor(boards_np, device=DEVICE)

        # Build move tensors outside autocast — they are float32 inputs, not computed tensors
        pol_t = None
        if all_planes:
            moves_np = np.stack(all_planes)
            if _CUDA:
                moves_t = (torch.from_numpy(moves_np).pin_memory()
                           .to(DEVICE, non_blocking=True))
            else:
                moves_t = torch.tensor(moves_np, device=DEVICE)
            row_idx = torch.tensor(feat_row_idx, dtype=torch.long, device=DEVICE)

        dev_type = "cuda" if _CUDA else "cpu"
        with torch.amp.autocast(device_type=dev_type):
            with torch.no_grad():
                features = self.net.trunk(boards_t)          # [B, C, S, S]
                val_t    = self.net.value(features)           # [B]
                if all_planes:
                    pol_t = self.net.policy_logit(features[row_idx], moves_t)  # [M]

        values = val_t.float().cpu().numpy()
        logits = pol_t.float().cpu().numpy() if pol_t is not None else np.array([])

        offset = 0
        for i, (board_idx, resp, valid_moves, key) in enumerate(request_slices):
            n = len(valid_moves)
            policy = {m: float(logits[offset + j]) for j, m in enumerate(valid_moves)}
            offset += n
            res = (float(values[board_idx]), policy)
            with self._cache_lock:
                self.cache[key] = res
            with _persistent_cache_lock:
                _persistent_cache[key] = (float(values[board_idx]), policy, self.gen)
            resp.put(res)
