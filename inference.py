"""
Batched inference server for net-guided MCTS.

Problem: sequential GPU forward passes have ~6ms kernel launch overhead each,
regardless of batch size. 60 sims/move × 6ms = 360ms/move minimum.

Solution: run N games in parallel threads. Each game blocks waiting for a net
evaluation. The server collects requests from all games, batches them into one
GPU call, and returns results. GPU utilisation goes from ~1% to ~N%.

Throughput scales roughly linearly with num_workers until GPU is saturated.
RTX 2060 with our 355K-param net: expect useful gains up to ~16 workers.

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


class InferenceServer:
    def __init__(self, net: HexNet, batch_size: int = 8, timeout_ms: float = 5.0):
        self.net = net
        self.batch_size = batch_size
        self.timeout = timeout_ms / 1000.0
        self._req_queue: queue.Queue = queue.Queue()
        self._thread = threading.Thread(target=self._serve, daemon=True,
                                        name="inference")
        self._running = False
        # Evaluation cache: frozenset(board.items()) -> (value, policy)
        self.cache: dict[frozenset, tuple[float, dict]] = {}
        self._cache_lock = threading.Lock()
        # Stats
        self.total_calls = 0
        self.total_batches = 0
        self.avg_batch_size = 0.0
        self.cache_hits = 0

    def start(self):
        self._running = True
        # 1b: compile net for reduced kernel-launch overhead on CUDA.
        # dynamic=True handles variable batch sizes without recompilation.
        # Falls back silently if torch.compile is unavailable (PyTorch < 2.0).
        if torch.cuda.is_available() and hasattr(torch, "compile"):
            try:
                self.net = torch.compile(self.net, dynamic=True)
            except Exception:
                pass  # non-fatal: older PyTorch or unsupported configuration
        self._thread.start()

    def stop(self):
        self._running = False
        self._req_queue.put(_SENTINEL)

    def evaluate(self, game) -> tuple[float, dict]:
        """
        Thread-safe. Encodes the game state, submits to batch queue, blocks
        until inference completes. Returns (value, {move: logit}).
        """
        # Cache check
        key = frozenset(game.board.items())
        with self._cache_lock:
            if key in self.cache:
                self.cache_hits += 1
                return self.cache[key]

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

            self._process_batch(batch)

            # Update stats
            self.total_batches += 1
            self.total_calls += len(batch)
            self.avg_batch_size = self.total_calls / self.total_batches

    def _process_batch(self, batch: list):
        """Run one batched forward pass for all requests."""
        # Each request may have multiple moves (one row per move in the batch)
        # We stack: all moves from all requests into one big tensor
        board_rows = []    # [3, S, S] repeated per move
        move_planes = []   # [1, S, S] per move
        request_slices = []  # (start, end, resp_queue, moves, value_idx, cache_key)

        for board_arr, moves, oq, or_, key, resp in batch:
            valid_moves = []
            planes = []
            for m in moves:
                p = encode_move(m[0], m[1], oq, or_)
                if p is not None:
                    valid_moves.append(m)
                    planes.append(p)
            
            n = len(valid_moves)
            start = len(board_rows)
            for board_arr_p, p in zip([board_arr] * n, planes):
                board_rows.append(board_arr_p)
                move_planes.append(p)
            request_slices.append((start, start + n, resp, valid_moves, key))

        if not board_rows:
            # All moves in all requests were clipped (extremely rare)
            for board_arr, moves, oq, or_, key, resp in batch:
                # Still need a value head estimate
                net_in = torch.tensor(board_arr, device=DEVICE).unsqueeze(0)
                with torch.no_grad():
                    v = self.net.value(self.net.trunk(net_in)).item()
                res = (float(v), {})
                with self._cache_lock:
                    self.cache[key] = res
                resp.put(res)
            return

        boards_t = torch.tensor(np.stack(board_rows), device=DEVICE)      # [N,3,S,S]
        moves_t  = torch.tensor(np.stack(move_planes), device=DEVICE)     # [N,1,S,S]

        self.net.eval()
        with torch.amp.autocast(device_type="cuda" if "cuda" in str(DEVICE) else "cpu"):
            with torch.no_grad():
                features = self.net.trunk(boards_t)
                # Value: evaluate only the first row of each request
                val_idxs = [s for s, e, r, m, k in request_slices]
                val_t = self.net.value(features[val_idxs])        # [batch]
                pol_t = self.net.policy_logit(features, moves_t)  # [N]

        values  = val_t.cpu().numpy()
        logits  = pol_t.cpu().numpy()

        for i, (start, end, resp, moves, key) in enumerate(request_slices):
            policy = {m: float(logits[start + j]) for j, m in enumerate(moves)}
            res = (float(values[i]), policy)
            with self._cache_lock:
                self.cache[key] = res
            resp.put(res)
