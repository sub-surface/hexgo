# HexGo — Inference Server (`inference.py`)

## Problem and Solution

Sequential GPU forward passes incur ~6ms kernel launch overhead regardless of
batch size. With 60 MCTS sims/move, sequential calls would cost 360ms/move minimum.

**Solution:** `InferenceServer` collects requests from all parallel self-play threads,
batches them into one GPU call per timeout window, and distributes results back.
GPU utilisation scales roughly linearly with `num_workers` until the GPU saturates.

---

## Architecture

```
N worker threads
  each calls server.evaluate(game) → blocks on resp_queue.get()

InferenceServer._serve() loop:
  1. Wait for first request (blocking, up to 100ms)
  2. Drain queue for up to timeout_ms (default 30ms)
  3. Batch all collected requests → single GPU forward pass
  4. Return (value, policy) to each blocked thread
```

---

## Evaluation Cache

Two-level caching:

### Per-server cache
- Key: `frozenset(game.board.items())` — board pieces only
- Thread-safe via `_cache_lock`
- Clears when `InferenceServer` is re-created (each generation)
- Typical hit rate: 20–40% during MCTS

**Known bug**: cache key ignores `current_player` and `placements_in_turn`.
Under the 1-2-2 rule, two board states with identical piece placement but
different `current_player` (mid-turn vs. start-of-turn) will collide. The
network's `to-move` channel (ch 10) returns different values in each case.
Fix: key should include `(frozenset(board.items()), current_player, placements_in_turn)`.

### Persistent cross-generation cache
- Module-level dict `_persistent_cache` survives across `InferenceServer` instances
- Entries tagged with generation number; evicted after `CACHE_MAX_AGE=5` gens
- `evict_stale_cache(gen)` called before each generation starts
- Trades inference quality (stale weights) for speed on common opening positions
- 5-gen retention is conservative — consider reducing to 2–3 once training stabilizes

---

## CUDA Graphs

`InferenceServer.start()` attempts to capture a CUDA Graph for the full `batch_size`:

```python
with torch.cuda.graph(g):
    self._graph_feat = self.net.trunk(self._graph_boards)   # BUG: rebind, not in-place
    self._graph_val  = self.net.value(self._graph_feat)     # BUG: rebind
    self._graph_pol  = self.net.policy_logit(...)           # BUG: rebind
```

**Critical bug**: The assignments inside `with torch.cuda.graph(g):` rebind Python
names rather than writing into pre-allocated output buffers. CUDA Graph replay
writes into the captured tensor storage, which is no longer referenced by
`self._graph_val` / `self._graph_pol`. After `g.replay()`, reading these variables
returns zeros. The graph path silently returns stale/incorrect values.

Additionally, `_graph_val` is pre-allocated as `[B, 1]` but `net.value()` returns
shape `[B]`, making the `[val_idxs, 0]` indexing in the graph path raise
`IndexError` in any execution that doesn't crash first.

Because all exceptions are caught and discarded, this silently falls back to the
non-graph path — meaning CUDA Graphs are currently a no-op in practice. The
expected 30–50% latency reduction from the roadmap item is not being realized.

---

## `torch.compile`

Applied in `start()` with `dynamic=True` when CUDA is available:
```python
self.net = torch.compile(self.net, dynamic=True)
```

On Windows, Triton (the default compile backend) is poorly supported on native
installs. Failures are silently swallowed (`except Exception: pass`). Whether
`torch.compile` is active or not is unobservable from logs.

`torch.compile` and CUDA Graphs interact poorly (both share the same `net` object;
CUDA Graphs need static shapes while `dynamic=True` generates dynamic dispatch).

---

## Batching Performance

`avg_batch_size ≈ 1.0` is commonly observed. The root cause is the Python GIL
serializing CPU-bound MCTS tree traversal, not the timeout setting. When threads
spend most time in Python-level tree walks between evaluations, they rarely arrive
at the server concurrently within the 30ms window.

True fix: move MCTS tree traversal to C++/Rust (roadmap item 5b). Workaround:
increase `NUM_WORKERS` so more threads are waiting simultaneously.

---

## API

```python
server = InferenceServer(net, batch_size=8, timeout_ms=30.0, gen=0)
server.start()
value, policy = server.evaluate(game)   # blocks, thread-safe
                                         # value ∈ [-1,1] for current player
                                         # policy: dict[(q,r) → logit]
server.stop()
server.latency_summary()                 # min/avg/max batch latency string
```

---

## Known Issues

| Issue | Severity | Status |
|-------|----------|--------|
| CUDA Graph rebinding bug — graph always falls back to eager | Critical | Not fixed |
| `_graph_val` shape `[B,1]` vs value() output `[B]` — IndexError | Critical | Not fixed |
| Cache key ignores `current_player`/`placements_in_turn` | Important | Not fixed |
| `torch.compile` failure silent on Windows | Moderate | By design |
| Persistent cache 5-gen staleness | Suggestion | Tunable |
| `avg_batch_size≈1` root cause is GIL, not timeout | Design note | |
| Stale "355K-param net" comment in module docstring | Minor | Outdated from old arch |
