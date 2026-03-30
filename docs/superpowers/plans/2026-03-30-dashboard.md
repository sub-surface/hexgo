# HexGo Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the tkinter `app.py` self-play monitor with a FastAPI + vanilla JS local dashboard that controls training, visualises metrics, and replays saved games — without the UI ever importing PyTorch.

**Architecture:** `server.py` is a FastAPI app with a `ProcessSingleton` that manages the `train.py` subprocess; it exposes REST endpoints and an SSE stream. `dashboard.html` is a single-file frontend (no build step) with three tabs — Training, Replay, Config — using Chart.js for graphs and a `<canvas>` hex board. `app.py` becomes a 20-line launcher. `train.py` gets one new hook: appending a JSON line to `metrics.jsonl` at the end of each generation.

**Tech Stack:** Python 3.12, FastAPI, uvicorn, SSE (`text/event-stream`), Chart.js 4 (CDN), vanilla JS/CSS, existing `config.py` + `elo.json` + `train.log` + `replays/*.json`

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `server.py` | **Create** | FastAPI app, ProcessSingleton, all API routes |
| `dashboard.html` | **Create** | Full single-file frontend |
| `app.py` | **Rewrite** | Launch server + open browser, nothing else |
| `train.py` | **Modify** | Append to `metrics.jsonl` each generation |

---

### Task 1: Add `metrics.jsonl` hook to `train.py`

**Files:**
- Modify: `train.py` (end of generation loop, around line 771)

- [ ] **Step 1: Add the metrics append block**

In `train.py`, after the existing `tune_result.json` block (after line 771) and before the latency summary, add:

```python
        # Dashboard metrics hook — append one line per gen to metrics.jsonl.
        # Written unconditionally so the dashboard always has data.
        avg_loss = sum(losses) / len(losses) if losses else None
        avg_ent  = sum(entropies) / len(entropies) if entropies else None
        _metrics_line = {
            "gen":          gen,
            "avg_loss":     round(avg_loss, 4) if avg_loss is not None else None,
            "avg_ent":      round(avg_ent,  4) if avg_ent  is not None else None,
            "eis_winrate":  round(eis_wins / eis_n, 3),
            "gen_time_s":   round(time.perf_counter() - t_gen, 1),
            "buffer_size":  len(buffer),
            "positions":    total_positions,
        }
        with open("metrics.jsonl", "a", encoding="utf-8") as _mf:
            _mf.write(json.dumps(_metrics_line) + "\n")
```

Note: `avg_loss` and `avg_ent` are already computed locally in the gen loop. `eis_wins` and `eis_n` are already in scope. No new imports needed.

- [ ] **Step 2: Verify the hook works**

```bash
cd "C:\Users\Leon\Desktop\Psychograph\hexgo"
"C:\Program Files\Python312\python.exe" train.py --gens 1 --games 2 --sims 10
```

Expected: `metrics.jsonl` created, contains one JSON line like:
```json
{"gen": 7, "avg_loss": 0.4231, "avg_ent": 2.1, "eis_winrate": 0.0, "gen_time_s": 45.2, "buffer_size": 120, "positions": 120}
```

- [ ] **Step 3: Commit**

```bash
git add train.py
git commit -m "feat: append per-gen metrics to metrics.jsonl for dashboard"
```

---

### Task 2: Create `server.py` — ProcessSingleton + API skeleton

**Files:**
- Create: `server.py`

- [ ] **Step 1: Write `server.py`**

```python
"""
server.py — HexGo dashboard backend.

Endpoints:
  GET  /                       → serve dashboard.html
  GET  /api/status             → process state, current gen
  POST /api/train/start        → launch train.py subprocess
  POST /api/train/stop         → terminate subprocess gracefully
  GET  /api/metrics            → all metrics.jsonl lines as JSON array
  GET  /api/elo                → elo.json contents
  GET  /api/config             → current config.py CFG dict
  POST /api/config             → write staged config to config.py
  GET  /api/replays            → list replays/*.json filenames
  GET  /api/replay/{filename}  → contents of one replay file
  GET  /api/log                → last N lines of train.log
  GET  /events                 → SSE stream: metrics + status updates
"""

import asyncio
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

PYTHON      = sys.executable
TRAIN_CMD   = [PYTHON, "train.py"]
METRICS_FILE = Path("metrics.jsonl")
ELO_FILE     = Path("elo.json")
CONFIG_FILE  = Path("config.py")
LOG_FILE     = Path("train.log")
REPLAYS_DIR  = Path("replays")

app = FastAPI()


# ── Process singleton ─────────────────────────────────────────────────────────

class ProcessSingleton:
    """Manages at most one training subprocess at a time."""

    def __init__(self):
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._start_time: float | None = None
        self._args: dict = {}

    @property
    def running(self) -> bool:
        with self._lock:
            return self._proc is not None and self._proc.poll() is None

    @property
    def pid(self) -> int | None:
        with self._lock:
            return self._proc.pid if self._proc else None

    @property
    def returncode(self) -> int | None:
        with self._lock:
            return self._proc.returncode if self._proc else None

    def start(self, gens: int = 50, games: int = 20, sims: int = 100) -> dict:
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                return {"ok": False, "error": "already running", "pid": self._proc.pid}
            cmd = TRAIN_CMD + [
                "--gens",  str(gens),
                "--games", str(games),
                "--sims",  str(sims),
            ]
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            self._start_time = time.time()
            self._args = {"gens": gens, "games": games, "sims": sims}
            return {"ok": True, "pid": self._proc.pid, "cmd": " ".join(cmd)}

    def stop(self) -> dict:
        with self._lock:
            if self._proc is None:
                return {"ok": False, "error": "not running"}
            if self._proc.poll() is not None:
                return {"ok": False, "error": "already stopped"}
            # Graceful SIGTERM first, then SIGKILL after 5s
            try:
                if sys.platform == "win32":
                    self._proc.terminate()
                else:
                    os.kill(self._proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            return {"ok": True, "pid": self._proc.pid}

    def status(self) -> dict:
        with self._lock:
            if self._proc is None:
                return {"state": "idle", "pid": None}
            rc = self._proc.poll()
            if rc is None:
                elapsed = round(time.time() - (self._start_time or 0), 1)
                return {"state": "running", "pid": self._proc.pid,
                        "elapsed_s": elapsed, "args": self._args}
            return {"state": "stopped", "pid": self._proc.pid, "returncode": rc}


_singleton = ProcessSingleton()


# ── SSE event bus ─────────────────────────────────────────────────────────────

_sse_subscribers: list[asyncio.Queue] = []
_sse_lock = threading.Lock()


def _broadcast(event: dict):
    """Push an event to all SSE subscribers (thread-safe)."""
    data = json.dumps(event)
    with _sse_lock:
        dead = []
        for q in _sse_subscribers:
            try:
                q.put_nowait(data)
            except asyncio.QueueFull:
                dead.append(q)
        for q in dead:
            _sse_subscribers.remove(q)


def _metrics_watcher():
    """Background thread: tail metrics.jsonl and broadcast new lines."""
    pos = 0
    while True:
        time.sleep(1)
        if not METRICS_FILE.exists():
            continue
        try:
            with open(METRICS_FILE, "r", encoding="utf-8") as f:
                f.seek(pos)
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            entry = json.loads(line)
                            _broadcast({"type": "metrics", "data": entry})
                        except json.JSONDecodeError:
                            pass
                pos = f.tell()
        except Exception:
            pass


threading.Thread(target=_metrics_watcher, daemon=True, name="metrics_watcher").start()


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return FileResponse("dashboard.html")


@app.get("/api/status")
def api_status():
    return _singleton.status()


@app.post("/api/train/start")
def api_train_start(gens: int = 50, games: int = 20, sims: int = 100):
    result = _singleton.start(gens=gens, games=games, sims=sims)
    if result["ok"]:
        _broadcast({"type": "status", "data": _singleton.status()})
    return result


@app.post("/api/train/stop")
def api_train_stop():
    result = _singleton.stop()
    _broadcast({"type": "status", "data": _singleton.status()})
    return result


@app.get("/api/metrics")
def api_metrics():
    if not METRICS_FILE.exists():
        return []
    lines = []
    for line in METRICS_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                lines.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return lines


@app.get("/api/elo")
def api_elo():
    if not ELO_FILE.exists():
        return {}
    return json.loads(ELO_FILE.read_text(encoding="utf-8"))


@app.get("/api/config")
def api_config():
    ns: dict = {}
    exec(CONFIG_FILE.read_text(encoding="utf-8"), ns)
    return ns.get("CFG", {})


@app.post("/api/config")
def api_config_write(cfg: dict):
    """Overwrite config.py with the staged values from the frontend."""
    # Validate: all keys must be present in current config
    ns: dict = {}
    exec(CONFIG_FILE.read_text(encoding="utf-8"), ns)
    current = ns.get("CFG", {})
    unknown = [k for k in cfg if k not in current]
    if unknown:
        raise HTTPException(status_code=400, detail=f"Unknown config keys: {unknown}")
    # Write new config.py preserving the header comment
    lines = [
        "# config.py — tunable hyperparameters for HexGo autotune",
        "# Edit this file to propose a new trial config.",
        "# Imported by train.py and mcts.py at startup.",
        "",
        "CFG = {",
    ]
    for k, v in cfg.items():
        if isinstance(v, float):
            lines.append(f'    "{k}": {v},')
        else:
            lines.append(f'    "{k}": {v},')
    lines.append("}")
    lines.append("")
    CONFIG_FILE.write_text("\n".join(lines), encoding="utf-8")
    return {"ok": True}


@app.get("/api/replays")
def api_replays():
    if not REPLAYS_DIR.exists():
        return []
    files = sorted(REPLAYS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return [p.name for p in files]


@app.get("/api/replay/{filename}")
def api_replay(filename: str):
    # Sanitise: no path traversal
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    path = REPLAYS_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Not found")
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/api/log")
def api_log(n: int = 100):
    if not LOG_FILE.exists():
        return {"lines": []}
    lines = LOG_FILE.read_text(encoding="utf-8").splitlines()
    return {"lines": lines[-n:]}


@app.get("/events")
async def api_events():
    """SSE endpoint — pushes metrics and status changes as they happen."""
    q: asyncio.Queue = asyncio.Queue(maxsize=200)
    with _sse_lock:
        _sse_subscribers.append(q)

    async def stream():
        try:
            # Send current status immediately on connect
            yield f"data: {json.dumps({'type': 'status', 'data': _singleton.status()})}\n\n"
            while True:
                try:
                    msg = await asyncio.wait_for(q.get(), timeout=15)
                    yield f"data: {msg}\n\n"
                except asyncio.TimeoutError:
                    yield ": heartbeat\n\n"  # keep connection alive
        finally:
            with _sse_lock:
                try:
                    _sse_subscribers.remove(q)
                except ValueError:
                    pass

    return StreamingResponse(stream(), media_type="text/event-stream",
                              headers={"Cache-Control": "no-cache",
                                       "X-Accel-Buffering": "no"})


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=7860, log_level="warning")
```

- [ ] **Step 2: Install dependencies**

```bash
"C:\Program Files\Python312\python.exe" -m pip install fastapi uvicorn[standard]
```

Expected: both install without error.

- [ ] **Step 3: Smoke-test the server starts**

```bash
cd "C:\Users\Leon\Desktop\Psychograph\hexgo"
"C:\Program Files\Python312\python.exe" server.py
```

Expected: uvicorn starts on `http://127.0.0.1:7860` with no errors. Ctrl+C to stop.

- [ ] **Step 4: Test key endpoints**

With the server running, in a second terminal:

```bash
curl http://127.0.0.1:7860/api/status
# Expected: {"state":"idle","pid":null}

curl http://127.0.0.1:7860/api/config
# Expected: {"LR":0.001,"WEIGHT_DECAY":0.0001,...}

curl http://127.0.0.1:7860/api/metrics
# Expected: [] or array of gen objects if metrics.jsonl exists

curl http://127.0.0.1:7860/api/replays
# Expected: array of replay filenames
```

- [ ] **Step 5: Commit**

```bash
git add server.py
git commit -m "feat: add FastAPI server with ProcessSingleton and REST+SSE API"
```

---

### Task 3: Create `dashboard.html` — shell, tabs, dark theme

**Files:**
- Create: `dashboard.html`

- [ ] **Step 1: Write the HTML shell with tabs and dark theme**

```html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>HexGo Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  :root {
    --bg:       #0d1117;
    --surface:  #161b22;
    --border:   #21262d;
    --text:     #e6edf3;
    --dim:      #7d8590;
    --accent:   #388bfd;
    --green:    #3fb950;
    --orange:   #d29922;
    --red:      #f85149;
    --p1:       #e8a020;
    --p2:       #1a6fb5;
    --font:     'Courier New', monospace;
  }
  body { background: var(--bg); color: var(--text); font-family: var(--font); font-size: 13px; height: 100vh; display: flex; flex-direction: column; overflow: hidden; }

  /* Tab bar */
  #tabbar { display: flex; border-bottom: 1px solid var(--border); background: var(--surface); padding: 0 16px; gap: 2px; flex-shrink: 0; }
  .tab { padding: 10px 20px; cursor: pointer; border-bottom: 2px solid transparent; color: var(--dim); transition: color .15s, border-color .15s; user-select: none; }
  .tab:hover { color: var(--text); }
  .tab.active { color: var(--accent); border-bottom-color: var(--accent); }
  #title { margin-left: auto; padding: 10px 0; color: var(--dim); letter-spacing: 3px; font-size: 11px; }

  /* Tab content */
  .tabpanel { display: none; flex: 1; overflow: hidden; }
  .tabpanel.active { display: flex; }

  /* Shared layout primitives */
  .panel { background: var(--surface); border: 1px solid var(--border); border-radius: 6px; padding: 14px; }
  .panel-title { color: var(--dim); font-size: 10px; letter-spacing: 2px; margin-bottom: 10px; }
  .row { display: flex; gap: 12px; }
  .col { display: flex; flex-direction: column; gap: 12px; }
  .badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px; }
  .badge-green { background: #1a3a1e; color: var(--green); }
  .badge-red   { background: #3a1a1a; color: var(--red); }
  .badge-dim   { background: var(--border); color: var(--dim); }
  btn, button { background: var(--border); color: var(--text); border: 1px solid #30363d; border-radius: 4px; padding: 6px 14px; cursor: pointer; font-family: var(--font); font-size: 12px; transition: background .15s; }
  button:hover { background: #21262d; }
  button.primary { background: #1f3a6e; border-color: var(--accent); color: var(--accent); }
  button.primary:hover { background: #2a4a8e; }
  button.danger { background: #3a1a1a; border-color: var(--red); color: var(--red); }
  button.danger:hover { background: #4a2020; }
  button:disabled { opacity: .4; cursor: not-allowed; }
  input[type=number], input[type=text], select {
    background: var(--bg); border: 1px solid var(--border); color: var(--text);
    border-radius: 4px; padding: 5px 8px; font-family: var(--font); font-size: 12px; width: 100%;
  }
  input:focus, select:focus { outline: none; border-color: var(--accent); }
  label { color: var(--dim); font-size: 11px; margin-bottom: 3px; display: block; }
  .stat-row { display: flex; justify-content: space-between; padding: 4px 0; border-bottom: 1px solid var(--border); }
  .stat-row:last-child { border-bottom: none; }
  .stat-key { color: var(--dim); }
  .stat-val { color: var(--text); font-weight: bold; }
</style>
</head>
<body>

<div id="tabbar">
  <div class="tab active" data-tab="training">Training</div>
  <div class="tab" data-tab="replay">Replay</div>
  <div class="tab" data-tab="config">Config</div>
  <div id="title">HEXGO</div>
</div>

<!-- Training tab -->
<div class="tabpanel active" id="tab-training" style="padding:14px; gap:12px; overflow:auto;">
  <!-- row 1: controls + status -->
  <div class="row" style="flex-shrink:0;">
    <div class="panel col" style="flex:0 0 220px;">
      <div class="panel-title">CONTROLS</div>
      <div class="col" style="gap:8px;">
        <div><label>Generations</label><input type="number" id="c-gens" value="50" min="1" max="9999"></div>
        <div><label>Games / gen</label><input type="number" id="c-games" value="20" min="1" max="200"></div>
        <div><label>Sims / move</label><input type="number" id="c-sims" value="100" min="5" max="2000"></div>
      </div>
      <div class="row" style="margin-top:10px; gap:8px;">
        <button class="primary" id="btn-start" onclick="trainStart()">▶ START</button>
        <button class="danger"  id="btn-stop"  onclick="trainStop()" disabled>■ STOP</button>
      </div>
    </div>
    <div class="panel col" style="flex:1;">
      <div class="panel-title">STATUS</div>
      <div id="status-badge" class="badge badge-dim">IDLE</div>
      <div id="status-stats" style="margin-top:8px;"></div>
    </div>
    <div class="panel col" style="flex:1;">
      <div class="panel-title">LAST GEN</div>
      <div id="last-gen-stats"></div>
    </div>
  </div>
  <!-- row 2: charts -->
  <div class="row" style="flex:1; min-height:0;">
    <div class="panel" style="flex:1; min-height:0;">
      <div class="panel-title">LOSS</div>
      <canvas id="chart-loss"></canvas>
    </div>
    <div class="panel" style="flex:1; min-height:0;">
      <div class="panel-title">EIS WINRATE</div>
      <canvas id="chart-eis"></canvas>
    </div>
    <div class="panel" style="flex:1; min-height:0;">
      <div class="panel-title">ENTROPY</div>
      <canvas id="chart-ent"></canvas>
    </div>
  </div>
  <!-- row 3: log tail -->
  <div class="panel" style="flex-shrink:0; max-height:160px;">
    <div class="panel-title">TRAIN LOG <span style="float:right;cursor:pointer;color:var(--accent)" onclick="refreshLog()">↻</span></div>
    <pre id="log-tail" style="color:var(--dim); font-size:11px; overflow-y:auto; max-height:120px; white-space:pre-wrap;"></pre>
  </div>
</div>

<!-- Replay tab -->
<div class="tabpanel" id="tab-replay" style="padding:14px; gap:12px; overflow:hidden;">
  <div class="row" style="flex:1; min-height:0;">
    <!-- file list -->
    <div class="panel col" style="flex:0 0 220px; overflow:hidden;">
      <div class="panel-title">REPLAYS <span style="float:right;cursor:pointer;color:var(--accent)" onclick="loadReplayList()">↻</span></div>
      <div id="replay-list" style="overflow-y:auto; flex:1; gap:4px; display:flex; flex-direction:column;"></div>
    </div>
    <!-- board -->
    <div class="panel col" style="flex:1; align-items:center;">
      <div class="panel-title" id="replay-title">SELECT A REPLAY</div>
      <canvas id="replay-canvas" style="flex:1; width:100%; max-height:100%;"></canvas>
    </div>
    <!-- controls + info -->
    <div class="panel col" style="flex:0 0 200px;">
      <div class="panel-title">PLAYBACK</div>
      <div class="stat-row"><span class="stat-key">Move</span><span class="stat-val" id="r-move">—</span></div>
      <div class="stat-row"><span class="stat-key">Player</span><span class="stat-val" id="r-player">—</span></div>
      <div class="stat-row"><span class="stat-key">Winner</span><span class="stat-val" id="r-winner">—</span></div>
      <div class="stat-row"><span class="stat-key">Total moves</span><span class="stat-val" id="r-total">—</span></div>
      <div style="margin-top:12px;">
        <label>Speed (ms/move)</label>
        <input type="number" id="r-speed" value="300" min="50" max="2000">
      </div>
      <input type="range" id="r-scrub" min="0" value="0" style="width:100%; margin:10px 0; accent-color:var(--accent);">
      <div class="row" style="flex-wrap:wrap; gap:6px; margin-top:4px;">
        <button onclick="replayStep(-1)">◀</button>
        <button onclick="replayStep(1)">▶</button>
        <button onclick="replayTogglePlay()" id="r-playbtn">▶▶</button>
        <button onclick="replayReset()">↺</button>
      </div>
    </div>
  </div>
</div>

<!-- Config tab -->
<div class="tabpanel" id="tab-config" style="padding:14px; overflow:auto;">
  <div class="row" style="max-width:800px; flex-wrap:wrap; gap:12px;">
    <div class="panel col" style="flex:1; min-width:280px;">
      <div class="panel-title">HYPERPARAMETERS</div>
      <div id="config-fields" class="col" style="gap:10px;"></div>
      <div class="row" style="margin-top:14px; gap:8px; align-items:center;">
        <button class="primary" onclick="configApply()">✓ APPLY TO config.py</button>
        <button onclick="configReset()">↺ RESET</button>
        <span id="config-status" style="color:var(--dim); font-size:11px;"></span>
      </div>
      <div style="margin-top:8px; color:var(--dim); font-size:11px;">Changes are staged until APPLY. Takes effect on next training run.</div>
    </div>
    <div class="panel col" style="flex:1; min-width:280px;">
      <div class="panel-title">PARAMETER GUIDE</div>
      <div id="config-guide" style="color:var(--dim); font-size:11px; line-height:1.7;"></div>
    </div>
  </div>
</div>

<script>
// ── Tab switching ─────────────────────────────────────────────────────────────
document.querySelectorAll('.tab').forEach(t => {
  t.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach(x => x.classList.remove('active'));
    document.querySelectorAll('.tabpanel').forEach(x => x.classList.remove('active'));
    t.classList.add('active');
    document.getElementById('tab-' + t.dataset.tab).classList.add('active');
    if (t.dataset.tab === 'replay') loadReplayList();
    if (t.dataset.tab === 'config') loadConfig();
  });
});

// ── Shared fetch helper ───────────────────────────────────────────────────────
async function api(path, opts={}) {
  const r = await fetch(path, opts);
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

// ── Charts setup ──────────────────────────────────────────────────────────────
const CHART_OPTS = (label, color) => ({
  type: 'line',
  data: { labels: [], datasets: [{ label, data: [], borderColor: color,
    backgroundColor: color + '22', borderWidth: 1.5, pointRadius: 2, tension: 0.3 }] },
  options: {
    animation: false, responsive: true, maintainAspectRatio: false,
    plugins: { legend: { display: false } },
    scales: {
      x: { ticks: { color: '#7d8590', font: { family: 'Courier New', size: 10 } }, grid: { color: '#21262d' } },
      y: { ticks: { color: '#7d8590', font: { family: 'Courier New', size: 10 } }, grid: { color: '#21262d' } },
    }
  }
});

const chartLoss = new Chart(document.getElementById('chart-loss'), CHART_OPTS('loss', '#388bfd'));
const chartEis  = new Chart(document.getElementById('chart-eis'),  CHART_OPTS('eis_wr', '#3fb950'));
const chartEnt  = new Chart(document.getElementById('chart-ent'),  CHART_OPTS('entropy', '#d29922'));

function pushMetric(d) {
  const gen = String(d.gen);
  [chartLoss, chartEis, chartEnt].forEach(c => {
    if (!c.data.labels.includes(gen)) c.data.labels.push(gen);
  });
  const idx = chartLoss.data.labels.indexOf(gen);
  chartLoss.data.datasets[0].data[idx] = d.avg_loss;
  chartEis.data.datasets[0].data[idx]  = d.eis_winrate;
  chartEnt.data.datasets[0].data[idx]  = d.avg_ent;
  [chartLoss, chartEis, chartEnt].forEach(c => c.update('none'));

  // last gen stats
  document.getElementById('last-gen-stats').innerHTML = `
    <div class="stat-row"><span class="stat-key">Gen</span><span class="stat-val">${d.gen}</span></div>
    <div class="stat-row"><span class="stat-key">Loss</span><span class="stat-val">${d.avg_loss ?? '—'}</span></div>
    <div class="stat-row"><span class="stat-key">Eis WR</span><span class="stat-val">${d.eis_winrate ?? '—'}</span></div>
    <div class="stat-row"><span class="stat-key">Buffer</span><span class="stat-val">${d.buffer_size ?? '—'}</span></div>
    <div class="stat-row"><span class="stat-key">Gen time</span><span class="stat-val">${d.gen_time_s ?? '—'}s</span></div>
  `;
}

// Load historical metrics on startup
api('/api/metrics').then(arr => arr.forEach(pushMetric)).catch(() => {});

// ── SSE stream ────────────────────────────────────────────────────────────────
function connectSSE() {
  const es = new EventSource('/events');
  es.onmessage = e => {
    try {
      const msg = JSON.parse(e.data);
      if (msg.type === 'metrics') pushMetric(msg.data);
      if (msg.type === 'status')  updateStatus(msg.data);
    } catch(_) {}
  };
  es.onerror = () => setTimeout(connectSSE, 3000);
}
connectSSE();

// ── Status panel ──────────────────────────────────────────────────────────────
function updateStatus(s) {
  const badge = document.getElementById('status-badge');
  const stats = document.getElementById('status-stats');
  const start = document.getElementById('btn-start');
  const stop  = document.getElementById('btn-stop');

  if (s.state === 'running') {
    badge.className = 'badge badge-green'; badge.textContent = '● RUNNING';
    start.disabled = true; stop.disabled = false;
    stats.innerHTML = `<div class="stat-row"><span class="stat-key">PID</span><span class="stat-val">${s.pid}</span></div>
      <div class="stat-row"><span class="stat-key">Elapsed</span><span class="stat-val">${s.elapsed_s}s</span></div>`;
  } else {
    badge.className = 'badge badge-dim'; badge.textContent = 'IDLE';
    start.disabled = false; stop.disabled = true;
    stats.innerHTML = s.returncode !== undefined
      ? `<div class="stat-row"><span class="stat-key">Exit code</span><span class="stat-val">${s.returncode}</span></div>` : '';
  }
}
// Poll status every 5s (SSE handles fast updates; this is a fallback)
setInterval(() => api('/api/status').then(updateStatus).catch(() => {}), 5000);
api('/api/status').then(updateStatus).catch(() => {});

// ── Training controls ─────────────────────────────────────────────────────────
async function trainStart() {
  const gens  = document.getElementById('c-gens').value;
  const games = document.getElementById('c-games').value;
  const sims  = document.getElementById('c-sims').value;
  try {
    await api(`/api/train/start?gens=${gens}&games=${games}&sims=${sims}`, {method:'POST'});
  } catch(e) { alert('Start failed: ' + e.message); }
}
async function trainStop() {
  try { await api('/api/train/stop', {method:'POST'}); }
  catch(e) { alert('Stop failed: ' + e.message); }
}

// ── Log tail ──────────────────────────────────────────────────────────────────
async function refreshLog() {
  try {
    const data = await api('/api/log?n=60');
    document.getElementById('log-tail').textContent = data.lines.join('\n');
  } catch(_) {}
}
setInterval(refreshLog, 3000);
refreshLog();

// ── Config tab ────────────────────────────────────────────────────────────────
const CONFIG_GUIDE = {
  LR:                 'Adam learning rate',
  WEIGHT_DECAY:       'L2 regularisation weight',
  BATCH_SIZE:         'Training batch size',
  SIMS:               'Full MCTS sim budget (25% of games)',
  SIMS_MIN:           'Reduced sim budget floor (75% of games) — keep << SIMS',
  CAP_FULL_FRAC:      'Fraction of games using full SIMS',
  CPUCT:              'PUCT exploration constant',
  DIRICHLET_ALPHA:    'Root Dirichlet noise concentration',
  DIRICHLET_EPS:      'Root Dirichlet noise weight',
  ZOI_MARGIN:         'Zone-of-influence pruning radius (hex distance)',
  TD_GAMMA:           'TD-lambda discount for value targets',
  TEMP_HORIZON:       'Cosine temp annealing full-life (moves)',
  WEIGHT_SYNC_BATCHES:'Batches between weight sync to inference server',
};
let _cfgStaged = null;

async function loadConfig() {
  const cfg = await api('/api/config');
  _cfgStaged = Object.assign({}, cfg);
  const fields = document.getElementById('config-fields');
  fields.innerHTML = '';
  const guide = document.getElementById('config-guide');
  guide.innerHTML = Object.entries(CONFIG_GUIDE).map(([k,v]) =>
    `<div><span style="color:var(--text)">${k}</span>: ${v}</div>`).join('');
  for (const [k, v] of Object.entries(cfg)) {
    const wrap = document.createElement('div');
    wrap.innerHTML = `<label>${k}</label>
      <input type="text" id="cfg-${k}" value="${v}" data-key="${k}" oninput="cfgStage(this)">`;
    fields.appendChild(wrap);
  }
}

function cfgStage(el) {
  const key = el.dataset.key;
  const raw = el.value;
  const num = Number(raw);
  _cfgStaged[key] = isNaN(num) ? raw : num;
  document.getElementById('config-status').textContent = '● unsaved changes';
}

function configReset() {
  loadConfig();
  document.getElementById('config-status').textContent = '';
}

async function configApply() {
  try {
    await api('/api/config', { method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(_cfgStaged) });
    document.getElementById('config-status').textContent = '✓ saved';
    setTimeout(() => document.getElementById('config-status').textContent = '', 2000);
  } catch(e) {
    document.getElementById('config-status').textContent = '✗ ' + e.message;
  }
}

// ── Replay tab ────────────────────────────────────────────────────────────────
let _replayData   = null;   // full replay JSON
let _replayIdx    = 0;      // current move index (0 = empty board)
let _replayBoard  = {};     // (q,r) -> player
let _replayTimer  = null;   // auto-play interval ID
let _replayPlaying = false;

async function loadReplayList() {
  const list = await api('/api/replays');
  const el = document.getElementById('replay-list');
  el.innerHTML = '';
  list.forEach(name => {
    const btn = document.createElement('div');
    btn.textContent = name.replace('game_','').replace('.json','');
    btn.style.cssText = 'padding:5px 8px;cursor:pointer;border-radius:4px;color:var(--dim);font-size:11px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;';
    btn.title = name;
    btn.onclick = () => loadReplay(name);
    el.appendChild(btn);
  });
}

async function loadReplay(filename) {
  _replayData = await api('/api/replay/' + filename);
  _replayIdx = 0;
  _replayBoard = {};
  replayStopPlay();
  const total = _replayData.moves.length;
  document.getElementById('r-total').textContent = total;
  document.getElementById('r-winner').textContent =
    _replayData.winner === 1 ? 'X' : _replayData.winner === 2 ? 'O' : 'draw';
  document.getElementById('replay-title').textContent =
    filename.replace('game_','').replace('.json','');
  const scrub = document.getElementById('r-scrub');
  scrub.max = total;
  scrub.value = 0;
  scrub.oninput = () => replayJump(parseInt(scrub.value));
  replayDraw();
}

function replayJump(idx) {
  if (!_replayData) return;
  _replayIdx = Math.max(0, Math.min(idx, _replayData.moves.length));
  // Rebuild board from scratch up to idx
  _replayBoard = {};
  let cur = 1, placed = 0;
  for (let i = 0; i < _replayIdx; i++) {
    const [q, r] = _replayData.moves[i];
    _replayBoard[`${q},${r}`] = cur;
    placed++;
    const limit = (i < 1) ? 1 : 2;
    if (placed >= limit) { cur = 3 - cur; placed = 0; }
  }
  document.getElementById('r-scrub').value = _replayIdx;
  replayDraw();
}

function replayStep(delta) {
  if (!_replayData) return;
  replayJump(_replayIdx + delta);
}

function replayReset() { replayJump(0); }

function replayTogglePlay() {
  if (_replayPlaying) { replayStopPlay(); return; }
  _replayPlaying = true;
  document.getElementById('r-playbtn').textContent = '⏸';
  function tick() {
    if (!_replayPlaying || !_replayData) return;
    if (_replayIdx >= _replayData.moves.length) { replayStopPlay(); return; }
    replayStep(1);
    const ms = parseInt(document.getElementById('r-speed').value) || 300;
    _replayTimer = setTimeout(tick, ms);
  }
  tick();
}

function replayStopPlay() {
  _replayPlaying = false;
  document.getElementById('r-playbtn').textContent = '▶▶';
  if (_replayTimer) { clearTimeout(_replayTimer); _replayTimer = null; }
}

// ── Hex canvas renderer ───────────────────────────────────────────────────────
const HEX_COLORS = {
  bg:    '#0d1117', gridLine: '#1e2a38',
  p1:    '#e8a020', p1b: '#ffcc66',
  p2:    '#1a6fb5', p2b: '#66aaee',
  empty: '#141c26', emptyB: '#1e2a38',
  last:  '#ffffff',
};

function replayDraw() {
  const canvas = document.getElementById('replay-canvas');
  const ctx = canvas.getContext('2d');
  // Resize canvas to match display size
  canvas.width  = canvas.offsetWidth  || 600;
  canvas.height = canvas.offsetHeight || 400;

  const W = canvas.width, H = canvas.height;
  ctx.fillStyle = HEX_COLORS.bg;
  ctx.fillRect(0, 0, W, H);

  if (!_replayData) return;

  // Determine board extent from all moves
  const allMoves = _replayData.moves;
  if (!allMoves.length) return;
  const qs = allMoves.map(m => m[0]);
  const rs = allMoves.map(m => m[1]);
  const qMin = Math.min(...qs) - 1, qMax = Math.max(...qs) + 1;
  const rMin = Math.min(...rs) - 1, rMax = Math.max(...rs) + 1;

  // Auto-fit hex size
  const cols = qMax - qMin + 1, rows = rMax - rMin + 1;
  const size = Math.max(8, Math.min(30, Math.floor(Math.min(W / (cols * 1.6), H / (rows * 1.2)))));
  const ox = W / 2 - (size * 1.5 * ((qMin + qMax) / 2));
  const oy = H / 2 - (size * Math.sqrt(3) * ((rMin + rMax) / 2 + (qMin + qMax) / 4));

  function hexToPixel(q, r) {
    return [ox + size * 1.5 * q, oy + size * Math.sqrt(3) * (r + q / 2)];
  }
  function hexPath(cx, cy, s) {
    ctx.beginPath();
    for (let i = 0; i < 6; i++) {
      const a = Math.PI / 180 * 60 * i;
      const x = cx + (s - 1) * Math.cos(a), y = cy + (s - 1) * Math.sin(a);
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }
    ctx.closePath();
  }

  const lastMove = _replayIdx > 0 ? _replayData.moves[_replayIdx - 1] : null;

  // Draw background cells
  for (let q = qMin; q <= qMax; q++) {
    for (let r = rMin; r <= rMax; r++) {
      const [cx, cy] = hexToPixel(q, r);
      const key = `${q},${r}`;
      const p = _replayBoard[key];
      hexPath(cx, cy, size);
      if (p == null) {
        ctx.fillStyle = HEX_COLORS.empty;
        ctx.strokeStyle = HEX_COLORS.emptyB;
        ctx.lineWidth = 0.5;
      } else {
        ctx.fillStyle = p === 1 ? HEX_COLORS.p1 : HEX_COLORS.p2;
        const isLast = lastMove && lastMove[0] === q && lastMove[1] === r;
        ctx.strokeStyle = isLast ? HEX_COLORS.last : (p === 1 ? HEX_COLORS.p1b : HEX_COLORS.p2b);
        ctx.lineWidth = isLast ? 2.5 : 1;
      }
      ctx.fill(); ctx.stroke();
      if (p != null) {
        ctx.fillStyle = p === 1 ? HEX_COLORS.p1b : HEX_COLORS.p2b;
        ctx.font = `bold ${Math.max(6, Math.floor(size * 0.6))}px Courier New`;
        ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        ctx.fillText(p === 1 ? 'X' : 'O', cx, cy);
      }
    }
  }

  // Update info
  const move = _replayIdx;
  document.getElementById('r-move').textContent = move;
  if (move > 0) {
    let cur = 1, placed = 0;
    for (let i = 0; i < move - 1; i++) {
      placed++;
      if (placed >= (i < 1 ? 1 : 2)) { cur = 3 - cur; placed = 0; }
    }
    document.getElementById('r-player').textContent = cur === 1 ? 'X' : 'O';
  } else {
    document.getElementById('r-player').textContent = '—';
  }
}

// Resize canvas when window resizes
window.addEventListener('resize', () => { if (_replayData) replayDraw(); });
</script>
</body>
</html>
```

- [ ] **Step 2: Verify it loads**

Start server, open `http://127.0.0.1:7860` in browser. Check:
- Three tabs visible: Training, Replay, Config
- Training tab shows controls panel, three chart areas, log tail
- No JS errors in browser console

- [ ] **Step 3: Commit**

```bash
git add dashboard.html
git commit -m "feat: add single-file dashboard with training/replay/config tabs"
```

---

### Task 4: Rewrite `app.py` as a thin launcher

**Files:**
- Modify: `app.py` (full rewrite)

- [ ] **Step 1: Rewrite `app.py`**

```python
"""
app.py — HexGo dashboard launcher.

Starts the FastAPI server (server.py) and opens the dashboard in the
default browser. The server runs until Ctrl+C.

Usage: python app.py [--port 7860] [--no-browser]
"""

import argparse
import threading
import time
import webbrowser

import uvicorn

from server import app


def main():
    parser = argparse.ArgumentParser(description="HexGo Dashboard")
    parser.add_argument("--port",       type=int, default=7860)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    url = f"http://127.0.0.1:{args.port}"
    print(f"HexGo Dashboard → {url}")

    if not args.no-browser:
        # Open browser after a short delay so the server is ready
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()

    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
```

Note: fix the `args.no-browser` → `args.no_browser` (argparse converts `-` to `_`).

- [ ] **Step 2: Fix the hyphen typo**

The `--no-browser` flag becomes `args.no_browser` in argparse. The actual write should be:

```python
    if not args.no_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
```

- [ ] **Step 3: Test the launcher**

```bash
cd "C:\Users\Leon\Desktop\Psychograph\hexgo"
"C:\Program Files\Python312\python.exe" app.py
```

Expected: browser opens to `http://127.0.0.1:7860`, dashboard loads. Ctrl+C cleanly stops.

- [ ] **Step 4: Commit**

```bash
git add app.py
git commit -m "refactor: app.py is now a thin launcher for the FastAPI dashboard"
```

---

### Task 5: End-to-end smoke test

**Files:** none changed

- [ ] **Step 1: Full flow test**

```bash
# Terminal 1: start the dashboard
cd "C:\Users\Leon\Desktop\Psychograph\hexgo"
"C:\Program Files\Python312\python.exe" app.py --no-browser
```

In browser at `http://127.0.0.1:7860`:

1. **Config tab**: values load from `config.py`. Change `SIMS` to 60. Click APPLY. Open `config.py` and confirm value changed.
2. **Config tab**: click RESET. Confirm value reverts in the UI (but `config.py` keeps the applied value — reset only reloads from disk).
3. **Training tab**: set Gens=1, Games=2, Sims=15. Click START. Badge turns green. After ~1 min, badge returns to IDLE. Check `metrics.jsonl` has a new line.
4. **Training tab**: charts update with the new gen's loss/eis_winrate/entropy.
5. **Replay tab**: click refresh, select the newly saved replay. Board renders. Step forward with ▶. Auto-play works. Scrubber jumps to any move.
6. **Training tab** log tail: shows recent `train.log` lines.

- [ ] **Step 2: Commit**

```bash
git add metrics.jsonl  # if created during test
git commit -m "test: smoke-tested full dashboard flow end-to-end"
```

---

### Task 6: Update `.gitignore` and clean up

**Files:**
- Modify: `.gitignore` (or create if absent)

- [ ] **Step 1: Ensure metrics.jsonl and logs aren't committed accidentally**

Check if `.gitignore` exists:

```bash
cat .gitignore 2>/dev/null || echo "(none)"
```

Add if not already present:

```
metrics.jsonl
train.log
hexgo.log
tune_log.jsonl
tune_result.json
*.bak
__pycache__/
*.pyc
```

- [ ] **Step 2: Commit**

```bash
git add .gitignore
git commit -m "chore: update .gitignore for dashboard artefacts"
```

---

## Self-Review

**Spec coverage:**
- ✅ `app.py` as launcher only — Task 4
- ✅ Dashboard with tabbed layout — Task 3
- ✅ Training tab: start/stop/controls, live charts, log tail — Task 3
- ✅ Config tab: staged edit + apply — Task 3 + Task 2
- ✅ Replay tab: file list, hex canvas, step + auto-play — Task 3
- ✅ ProcessSingleton for safety — Task 2
- ✅ `train.py` metrics hook — Task 1
- ✅ SSE for live updates — Task 2
- ✅ Dark mode — Task 3

**Placeholder scan:** None found. All code blocks are complete.

**Type consistency:** `pushMetric(d)` in Task 3 matches the `_metrics_line` dict keys written in Task 1 (`gen`, `avg_loss`, `avg_ent`, `eis_winrate`, `gen_time_s`, `buffer_size`, `positions`). `api('/api/replay/' + filename)` in Task 3 matches `api_replay(filename: str)` route in Task 2. Config POST in Task 3 sends `JSON.stringify(_cfgStaged)` as a dict, matches `api_config_write(cfg: dict)` in Task 2. ✅
