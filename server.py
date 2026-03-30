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
from fastapi.responses import FileResponse, StreamingResponse

PYTHON       = sys.executable
TRAIN_CMD    = [PYTHON, "train.py"]
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
    ns: dict = {}
    exec(CONFIG_FILE.read_text(encoding="utf-8"), ns)
    current = ns.get("CFG", {})
    unknown = [k for k in cfg if k not in current]
    if unknown:
        raise HTTPException(status_code=400, detail=f"Unknown config keys: {unknown}")
    lines = [
        "# config.py — tunable hyperparameters for HexGo autotune",
        "# Edit this file to propose a new trial config.",
        "# Imported by train.py and mcts.py at startup.",
        "",
        "CFG = {",
    ]
    for k, v in cfg.items():
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
            yield f"data: {json.dumps({'type': 'status', 'data': _singleton.status()})}\n\n"
            while True:
                try:
                    msg = await asyncio.wait_for(q.get(), timeout=15)
                    yield f"data: {msg}\n\n"
                except asyncio.TimeoutError:
                    yield ": heartbeat\n\n"
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
