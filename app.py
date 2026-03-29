"""
Hexagonal self-play monitor — tkinter, stdlib only.

UI fix: incremental canvas updates — only draw new/changed cells, never delete-all.
Board resets between games by clearing all items and the item registry.

Layout:
  Left  — live hex board (Canvas)
  Right — stats panel: game#, move#, sims/s, wins, scrolling game log
"""

import logging
import math
import queue
import sys
import threading
import time
import traceback
import tkinter as tk
from pathlib import Path

from game import HexGame
from mcts import mcts

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s.%(msecs)03d %(levelname)-5s %(threadName)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler("hexgo.log", mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("hexgo")

def _thread_excepthook(args):
    log.error("Uncaught thread exception:\n%s", "".join(
        traceback.format_exception(args.exc_type, args.exc_value, args.exc_traceback)))

threading.excepthook = _thread_excepthook

# ── Colours ───────────────────────────────────────────────────────────────────
BG        = "#0d1117"
GRID_LINE = "#1e2a38"
P1_FILL   = "#e8a020"
P1_BDR    = "#ffcc66"
P2_FILL   = "#1a6fb5"
P2_BDR    = "#66aaee"
EMPTY_F   = "#141c26"
EMPTY_B   = "#1e2a38"
LAST_BDR  = "#ffffff"
TEXT_DIM  = "#4a5568"
TEXT_MAIN = "#e2e8f0"
TEXT_ACC  = "#63b3ed"

HEX_SIZE  = 22
SIMS      = 120
POLL_MS   = 80


# ── Hex geometry ──────────────────────────────────────────────────────────────

def hex_to_pixel(q, r, size, ox, oy):
    x = ox + size * 1.5 * q
    y = oy + size * math.sqrt(3) * (r + q / 2)
    return x, y

def hex_corners(cx, cy, size):
    pts = []
    for i in range(6):
        a = math.radians(60 * i)
        pts += [cx + size * math.cos(a), cy + size * math.sin(a)]
    return pts


# ── Stats panel ───────────────────────────────────────────────────────────────

class StatsPanel(tk.Frame):
    def __init__(self, master):
        super().__init__(master, bg=BG, width=200)
        self.pack_propagate(False)

        tk.Label(self, text="HEXGO", bg=BG, fg=TEXT_ACC,
                 font=("Courier", 15, "bold")).pack(pady=(24, 2))
        tk.Label(self, text="self-play monitor", bg=BG, fg=TEXT_DIM,
                 font=("Courier", 8)).pack(pady=(0, 20))

        self._rows = {}
        for key in ("game", "move", "player", "sims/s", "last", "wins"):
            self._add_row(key)

        tk.Frame(self, bg=GRID_LINE, height=1).pack(fill="x", padx=16, pady=8)

        # Game log
        tk.Label(self, text="GAME LOG", bg=BG, fg=TEXT_DIM,
                 font=("Courier", 7)).pack(anchor="w", padx=16)
        self._log = tk.Text(self, bg=BG, fg=TEXT_DIM, font=("Courier", 7),
                            width=22, height=6, bd=0, highlightthickness=0,
                            state="disabled", wrap="word")
        self._log.pack(padx=12, fill="x")

        tk.Frame(self, bg=GRID_LINE, height=1).pack(fill="x", padx=16, pady=8)

        # Training log (reads train.log if present)
        tk.Label(self, text="TRAINING", bg=BG, fg=TEXT_DIM,
                 font=("Courier", 7)).pack(anchor="w", padx=16)
        self._train_log = tk.Text(self, bg=BG, fg="#4a7a58", font=("Courier", 7),
                                  width=22, height=7, bd=0, highlightthickness=0,
                                  state="disabled", wrap="word")
        self._train_log.pack(padx=12, fill="x", expand=True)
        self._train_log_pos = 0  # byte offset into train.log

        # Controls
        tk.Frame(self, bg=GRID_LINE, height=1).pack(fill="x", padx=16, pady=8)
        self.btn_pause = tk.Button(self, text="PAUSE", bg="#1f2937", fg=TEXT_MAIN,
                                   font=("Courier", 8, "bold"), bd=0, 
                                   padx=10, pady=5, command=self._toggle_pause)
        self.btn_pause.pack(pady=5)
        
        tk.Button(self, text="RESET WINS", bg="#1f2937", fg=TEXT_DIM,
                  font=("Courier", 7), bd=0, padx=10, pady=2,
                  command=self._reset_wins).pack(pady=5)

    def _add_row(self, key):
        f = tk.Frame(self, bg=BG)
        f.pack(fill="x", padx=16, pady=2)
        tk.Label(f, text=key.upper(), bg=BG, fg=TEXT_DIM,
                 font=("Courier", 7), anchor="w", width=8).pack(side="left")
        v = tk.Label(f, text="—", bg=BG, fg=TEXT_MAIN,
                     font=("Courier", 9, "bold"), anchor="e")
        v.pack(side="right")
        self._rows[key] = v

    def update(self, **kw):
        for k, v in kw.items():
            if k in self._rows:
                self._rows[k].config(text=str(v))

    def log_line(self, msg):
        self._log.config(state="normal")
        self._log.insert("end", msg + "\n")
        self._log.see("end")
        self._log.config(state="disabled")

    def poll_train_log(self):
        """Append any new lines from train.log since last read."""
        train_log = Path("train.log")
        if not train_log.exists():
            return
        try:
            with open(train_log, "r") as f:
                f.seek(self._train_log_pos)
                new = f.read()
                self._train_log_pos = f.tell()
            if new.strip():
                lines = []
                for line in new.splitlines():
                    if " INFO  |" in line or " INFO | " in line:
                        parts = line.split("|", 1)
                        lines.append(parts[-1].strip() if len(parts) > 1 else line.strip())
                if lines:
                    self._train_log.config(state="normal")
                    self._train_log.insert("end", "\n".join(lines) + "\n")
                    self._train_log.see("end")
                    self._train_log.config(state="disabled")
        except Exception:
            pass

    def _toggle_pause(self):
        if self.master._paused.is_set():
            self.master._paused.clear()
            self.btn_pause.config(text="PAUSE", bg="#1f2937")
        else:
            self.master._paused.set()
            self.btn_pause.config(text="RESUME", bg="#4a5568")

    def _reset_wins(self):
        self.master._reset_wins()


# ── Board canvas — incremental + resize-safe + zoom ──────────────────────────

class BoardCanvas(tk.Canvas):
    HEX_MIN = 8
    HEX_MAX = 48

    def __init__(self, master):
        super().__init__(master, bg=BG, bd=0, highlightthickness=0)
        self._hex_size = HEX_SIZE
        self._items = {}         # (q, r) -> list of canvas IDs
        self._items_player = {}  # (q, r) -> player ID
        self._background = set()
        self._last = None

        self.bind("<Configure>", self._on_resize)
        self.bind("<MouseWheel>", self._on_zoom)
        self.bind("<Button-4>", self._on_zoom)
        self.bind("<Button-5>", self._on_zoom)
        self._resize_after = None

    def _draw_cell(self, q, r, player, is_last=False):
        ox, oy = self.winfo_width() / 2, self.winfo_height() / 2
        cx, cy = hex_to_pixel(q, r, self._hex_size, ox, oy)
        pts = hex_corners(cx, cy, self._hex_size - 1)
        
        ids = []
        if player is None:
            ids.append(self.create_polygon(pts, fill=EMPTY_F, outline=EMPTY_B))
        else:
            fill = P1_FILL if player == 1 else P2_FILL
            bdr  = P1_BDR if player == 1 else P2_BDR
            if is_last: bdr = LAST_BDR
            ids.append(self.create_polygon(pts, fill=fill, outline=bdr, width=2 if is_last else 1))
            
            sym = "X" if player == 1 else "O"
            col = P1_BDR if player == 1 else P2_BDR
            fs = max(6, int(self._hex_size * 0.65))
            ids.append(self.create_text(cx, cy, text=sym, fill=col,
                                        font=("Courier", fs, "bold")))
        return ids

    def _full_redraw(self):
        """Redraw everything from stored state — called on resize or zoom."""
        self.delete("all")
        self._items.clear()
        for q, r in self._background:
            if (q, r) not in self._items_player:
                self._items[(q, r)] = self._draw_cell(q, r, None)
        for (q, r), p in self._items_player.items():
            is_last = (q, r) == self._last
            self._items[(q, r)] = self._draw_cell(q, r, p, is_last=is_last)

    def _on_resize(self, event):
        if self._resize_after:
            self.after_cancel(self._resize_after)
        self._resize_after = self.after(80, self._full_redraw)

    def _on_zoom(self, event):
        if hasattr(event, "delta"):
            step = 2 if event.delta > 0 else -2
        else:
            step = 2 if event.num == 4 else -2
        new = max(self.HEX_MIN, min(self.HEX_MAX, self._hex_size + step))
        if new != self._hex_size:
            self._hex_size = new
            self._full_redraw()

    def reset(self):
        self.delete("all")
        self._items.clear()
        self._items_player.clear()
        self._background.clear()
        self._last = None
        log.debug("Board reset")

    def add_piece(self, q, r, player):
        """Incremental draw: place piece, update last-move highlight."""
        prev_last = self._last

        if prev_last and prev_last != (q, r) and prev_last in self._items_player:
            pq, pr = prev_last
            for item in self._items.pop((pq, pr), []):
                self.delete(item)
            self._items[(pq, pr)] = self._draw_cell(pq, pr,
                                                    self._items_player[(pq, pr)],
                                                    is_last=False)

        for item in self._items.pop((q, r), []):
            self.delete(item)
        self._items[(q, r)] = self._draw_cell(q, r, player, is_last=True)
        self._items_player[(q, r)] = player
        self._last = (q, r)

    def ensure_background(self, q, r):
        if (q, r) not in self._items_player and (q, r) not in self._background:
            self._background.add((q, r))
            self._items[(q, r)] = self._draw_cell(q, r, None)


# ── Worker ────────────────────────────────────────────────────────────────────

def worker(q: queue.Queue, stop: threading.Event, paused: threading.Event):
    log.info("Worker started SIMS=%d", SIMS)
    game_num = 0
    while not stop.is_set():
        game = HexGame()
        game_num += 1
        move_num = 0
        t0 = time.perf_counter()
        log.debug("Game %d start", game_num)
        q.put({"type": "new_game", "game": game_num})

        while game.winner is None and not stop.is_set():
            if paused.is_set():
                time.sleep(0.1)
                continue
                
            legal = game.legal_moves()
            if not legal:
                log.warning("G%d no legal moves at M%d", game_num, move_num)
                break
            t1 = time.perf_counter()
            try:
                move = mcts(game, SIMS)
            except Exception:
                log.exception("mcts failed G%d M%d", game_num, move_num)
                break
            sps = SIMS / max(time.perf_counter() - t1, 1e-6)
            player = game.current_player
            game.make(*move)
            move_num += 1

            if move_num % 10 == 0:
                log.debug("G%d M%d %s sps=%.0f cands=%d",
                          game_num, move_num, move, sps, len(game.candidates))

            q.put({
                "type": "move",
                "q": move[0], "r": move[1],
                "player": player,
                "next_player": game.current_player,
                "game": game_num,
                "move": move_num,
                "sps": sps,
                "winner": game.winner,
            })

        dur = time.perf_counter() - t0
        log.info("G%d over winner=%s moves=%d dur=%.1fs", game_num, game.winner, move_num, dur)
        q.put({"type": "game_over", "winner": game.winner,
               "moves": move_num, "game": game_num, "duration": dur})


# ── App ───────────────────────────────────────────────────────────────────────

class App(tk.Tk):
    def __init__(self):
        log.info("App init")
        super().__init__()
        self.title("hexgo — self-play")
        self.configure(bg=BG)
        self.geometry("900x640")
        self.minsize(600, 400)

        self._stats = StatsPanel(self)
        self._stats.pack(side="right", fill="y")

        self._board = BoardCanvas(self)
        self._board.pack(side="left", fill="both", expand=True)

        self._queue: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._paused = threading.Event()
        self._wins = {1: 0, 2: 0}

        self._worker = threading.Thread(target=worker, args=(self._queue, self._stop, self._paused),
                                        daemon=True, name="worker")
        self._worker.start()
        self._poll()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        log.info("App ready")

    def _poll(self):
        try:
            processed = 0
            while processed < 20:   # cap per poll cycle to keep UI responsive
                msg = self._queue.get_nowait()
                self._handle(msg)
                processed += 1
        except queue.Empty:
            pass
        except Exception:
            log.exception("UI poll error")
        # Poll train log every ~2s (every 25 UI cycles)
        if not hasattr(self, '_train_poll_count'):
            self._train_poll_count = 0
        self._train_poll_count += 1
        if self._train_poll_count >= 25:
            self._stats.poll_train_log()
            self._train_poll_count = 0
        self.after(POLL_MS, self._poll)

    def _handle(self, msg):
        t = msg["type"]
        if t == "new_game":
            self._board.reset()

        elif t == "move":
            q, r, player = msg["q"], msg["r"], msg["player"]
            for dq, dr in ((1,0),(-1,0),(0,1),(0,-1),(1,-1),(-1,1)):
                self._board.ensure_background(q+dq, r+dr)
            self._board.add_piece(q, r, player)
            self._stats.update(
                game=msg["game"],
                move=msg["move"],
                player=f"{'X' if msg['next_player']==1 else 'O'}",
                **{"sims/s": f"{msg['sps']:.0f}"},
                last=f"({q},{r})",
                wins=f"X{self._wins[1]} O{self._wins[2]}",
            )

        elif t == "game_over":
            w = msg["winner"]
            if w:
                self._wins[w] += 1
            sym = "X" if w == 1 else "O" if w == 2 else "?"
            self._stats.log_line(
                f"G{msg['game']:03d} {sym} {msg['moves']}mv {msg['duration']:.1f}s")
            self._stats.update(wins=f"X{self._wins[1]} O{self._wins[2]}")
            log.debug("UI handled game_over G%d", msg["game"])

    def _reset_wins(self):
        self._wins = {1: 0, 2: 0}
        self._stats.update(wins="X0 O0")

    def _on_close(self):
        log.info("Closing")
        self._stop.set()
        self.destroy()


if __name__ == "__main__":
    try:
        App().mainloop()
        log.info("Clean exit")
    except Exception:
        log.exception("App crashed")
        sys.exit(1)
