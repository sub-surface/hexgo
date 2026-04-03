# HexGo AutoResearch Protocol

## Setup

- **Project**: `C:\Users\landa\.claude\projects\hexgo2`
- **Python**: `bash run.sh <script.py>` (hardcodes correct interpreter)
- **GPU**: RTX 5070 Ti (16GB VRAM)
- **Metric**: Net win rate vs Eisenstein (24 games, Rust-batched)
- **Secondary**: Value loss (v=), policy loss (p=), avg_moves
- **Budget**: 10 generations per trial (~20 min)

## The Loop

### 1. READ BASELINE
```
tail -50 train.log   # find last ELO eval line + last gen metrics
```
Record: `v=`, `p=`, `avg_moves`, `eval_eis_wins`, `eval_mm_wins`.

### 2. PROPOSE ONE CHANGE
- Pick ONE thing to change. Not two. ONE.
- Write hypothesis: "Changing X from A to B should improve Y because Z"
- Allowed edits: `train.py`, `net.py`, `config.py`
- Do NOT touch: `game.py`, `hexgo-rs/`, `mcts.py`

### 3. IMPLEMENT + COMMIT
```
# edit the file(s)
git add -A && git commit -m "autoresearch: <short description>"
```

### 4. RUN TRIAL
```
bash run.sh train.py --gens 10 --sims 200 --games 128
```
Wait for completion. Training logs to `train.log`. ELO eval runs every 10 gens.

### 5. READ RESULTS
```
tail -30 train.log   # find ELO eval + final gen metrics
```
Record same metrics as baseline.

### 6. DECIDE
**Keep** if ANY:
- `eval_eis_wins` improved by 2+ (e.g. 2/24 → 4/24)
- `v` dropped by 0.01+ AND `eval_eis_wins` did not regress
- `eval_mm_wins` improved by 2+ without Eisenstein regression

**Discard** if ANY:
- `eval_eis_wins` dropped by 2+
- `v` increased by 0.02+
- Training crashed or NaN

**Neutral** (keep if simpler, discard if adds complexity):
- Metrics within noise band (+-1 win, +-0.005 v)

### 7. KEEP OR REVERT
- **Keep**: Log to `results.tsv`. This is the new baseline.
- **Discard**: `git revert HEAD --no-edit`. Restore checkpoint:
  ```
  cp checkpoints/net_baseline_trial.pt checkpoints/net_latest.pt
  ```

### 8. LOG
Append one line to `autoresearch/results.tsv`:
```
timestamp	experiment	hypothesis	v_before	v_after	eis_before	eis_after	mm_before	mm_after	kept	reason
```

### 9. LOOP TO STEP 1

## Rules
- NEVER delete `train.lock`
- NEVER run training while another is running
- Save baseline checkpoint before each trial: `cp checkpoints/net_latest.pt checkpoints/net_baseline_trial.pt`
- One change at a time — isolate variables
- If 3 consecutive experiments fail, stop and reconsider approach
- Git branch tip is always the best known config
- Simpler is better — reject marginal gains that add complexity
