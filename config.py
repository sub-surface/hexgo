# config.py — tunable hyperparameters for HexGo autotune
# Edit this file to propose a new trial config.
# Imported by train.py and mcts.py at startup.

CFG = {
    # Optimiser
    "LR":                   1e-3,
    "WEIGHT_DECAY":         1e-4,
    "BATCH_SIZE":           64,

    # Self-play search
    "SIMS":                 50,       # full sim budget (25% of games)
    "SIMS_MIN":             6,        # reduced budget floor (75% of games) — must be << SIMS for playout-cap diversity
    "CAP_FULL_FRAC":        0.25,     # fraction of games at full SIMS
    "CPUCT":                2.0,      # PUCT exploration constant — research target 2.0–2.5
    "DIRICHLET_ALPHA":      0.09,     # root noise concentration — 10/|ZoI| ≈ 0.09 for ZOI_MARGIN=6
    "DIRICHLET_EPS":        0.25,     # root noise weight
    "ZOI_MARGIN":           6,        # hex-distance ZOI pruning radius

    # Training dynamics
    "TD_GAMMA":             0.99,     # TD-lambda discount for value targets
    "TEMP_HORIZON":         40,       # cosine temp annealing half-life (moves)
    "WEIGHT_SYNC_BATCHES":  20,       # batches between weight sync to inference server
    "RECENCY_WEIGHT":       0.75,     # fraction of each batch drawn from recent half of buffer
}
