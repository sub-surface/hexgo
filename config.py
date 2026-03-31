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
    "ZOI_LOOKBACK":         16,       # recent moves used to define ZOI focus (was hardcoded 8)

    # Search
    "GUMBEL_SELECTION":     True,     # Gumbel argmax root selection (vs softmax-temp sampling)

    # Training dynamics
    "TD_GAMMA":             0.99,     # TD-lambda discount for value targets
    "TEMP_HORIZON":         40,       # cosine temp annealing half-life (moves)
    "WEIGHT_SYNC_BATCHES":  20,       # batches between weight sync to inference server
    "RECENCY_WEIGHT":       0.75,     # fraction of each batch drawn from recent half of buffer

    # Network architecture
    "TRUNK_BLOCKS":         4,        # residual blocks in trunk (was 2)
    "TRUNK_CHANNELS":       64,       # hidden channels throughout trunk (was 32)
    "WEIGHT_INIT":          "ca",     # "ca" = hex NCA Laplacian priors | "xavier" = standard

    # Loss weighting
    "VALUE_LOSS_WEIGHT":    2.0,      # multiplier on MSE value loss — policy CE dominates by ~20x without this
    "ENTROPY_REG":          0.01,     # policy entropy regularization weight (-β·H(π) bonus; 0 = disabled)

    # Auxiliary head loss weights — kept small per bitter-lesson principle
    # (aux heads guide representation; main value+policy loss drives play quality)
    "AUX_LOSS_OWN":         0.1,      # ownership prediction loss weight
    "AUX_LOSS_THREAT":      0.1,      # threat prediction loss weight
    "UNC_LOSS_WEIGHT":      0.1,      # value uncertainty head (Gaussian NLL) loss weight
}
