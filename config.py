# config.py — tunable hyperparameters for HexGo autotune
# Edit this file to propose a new trial config.
# Imported by train.py and mcts.py at startup.

CFG = {
    "LR":               1e-3,
    "WEIGHT_DECAY":     1e-4,
    "BATCH_SIZE":       64,
    "SIMS":             50,
    "SIMS_MIN":         25,
    "CAP_FULL_FRAC":    0.25,
    "CPUCT":            1.4,
    "DIRICHLET_ALPHA":  0.3,
    "DIRICHLET_EPS":    0.25,
    "ZOI_MARGIN":       6,
}
