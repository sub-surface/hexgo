# Research Notes — HexGo

## Cellular Automata Weight Initialisation

**To investigate:** recent work on initialising neural network weights using cellular automata
patterns rather than standard random initialisation (Xavier/He). The hypothesis for HexGo:
the hex lattice has natural CA dynamics (each cell's state is a function of its 6 neighbours),
so a CA-derived initialisation might produce weight priors already aligned with the game's
local structure — potentially faster convergence and better early self-play quality.

Relevant prior art to find:
- Mordvintsev et al., "Growing Neural Cellular Automata" (2020) — self-organising NCA
- Any work specifically on using CA rules to seed CNN weight tensors
- Connection to the Z[ω] lattice: the hex-7 neighbourhood IS the standard NCA update kernel

**Why it fits here:** HexConv2d uses 7-cell hex kernels. A CA initialisation would set those
kernels to patterns that already propagate local board state coherently, rather than starting
from Gaussian noise. Could be especially useful for the ResBlocks which need to learn
spatial propagation of threat information.

**Action:** search for the specific paper the user referenced, evaluate whether it applies
to a 7-cell hex kernel, prototype as an optional `init_weights_ca(net)` function.

---

## Eisenstein Integer Isomorphism — Key Finding

The hex grid with axial coordinates (q, r) is isomorphic to the Eisenstein integer ring
Z[ω] where ω = e^(2πi/3). The three win axes correspond to the unit directions {1, ω, ω²}.
A win in HexGo is exactly an arithmetic progression of length 6 in Z[ω] with a unit step.

Implications documented in DESIGN.md and ROADMAP.md.
