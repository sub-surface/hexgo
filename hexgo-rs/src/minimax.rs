//! MinimaxBot — alpha-beta search with learned pattern evaluation.
//!
//! Ported from ext/HexTicTacToe/ai.py.  Uses 6-cell windows for win/threat
//! detection and 6-cell windows for pattern-based evaluation (loaded from
//! pattern_values.json).  Incremental state updates keep make/unmake O(1)
//! per window touched.

use std::collections::HashMap;

use rand::Rng;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::game::HexGame;
use crate::types::*;

// ── Constants ───────────────────────────────────────────────────────────────

const WIN_LEN: usize = 6;
const NEIGHBOR_DIST: i16 = 2;
const ROOT_CANDIDATE_CAP: usize = 16;
const INNER_CANDIDATE_CAP: usize = 11;
const MAX_QDEPTH: i32 = 16;
const WIN_SCORE: f64 = 100_000_000.0;
const DELTA_WEIGHT: f64 = 1.5;
const NUM_PATTERNS: usize = 729; // 3^6

/// Direction vectors — same order as Python HEX_DIRECTIONS.
const DIR_VECTORS: [(i16, i16); 3] = [(1, 0), (0, 1), (1, -1)];

/// Colony direction vectors for far-away group exploration.
const COLONY_DIRS: [(i16, i16); 6] = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)];

/// 6-cell window offsets: (dir_idx, offset_k, dq*k, dr*k).
/// For a piece at (q, r), the windows it belongs to are keyed by
/// (dir_idx, q - k*dq, r - k*dr) for k in 0..WIN_LEN.
struct WinOffset {
    dir_idx: u8,
    oq: i16,
    or_: i16,
}

fn build_win_offsets() -> Vec<WinOffset> {
    let mut v = Vec::with_capacity(3 * WIN_LEN);
    for (d_idx, &(dq, dr)) in DIR_VECTORS.iter().enumerate() {
        for k in 0..WIN_LEN {
            v.push(WinOffset {
                dir_idx: d_idx as u8,
                oq: k as i16 * dq,
                or_: k as i16 * dr,
            });
        }
    }
    v
}

/// Eval offsets — for N-cell patterns (same as win offsets when N=6, but
/// keyed with dir_idx offset by 3 to avoid collision with 6-cell win keys).
struct EvalOffset {
    dir_idx: u8,
    k: usize,
    oq: i16,
    or_: i16,
}

fn build_eval_offsets() -> Vec<EvalOffset> {
    let mut v = Vec::with_capacity(3 * WIN_LEN);
    for (d_idx, &(dq, dr)) in DIR_VECTORS.iter().enumerate() {
        for k in 0..WIN_LEN {
            v.push(EvalOffset {
                dir_idx: d_idx as u8,
                k,
                oq: k as i16 * dq,
                or_: k as i16 * dr,
            });
        }
    }
    v
}

/// Neighbor offsets within hex distance 2 (excluding origin).
fn build_neighbor_offsets() -> Vec<(i16, i16)> {
    let mut v = Vec::new();
    for dq in -NEIGHBOR_DIST..=NEIGHBOR_DIST {
        for dr in -NEIGHBOR_DIST..=NEIGHBOR_DIST {
            let ds = -dq - dr;
            let dist = dq.abs().max(dr.abs()).max(ds.abs());
            if dist <= NEIGHBOR_DIST && (dq, dr) != (0, 0) {
                v.push((dq, dr));
            }
        }
    }
    v
}

/// Powers of 3 for base-3 encoding.
const POW3: [u32; 6] = [1, 3, 9, 27, 81, 243];

// ── Pair masks ──────────────────────────────────────────────────────────────

/// Root pair mask (16x16 upper triangle).
#[rustfmt::skip]
const ROOT_PAIR_MASK: [[u8; 16]; 16] = [
    [0, 2, 3, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
];

/// Inner pair mask (11x11 upper triangle).
#[rustfmt::skip]
const INNER_PAIR_MASK: [[u8; 11]; 11] = [
    [0, 2, 3, 4, 6, 8, 11, 12, 13, 14, 15],
    [0, 0, 5, 7, 9,  0,  0,  0,  0,  0,  0],
    [0, 0, 0, 10, 0,  0,  0,  0,  0,  0,  0],
    [0, 0, 0,  0, 0,  0,  0,  0,  0,  0,  0],
    [0, 0, 0,  0, 0,  0,  0,  0,  0,  0,  0],
    [0, 0, 0,  0, 0,  0,  0,  0,  0,  0,  0],
    [0, 0, 0,  0, 0,  0,  0,  0,  0,  0,  0],
    [0, 0, 0,  0, 0,  0,  0,  0,  0,  0,  0],
    [0, 0, 0,  0, 0,  0,  0,  0,  0,  0,  0],
    [0, 0, 0,  0, 0,  0,  0,  0,  0,  0,  0],
    [0, 0, 0,  0, 0,  0,  0,  0,  0,  0,  0],
];

/// Build ordered pair list from a mask.
/// 0 = skip, 1 = auto-order (by i+j then i), >=2 = explicit priority (lowest first).
fn build_pairs<const N: usize>(mask: &[[u8; N]; N]) -> Vec<(usize, usize)> {
    let mut explicit: Vec<(u8, usize, usize)> = Vec::new();
    let mut auto: Vec<(usize, usize, usize)> = Vec::new();
    for i in 0..N {
        for j in 0..N {
            let v = mask[i][j];
            if v >= 2 {
                explicit.push((v, i, j));
            } else if v == 1 {
                auto.push((i + j, i, j));
            }
        }
    }
    explicit.sort();
    auto.sort();
    let mut result: Vec<(usize, usize)> = explicit.iter().map(|&(_, i, j)| (i, j)).collect();
    result.extend(auto.iter().map(|&(_, i, j)| (i, j)));
    result
}

// ── TT flags ────────────────────────────────────────────────────────────────

const TT_EXACT: u8 = 0;
const TT_LOWER: u8 = 1;
const TT_UPPER: u8 = 2;

/// Transposition table entry.
#[derive(Clone)]
struct TTEntry {
    depth: i32,
    score: f64,
    flag: u8,
    best_move: Option<(Coord, Coord)>,
}

// ── Zobrist ─────────────────────────────────────────────────────────────────

/// Simple deterministic 64-bit RNG (xorshift64) seeded from a u64.
struct DetRng {
    state: u64,
}

impl DetRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        // xorshift64
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }
}

// ── Pattern loading ─────────────────────────────────────────────────────────

/// Decode an integer into a 6-digit base-3 pattern.
fn int_to_pattern(mut n: u32) -> [u8; 6] {
    let mut pat = [0u8; 6];
    for p in pat.iter_mut() {
        *p = (n % 3) as u8;
        n /= 3;
    }
    pat
}

/// Encode a 6-digit base-3 pattern into an integer.
fn pattern_to_int(pat: &[u8; 6]) -> u32 {
    let mut n: u32 = 0;
    for i in (0..6).rev() {
        n = n * 3 + pat[i] as u32;
    }
    n
}

/// Reverse a pattern (flip symmetry).
fn reverse_pattern(pat: &[u8; 6]) -> [u8; 6] {
    let mut r = *pat;
    r.reverse();
    r
}

/// Load pattern_values.json and build the flat pat_value[729] array.
///
/// The JSON has "_meta" with score_scale and window_length=6, plus entries
/// like "000012": -0.005.  piece_swap_symmetry is false for our data, so we
/// only use flip (reverse) symmetry.
///
/// For each pattern int `pi`, we find its canonical form (min of pi and
/// reverse(pi)), look up the JSON value, and apply the flip sign.
pub fn load_pattern_values(json_str: &str) -> [f64; NUM_PATTERNS] {
    let raw: serde_json::Value =
        serde_json::from_str(json_str).expect("Failed to parse pattern_values.json");

    let meta = raw.get("_meta").expect("Missing _meta in pattern JSON");
    let score_scale = meta
        .get("score_scale")
        .and_then(|v| v.as_f64())
        .unwrap_or(1.0);
    let piece_swap = meta
        .get("piece_swap_symmetry")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    // Build canon_index and canon_sign arrays (same logic as pattern_table.py)
    let mut canon_index = vec![-1i32; NUM_PATTERNS];
    let mut canon_sign = vec![0i8; NUM_PATTERNS];
    let mut canon_patterns: Vec<[u8; 6]> = Vec::new();
    let mut canon_lookup: HashMap<[u8; 6], usize> = HashMap::new();

    for i in 0..NUM_PATTERNS as u32 {
        if canon_index[i as usize] != -1 || canon_sign[i as usize] != 0 {
            // Already assigned via a variant
            continue;
        }

        let pat = int_to_pattern(i);

        // All-zero pattern: no value
        if pat.iter().all(|&c| c == 0) {
            canon_index[i as usize] = -1;
            canon_sign[i as usize] = 0;
            continue;
        }

        let p_flip = reverse_pattern(&pat);

        if piece_swap {
            let p_swap = swap_pieces(&pat);
            let p_swap_flip = swap_pieces(&p_flip);

            let pos_variants = [pat, p_flip];
            let neg_variants = [p_swap, p_swap_flip];

            let mut all_variants = Vec::with_capacity(4);
            all_variants.extend_from_slice(&pos_variants);
            all_variants.extend_from_slice(&neg_variants);
            let canon = *all_variants.iter().min().unwrap();

            let canon_s = if pos_variants.contains(&canon) { 1i8 } else { -1i8 };

            let cidx = *canon_lookup.entry(canon).or_insert_with(|| {
                let idx = canon_patterns.len();
                canon_patterns.push(canon);
                idx
            });

            // Check self-symmetry
            let pos_set: FxHashSet<[u8; 6]> = pos_variants.iter().copied().collect();
            let neg_set: FxHashSet<[u8; 6]> = neg_variants.iter().copied().collect();
            let is_self_symmetric = pos_set.intersection(&neg_set).next().is_some();

            for p in &pos_variants {
                let pi = pattern_to_int(p) as usize;
                if canon_index[pi] == -1 && canon_sign[pi] == 0 && !pat_is_zero(p) {
                    canon_index[pi] = cidx as i32;
                    canon_sign[pi] = if is_self_symmetric { 0 } else { canon_s };
                }
            }
            for p in &neg_variants {
                let pi = pattern_to_int(p) as usize;
                if canon_index[pi] == -1 && canon_sign[pi] == 0 && !pat_is_zero(p) {
                    canon_index[pi] = cidx as i32;
                    canon_sign[pi] = if is_self_symmetric { 0 } else { -canon_s };
                }
            }
        } else {
            // No piece swap — only flip symmetry
            let pos_variants = [pat, p_flip];
            let canon = *pos_variants.iter().min().unwrap();

            let cidx = *canon_lookup.entry(canon).or_insert_with(|| {
                let idx = canon_patterns.len();
                canon_patterns.push(canon);
                idx
            });

            for p in &pos_variants {
                let pi = pattern_to_int(p) as usize;
                if canon_index[pi] == -1 && canon_sign[pi] == 0 && !pat_is_zero(p) {
                    canon_index[pi] = cidx as i32;
                    canon_sign[pi] = 1;
                }
            }
        }
    }

    // Parse canonical values from JSON strings
    let num_canon = canon_patterns.len();
    let mut params = vec![0.0f64; num_canon];
    for (i, pat) in canon_patterns.iter().enumerate() {
        let pat_str: String = pat.iter().map(|c| char::from(b'0' + c)).collect();
        if let Some(val) = raw.get(&pat_str).and_then(|v| v.as_f64()) {
            params[i] = val * score_scale;
        }
    }

    // Build final PAT_VALUE array
    let mut pat_value = [0.0f64; NUM_PATTERNS];
    for pi in 0..NUM_PATTERNS {
        let ci = canon_index[pi];
        let cs = canon_sign[pi];
        if cs != 0 && ci >= 0 {
            pat_value[pi] = cs as f64 * params[ci as usize];
        }
    }

    pat_value
}

fn swap_pieces(pat: &[u8; 6]) -> [u8; 6] {
    let mut s = [0u8; 6];
    for i in 0..6 {
        s[i] = match pat[i] {
            1 => 2,
            2 => 1,
            _ => 0,
        };
    }
    s
}

fn pat_is_zero(pat: &[u8; 6]) -> bool {
    pat.iter().all(|&c| c == 0)
}

// ── Window key type ─────────────────────────────────────────────────────────
// (dir_idx, start_q, start_r)
type WKey = (u8, i16, i16);

// ── TimeUp sentinel ─────────────────────────────────────────────────────────

/// Returned from search when the deadline is exceeded.
struct TimeUp;

// ── Minimax State ───────────────────────────────────────────────────────────

/// Incremental state for one minimax search invocation.
struct MinimaxState<'a> {
    pat_value: &'a [f64; NUM_PATTERNS],

    // Zobrist
    zobrist_table: FxHashMap<(i16, i16, u8), u64>,
    zobrist_rng: DetRng,
    hash: u64,

    // 6-cell window counts: [a_count, b_count]
    wc: FxHashMap<WKey, [u32; 2]>,
    // N-cell pattern ints (key dir offset by 3)
    wp: FxHashMap<WKey, u32>,
    // Running eval score
    eval_score: f64,

    // Hot windows: those with 4+ pieces for one side
    hot_a: FxHashSet<WKey>,
    hot_b: FxHashSet<WKey>,

    // Candidate set + refcounts
    cand_set: FxHashSet<Coord>,
    cand_refcount: FxHashMap<Coord, u32>,
    rc_stack: Vec<u32>, // saved refcount of placed cell

    // Which player "we" are (the root caller)
    me: u8,
    // Cell value mapping: if me=P1, cell_a=1 (me), cell_b=2 (opp)
    cell_a: u32, // value for Player A stones in pattern encoding
    cell_b: u32, // value for Player B stones in pattern encoding

    // Transposition table
    tt: FxHashMap<(u64, u8, u8), TTEntry>,

    // History heuristic
    history: FxHashMap<Coord, f64>,

    // Node counter + deadline
    nodes: u64,
    deadline: std::time::Instant,

    // Precomputed offsets
    win_offsets: Vec<WinOffset>,
    eval_offsets: Vec<EvalOffset>,
    neighbor_offsets: Vec<(i16, i16)>,

    // Precomputed pair orders
    root_pairs: Vec<(usize, usize)>,
    inner_pairs: Vec<(usize, usize)>,
}

impl<'a> MinimaxState<'a> {
    fn new(
        game: &HexGame,
        pat_value: &'a [f64; NUM_PATTERNS],
        time_limit_ms: u64,
    ) -> Self {
        let win_offsets = build_win_offsets();
        let eval_offsets = build_eval_offsets();
        let neighbor_offsets = build_neighbor_offsets();
        let root_pairs = build_pairs(&ROOT_PAIR_MASK);
        let inner_pairs = build_pairs(&INNER_PAIR_MASK);

        let me = game.current_player;
        // Cell value mapping: me's stones = 1, opponent's = 2 in pattern encoding
        let (cell_a, cell_b) = if me == P1 {
            (1u32, 2u32) // A=me=1, B=opp=2
        } else {
            (2u32, 1u32) // A=opp=2, B=me=1
        };

        let deadline = std::time::Instant::now()
            + std::time::Duration::from_millis(time_limit_ms);

        let mut state = MinimaxState {
            pat_value,
            zobrist_table: FxHashMap::default(),
            zobrist_rng: DetRng::new(42),
            hash: 0,
            wc: FxHashMap::default(),
            wp: FxHashMap::default(),
            eval_score: 0.0,
            hot_a: FxHashSet::default(),
            hot_b: FxHashSet::default(),
            cand_set: FxHashSet::default(),
            cand_refcount: FxHashMap::default(),
            rc_stack: Vec::new(),
            me,
            cell_a,
            cell_b,
            tt: FxHashMap::default(),
            history: FxHashMap::default(),
            nodes: 0,
            deadline,
            win_offsets,
            eval_offsets,
            neighbor_offsets,
            root_pairs,
            inner_pairs,
        };

        state.init_from_board(game);
        state
    }

    /// Initialize all incremental structures from the current board.
    fn init_from_board(&mut self, game: &HexGame) {
        let board = game.board_ref();

        // Zobrist hash
        self.hash = 0;
        for (&(q, r), &p) in board.iter() {
            let zval = self.get_zobrist(q, r, p);
            self.hash ^= zval;
        }

        // 6-cell windows: scan all windows that contain at least one piece
        let mut seen6: FxHashSet<WKey> = FxHashSet::default();
        for &(q, r) in board.keys() {
            for wo in &self.win_offsets {
                let wkey: WKey = (wo.dir_idx, q - wo.oq, r - wo.or_);
                if !seen6.insert(wkey) {
                    continue;
                }
                let (dq, dr) = DIR_VECTORS[wo.dir_idx as usize];
                let (sq, sr) = (wkey.1, wkey.2);
                let mut a_count = 0u32;
                let mut b_count = 0u32;
                for j in 0..WIN_LEN as i16 {
                    if let Some(&cp) = board.get(&(sq + j * dq, sr + j * dr)) {
                        if cp == P1 {
                            a_count += 1;
                        } else if cp == P2 {
                            b_count += 1;
                        }
                    }
                }
                if a_count > 0 || b_count > 0 {
                    self.wc.insert(wkey, [a_count, b_count]);
                }
            }
        }

        // Hot sets
        for (&wkey, counts) in &self.wc {
            if counts[0] >= 4 {
                self.hot_a.insert(wkey);
            }
            if counts[1] >= 4 {
                self.hot_b.insert(wkey);
            }
        }

        // N-cell eval windows
        let mut seen_eval: FxHashSet<WKey> = FxHashSet::default();
        self.eval_score = 0.0;
        for &(q, r) in board.keys() {
            for eo in &self.eval_offsets {
                let wkey: WKey = (3 + eo.dir_idx, q - eo.oq, r - eo.or_);
                if !seen_eval.insert(wkey) {
                    continue;
                }
                let (dq, dr) = DIR_VECTORS[eo.dir_idx as usize];
                let (sq, sr) = (wkey.1, wkey.2);
                let mut pat_int = 0u32;
                let mut has_piece = false;
                for j in 0..WIN_LEN as i16 {
                    if let Some(&cp) = board.get(&(sq + j * dq, sr + j * dr)) {
                        let cv = if cp == P1 { self.cell_a } else { self.cell_b };
                        pat_int += cv * POW3[j as usize];
                        has_piece = true;
                    }
                }
                if has_piece {
                    self.wp.insert(wkey, pat_int);
                    self.eval_score += self.pat_value[pat_int as usize];
                }
            }
        }

        // Candidate set
        for &(q, r) in board.keys() {
            for &(dq, dr) in &self.neighbor_offsets {
                let nb = (q + dq, r + dr);
                if !board.contains_key(&nb) {
                    *self.cand_refcount.entry(nb).or_insert(0) += 1;
                    self.cand_set.insert(nb);
                }
            }
        }
    }

    // ── Zobrist ─────────────────────────────────────────────────────────────

    fn get_zobrist(&mut self, q: i16, r: i16, player: u8) -> u64 {
        let key = (q, r, player);
        if let Some(&v) = self.zobrist_table.get(&key) {
            return v;
        }
        let v = self.zobrist_rng.next_u64();
        self.zobrist_table.insert(key, v);
        v
    }

    // ── Make / Unmake ───────────────────────────────────────────────────────

    /// Make a single stone placement.  Updates all incremental state.
    /// Returns true if the placement caused a win.
    fn make_stone(&mut self, game: &mut HexGame, q: i16, r: i16) -> bool {
        let player = game.current_player;

        // Zobrist
        let zval = self.get_zobrist(q, r, player);
        self.hash ^= zval;

        let cell_val = if player == P1 { self.cell_a } else { self.cell_b };

        // 6-cell windows: update counts
        let mut won = false;
        let p_idx = if player == P1 { 0usize } else { 1usize };
        for wo in &self.win_offsets {
            let wkey: WKey = (wo.dir_idx, q - wo.oq, r - wo.or_);
            let counts = self.wc.entry(wkey).or_insert([0, 0]);
            counts[p_idx] += 1;
            if counts[p_idx] >= 4 {
                if player == P1 {
                    self.hot_a.insert(wkey);
                } else {
                    self.hot_b.insert(wkey);
                }
            }
            if counts[p_idx] == WIN_LEN as u32 && counts[1 - p_idx] == 0 {
                won = true;
            }
        }

        // N-cell eval windows: update pattern ints
        for eo in &self.eval_offsets {
            let wkey: WKey = (3 + eo.dir_idx, q - eo.oq, r - eo.or_);
            let old_pi = *self.wp.get(&wkey).unwrap_or(&0);
            let new_pi = old_pi + cell_val * POW3[eo.k];
            self.eval_score += self.pat_value[new_pi as usize]
                - self.pat_value[old_pi as usize];
            self.wp.insert(wkey, new_pi);
        }

        // Candidates: remove placed cell, add neighbors
        self.cand_set.remove(&(q, r));
        let saved_rc = self.cand_refcount.remove(&(q, r)).unwrap_or(0);
        self.rc_stack.push(saved_rc);

        let board = game.board_ref();
        for &(dq, dr) in &self.neighbor_offsets {
            let nb = (q + dq, r + dr);
            *self.cand_refcount.entry(nb).or_insert(0) += 1;
            if !board.contains_key(&nb) && nb != (q, r) {
                self.cand_set.insert(nb);
            }
        }

        // Make the move on the game
        game.make_move(q, r);

        won
    }

    /// Undo a single stone placement.
    fn unmake_stone(&mut self, game: &mut HexGame, q: i16, r: i16, player: u8) {
        // Unmake the game move first (restores board state)
        game.unmake_move();

        // Zobrist
        let zval = self.get_zobrist(q, r, player);
        self.hash ^= zval;

        let cell_val = if player == P1 { self.cell_a } else { self.cell_b };
        let p_idx = if player == P1 { 0usize } else { 1usize };

        // 6-cell windows: decrement counts
        for wo in &self.win_offsets {
            let wkey: WKey = (wo.dir_idx, q - wo.oq, r - wo.or_);
            if let Some(counts) = self.wc.get_mut(&wkey) {
                counts[p_idx] -= 1;
                if counts[p_idx] < 4 {
                    if player == P1 {
                        self.hot_a.remove(&wkey);
                    } else {
                        self.hot_b.remove(&wkey);
                    }
                }
            }
        }

        // N-cell eval windows: reverse pattern update
        for eo in &self.eval_offsets {
            let wkey: WKey = (3 + eo.dir_idx, q - eo.oq, r - eo.or_);
            let old_pi = *self.wp.get(&wkey).unwrap_or(&0);
            let new_pi = old_pi - cell_val * POW3[eo.k];
            self.eval_score += self.pat_value[new_pi as usize]
                - self.pat_value[old_pi as usize];
            if new_pi == 0 {
                self.wp.remove(&wkey);
            } else {
                self.wp.insert(wkey, new_pi);
            }
        }

        // Candidates: restore
        for &(dq, dr) in &self.neighbor_offsets {
            let nb = (q + dq, r + dr);
            let rc = self.cand_refcount.get_mut(&nb);
            if let Some(rc) = rc {
                *rc -= 1;
                if *rc == 0 {
                    self.cand_refcount.remove(&nb);
                    self.cand_set.remove(&nb);
                }
            }
        }
        let saved_rc = self.rc_stack.pop().unwrap_or(0);
        if saved_rc > 0 {
            self.cand_refcount.insert((q, r), saved_rc);
            self.cand_set.insert((q, r));
        }
    }

    // ── Turn-level make/unmake (pair of stones) ─────────────────────────────

    /// Place a turn (pair of moves).  Returns undo info.
    fn make_turn(
        &mut self,
        game: &mut HexGame,
        m1: Coord,
        m2: Coord,
    ) -> Vec<(Coord, u8)> {
        let p1 = game.current_player;
        self.make_stone(game, m1.0, m1.1);

        if game.winner_raw() != NO_PLAYER {
            return vec![(m1, p1)];
        }

        let p2 = game.current_player;
        self.make_stone(game, m2.0, m2.1);
        vec![(m1, p1), (m2, p2)]
    }

    /// Undo a turn.
    fn unmake_turn(&mut self, game: &mut HexGame, undo_info: &[(Coord, u8)]) {
        for &(coord, player) in undo_info.iter().rev() {
            self.unmake_stone(game, coord.0, coord.1, player);
        }
    }

    // ── Time check ──────────────────────────────────────────────────────────

    fn check_time(&mut self) -> Result<(), TimeUp> {
        self.nodes += 1;
        if self.nodes % 1024 == 0 && std::time::Instant::now() >= self.deadline {
            return Err(TimeUp);
        }
        Ok(())
    }

    // ── Move delta (read-only eval prediction) ──────────────────────────────

    fn move_delta(&self, q: i16, r: i16, is_player_a: bool) -> f64 {
        let cell_val = if is_player_a { self.cell_a } else { self.cell_b };
        let mut delta = 0.0;
        for eo in &self.eval_offsets {
            let wkey: WKey = (3 + eo.dir_idx, q - eo.oq, r - eo.or_);
            let old_pi = *self.wp.get(&wkey).unwrap_or(&0);
            let new_pi = old_pi + cell_val * POW3[eo.k];
            delta += self.pat_value[new_pi as usize]
                - self.pat_value[old_pi as usize];
        }
        delta
    }

    // ── Threat detection ────────────────────────────────────────────────────

    /// Find an instant win for `player`: a window with 5+ own and 0 opponent,
    /// and at most 2 empty cells to fill.
    fn find_instant_win(
        &self,
        game: &HexGame,
        player: u8,
    ) -> Option<(Coord, Coord)> {
        let p_idx = if player == P1 { 0usize } else { 1 };
        let o_idx = 1 - p_idx;
        let hot = if player == P1 { &self.hot_a } else { &self.hot_b };
        let board = game.board_ref();

        for &wkey in hot {
            let counts = match self.wc.get(&wkey) {
                Some(c) => c,
                None => continue,
            };
            if counts[p_idx] >= (WIN_LEN - 2) as u32 && counts[o_idx] == 0 {
                let (dq, dr) = DIR_VECTORS[wkey.0 as usize];
                let (sq, sr) = (wkey.1, wkey.2);
                let mut cells: Vec<Coord> = Vec::new();
                for j in 0..WIN_LEN as i16 {
                    let cell = (sq + j * dq, sr + j * dr);
                    if !board.contains_key(&cell) {
                        cells.push(cell);
                    }
                }
                if cells.len() == 1 {
                    // Need a second arbitrary move
                    let other = self
                        .cand_set
                        .iter()
                        .find(|&&c| c != cells[0])
                        .copied()
                        .unwrap_or(cells[0]);
                    return Some((cells[0].min(other), cells[0].max(other)));
                } else if cells.len() == 2 {
                    return Some((
                        cells[0].min(cells[1]),
                        cells[0].max(cells[1]),
                    ));
                }
            }
        }
        None
    }

    /// Find all empty cells in opponent's hot windows (unblocked threats).
    fn find_threat_cells(&self, game: &HexGame, player: u8) -> FxHashSet<Coord> {
        let mut threat_cells = FxHashSet::default();
        let p_idx = if player == P1 { 0usize } else { 1 };
        let o_idx = 1 - p_idx;
        let hot = if player == P1 { &self.hot_a } else { &self.hot_b };
        let board = game.board_ref();

        for &wkey in hot {
            let counts = match self.wc.get(&wkey) {
                Some(c) => c,
                None => continue,
            };
            if counts[o_idx] == 0 {
                let (dq, dr) = DIR_VECTORS[wkey.0 as usize];
                let (sq, sr) = (wkey.1, wkey.2);
                for j in 0..WIN_LEN as i16 {
                    let cell = (sq + j * dq, sr + j * dr);
                    if !board.contains_key(&cell) {
                        threat_cells.insert(cell);
                    }
                }
            }
        }
        threat_cells
    }

    /// Filter turns to only those blocking all opponent near-win windows.
    fn filter_turns_by_threats(
        &self,
        game: &HexGame,
        turns: &[(Coord, Coord)],
    ) -> Vec<(Coord, Coord)> {
        let opponent = 3 - game.current_player;
        let p_idx = if opponent == P1 { 0usize } else { 1 };
        let o_idx = 1 - p_idx;
        let hot = if opponent == P1 { &self.hot_a } else { &self.hot_b };
        let board = game.board_ref();

        let mut must_hit: Vec<FxHashSet<Coord>> = Vec::new();
        for &wkey in hot {
            let counts = match self.wc.get(&wkey) {
                Some(c) => c,
                None => continue,
            };
            if counts[p_idx] >= (WIN_LEN - 2) as u32 && counts[o_idx] == 0 {
                let (dq, dr) = DIR_VECTORS[wkey.0 as usize];
                let (sq, sr) = (wkey.1, wkey.2);
                let mut empties = FxHashSet::default();
                for j in 0..WIN_LEN as i16 {
                    let cell = (sq + j * dq, sr + j * dr);
                    if !board.contains_key(&cell) {
                        empties.insert(cell);
                    }
                }
                must_hit.push(empties);
            }
        }

        if must_hit.is_empty() {
            return turns.to_vec();
        }

        turns
            .iter()
            .filter(|(m1, m2)| {
                must_hit
                    .iter()
                    .all(|w| w.contains(m1) || w.contains(m2))
            })
            .copied()
            .collect()
    }

    // ── Move generation ─────────────────────────────────────────────────────

    /// Generate root-level turns (pair moves).
    fn generate_root_turns(&self, game: &HexGame) -> Vec<(Coord, Coord)> {
        // Check for instant win
        if let Some(win_turn) = self.find_instant_win(game, game.current_player) {
            return vec![win_turn];
        }

        let mut candidates: Vec<Coord> = self.cand_set.iter().copied().collect();
        if candidates.len() < 2 {
            if candidates.len() == 1 {
                return vec![(candidates[0], candidates[0])];
            }
            return Vec::new();
        }

        let is_a = game.current_player == P1;
        let maximizing = game.current_player == self.me;

        candidates.sort_by(|a, b| {
            let da = self.move_delta(a.0, a.1, is_a);
            let db = self.move_delta(b.0, b.1, is_a);
            if maximizing {
                db.partial_cmp(&da).unwrap_or(std::cmp::Ordering::Equal)
            } else {
                da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
            }
        });
        candidates.truncate(ROOT_CANDIDATE_CAP);

        // Colony candidate: a far-away hex for starting a separate group.
        // Pick the direction that's least occupied.
        let board = game.board_ref();
        if !board.is_empty() {
            let occupied: Vec<Coord> = board.keys().copied().collect();
            let n_occ = occupied.len() as i16;
            let cq = occupied.iter().map(|c| c.0).sum::<i16>() / n_occ;
            let cr = occupied.iter().map(|c| c.1).sum::<i16>() / n_occ;
            let max_r = occupied
                .iter()
                .map(|&(q, r)| {
                    let dq = q - cq;
                    let dr = r - cr;
                    let ds = -dq - dr;
                    dq.abs().max(dr.abs()).max(ds.abs())
                })
                .max()
                .unwrap_or(0);
            let colony_dist = max_r + 3;
            // Pick colony direction based on position hash (varies per board state)
            let start = self.hash as usize % COLONY_DIRS.len();
            for i in 0..COLONY_DIRS.len() {
                let (dq, dr) = COLONY_DIRS[(start + i) % COLONY_DIRS.len()];
                let colony = (cq + dq * colony_dist, cr + dr * colony_dist);
                if !board.contains_key(&colony) {
                    candidates.push(colony);
                    break;
                }
            }
        }

        let n = candidates.len();
        let turns: Vec<(Coord, Coord)> = self
            .root_pairs
            .iter()
            .filter_map(|&(i, j)| {
                if i < n && j < n {
                    Some((candidates[i], candidates[j]))
                } else {
                    None
                }
            })
            .collect();

        self.filter_turns_by_threats(game, &turns)
    }

    /// Generate inner-node turns (pair moves).
    fn generate_inner_turns(&self, game: &HexGame) -> Vec<(Coord, Coord)> {
        let mut candidates: Vec<Coord> = self.cand_set.iter().copied().collect();
        if candidates.len() < 2 {
            if candidates.len() == 1 {
                return vec![(candidates[0], candidates[0])];
            }
            return Vec::new();
        }

        let is_a = game.current_player == P1;
        let maximizing = game.current_player == self.me;
        let delta_sign = if maximizing { DELTA_WEIGHT } else { -DELTA_WEIGHT };

        candidates.sort_by(|a, b| {
            let sa = *self.history.get(a).unwrap_or(&0.0)
                + self.move_delta(a.0, a.1, is_a) * delta_sign;
            let sb = *self.history.get(b).unwrap_or(&0.0)
                + self.move_delta(b.0, b.1, is_a) * delta_sign;
            sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
        });
        candidates.truncate(INNER_CANDIDATE_CAP);

        let n = candidates.len();
        let turns: Vec<(Coord, Coord)> = self
            .inner_pairs
            .iter()
            .filter_map(|&(i, j)| {
                if i < n && j < n {
                    Some((candidates[i], candidates[j]))
                } else {
                    None
                }
            })
            .collect();

        self.filter_turns_by_threats(game, &turns)
    }

    /// Generate threat-response turns for quiescence.
    fn generate_threat_turns(
        &self,
        game: &HexGame,
        my_threats: &FxHashSet<Coord>,
        opp_threats: &FxHashSet<Coord>,
    ) -> Vec<(Coord, Coord)> {
        if let Some(win_turn) = self.find_instant_win(game, game.current_player) {
            return vec![win_turn];
        }

        let is_a = game.current_player == P1;
        let maximizing = game.current_player == self.me;
        let sign: f64 = if maximizing { 1.0 } else { -1.0 };

        let opp_cells: Vec<Coord> = opp_threats
            .iter()
            .filter(|c| self.cand_set.contains(c))
            .copied()
            .collect();
        let my_cells: Vec<Coord> = my_threats
            .iter()
            .filter(|c| self.cand_set.contains(c))
            .copied()
            .collect();

        let primary = if !opp_cells.is_empty() {
            &opp_cells
        } else if !my_cells.is_empty() {
            &my_cells
        } else {
            return Vec::new();
        };

        if primary.len() >= 2 {
            let mut pairs: Vec<(Coord, Coord)> = Vec::new();
            for i in 0..primary.len() {
                for j in (i + 1)..primary.len() {
                    pairs.push((primary[i], primary[j]));
                }
            }
            pairs.sort_by(|a, b| {
                let da = self.move_delta(a.0 .0, a.0 .1, is_a)
                    + self.move_delta(a.1 .0, a.1 .1, is_a);
                let db = self.move_delta(b.0 .0, b.0 .1, is_a)
                    + self.move_delta(b.1 .0, b.1 .1, is_a);
                if maximizing {
                    db.partial_cmp(&da).unwrap_or(std::cmp::Ordering::Equal)
                } else {
                    da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
                }
            });
            return pairs;
        }

        // Single threat cell — pair with best companion
        let tc = primary[0];
        let mut best_comp: Option<Coord> = None;
        let mut best_delta = f64::NEG_INFINITY;
        for &c in &self.cand_set {
            if c != tc {
                let d = self.move_delta(c.0, c.1, is_a) * sign;
                if d > best_delta {
                    best_delta = d;
                    best_comp = Some(c);
                }
            }
        }
        match best_comp {
            Some(comp) => vec![(tc.min(comp), tc.max(comp))],
            None => Vec::new(),
        }
    }

    // ── TT key ──────────────────────────────────────────────────────────────

    fn tt_key(&self, game: &HexGame) -> (u64, u8, u8) {
        (self.hash, game.current_player, game.placements_in_turn)
    }

    // ── Quiescence search ───────────────────────────────────────────────────

    fn quiescence(
        &mut self,
        game: &mut HexGame,
        mut alpha: f64,
        mut beta: f64,
        qdepth: i32,
    ) -> Result<f64, TimeUp> {
        self.check_time()?;

        if game.winner_raw() != NO_PLAYER {
            if game.winner_raw() == self.me {
                return Ok(WIN_SCORE);
            } else {
                return Ok(-WIN_SCORE);
            }
        }

        if let Some(win_turn) = self.find_instant_win(game, game.current_player) {
            let undo = self.make_turn(game, win_turn.0, win_turn.1);
            let score = if game.winner_raw() == self.me {
                WIN_SCORE
            } else {
                -WIN_SCORE
            };
            self.unmake_turn(game, &undo);
            return Ok(score);
        }

        let stand_pat = self.eval_score;
        let current = game.current_player;
        let opponent = 3 - current;
        let my_threats = self.find_threat_cells(game, current);
        let opp_threats = self.find_threat_cells(game, opponent);

        if (my_threats.is_empty() && opp_threats.is_empty()) || qdepth <= 0 {
            return Ok(stand_pat);
        }

        let maximizing = current == self.me;

        if maximizing {
            if stand_pat >= beta {
                return Ok(stand_pat);
            }
            alpha = alpha.max(stand_pat);
        } else {
            if stand_pat <= alpha {
                return Ok(stand_pat);
            }
            beta = beta.min(stand_pat);
        }

        let threat_turns =
            self.generate_threat_turns(game, &my_threats, &opp_threats);
        if threat_turns.is_empty() {
            return Ok(stand_pat);
        }

        if maximizing {
            let mut value = stand_pat;
            for &(m1, m2) in &threat_turns {
                let undo = self.make_turn(game, m1, m2);
                let child_val = if game.winner_raw() != NO_PLAYER {
                    if game.winner_raw() == self.me {
                        WIN_SCORE
                    } else {
                        -WIN_SCORE
                    }
                } else {
                    self.quiescence(game, alpha, beta, qdepth - 1)?
                };
                self.unmake_turn(game, &undo);
                if child_val > value {
                    value = child_val;
                }
                alpha = alpha.max(value);
                if alpha >= beta {
                    break;
                }
            }
            Ok(value)
        } else {
            let mut value = stand_pat;
            for &(m1, m2) in &threat_turns {
                let undo = self.make_turn(game, m1, m2);
                let child_val = if game.winner_raw() != NO_PLAYER {
                    if game.winner_raw() == self.me {
                        WIN_SCORE
                    } else {
                        -WIN_SCORE
                    }
                } else {
                    self.quiescence(game, alpha, beta, qdepth - 1)?
                };
                self.unmake_turn(game, &undo);
                if child_val < value {
                    value = child_val;
                }
                beta = beta.min(value);
                if alpha >= beta {
                    break;
                }
            }
            Ok(value)
        }
    }

    // ── Alpha-beta search ───────────────────────────────────────────────────

    fn minimax(
        &mut self,
        game: &mut HexGame,
        depth: i32,
        mut alpha: f64,
        mut beta: f64,
    ) -> Result<f64, TimeUp> {
        self.check_time()?;

        if game.winner_raw() != NO_PLAYER {
            if game.winner_raw() == self.me {
                return Ok(WIN_SCORE);
            } else {
                return Ok(-WIN_SCORE);
            }
        }

        let tt_key = self.tt_key(game);
        let mut tt_move: Option<(Coord, Coord)> = None;
        if let Some(entry) = self.tt.get(&tt_key) {
            tt_move = entry.best_move;
            if entry.depth >= depth {
                match entry.flag {
                    TT_EXACT => return Ok(entry.score),
                    TT_LOWER => alpha = alpha.max(entry.score),
                    TT_UPPER => beta = beta.min(entry.score),
                    _ => {}
                }
                if alpha >= beta {
                    return Ok(entry.score);
                }
            }
        }

        if depth == 0 {
            let score = self.quiescence(game, alpha, beta, MAX_QDEPTH)?;
            self.tt.insert(
                tt_key,
                TTEntry {
                    depth: 0,
                    score,
                    flag: TT_EXACT,
                    best_move: None,
                },
            );
            return Ok(score);
        }

        // Instant win check
        if let Some(win_turn) = self.find_instant_win(game, game.current_player) {
            let undo = self.make_turn(game, win_turn.0, win_turn.1);
            let score = if game.winner_raw() == self.me {
                WIN_SCORE
            } else {
                -WIN_SCORE
            };
            self.unmake_turn(game, &undo);
            self.tt.insert(
                tt_key,
                TTEntry {
                    depth,
                    score,
                    flag: TT_EXACT,
                    best_move: Some(win_turn),
                },
            );
            return Ok(score);
        }

        // Opponent instant win — check if blockable
        let opponent = 3 - game.current_player;
        if let Some(_opp_win) = self.find_instant_win(game, opponent) {
            // Check if opponent has multiple must-block windows
            let p_idx = if opponent == P1 { 0usize } else { 1 };
            let o_idx = 1 - p_idx;
            let hot = if opponent == P1 { &self.hot_a } else { &self.hot_b };
            let board = game.board_ref();

            let mut must_hit: Vec<FxHashSet<Coord>> = Vec::new();
            for &wkey in hot.iter() {
                if let Some(counts) = self.wc.get(&wkey) {
                    if counts[p_idx] >= (WIN_LEN - 2) as u32 && counts[o_idx] == 0 {
                        let (dq, dr) = DIR_VECTORS[wkey.0 as usize];
                        let (sq, sr) = (wkey.1, wkey.2);
                        let mut empties = FxHashSet::default();
                        for j in 0..WIN_LEN as i16 {
                            let cell = (sq + j * dq, sr + j * dr);
                            if !board.contains_key(&cell) {
                                empties.insert(cell);
                            }
                        }
                        must_hit.push(empties);
                    }
                }
            }

            if must_hit.len() > 1 {
                // Check if any pair of cells can cover all windows
                let mut all_cells = FxHashSet::default();
                for s in &must_hit {
                    for &c in s {
                        all_cells.insert(c);
                    }
                }
                let mut can_block = false;
                'outer: for &c1 in &all_cells {
                    for &c2 in &all_cells {
                        if must_hit.iter().all(|w| w.contains(&c1) || w.contains(&c2))
                        {
                            can_block = true;
                            break 'outer;
                        }
                    }
                }
                if !can_block {
                    let score = if opponent != self.me {
                        -WIN_SCORE
                    } else {
                        WIN_SCORE
                    };
                    self.tt.insert(
                        tt_key,
                        TTEntry {
                            depth,
                            score,
                            flag: TT_EXACT,
                            best_move: None,
                        },
                    );
                    return Ok(score);
                }
            }
        }

        let orig_alpha = alpha;
        let orig_beta = beta;
        let maximizing = game.current_player == self.me;

        let mut turns = self.generate_inner_turns(game);
        if turns.is_empty() {
            let score = self.eval_score;
            self.tt.insert(
                tt_key,
                TTEntry {
                    depth,
                    score,
                    flag: TT_EXACT,
                    best_move: None,
                },
            );
            return Ok(score);
        }

        // Move TT best move to front
        if let Some(tt_mv) = tt_move {
            if let Some(idx) = turns.iter().position(|&t| t == tt_mv) {
                turns.swap(0, idx);
            }
        }

        let mut best_move: Option<(Coord, Coord)> = None;

        if maximizing {
            let mut value = f64::NEG_INFINITY;
            for &(m1, m2) in &turns {
                let undo = self.make_turn(game, m1, m2);
                let child_val = if game.winner_raw() != NO_PLAYER {
                    if game.winner_raw() == self.me {
                        WIN_SCORE
                    } else {
                        -WIN_SCORE
                    }
                } else {
                    self.minimax(game, depth - 1, alpha, beta)?
                };
                self.unmake_turn(game, &undo);
                if child_val > value {
                    value = child_val;
                    best_move = Some((m1, m2));
                }
                alpha = alpha.max(value);
                if alpha >= beta {
                    // History heuristic update
                    let d2 = (depth * depth) as f64;
                    *self.history.entry(m1).or_insert(0.0) += d2;
                    *self.history.entry(m2).or_insert(0.0) += d2;
                    break;
                }
            }

            let flag = if value <= orig_alpha {
                TT_UPPER
            } else if value >= orig_beta {
                TT_LOWER
            } else {
                TT_EXACT
            };
            self.tt.insert(
                tt_key,
                TTEntry {
                    depth,
                    score: value,
                    flag,
                    best_move,
                },
            );
            Ok(value)
        } else {
            let mut value = f64::INFINITY;
            for &(m1, m2) in &turns {
                let undo = self.make_turn(game, m1, m2);
                let child_val = if game.winner_raw() != NO_PLAYER {
                    if game.winner_raw() == self.me {
                        WIN_SCORE
                    } else {
                        -WIN_SCORE
                    }
                } else {
                    self.minimax(game, depth - 1, alpha, beta)?
                };
                self.unmake_turn(game, &undo);
                if child_val < value {
                    value = child_val;
                    best_move = Some((m1, m2));
                }
                beta = beta.min(value);
                if alpha >= beta {
                    let d2 = (depth * depth) as f64;
                    *self.history.entry(m1).or_insert(0.0) += d2;
                    *self.history.entry(m2).or_insert(0.0) += d2;
                    break;
                }
            }

            let flag = if value <= orig_alpha {
                TT_UPPER
            } else if value >= orig_beta {
                TT_LOWER
            } else {
                TT_EXACT
            };
            self.tt.insert(
                tt_key,
                TTEntry {
                    depth,
                    score: value,
                    flag,
                    best_move,
                },
            );
            Ok(value)
        }
    }

    /// Root search: evaluates all turns without pruning (for sorting).
    fn search_root(
        &mut self,
        game: &mut HexGame,
        turns: &[(Coord, Coord)],
        depth: i32,
    ) -> Result<((Coord, Coord), FxHashMap<(Coord, Coord), f64>), TimeUp> {
        let maximizing = game.current_player == self.me;
        let mut best_turn = turns[0];
        let mut alpha = f64::NEG_INFINITY;
        let mut beta = f64::INFINITY;
        let mut scores: FxHashMap<(Coord, Coord), f64> = FxHashMap::default();

        for &(m1, m2) in turns {
            self.check_time()?;
            let undo = self.make_turn(game, m1, m2);
            let score = if game.winner_raw() != NO_PLAYER {
                if game.winner_raw() == self.me {
                    WIN_SCORE
                } else {
                    -WIN_SCORE
                }
            } else {
                self.minimax(game, depth - 1, alpha, beta)?
            };
            self.unmake_turn(game, &undo);
            scores.insert((m1, m2), score);

            if maximizing && score > alpha {
                alpha = score;
                best_turn = (m1, m2);
            } else if !maximizing && score < beta {
                beta = score;
                best_turn = (m1, m2);
            }
        }

        // Store in TT
        let best_score = if maximizing { alpha } else { beta };
        let tt_key = self.tt_key(game);
        self.tt.insert(
            tt_key,
            TTEntry {
                depth,
                score: best_score,
                flag: TT_EXACT,
                best_move: Some(best_turn),
            },
        );

        Ok((best_turn, scores))
    }
}

// ── Cached pair move state ──────────────────────────────────────────────────

/// Holds a queued second move from a pair search.
/// The bot returns one move at a time (called twice per turn by batched.rs).
/// On the first call it computes the pair and returns move 1, caching move 2.
/// On the second call it returns the cached move 2.
pub struct PairCache {
    /// The cached second move, if any.
    second_move: Option<Coord>,
}

// ── Public interface ────────────────────────────────────────────────────────

/// Thread-local storage for the cached second move of a pair.
/// We use a simple static-like approach via an explicit cache parameter.

/// Choose a single move for the current player.
///
/// Because the game uses pair placements (2 stones per turn after move 1),
/// but this function returns a single (q, r), we compute the full pair on the
/// first call and cache the second stone.  The `pair_cache` parameter persists
/// across the two calls within one turn.
///
/// `pat_value` is the preloaded 729-entry pattern table.
/// `time_limit_ms` is the total time budget for the pair computation.
/// `rng` is used for colony direction randomization.
pub fn minimax_choose_move(
    game: &HexGame,
    pat_value: &[f64; NUM_PATTERNS],
    time_limit_ms: u64,
    rng: &mut impl Rng,
    pair_cache: &mut Option<PairCache>,
) -> Coord {
    // Check if we have a cached second move
    if let Some(ref mut cache) = pair_cache {
        if let Some(second) = cache.second_move.take() {
            // Verify the cached move is still valid (cell is empty)
            if !game.board_ref().contains_key(&second) {
                *pair_cache = None;
                return second;
            }
        }
    }
    *pair_cache = None;

    // Empty board — play center
    if game.board_ref().is_empty() {
        return (0, 0);
    }

    // Build minimax state
    let mut state = MinimaxState::new(game, pat_value, time_limit_ms);
    let mut game_clone = game.clone();

    if state.cand_set.is_empty() {
        return (0, 0);
    }

    // Determine if this is a single-stone turn (first move of game)
    let is_single = game.move_history.len() == 0;

    if is_single {
        // Single stone placement — just find best single move
        let mut candidates: Vec<Coord> = state.cand_set.iter().copied().collect();
        let is_a = game.current_player == P1;
        candidates.sort_by(|a, b| {
            let da = state.move_delta(a.0, a.1, is_a);
            let db = state.move_delta(b.0, b.1, is_a);
            db.partial_cmp(&da).unwrap_or(std::cmp::Ordering::Equal)
        });
        return candidates.first().copied().unwrap_or((0, 0));
    }

    // Generate pair turns
    let turns = state.generate_root_turns(&game_clone);
    if turns.is_empty() {
        return (0, 0);
    }

    let mut best_pair = turns[0];
    let mut sorted_turns = turns.clone();

    // Save state for TimeUp rollback
    let saved_hash = state.hash;
    let saved_eval = state.eval_score;
    let saved_wc: FxHashMap<WKey, [u32; 2]> =
        state.wc.iter().map(|(&k, v)| (k, *v)).collect();
    let saved_wp: FxHashMap<WKey, u32> = state.wp.clone();
    let saved_cand_set: FxHashSet<Coord> = state.cand_set.clone();
    let saved_cand_rc: FxHashMap<Coord, u32> = state.cand_refcount.clone();
    let saved_hot_a: FxHashSet<WKey> = state.hot_a.clone();
    let saved_hot_b: FxHashSet<WKey> = state.hot_b.clone();
    let saved_rc_stack_len = state.rc_stack.len();

    // Save game state for rollback
    let game_backup = game_clone.clone();

    let maximizing = game.current_player == state.me;

    // Iterative deepening
    for depth in 1..200 {
        match state.search_root(&mut game_clone, &sorted_turns, depth) {
            Ok((result, scores)) => {
                best_pair = result;
                // Re-sort turns for next iteration
                sorted_turns.sort_by(|a, b| {
                    let sa = scores.get(a).copied().unwrap_or(0.0);
                    let sb = scores.get(b).copied().unwrap_or(0.0);
                    if maximizing {
                        sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
                    } else {
                        sa.partial_cmp(&sb).unwrap_or(std::cmp::Ordering::Equal)
                    }
                });
                // Early exit on proven win/loss
                if let Some(&score) = scores.get(&result) {
                    if score.abs() >= WIN_SCORE {
                        break;
                    }
                }
            }
            Err(TimeUp) => {
                // Rollback state — game_clone not needed further, only state matters
                let _ = game_backup;
                state.hash = saved_hash;
                state.eval_score = saved_eval;
                state.wc = saved_wc;
                state.wp = saved_wp;
                state.cand_set = saved_cand_set;
                state.cand_refcount = saved_cand_rc;
                state.hot_a = saved_hot_a;
                state.hot_b = saved_hot_b;
                state.rc_stack.truncate(saved_rc_stack_len);
                break;
            }
        }
    }

    // Cache the second move
    let (m1, m2) = best_pair;
    if m1 != m2 {
        *pair_cache = Some(PairCache {
            second_move: Some(m2),
        });
    }

    m1
}

// ── Convenience: load pattern values from file path ─────────────────────────

/// Load pattern values from a JSON file on disk.
pub fn load_pattern_values_from_file(
    path: &str,
) -> Result<[f64; NUM_PATTERNS], Box<dyn std::error::Error>> {
    let json_str = std::fs::read_to_string(path)?;
    Ok(load_pattern_values(&json_str))
}
