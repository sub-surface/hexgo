use pyo3::prelude::*;

mod types;
mod game;
mod node;
mod mcts;
mod parallel;
mod encode;
mod minimax;
mod batched;

/// HexGo Rust engine — drop-in replacement for game.py + mcts.py.
///
/// Usage from Python:
///   from hexgo import HexGame, mcts, mcts_with_net
///   game = HexGame()
///   game.make(0, 0)
///   move = mcts(game, num_sims=200)
#[pymodule]
fn hexgo(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Game engine
    m.add_class::<game::HexGame>()?;

    // MCTS functions
    m.add_function(wrap_pyfunction!(mcts::mcts, m)?)?;
    m.add_function(wrap_pyfunction!(mcts::mcts_with_net, m)?)?;
    m.add_function(wrap_pyfunction!(mcts::self_play_game, m)?)?;

    // Parallel self-play
    m.add_class::<parallel::GameResult>()?;
    m.add_function(wrap_pyfunction!(parallel::parallel_self_play, m)?)?;
    m.add_function(wrap_pyfunction!(parallel::parallel_eisenstein_games, m)?)?;

    // Batched net-guided self-play
    m.add_class::<batched::GameTrainingResult>()?;
    m.add_class::<batched::PositionData>()?;
    m.add_function(wrap_pyfunction!(batched::batched_self_play, m)?)?;

    // Minimax
    m.add_function(wrap_pyfunction!(minimax_choose_move_py, m)?)?;
    m.add_function(wrap_pyfunction!(minimax_load_patterns, m)?)?;

    // Constants
    m.add("WIN_LENGTH", types::WIN_LENGTH)?;
    m.add("PLACEMENT_RADIUS", types::PLACEMENT_RADIUS)?;
    m.add("P1", types::P1)?;
    m.add("P2", types::P2)?;

    Ok(())
}

/// Load minimax pattern values from a JSON file. Returns an opaque handle.
#[pyfunction]
fn minimax_load_patterns(path: &str) -> PyResult<Vec<f64>> {
    let pats = minimax::load_pattern_values_from_file(path)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{}", e)))?;
    Ok(pats.to_vec())
}

/// Choose a move using minimax alpha-beta search.
/// Returns (q, r). Manages pair caching internally via a module-level cache.
#[pyfunction]
#[pyo3(signature = (game, patterns, time_limit_ms=50))]
fn minimax_choose_move_py(
    game: &game::HexGame,
    patterns: Vec<f64>,
    time_limit_ms: u64,
) -> PyResult<(i16, i16)> {
    use std::cell::RefCell;
    thread_local! {
        static PAIR_CACHE: RefCell<Option<minimax::PairCache>> = RefCell::new(None);
    }
    let mut pat_arr = [0.0f64; 729];
    pat_arr.copy_from_slice(&patterns[..729]);
    let mut rng = rand::thread_rng();
    PAIR_CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        let mv = minimax::minimax_choose_move(game, &pat_arr, time_limit_ms, &mut rng, &mut cache);
        Ok(mv)
    })
}
