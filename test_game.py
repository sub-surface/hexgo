"""Quick sanity tests."""

from game import HexGame


def test_basic_play():
    g = HexGame()
    assert g.play(0, 0)
    assert g.board[(0, 0)] == 1
    assert g.current_player == 2
    assert not g.play(0, 0)  # occupied


def test_move_rule_1_2_2():
    g = HexGame()
    # Turn 1: Player 1 places 1 tile
    g.make(0, 0)
    assert g.current_player == 2
    assert g.placements_in_turn == 0
    
    # Turn 2: Player 2 places 1st tile
    g.make(1, 0)
    assert g.current_player == 2
    assert g.placements_in_turn == 1
    
    # Turn 2: Player 2 places 2nd tile
    g.make(1, 1)
    assert g.current_player == 1
    assert g.placements_in_turn == 0
    
    # Turn 3: Player 1 places 1st tile
    g.make(0, 1)
    assert g.current_player == 1
    
    # Undo 3: back to player 1, 0 placements
    g.unmake()
    assert g.current_player == 1
    assert g.placements_in_turn == 0
    
    # Undo 2 (2nd tile): back to player 2, 1 placement
    g.unmake()
    assert g.current_player == 2
    assert g.placements_in_turn == 1
    
    # Undo 2 (1st tile): back to player 2, 0 placements
    g.unmake()
    assert g.current_player == 2
    assert g.placements_in_turn == 0
    
    # Undo 1: back to player 1, 0 placements
    g.unmake()
    assert g.current_player == 1
    assert g.placements_in_turn == 0
    assert len(g.board) == 0

def test_six_in_a_row():
    g = HexGame()
    # P1: (0,0) [1 tile]
    g.make(0, 0)
    # P2: (10,10), (10,11) [2 tiles]
    g.make(10, 10)
    g.make(10, 11)
    # P1: (1,0), (2,0) [2 tiles]
    g.make(1, 0)
    g.make(2, 0)
    # P2: (11,10), (11,11) [2 tiles]
    g.make(11, 10)
    g.make(11, 11)
    # P1: (3,0), (4,0) [2 tiles]
    g.make(3, 0)
    g.make(4, 0)
    # P2: (12,10), (12,11) [2 tiles]
    g.make(12, 10)
    g.make(12, 11)
    # P1: (5,0) -> WIN
    g.make(5, 0)
    assert g.winner == 1


def test_no_false_win():
    g = HexGame()
    # P1: (0,0)
    g.make(0, 0)
    # P2: (1,0), (2,0)
    g.make(1, 0)
    g.make(2, 0)
    # P1: (0,1), (0,2)
    g.make(0, 1)
    g.make(0, 2)
    assert g.winner is None


def test_clone():
    g = HexGame()
    g.play(0, 0)
    c = g.clone()
    c.play(1, 0)
    assert (1, 0) not in g.board


def test_legal_moves_empty():
    g = HexGame()
    assert g.legal_moves() == [(0, 0)]


def test_legal_moves_after_play():
    g = HexGame()
    g.play(0, 0)
    moves = g.legal_moves()
    assert (0, 0) not in moves
    assert len(moves) == 6  # exactly 6 neighbors


def test_tensor_shape():
    g = HexGame()
    g.play(0, 0)
    t = g.to_tensor(size=5)
    assert len(t) == 3
    assert len(t[0]) == 11
    assert len(t[0][0]) == 11


if __name__ == "__main__":
    tests = [
        test_basic_play, 
        test_move_rule_1_2_2, 
        test_six_in_a_row, 
        test_no_false_win, 
        test_clone,
        test_legal_moves_empty, 
        test_legal_moves_after_play,
        test_tensor_shape
    ]
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
