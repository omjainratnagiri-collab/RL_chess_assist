
import random
import chess
import numpy as np
import sys
from halfkp_encoding import NUM_FEATURES
import zobrist
from accumulator import AccumulatorPair

EMBEDDING_DIM = 4                                                        

fake_weights = np.random.randn(NUM_FEATURES, EMBEDDING_DIM).astype(np.float32)

failures = 0
checked = 0
num_games = 30
plies_per_game = 60
rng = random.Random(0)

for g in range(num_games):
    board = chess.Board()
    acc = AccumulatorPair(fake_weights, board=board)

    expected = zobrist.compute_full_zobrist(board)
    if acc.zobrist != expected:
        print(f"  [FAIL] game {g} initial position: incremental={acc.zobrist} full={expected}")
        failures += 1
    checked += 1

    for ply in range(plies_per_game):
        legal = list(board.legal_moves)
        if not legal:
            break
        move = rng.choice(legal)

        acc.push(board, move)
        board.push(move)

        expected = zobrist.compute_full_zobrist(board)
        checked += 1
        if acc.zobrist != expected:
            failures += 1
            print(f"  [FAIL] game {g} ply {ply} ({move}): incremental={acc.zobrist} full={expected} "
                  f"fen={board.fen()}")

status = "PASS" if failures == 0 else "FAIL"
print(f"\n[{status}] real-move checks: {checked} positions across {num_games} random games, {failures} mismatches.")

                                                                    
                                
null_failures = 0
null_checked = 0
num_null_games = 20
plies_before_null = 10                                                 
                                                       

for g in range(num_null_games):
    board = chess.Board()
    acc = AccumulatorPair(fake_weights, board=board)

    for _ in range(plies_before_null):
        legal = list(board.legal_moves)
        if not legal:
            break
        move = rng.choice(legal)
        acc.push(board, move)
        board.push(move)

    if board.is_game_over(claim_draw=True):
        continue

                                                          
    acc.push_null(board)
    board.push(chess.Move.null())

    expected = zobrist.compute_full_zobrist(board)
    null_checked += 1
    if acc.zobrist != expected:
        null_failures += 1
        print(f"  [FAIL] null-move game {g}: incremental={acc.zobrist} full={expected} "
              f"fen_before_null={board.fen()}")

                                                                          
                                                                   
    pre_null_zobrist = None                                                                   
    board.pop()
    acc.pop_null()
    expected_after_pop = zobrist.compute_full_zobrist(board)
    null_checked += 1
    if acc.zobrist != expected_after_pop:
        null_failures += 1
        print(f"  [FAIL] null-move UNDO game {g}: incremental={acc.zobrist} full={expected_after_pop} "
              f"fen={board.fen()}")

null_status = "PASS" if null_failures == 0 else "FAIL"
print(f"[{null_status}] null-move checks: {null_checked} checks across {num_null_games} games, {null_failures} mismatches.")

failures += null_failures
status = "PASS" if failures == 0 else "FAIL"
print(f"\n[{status}] OVERALL: {failures} total mismatches (real-move + null-move).")
if failures:
    sys.exit(1)
