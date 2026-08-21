
import sys
                        

import chess
import torch
import numpy as np
import random

from nnue import NNUE
from search import SearchEngine, MATE_SCORE
                           
SEED = 0
random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

model = NNUE().to(device)
model.eval()
engine = SearchEngine(model, device)

results = []

                                                                          
                                                                         
                                                
print("\n=== Test 1: mate-in-1 detection ===")
board = chess.Board("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1")
acc = engine.new_accumulator(board)
move, score, info = engine.search(board, acc, max_depth=3, time_limit=10)
expected_move = chess.Move.from_uci("a1a8")
ok = (move == expected_move) and (score > MATE_SCORE - 0.5)
results.append(ok)
print(f"  [{'PASS' if ok else 'FAIL'}] found={move} score={score:.3f} "
      f"(expected {expected_move}, score near +{MATE_SCORE})")
print(f"  nodes={info['nodes']} depth={info['depth']} nps={info['nps']}")

                                                                                   
print("\n=== Test 2: mate-in-1 detection (Black to move) ===")
board2 = chess.Board("r5k1/8/8/8/8/8/5PPP/6K1 b - - 0 1")
acc2 = engine.new_accumulator(board2)
move2, score2, info2 = engine.search(board2, acc2, max_depth=3, time_limit=10)
expected_move2 = chess.Move.from_uci("a8a1")
ok2 = (move2 == expected_move2) and (score2 > MATE_SCORE - 0.5)
results.append(ok2)
print(f"  [{'PASS' if ok2 else 'FAIL'}] found={move2} score={score2:.3f} "
      f"(expected {expected_move2}, score near +{MATE_SCORE})")

                                                                         
                                                                          
 
                                                                         
                                                                       
                                                                       
                                                                         
                                                                        
                                                                       
                                                                         
                                                                     
                                         
     
print("\n=== Test 3: board/accumulator state preserved after search ===")
board3 = chess.Board()
fen_before = board3.fen()
acc3 = engine.new_accumulator(board3)
white_before = acc3.white.copy()
black_before = acc3.black.copy()
undo_depth_before = len(acc3._undo_stack)

engine.search(board3, acc3, max_depth=3, time_limit=10)

fen_ok = board3.fen() == fen_before
acc_ok = (
    np.allclose(acc3.white, white_before, atol=1e-5)
    and np.allclose(acc3.black, black_before, atol=1e-5)
)
stack_ok = len(acc3._undo_stack) == undo_depth_before
ok3 = fen_ok and acc_ok and stack_ok
results.append(ok3)
max_white_diff = float(np.max(np.abs(acc3.white - white_before)))
max_black_diff = float(np.max(np.abs(acc3.black - black_before)))
print(f"  [{'PASS' if ok3 else 'FAIL'}] fen_unchanged={fen_ok} "
      f"accumulator_unchanged={acc_ok} (max diff white={max_white_diff:.2e}, "
      f"black={max_black_diff:.2e}) undo_stack_balanced={stack_ok}")

                                                                          
                
print("\n=== Test 4: stability across random positions ===")
rng = random.Random(3)
crash_free = True
state_clean = True
for i in range(15):
    b = chess.Board()
    for _ in range(rng.randint(0, 40)):
        if b.is_game_over(claim_draw=True):
            break
        b.push(rng.choice(list(b.legal_moves)))
    if b.is_game_over(claim_draw=True):
        continue

    fen_before_i = b.fen()
    a = engine.new_accumulator(b)
    try:
        _, _, info_i = engine.search(b, a, max_depth=3, time_limit=5)
    except Exception as exc:
        crash_free = False
        print(f"  [FAIL] crashed on position {i}: {fen_before_i} -> {exc}")
        continue

    if b.fen() != fen_before_i or len(a._undo_stack) != 0:
        state_clean = False
        print(f"  [FAIL] state corrupted after search on position {i}: {fen_before_i}")

    print(f"  position {i}: depth={info_i['depth']} nodes={info_i['nodes']} time={info_i['time']:.2f}s")

ok4 = crash_free and state_clean
results.append(ok4)
print(f"  [{'PASS' if ok4 else 'FAIL'}] 15 random positions, crash_free={crash_free}, state_clean={state_clean}")

                                                                           
                                  
print("\n=== Test 5: iterative deepening reaches requested depth ===")
board5 = chess.Board()
acc5 = engine.new_accumulator(board5)
_, _, info5 = engine.search(board5, acc5, max_depth=4, time_limit=30)
ok5 = info5["depth"] == 4
results.append(ok5)
print(f"  [{'PASS' if ok5 else 'FAIL'}] requested max_depth=4, reached depth={info5['depth']}")

print()
print("=" * 60)
print(f"TOTAL: {sum(results)}/{len(results)} tests passed")
print("=" * 60)
if not all(results):
    sys.exit(1)