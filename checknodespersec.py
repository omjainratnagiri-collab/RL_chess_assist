import chess, torch
from nnue import NNUE
from search import SearchEngine

device = torch.device("cpu")
model = NNUE().to(device)
checkpoint = torch.load("checkpoints/td_selfplay_best.pt", map_location=device)
model.load_state_dict(checkpoint["model"] if "model" in checkpoint else checkpoint)
model.eval()

engine = SearchEngine(model, device)                                      
board = chess.Board('r1bq1rk1/pp2ppbp/2np1np1/8/3NP3/2N1BP2/PPPQ2PP/R3KB1R b KQ - 0 9')                            
acc = engine.new_accumulator(board)

move, score, info = engine.search(board, acc, max_depth=6, time_limit=2)
print(f"move={move} score={score:.3f}")
print(f"depth={info['depth']} nodes={info['nodes']} time={info['time']:.3f}s nps={info['nps']}")
