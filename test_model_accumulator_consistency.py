
import torch
import numpy as np
import chess

from nnue import NNUE
from halfkp_encoding import encode_stm, encode_halfkp
from accumulator import AccumulatorPair

model = NNUE()
model.eval()

board = chess.Board()
board.push_san("e4")
board.push_san("c5")
board.push_san("Nf3")

                                                                
us, them = encode_stm(board)
with torch.no_grad():
    policy1, value1 = model([us], [them])

                                                  
weights_np = model.feature_transformer.embedding.weight.detach().cpu().numpy()
acc = AccumulatorPair(weights_np, board=board)
stm_vec, opp_vec = acc.stm_and_opponent(board.turn == chess.WHITE)
with torch.no_grad():
    policy2, value2 = model.forward_from_accumulators(stm_vec, opp_vec)

print("value1 (training path):", value1.item())
print("value2 (accumulator path):", value2.item())
print("value match:", torch.allclose(value1, value2, atol=1e-4))
print("policy match:", torch.allclose(policy1, policy2, atol=1e-3))
