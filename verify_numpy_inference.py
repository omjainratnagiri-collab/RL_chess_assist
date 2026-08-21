
import sys
from pathlib import Path

import numpy as np
import torch

from nnue import NNUE
from numpy_inference import NumpyValueHead

torch.manual_seed(0)
np.random.seed(0)

device = torch.device("cpu")
model = NNUE().to(device)
model.eval()

numpy_head = NumpyValueHead(model)

embedding_dim = model.feature_transformer.embedding.weight.shape[1]

max_diff = 0.0
num_trials = 200
failures = 0

for i in range(num_trials):
                                                                     
                                                                       
                                                            
    us_vec = np.random.randn(embedding_dim).astype(np.float32) * 0.5
    them_vec = np.random.randn(embedding_dim).astype(np.float32) * 0.5

    with torch.no_grad():
        _, torch_value = model.forward_from_accumulators(us_vec, them_vec, device=device)
    torch_value = float(torch_value.item())

    numpy_value = numpy_head.evaluate(us_vec, them_vec)

    diff = abs(torch_value - numpy_value)
    max_diff = max(max_diff, diff)
    if diff > 1e-4:
        failures += 1
        print(f"  [FAIL] trial {i}: torch={torch_value:.6f} numpy={numpy_value:.6f} diff={diff:.6f}")

print(f"\n{num_trials - failures}/{num_trials} trials matched within 1e-4, max diff={max_diff:.6f}")
if failures == 0:
    print("PASS -- numpy value head matches the torch path. Safe to use in search.")
else:
    print("FAIL -- do not use the numpy path until this is fixed.")
    sys.exit(1)