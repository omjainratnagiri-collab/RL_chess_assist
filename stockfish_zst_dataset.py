
import json
import math
import io
from pathlib import Path

import chess
import torch
from torch.utils.data import IterableDataset, get_worker_info
import zstandard as zstd

from move_encoder import move_to_index
from halfkp_encoding import encode_stm


def cp_to_value(cp, scale=400.0):
    return math.tanh(float(cp) / scale)


def cp_to_policy_probs(cps, temperature=100.0):
    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    scaled = [float(cp) / temperature for cp in cps]
    max_scaled = max(scaled)
    weights = [math.exp(score - max_scaled) for score in scaled]
    total = sum(weights)
    return [weight / total for weight in weights]


def _select_eval(evals, max_depth=None, use_first_eval=True):
    if not evals:
        return None

    if use_first_eval:
        eval_item = evals[0]
        if max_depth is None or eval_item.get("depth", 0) <= max_depth:
            return eval_item
        return None

    candidates = [
        item for item in evals
        if max_depth is None or item.get("depth", 0) <= max_depth
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda item: item.get("depth", 0))


class StockfishZstDataset(IterableDataset):
    def __init__(
        self,
        zst_file,
        max_depth=None,
        max_positions=None,
        skip_positions=0,
        max_pvs=None,
        policy_temperature=100.0,
        value_scale=400.0,
        use_first_eval=True,
        skip_invalid=True
    ):
        self.zst_file = Path(zst_file)
        self.max_depth = max_depth
        self.max_positions = max_positions
        self.skip_positions = skip_positions
        self.max_pvs = max_pvs
        self.policy_temperature = policy_temperature
        self.value_scale = value_scale
        self.use_first_eval = use_first_eval
        self.skip_invalid = skip_invalid

        if not self.zst_file.exists():
            raise FileNotFoundError(f"{self.zst_file} does not exist.")

    def __iter__(self):
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        num_workers = worker.num_workers if worker is not None else 1

        yielded = 0
        skipped = 0
        dctx = zstd.ZstdDecompressor()

        with open(self.zst_file, "rb") as compressed:
            with dctx.stream_reader(compressed) as reader:
                text = io.TextIOWrapper(reader, encoding="utf-8")
                for line_no, line in enumerate(text):
                    if line_no % num_workers != worker_id:
                        continue

                    if (
                        self.max_positions is not None
                        and yielded >= self.max_positions
                    ):
                        break

                    line = line.strip()
                    if not line:
                        continue

                    try:
                        sample = self._parse_line(line)
                    except Exception:
                        if self.skip_invalid:
                            continue
                        raise

                    if sample is None:
                        continue

                    if skipped < self.skip_positions:
                        skipped += 1
                        continue

                    yielded += 1
                    yield sample

    def _parse_line(self, line):
        item = json.loads(line)
        board = chess.Board(item["fen"])
        eval_item = _select_eval(
            item.get("evals", []),
            self.max_depth,
            self.use_first_eval
        )
        if eval_item is None:
            return None

        move_indices = []
        cps = []
        pvs = eval_item.get("pvs", [])
        if self.max_pvs is not None:
            pvs = pvs[:self.max_pvs]

        for pv in pvs:
            cp = pv.get("cp")
            line_text = pv.get("line", "")
            if cp is None or not line_text:
                continue

            move_uci = line_text.split()[0]
            move = chess.Move.from_uci(move_uci)
            if move not in board.legal_moves:
                continue

            move_indices.append(move_to_index(move, board))
            cps.append(cp)

        if not move_indices:
            return None

                                                                           
                                                                   
        policy_cps = cps if board.turn == chess.WHITE else [-cp for cp in cps]
        probs = cp_to_policy_probs(policy_cps, self.policy_temperature)

                                                                   
        value_white = cp_to_value(cps[0], self.value_scale)
        value = value_white if board.turn == chess.WHITE else -value_white

        us_indices, them_indices = encode_stm(board)

                                                                            
                                                                          
                                                                    
                                                                       
                                              
        return (
            us_indices,
            them_indices,
            move_indices,
            probs,
            value,
        )