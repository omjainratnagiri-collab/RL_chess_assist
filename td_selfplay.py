

import argparse
import copy
import multiprocessing
import os
import pickle
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import chess
import torch
import torch.nn.functional as F

from nnue import NNUE
from search import SearchEngine
from halfkp_encoding import encode_stm


                                                                             
                                                                           
                                             
                                                                             

def td_lambda_targets(white_perspective_scores, outcome_white, td_lambda):
    
    T = len(white_perspective_scores)
    G = [0.0] * (T + 1)
    G[T] = outcome_white
    for t in range(T - 1, -1, -1):
        next_v = white_perspective_scores[t + 1] if t + 1 < T else outcome_white
        G[t] = (1 - td_lambda) * next_v + td_lambda * G[t + 1]
    return G[:T]


@dataclass
class Sample:
    us_indices: list
    them_indices: list
    value: float                                            


def load_checkpoint(path, model, optimizer, device, load_optimizer=False):
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
        if load_optimizer and optimizer is not None and "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        return checkpoint.get("cycle", 0)
    model.load_state_dict(checkpoint)
    return 0


def save_checkpoint(path, model, optimizer, cycle, args):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"cycle": cycle, "model": model.state_dict(),
         "optimizer": optimizer.state_dict(), "args": vars(args)},
        path,
    )


def save_replay(path, replay):
    
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "wb") as f:
        pickle.dump(replay, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp_path.replace(path)


def load_replay(path):
    
    path = Path(path)
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def random_opening(board, num_plies, rng):
    for _ in range(num_plies):
        if board.is_game_over(claim_draw=True):
            break
        board.push(rng.choice(list(board.legal_moves)))
    return board


def play_selfplay_game(engine, args, seed):
    rng = random.Random(seed)
    board = chess.Board()
    random_opening(board, args.opening_random_plies, rng)
    acc = engine.new_accumulator(board)

    ply_records = []                                                                       

    for _ply in range(args.max_plies):
        if board.is_game_over(claim_draw=True):
            break

        move, score, _info = engine.search(
            board, acc, max_depth=args.search_depth, time_limit=args.search_time_limit
        )
        if move is None:
            break

        mover_is_white = board.turn == chess.WHITE
        white_score = score if mover_is_white else -score
        us_indices, them_indices = encode_stm(board)
        ply_records.append((us_indices, them_indices, white_score, mover_is_white))

        acc.push(board, move)
        board.push(move)

    if not ply_records:
        return [], "no-moves"

    outcome = board.outcome(claim_draw=True)
    if outcome is None:
                                                                           
                                                                 
                                                                          
                                                       
        return [], "max-plies"

    if outcome.winner is None:
        outcome_white = 0.0
    else:
        outcome_white = 1.0 if outcome.winner == chess.WHITE else -1.0

    white_scores = [r[2] for r in ply_records]
    td_targets_white = td_lambda_targets(white_scores, outcome_white, args.td_lambda)

    samples = []
    for (us_indices, them_indices, _score, mover_is_white), target_white in zip(ply_records, td_targets_white):
        target = target_white if mover_is_white else -target_white
        samples.append(Sample(us_indices=us_indices, them_indices=them_indices, value=target))

    return samples, board.result(claim_draw=True)


PIECE_VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}


def material_score(board):
    score = 0
    for piece_type, val in PIECE_VALUES.items():
        score += val * len(board.pieces(piece_type, chess.WHITE))
        score -= val * len(board.pieces(piece_type, chess.BLACK))
    return score


def score_for_challenger(result, challenger_is_white):
    if result.startswith("1/2-1/2"):
        return 0.5
    if result.startswith("1-0"):
        return 1.0 if challenger_is_white else 0.0
    if result.startswith("0-1"):
        return 0.0 if challenger_is_white else 1.0
    return 0.5


def play_gate_game(challenger_engine, incumbent_engine, challenger_is_white, seed, args):
    rng = random.Random(seed)
    board = chess.Board()
    random_opening(board, args.gate_opening_plies, rng)

    challenger_acc = challenger_engine.new_accumulator(board)
    incumbent_acc = incumbent_engine.new_accumulator(board)

    ply = 0
    while not board.is_game_over(claim_draw=True) and ply < args.max_plies:
        is_challenger_turn = (board.turn == chess.WHITE) == challenger_is_white
        if is_challenger_turn:
            move, _s, _i = challenger_engine.search(
                board, challenger_acc, max_depth=args.gate_search_depth,
                time_limit=args.gate_search_time_limit,
            )
        else:
            move, _s, _i = incumbent_engine.search(
                board, incumbent_acc, max_depth=args.gate_search_depth,
                time_limit=args.gate_search_time_limit,
            )

        if move is None:
            winner_is_white = board.turn == chess.BLACK
            return score_for_challenger("1-0" if winner_is_white else "0-1", challenger_is_white)

        challenger_acc.push(board, move)
        incumbent_acc.push(board, move)
        board.push(move)
        ply += 1

    if ply >= args.max_plies and not board.is_game_over(claim_draw=True):
        score = material_score(board)
        if score >= args.gate_adjudicate_threshold:
            return score_for_challenger("1-0", challenger_is_white)
        if score <= -args.gate_adjudicate_threshold:
            return score_for_challenger("0-1", challenger_is_white)
        return 0.5

    return score_for_challenger(board.result(claim_draw=True), challenger_is_white)


def train_on_replay(model, optimizer, replay, device, args):
    model.train()
    total_loss = 0.0
    steps = 0

    for _epoch in range(args.train_epochs):
        order = list(range(len(replay)))
        random.shuffle(order)

        for start in range(0, len(order), args.batch_size):
            batch = [replay[i] for i in order[start:start + args.batch_size]]

            optimizer.zero_grad(set_to_none=True)
            _policy, value_pred = model(
                [s.us_indices for s in batch], [s.them_indices for s in batch]
            )
            value_target = torch.tensor([[s.value] for s in batch], dtype=torch.float32, device=device)
            loss = F.smooth_l1_loss(value_pred, value_target)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

            total_loss += loss.item()
            steps += 1

    return total_loss / steps


def trim_replay(replay, max_size):
    if max_size > 0 and len(replay) > max_size:
        del replay[:len(replay) - max_size]


def cpu_state_dict(model):
    
    return {k: v.detach().cpu() for k, v in model.state_dict().items()}


                                                                             
                                                                     
                                                                         
                                                                        
                                                                      
                                      
                                                                             

def _selfplay_worker(state_dict, seed, args):
    torch.set_num_threads(1)                                              
    device = torch.device("cpu")
    model = NNUE().to(device)
    model.load_state_dict(state_dict)
    model.eval()
    engine = SearchEngine(model, device)
    return play_selfplay_game(engine, args, seed)


def _gate_worker(challenger_state_dict, incumbent_state_dict, challenger_is_white, seed, args):
    torch.set_num_threads(1)
    device = torch.device("cpu")

    challenger_model = NNUE().to(device)
    challenger_model.load_state_dict(challenger_state_dict)
    challenger_model.eval()
    challenger_engine = SearchEngine(challenger_model, device)

    incumbent_model = NNUE().to(device)
    incumbent_model.load_state_dict(incumbent_state_dict)
    incumbent_model.eval()
    incumbent_engine = SearchEngine(incumbent_model, device)

    return play_gate_game(challenger_engine, incumbent_engine, challenger_is_white, seed, args)


def run_selfplay_cycle(pool, state_dict, cycle, args):
    
    tasks = [(state_dict, cycle * 100_000 + game_idx, args) for game_idx in range(args.games_per_cycle)]
    return pool.starmap(_selfplay_worker, tasks)


def run_gate_match(pool, challenger_state_dict, incumbent_state_dict, args, cycle):
    tasks = [
        (challenger_state_dict, incumbent_state_dict, game_idx % 2 == 0, cycle * 10_000 + game_idx, args)
        for game_idx in range(args.gate_games)
    ]
    scores = pool.starmap(_gate_worker, tasks)
    return sum(scores) / len(scores)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial-checkpoint", required=True)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--checkpoint-prefix", default="td_selfplay")
    parser.add_argument("--cycles", type=int, default=200)
    parser.add_argument("--games-per-cycle", type=int, default=8)
    parser.add_argument("--search-depth", type=int, default=5)
    parser.add_argument("--search-time-limit", type=float, default=2.0)
    parser.add_argument("--max-plies", type=int, default=200)
    parser.add_argument("--opening-random-plies", type=int, default=4)
    parser.add_argument("--td-lambda", type=float, default=0.7)
    parser.add_argument("--replay-size", type=int, default=20000)
    parser.add_argument("--min-replay-size", type=int, default=4000)
    parser.add_argument("--replay-file", default=None,
                         help="Path to save/load the replay buffer. Defaults to "
                              "<checkpoint-dir>/<checkpoint-prefix>_replay.pkl")
    parser.add_argument("--fresh-replay", action="store_true",
                         help="Ignore any saved replay buffer at --replay-file and start empty. "
                              "Use this after a deliberate change to td-lambda or the labeling "
                              "logic, where old samples' targets may no longer be representative.")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--train-epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gate-every", type=int, default=1)
    parser.add_argument("--gate-games", type=int, default=8)
    parser.add_argument("--gate-search-depth", type=int, default=4)
    parser.add_argument("--gate-search-time-limit", type=float, default=0.5)
    parser.add_argument("--gate-opening-plies", type=int, default=4)
    parser.add_argument("--gate-min-score", type=float, default=0.5)
    parser.add_argument("--gate-adjudicate-threshold", type=float, default=4)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1),
                         help="Number of worker processes for self-play/gate games. Defaults to "
                              "(CPU count - 1), leaving one core free for the main process "
                              "(training + orchestration). Set to 1 to fall back to the old "
                              "sequential behavior. If memory usage is too high, LOWER this "
                              "explicitly rather than relying on the default -- each worker is "
                              "a full separate process (its own torch/numpy/chess imports, its "
                              "own model copy), so memory scales roughly linearly with worker "
                              "count. Try e.g. --workers 4 first and watch actual RAM usage "
                              "before pushing it higher.")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using: {device} for training, {args.workers} CPU worker process(es) for self-play/gating")
    print(
        f"Settings: cycles={args.cycles}, games/cycle={args.games_per_cycle}, "
        f"search_depth={args.search_depth}, td_lambda={args.td_lambda}, "
        f"replay={args.replay_size}, gate_games={args.gate_games}"
    )

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_dir / f"{args.checkpoint_prefix}_best.pt"
    latest_path = checkpoint_dir / f"{args.checkpoint_prefix}_latest.pt"
    replay_path = Path(args.replay_file) if args.replay_file else checkpoint_dir / f"{args.checkpoint_prefix}_replay.pkl"

    model = NNUE().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    start_cycle = 0
    if args.resume and Path(args.resume).exists():
        start_cycle = load_checkpoint(args.resume, model, optimizer, device, load_optimizer=True)
        print(f"Resumed: {args.resume} (cycle {start_cycle})")
    elif latest_path.exists():
        start_cycle = load_checkpoint(latest_path, model, optimizer, device, load_optimizer=True)
        print(f"Resumed: {latest_path} (cycle {start_cycle})")
    else:
        load_checkpoint(args.initial_checkpoint, model, optimizer, device, load_optimizer=False)
        print(f"Loaded SL starting checkpoint: {args.initial_checkpoint}")

    best_model = NNUE().to(device)
    if best_path.exists():
        load_checkpoint(best_path, best_model, None, device, load_optimizer=False)
        print(f"Loaded gating baseline: {best_path}")
    else:
        best_model.load_state_dict(copy.deepcopy(model.state_dict()))
        save_checkpoint(best_path, best_model, optimizer, start_cycle, args)
        print(f"Initialized gating baseline from starting checkpoint: {best_path}")
    best_model.eval()

    model.load_state_dict(copy.deepcopy(best_model.state_dict()))

    replay = None if args.fresh_replay else load_replay(replay_path)
    if replay is not None:
        print(f"Loaded replay buffer: {replay_path} ({len(replay)} samples)")
        trim_replay(replay, args.replay_size)
    else:
        replay = []
        if args.fresh_replay:
            print("Starting with an empty replay buffer (--fresh-replay).")
        else:
            print(f"No saved replay buffer found at {replay_path} -- starting empty.")

    results = {"1-0": 0, "0-1": 0, "1/2-1/2": 0, "max-plies": 0, "no-moves": 0}
    trained_cycles_since_gate = 0

                                                                           
                                                                       
                                                                   
                                                                       
                                                             
                                                                    
                                                                        
                                                                       
                                                                     
                                                                       
                                                                        
                                                                      
                                                                         
                                                                      
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    with multiprocessing.Pool(processes=args.workers) as pool:
        for cycle in range(start_cycle + 1, start_cycle + args.cycles + 1):
            state_dict = cpu_state_dict(model)
            new_positions = 0

            game_outputs = run_selfplay_cycle(pool, state_dict, cycle, args)
            for samples, result in game_outputs:
                results[result] = results.get(result, 0) + 1
                if result in ("max-plies", "no-moves"):
                    continue
                replay.extend(samples)
                new_positions += len(samples)
            trim_replay(replay, args.replay_size)

                                                                            
                                                                          
                                                                      
                                                                           
                                                
            if cycle % args.save_every == 0:
                save_replay(replay_path, replay)

            if len(replay) < args.min_replay_size:
                print(f"Cycle {cycle:5d} | new {new_positions:4d} | replay {len(replay):5d} | "
                      f"warming up ({len(replay)}/{args.min_replay_size}) | results {results}")
                continue

            loss = train_on_replay(model, optimizer, replay, device, args)
            trained_cycles_since_gate += 1

            gate_msg = ""
            if trained_cycles_since_gate >= args.gate_every:
                trained_cycles_since_gate = 0
                challenger_state_dict = cpu_state_dict(model)
                incumbent_state_dict = cpu_state_dict(best_model)
                score = run_gate_match(pool, challenger_state_dict, incumbent_state_dict, args, cycle)

                if score >= args.gate_min_score:
                    best_model.load_state_dict(copy.deepcopy(model.state_dict()))
                    save_checkpoint(best_path, best_model, optimizer, cycle, args)
                    gate_msg = f" | GATE PASS ({score:.2f}) -> promoted"
                else:
                    model.load_state_dict(copy.deepcopy(best_model.state_dict()))
                    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
                    gate_msg = f" | GATE FAIL ({score:.2f}) -> reverted to best"

            print(f"Cycle {cycle:5d} | new {new_positions:4d} | replay {len(replay):5d} | "
                  f"loss {loss:.4f} | results {results}{gate_msg}")

            if cycle % args.save_every == 0:
                save_checkpoint(checkpoint_dir / f"{args.checkpoint_prefix}_{cycle}.pt", model, optimizer, cycle, args)
                save_checkpoint(latest_path, model, optimizer, cycle, args)
                print(f"Saved: {args.checkpoint_prefix}_{cycle}.pt")

    print(f"\nDone. Best (gated) checkpoint: {best_path}")


if __name__ == "__main__":
    main()
