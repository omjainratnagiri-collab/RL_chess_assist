

import argparse
import random
import sys
from pathlib import Path

import chess
import numpy as np
import torch
                        

from nnue import NNUE
from halfkp_encoding import encode_stm, NUM_FEATURES
from move_encoder import move_to_index, get_legal_mask, POLICY_SIZE
from stockfish_zst_dataset import StockfishZstDataset
from accumulator import AccumulatorPair


PASS = "PASS"
FAIL = "FAIL"


def load_model(checkpoint_path, device):
    model = NNUE().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    meta = {}
    if isinstance(checkpoint, dict):
        meta["chunk_or_epoch"] = checkpoint.get("chunk", checkpoint.get("epoch"))
        meta["loss"] = checkpoint.get("loss")
        meta["positions_seen"] = checkpoint.get("positions_seen")
    return model, meta


@torch.no_grad()
def eval_value(model, board, device):
    us, them = encode_stm(board)
    _, value = model([us], [them])
    return float(value.item())


@torch.no_grad()
def eval_policy_logits(model, board, device):
    us, them = encode_stm(board)
    policy, _ = model([us], [them])
    return policy.squeeze(0).cpu().numpy()


                                                                             
                                 
                                                                             

SANITY_POSITIONS = [
                                                                        
    ("start position (roughly balanced)",
     chess.STARTING_FEN, (-0.35, 0.35)),

    ("white up a queen, white to move",
     "4k3/8/8/8/8/8/8/3QK3 w - - 0 1", (0.5, 1.0)),

    ("white up a queen, black to move (mover is losing badly)",
     "4k3/8/8/8/8/8/8/3QK3 b - - 0 1", (-1.0, -0.5)),

    ("black up a rook, black to move",
     "4k2r/8/8/8/8/8/8/4K3 b - - 0 1", (0.35, 1.0)),

                                                                         
                                                                       
                                                                         
                                                                     
                                                       
    ("white to move, already checkmated (Fool's Mate)",
     "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 0 3", (-1.0, -0.6)),
]


def run_sanity_checks(model, device):
    print("\n=== 1. Known-position sanity checks ===")
    results = []
    for name, fen, (lo, hi) in SANITY_POSITIONS:
        board = chess.Board(fen)
        value = eval_value(model, board, device)
        ok = lo <= value <= hi
        status = PASS if ok else FAIL
        results.append(ok)
        print(f"  [{status}] {name}: value={value:+.3f} (expected [{lo:+.2f}, {hi:+.2f}])")
    passed = sum(results)
    print(f"  {passed}/{len(results)} sanity checks passed.")
    return passed, len(results)


                                                                             
                    
                                                                             

MIRROR_TEST_FENS = [
    chess.STARTING_FEN,
    "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 4 4",
    "8/5k2/8/3P4/8/3K4/8/8 w - - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 2",
]


def run_symmetry_checks(model, device, tol=0.05):
    print("\n=== 2. Mirror symmetry checks ===")
    print("  (eval(board) should ~= eval(board.mirror()) -- side-to-move-relative")
    print("   value must not depend on which color is which)")
    results = []
    for fen in MIRROR_TEST_FENS:
        board = chess.Board(fen)
        mirrored = board.mirror()
        v1 = eval_value(model, board, device)
        v2 = eval_value(model, mirrored, device)
        diff = abs(v1 - v2)
        ok = diff <= tol
        status = PASS if ok else FAIL
        results.append(ok)
        print(f"  [{status}] {fen[:40]:40s} orig={v1:+.3f} mirrored={v2:+.3f} diff={diff:.3f}")
    passed = sum(results)
    print(f"  {passed}/{len(results)} symmetry checks passed (tol={tol}).")
    return passed, len(results)


                                                                             
                                                                         
                                                                             

def run_accumulator_consistency(model, device, num_games=5, plies_per_game=30, seed=0, tol=1e-4):
    print("\n=== 3. Accumulator consistency (forward() vs forward_from_accumulators()) ===")
    rng = random.Random(seed)
    max_diff = 0.0
    checked = 0
    failures = 0

                                                                           
                                                                         
                                                    
    embedding_weights = model.feature_transformer.embedding.weight.detach().cpu().numpy()

    for g in range(num_games):
        board = chess.Board()
        acc = AccumulatorPair(embedding_weights, board=board)
        for ply in range(plies_per_game):
            legal = list(board.legal_moves)
            if not legal:
                break
            move = rng.choice(legal)

            acc.push(board, move)
            board.push(move)

            us_vec, them_vec = acc.stm_and_opponent(board.turn == chess.WHITE)
            _, value_acc = model.forward_from_accumulators(us_vec, them_vec, device=device)
            value_acc = float(value_acc.item())

            us_idx, them_idx = encode_stm(board)
            with torch.no_grad():
                _, value_fwd = model([us_idx], [them_idx])
            value_fwd = float(value_fwd.item())

            diff = abs(value_acc - value_fwd)
            max_diff = max(max_diff, diff)
            checked += 1
            if diff > tol:
                failures += 1
                print(f"  [FAIL] game {g} ply {ply}: acc={value_acc:+.6f} fwd={value_fwd:+.6f} diff={diff:.6f}")

    status = PASS if failures == 0 else FAIL
    print(f"  [{status}] checked {checked} positions across {num_games} random games, "
          f"max diff={max_diff:.6f} (tol={tol}), {failures} mismatches.")
    return (checked - failures), checked


                                                                             
                                                                            
                                                                             

@torch.no_grad()
def run_holdout_eval(model, device, zst_file, num_positions, max_depth, batch_size=256, batch_log_every=200):
    print(f"\n=== 4 & 5. Holdout evaluation on {zst_file} ({num_positions} positions) ===")
    print(f"  Opening dataset (max_depth={max_depth})...")

    dataset = StockfishZstDataset(
        zst_file=zst_file,
        max_depth=max_depth,
        max_positions=num_positions,
    )

    abs_errors = []
    sign_agree = 0
    sign_total = 0
    policy_top1 = 0
    policy_top3 = 0
    policy_total = 0
    n = 0

                                                                        
                                                                         
                                                                          
                                                                      
                                                                  
                                                                       
                                                                       
                                            
    buf_us, buf_them, buf_move_indices, buf_move_probs, buf_value_target = [], [], [], [], []

    def flush_batch():
        nonlocal n, sign_total, sign_agree, policy_total, policy_top1, policy_top3
        if not buf_us:
            return
        policy_logits_b, value_pred_b = model(buf_us, buf_them)
        value_pred_b = value_pred_b.squeeze(-1).cpu().numpy()
        policy_logits_b = policy_logits_b.cpu().numpy()

        for i in range(len(buf_us)):
            n += 1
            value_pred = float(value_pred_b[i])
            value_target = float(buf_value_target[i])

            abs_errors.append(abs(value_pred - value_target))
            if abs(value_target) > 0.03:
                sign_total += 1
                if (value_pred > 0) == (value_target > 0):
                    sign_agree += 1

            mi, mp = buf_move_indices[i], buf_move_probs[i]
            if mi and mp:
                best_idx = mi[int(np.argmax(mp))]
                logits = policy_logits_b[i]
                top3 = set(np.argsort(-logits)[:3].tolist())
                policy_total += 1
                policy_top1 += int(np.argmax(logits) == best_idx)
                policy_top3 += int(best_idx in top3)

            if n % batch_log_every == 0:
                print(f"  ...{n}/{num_positions} positions evaluated")

        buf_us.clear(); buf_them.clear(); buf_move_indices.clear()
        buf_move_probs.clear(); buf_value_target.clear()

    for us, them, move_indices, move_probs, value_target in dataset:
        buf_us.append(us)
        buf_them.append(them)
        buf_move_indices.append(move_indices)
        buf_move_probs.append(move_probs)
        buf_value_target.append(value_target)

        if len(buf_us) >= batch_size:
            flush_batch()

        if n + len(buf_us) >= num_positions:
            break

    flush_batch()

    if n == 0:
        print("  [FAIL] no positions were read from the holdout file -- check the path/format.")
        return 0, 1

    mae = float(np.mean(abs_errors))
    sign_acc = sign_agree / sign_total if sign_total else float("nan")
    top1_acc = policy_top1 / policy_total if policy_total else float("nan")
    top3_acc = policy_top3 / policy_total if policy_total else float("nan")

    print(f"  Positions evaluated: {n}")
    print(f"  Value MAE (tanh-scale, range [-1,1]): {mae:.4f}")
    print(f"  Value sign agreement (non-near-zero positions): {sign_acc:.1%} over {sign_total} positions")
    print(f"  Policy top-1 accuracy vs Stockfish best move: {top1_acc:.1%} over {policy_total} positions")
    print(f"  Policy top-3 accuracy vs Stockfish best move: {top3_acc:.1%} over {policy_total} positions")

                                                                        
    checks = [
        ("value MAE <= 0.15", mae <= 0.15),
        ("sign agreement >= 85%", not np.isnan(sign_acc) and sign_acc >= 0.85),
        ("policy top-1 >= 35%", not np.isnan(top1_acc) and top1_acc >= 0.35),
    ]
    passed = 0
    for label, ok in checks:
        status = PASS if ok else FAIL
        print(f"  [{status}] {label}")
        passed += int(ok)
    return passed, len(checks)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="Path to a .pt checkpoint saved by stockfish_zst.py")
    parser.add_argument("--zst-file", default=None, help="Held-out .jsonl.zst file for value/policy quality eval")
    parser.add_argument("--num-eval-positions", type=int, default=5000)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--skip-holdout", action="store_true", help="Skip checks 4/5 even if --zst-file is given")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    print(f"Loading checkpoint: {args.checkpoint} (device={device})")
    model, meta = load_model(args.checkpoint, device)
    if meta:
        print(f"  checkpoint metadata: {meta}")

    total_passed = 0
    total_checks = 0

    for fn, kwargs in [
        (run_sanity_checks, {}),
        (run_symmetry_checks, {}),
        (run_accumulator_consistency, {}),
    ]:
        p, t = fn(model, device, **kwargs)
        total_passed += p
        total_checks += t

    if args.zst_file and not args.skip_holdout:
        p, t = run_holdout_eval(model, device, args.zst_file, args.num_eval_positions, args.max_depth)
        total_passed += p
        total_checks += t
    else:
        print("\n=== 4 & 5. Holdout evaluation skipped (no --zst-file given, or --skip-holdout set) ===")

    print("\n" + "=" * 60)
    print(f"TOTAL: {total_passed}/{total_checks} checks passed")
    print("=" * 60)
    print(
        "\nNote: the holdout thresholds (MAE, sign agreement, top-1/top-3) are\n"
        "rough starting points, not ground truth -- tune them once you have a\n"
        "sense of your own dataset's difficulty and training budget. The\n"
        "sanity, symmetry, and accumulator checks (1-3) are the ones that\n"
        "should always pass regardless of how much training you've done --\n"
        "a failure there points to a real bug, not just 'needs more training'."
    )

    if total_passed < total_checks:
        sys.exit(1)


if __name__ == "__main__":
    main()
