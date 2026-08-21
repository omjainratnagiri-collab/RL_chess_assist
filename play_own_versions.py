import argparse
import math
import random
import chess
import torch

from nnue import NNUE
from search import SearchEngine
from td_selfplay import load_checkpoint, random_opening, material_score


def score_for_a(result, a_is_white):
    if result.startswith("1/2-1/2"):
        return 0.5
    if result.startswith("1-0"):
        return 1.0 if a_is_white else 0.0
    if result.startswith("0-1"):
        return 0.0 if a_is_white else 1.0
    return 0.5


def play_match_game(engine_a, engine_b, a_is_white, seed, args):
    rng = random.Random(seed)
    board = chess.Board()
    random_opening(board, args.opening_random_plies, rng)

    acc_a = engine_a.new_accumulator(board)
    acc_b = engine_b.new_accumulator(board)

    ply = 0
    while not board.is_game_over(claim_draw=True) and ply < args.max_plies:
        a_to_move = (board.turn == chess.WHITE) == a_is_white
        mover_engine = engine_a if a_to_move else engine_b
        mover_acc = acc_a if a_to_move else acc_b

        move, _score, _info = mover_engine.search(
            board,
            mover_acc,
            max_depth=args.search_depth,
            time_limit=args.search_time_limit,
        )

        if move is None:
            winner_is_white = board.turn == chess.BLACK
            return ("1-0" if winner_is_white else "0-1"), "no-legal-moves", ply

        acc_a.push(board, move)
        acc_b.push(board, move)
        board.push(move)
        ply += 1

    if ply >= args.max_plies and not board.is_game_over(claim_draw=True):
        score = material_score(board)
        if score >= args.adjudicate_threshold:
            return "1-0", "adjudicated", ply
        if score <= -args.adjudicate_threshold:
            return "0-1", "adjudicated", ply
        return "1/2-1/2", "adjudicated", ply

    return board.result(claim_draw=True), "normal", ply


def elo_diff_from_score(score):

    if score <= 0.0 or score >= 1.0:
        return None
    return -400.0 * math.log10(1.0 / score - 1.0)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint-a", required=True)
    parser.add_argument("--checkpoint-b", required=True)
    parser.add_argument("--label-a", default="A")
    parser.add_argument("--label-b", default="B")
    parser.add_argument(
        "--games",
        type=int,
        default=30,
        help="Total games. Alternates colors every game (A plays White on "
        "even game indices) so color isn't confounded with engine strength.",
    )
    parser.add_argument("--search-depth", type=int, default=6)
    parser.add_argument("--search-time-limit", type=float, default=2.0)
    parser.add_argument("--opening-random-plies", type=int, default=4)
    parser.add_argument("--max-plies", type=int, default=200)
    parser.add_argument("--adjudicate-threshold", type=float, default=4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--no-null-move",
        action="store_true",
        help="Disable null-move pruning for BOTH engines -- for bisecting a "
        "suspected search bug that only shows up at deeper depths. "
        "See verify_zobrist.py's null-move section for the related check.",
    )
    parser.add_argument(
        "--no-lmr",
        action="store_true",
        help="Disable late move reductions for BOTH engines -- same purpose "
        "as --no-null-move, to isolate which search feature (if either) "
        "is responsible for a depth-dependent result.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model_a = NNUE().to(device)
    load_checkpoint(args.checkpoint_a, model_a, None, device, load_optimizer=False)
    model_a.eval()
    engine_a = SearchEngine(
        model_a, device, use_null_move=not args.no_null_move, use_lmr=not args.no_lmr
    )

    model_b = NNUE().to(device)
    load_checkpoint(args.checkpoint_b, model_b, None, device, load_optimizer=False)
    model_b.eval()
    engine_b = SearchEngine(
        model_b, device, use_null_move=not args.no_null_move, use_lmr=not args.no_lmr
    )

    print(f"{args.label_a}: {args.checkpoint_a}")
    print(f"{args.label_b}: {args.checkpoint_b}")
    print(
        f"Playing {args.games} games, depth={args.search_depth}, "
        f"time_limit={args.search_time_limit}s, colors alternating"
    )
    print(
        f"null_move_pruning={'OFF' if args.no_null_move else 'ON'}, "
        f"lmr={'OFF' if args.no_lmr else 'ON'}\n"
    )

    wins_a = wins_b = draws = 0
    adjudicated_count = 0
    scores = []

    for game_idx in range(args.games):
        a_is_white = game_idx % 2 == 0
        seed = args.seed * 1_000_000 + game_idx

        result, reason, plies = play_match_game(
            engine_a, engine_b, a_is_white, seed, args
        )
        s = score_for_a(result, a_is_white)
        scores.append(s)

        if s == 1.0:
            wins_a += 1
        elif s == 0.0:
            wins_b += 1
        else:
            draws += 1
        if reason == "adjudicated":
            adjudicated_count += 1

        a_color = "White" if a_is_white else "Black"
        running_score = sum(scores) / len(scores)
        print(
            f"  Game {game_idx + 1:3d}/{args.games} | {args.label_a} as {a_color:5s} | "
            f"result={result:8s} ({reason}, {plies} plies) | "
            f"running score for {args.label_a} = {running_score:.3f}"
        )

    n = len(scores)
    mean_score = sum(scores) / n
    variance = sum((s - mean_score) ** 2 for s in scores) / max(n - 1, 1)
    se = (variance / n) ** 0.5

    print("\n" + "=" * 60)
    print(
        f"RESULT: {args.label_a} {wins_a}W {draws}D {wins_b}L vs {args.label_b}  "
        f"(over {n} games, {adjudicated_count} adjudicated)"
    )
    print(
        f"Score for {args.label_a}: {mean_score:.3f} (+/- {1.96 * se:.3f} at ~95% CI)"
    )

    elo = elo_diff_from_score(mean_score)
    if elo is not None:
        elo_lo = elo_diff_from_score(max(mean_score - 1.96 * se, 0.001))
        elo_hi = elo_diff_from_score(min(mean_score + 1.96 * se, 0.999))
        print(
            f"Estimated Elo difference: {elo:+.0f} (95% CI roughly [{elo_lo:+.0f}, {elo_hi:+.0f}])"
        )
        print(f"(positive = {args.label_a} stronger than {args.label_b})")
    else:
        print(
            f"{args.label_a} won every single game -- not enough variation to estimate an Elo "
            f"gap; the difference is real but this sample can't size it. Consider a larger "
            f"--games run or weaker/stronger search settings to get some decisive-but-mixed results."
        )

    print("=" * 60)
    print(
        f"\nNote: {n} games gives a rough estimate, not a precise one -- the wide CI above is "
        f"expected. 30 games is a reasonable first check; run more (or repeat with a different "
        f"--seed) if this result matters for a real decision."
    )


if __name__ == "__main__":
    main()
