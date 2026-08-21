
import argparse
import cProfile
import pstats
import chess
import torch

from nnue import NNUE
from search import SearchEngine


def load_model(checkpoint_path, device):
    model = NNUE().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--fen", default=chess.STARTING_FEN)
    parser.add_argument("--depth", type=int, default=3,
                         help="Fixed depth, no time limit -- lets this depth run to "
                              "completion so the profile reflects a full, comparable workload.")
    parser.add_argument("--top-n", type=int, default=20)
    args = parser.parse_args()

    device = torch.device("cpu")                                                                    
    print(f"Loading checkpoint: {args.checkpoint}")
    model = load_model(args.checkpoint, device)
    engine = SearchEngine(model, device)

    board = chess.Board(args.fen)
    acc = engine.new_accumulator(board)

    print(f"Profiling search at fixed depth={args.depth}, position: {args.fen}")
    print("(no time_limit -- letting this depth run to completion for a clean measurement)\n")

    profiler = cProfile.Profile()
    profiler.enable()
    move, score, info = engine.search(board, acc, max_depth=args.depth, time_limit=None)
    profiler.disable()

    print(f"Result: move={move} score={score:.3f} nodes={info['nodes']} "
          f"time={info['time']:.3f}s nps={info['nps']}\n")

    stats = pstats.Stats(profiler)

    print("=" * 70)
    print(f"Top {args.top_n} by CUMULATIVE time (includes time spent in functions it calls --")
    print("shows you the overall shape: e.g. negamax's cumulative time includes")
    print("everything quiescence/evaluate/python-chess do underneath it)")
    print("=" * 70)
    stats.sort_stats("cumulative").print_stats(args.top_n)

    print("=" * 70)
    print(f"Top {args.top_n} by SELF time (tottime -- excludes callees, so this points")
    print("at exactly which function's OWN code is expensive -- this is the more")
    print("direct answer to 'what should I actually optimize')")
    print("=" * 70)
    stats.sort_stats("tottime").print_stats(args.top_n)

    print(
        "\nWhat to look for:\n"
        "  - Lots of self-time in chess/__init__.py functions (legal_moves,\n"
        "    generate_legal_moves, is_check, push/pop, SquareSet ops) -> the\n"
        "    board representation itself is the bottleneck, not the eval or\n"
        "    Python search logic. This is what points toward a C++/Rust core\n"
        "    or a faster board library.\n"
        "  - Lots of self-time in numpy_inference.py / numpy internals ->\n"
        "    the eval, even in numpy, still has room to optimize (e.g. Numba,\n"
        "    or batching leaf evals instead of one at a time).\n"
        "  - Lots of self-time in zobrist_hash -> the TT hashing itself is a\n"
        "    meaningful cost, sometimes worth a custom incremental Zobrist\n"
        "    hash maintained alongside the accumulator instead of recomputing\n"
        "    it from scratch every node.\n"
        "  - Roughly even spread with nothing dominating -> there's no single\n"
        "    fix; general Python-loop overhead is the ceiling, which is the\n"
        "    strongest case for a lower-level rewrite."
    )


if __name__ == "__main__":
    main()
