import argparse
import sys
from pathlib import Path

import torch

from evaluate_match import load_model, StockfishOpponent, play_game
from estimate_elo import expected_score
from search import SearchEngine


def fit_elo(anchor_results):
   
    def total_squared_error(player_elo):
        error = 0.0
        for opponent_elo, wins, draws, losses in anchor_results:
            total = wins + draws + losses
            if total == 0:
                continue
            observed = (wins + 0.5 * draws) / total
            predicted = expected_score(player_elo, opponent_elo)
            error += total * (observed - predicted) ** 2
        return error

    best_elo = None
    best_error = float("inf")
    for elo in range(600, 3200, 25):
        err = total_squared_error(float(elo))
        if err < best_error:
            best_error = err
            best_elo = float(elo)

    step = 25.0
    for _ in range(8):
        step /= 2.0
        for candidate in (best_elo - step, best_elo + step):
            err = total_squared_error(candidate)
            if err < best_error:
                best_error = err
                best_elo = candidate

    return best_elo


def play_anchor_match(search_engine, stockfish_path, opponent_elo, games,
                       search_depth, search_time_limit, opening_plies,
                       max_plies, seed_base):
    stockfish = StockfishOpponent(stockfish_path, skill_level=20, think_time=0.1)
    calibration_ok = stockfish.configure_elo(opponent_elo)

    wins = draws = losses = 0
    for game_idx in range(games):
        engine_is_white = game_idx % 2 == 0
        seed = seed_base + game_idx

        score, _board = play_game(
            search_engine, stockfish, engine_is_white, opening_plies,
            seed, max_plies, search_depth, search_time_limit,
        )
        if score == 1.0:
            wins += 1
        elif score == 0.0:
            losses += 1
        else:
            draws += 1

        print(f"  Game {game_idx + 1}/{games} | engine as "
              f"{'White' if engine_is_white else 'Black'} | score={score}")

    stockfish.close()
    return wins, draws, losses, calibration_ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--stockfish-path", required=True)
    parser.add_argument("--anchors", type=int, nargs="+",
                         default=[1350, 1500, 1700, 1900, 2100],
                         help="Stockfish UCI_Elo targets. Stockfish's "
                              "supported range is roughly 1320-3190.")
    parser.add_argument("--games-per-anchor", type=int, default=6)
    parser.add_argument("--search-depth", type=int, default=8)
    parser.add_argument("--search-time-limit", type=float, default=2.0,
                         help="Per-move time budget (seconds) for the "
                              "engine's iterative deepening.")
    parser.add_argument("--opening-random-plies", type=int, default=4)
    parser.add_argument("--max-plies", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = load_model(args.checkpoint, device)
    search_engine = SearchEngine(model, device)

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Anchors: {args.anchors} | Games/anchor: {args.games_per_anchor}")
    print(f"Search: depth<= {args.search_depth}, time_limit={args.search_time_limit}s/move")
    print()

    anchor_results = []
    for i, opponent_elo in enumerate(args.anchors):
        print(f"{'=' * 60}")
        print(f"Anchor {i + 1}/{len(args.anchors)}: Stockfish @ Elo {opponent_elo}")
        print(f"{'=' * 60}")

        wins, draws, losses, calibration_ok = play_anchor_match(
            search_engine, args.stockfish_path, opponent_elo,
            args.games_per_anchor, args.search_depth, args.search_time_limit,
            args.opening_random_plies, args.max_plies,
            seed_base=args.seed + i * 1000,
        )

        total = wins + draws + losses
        score = (wins + 0.5 * draws) / total if total else 0.0
        print(f"Result: {wins}W {draws}D {losses}L | score {score:.3f}"
              f"{'' if calibration_ok else '  [uncalibrated Stockfish -- treat as approximate]'}")
        print()

        anchor_results.append((float(opponent_elo), wins, draws, losses))

    print(f"{'=' * 60}")
    print("Summary")
    print(f"{'=' * 60}")
    for opponent_elo, wins, draws, losses in anchor_results:
        total = wins + draws + losses
        score = (wins + 0.5 * draws) / total if total else 0.0
        print(f"  vs Elo {opponent_elo:6.0f} | {wins}W {draws}D {losses}L | score {score:.3f}")

    estimated_elo = fit_elo(anchor_results)
    print()
    print(f"Estimated Elo (best fit across all anchors): {estimated_elo:.0f}")
    print("Note: only as reliable as games-per-anchor allows -- widen "
          "--anchors or raise --games-per-anchor for a tighter estimate.")


if __name__ == "__main__":
    main()
