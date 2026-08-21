import argparse
import math


def score_rate(wins, draws, losses):
    total = wins + draws + losses
    if total <= 0:
        raise ValueError("Total games must be positive.")

    return (wins + 0.5 * draws) / total, total


def elo_from_score(score):
    eps = 1e-9
    score = min(max(score, eps), 1.0 - eps)
    return -400.0 * math.log10((1.0 / score) - 1.0)


def score_stddev(wins, draws, losses):
    total = wins + draws + losses
    mean = (wins + 0.5 * draws) / total
    second_moment = (wins + 0.25 * draws) / total
    variance = max(second_moment - mean * mean, 0.0)
    return math.sqrt(variance)


def elo_confidence_interval(wins, draws, losses, confidence=0.95):
    score, total = score_rate(wins, draws, losses)
    z_map = {
        0.90: 1.6448536269514722,
        0.95: 1.959963984540054,
        0.99: 2.5758293035489004,
    }
    z = z_map.get(confidence)
    if z is None:
        raise ValueError(
            "Supported confidence values: 0.90, 0.95, 0.99"
        )

    std = score_stddev(wins, draws, losses)
    stderr = std / math.sqrt(total)

    low_score = min(max(score - z * stderr, 1e-9), 1.0 - 1e-9)
    high_score = min(max(score + z * stderr, 1e-9), 1.0 - 1e-9)

    return (
        elo_from_score(low_score),
        elo_from_score(high_score),
    )


def expected_score(player_elo, opponent_elo):
    return 1.0 / (
        1.0
        + math.pow(
            10.0,
            (opponent_elo - player_elo) / 400.0
        )
    )


def update_player_elo(
    player_elo,
    opponent_elo,
    result_score,
    k_factor=32
):
    expected = expected_score(
        player_elo,
        opponent_elo
    )

    return round(
        player_elo
        + k_factor * (result_score - expected)
    )


def main():
    parser = argparse.ArgumentParser(
        description="Estimate Elo difference from wins, draws, and losses."
    )
    parser.add_argument("--wins", type=int, required=True)
    parser.add_argument("--draws", type=int, required=True)
    parser.add_argument("--losses", type=int, required=True)
    parser.add_argument(
        "--opponent-elo",
        type=float,
        default=None,
        help="If provided, also prints estimated engine Elo."
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        choices=[0.90, 0.95, 0.99]
    )
    args = parser.parse_args()

    score, total = score_rate(
        args.wins,
        args.draws,
        args.losses
    )
    elo_diff = elo_from_score(score)
    ci_low, ci_high = elo_confidence_interval(
        args.wins,
        args.draws,
        args.losses,
        confidence=args.confidence
    )

    print(f"Games           : {total}")
    print(f"Score rate      : {score:.4f}")
    print(f"Elo difference  : {elo_diff:.1f}")
    print(
        f"{int(args.confidence * 100)}% CI        : "
        f"[{ci_low:.1f}, {ci_high:.1f}]"
    )

    if args.opponent_elo is not None:
        engine_elo = args.opponent_elo + elo_diff
        print(f"Opponent Elo    : {args.opponent_elo:.1f}")
        print(f"Estimated Elo   : {engine_elo:.1f}")


if __name__ == "__main__":
    main()