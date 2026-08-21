

import argparse
import sys
from pathlib import Path
import chess
import chess.engine
import torch

from nnue import NNUE
from search import SearchEngine


PIECE_VALUES = {
    chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
    chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0,
}


def material_score(board):
    score = 0
    for piece_type, val in PIECE_VALUES.items():
        score += val * len(board.pieces(piece_type, chess.WHITE))
        score -= val * len(board.pieces(piece_type, chess.BLACK))
    return score


def load_model(checkpoint_path, device):
    model = NNUE().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model


class StockfishOpponent:
    def __init__(self, path, skill_level=20, think_time=0.1):
        self.think_time = think_time
        self.engine = chess.engine.SimpleEngine.popen_uci(path)
        try:
            self.engine.configure({"Skill Level": skill_level})
        except chess.engine.EngineError:
            pass

    def configure_elo(self, elo):
        try:
            self.engine.configure({"UCI_LimitStrength": True, "UCI_Elo": elo})
            return True
        except chess.engine.EngineError as exc:
            print(f"  Warning: Stockfish may not support UCI_LimitStrength/UCI_Elo "
                  f"({exc}). Results will be less precisely calibrated.")
            return False

    def choose_move(self, board):
        result = self.engine.play(board, chess.engine.Limit(time=self.think_time))
        return result.move

    def close(self):
        self.engine.quit()


def random_opening(board, num_plies, rng):
    for _ in range(num_plies):
        if board.is_game_over(claim_draw=True):
            break
        board.push(rng.choice(list(board.legal_moves)))
    return board


def score_for_engine(result, engine_is_white):
    if result.startswith("1/2-1/2"):
        return 0.5
    if result.startswith("1-0"):
        return 1.0 if engine_is_white else 0.0
    if result.startswith("0-1"):
        return 0.0 if engine_is_white else 1.0
    return 0.5


def play_game(search_engine, stockfish, engine_is_white, opening_plies, seed,
              max_plies, search_depth, search_time_limit, adjudicate_threshold=4):
    import random
    rng = random.Random(seed)
    board = chess.Board()
    random_opening(board, opening_plies, rng)

    acc = search_engine.new_accumulator(board)

    ply = 0
    while not board.is_game_over(claim_draw=True) and ply < max_plies:
        is_engine_turn = (board.turn == chess.WHITE) == engine_is_white

        if is_engine_turn:
            move, _score, _info = search_engine.search(
                board, acc, max_depth=search_depth, time_limit=search_time_limit
            )
        else:
            move = stockfish.choose_move(board)

        if move is None or move not in board.legal_moves:
            winner_is_white = board.turn == chess.BLACK
            result = "1-0" if winner_is_white else "0-1"
            return score_for_engine(result, engine_is_white), board

                                                                           
                                               
                                                             
                               
                                                                        

                                                                           
                                                                           
                                      
        acc.push(board, move)
        board.push(move)
                      
                 
        ply += 1

    if ply >= max_plies and not board.is_game_over(claim_draw=True):
        score = material_score(board)
        if score >= adjudicate_threshold:
            result = "1-0 (adjudicated, material)"
        elif score <= -adjudicate_threshold:
            result = "0-1 (adjudicated, material)"
        else:
            result = "1/2-1/2 (timeout, roughly even material)"
        return score_for_engine(result, engine_is_white), board

    return score_for_engine(board.result(claim_draw=True), engine_is_white), board


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--stockfish-path", required=True)
    parser.add_argument("--stockfish-skill", type=int, default=1)
    parser.add_argument("--stockfish-time", type=float, default=0.1)
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--search-depth", type=int, default=6)
    parser.add_argument("--search-time-limit", type=float, default=2.0)
    parser.add_argument("--opening-random-plies", type=int, default=4)
    parser.add_argument("--max-plies", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = load_model(args.checkpoint, device)
    search_engine = SearchEngine(model, device)
    stockfish = StockfishOpponent(args.stockfish_path, args.stockfish_skill, args.stockfish_time)

    total_score = 0.0
    wins = draws = losses = 0

    for game_idx in range(args.games):
        engine_is_white = game_idx % 2 == 0
        seed = args.seed + game_idx

        score, board = play_game(
            search_engine, stockfish, engine_is_white, args.opening_random_plies,
            seed, args.max_plies, args.search_depth, args.search_time_limit,
        )
        total_score += score
        if score == 1.0:
            wins += 1
        elif score == 0.0:
            losses += 1
        else:
            draws += 1

        print(f"Game {game_idx + 1}/{args.games} | engine as "
              f"{'White' if engine_is_white else 'Black'} | score={score} | "
              f"final FEN: {board.fen()}")

    stockfish.close()

    print()
    print(f"Record: {wins}W {draws}D {losses}L | Score: {total_score:.1f}/{args.games}")


if __name__ == "__main__":
    main()
