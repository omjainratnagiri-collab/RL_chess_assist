

import chess

from search import SearchEngine


DEFAULT_DEPTH = 6
DEFAULT_MOVETIME_MS = 2000


class UCIAdapter:
    def __init__(self, model, device):
        self.engine = SearchEngine(model, device)
        self.board = chess.Board()
        self.acc = self.engine.new_accumulator(self.board)
        self.last_info = []

    def sync_weights(self):
        """Call after loading new weights into the model -- refreshes the
        engine's cached numpy copy of them (SearchEngine.sync_weights)."""
        self.engine.sync_weights()

    def command(self, text):
        text = text.strip()
        if not text:
            return []
        parts = text.split()
        cmd = parts[0]

        if cmd == "uci":
            return [
                "id name RLChess-NNUE",
                "id author RL_chess project",
                "uciok",
            ]
        if cmd == "isready":
            return ["readyok"]
        if cmd == "ucinewgame":
            self._new_game()
            return []
        if cmd == "position":
            self._position(parts[1:])
            return []
        if cmd == "go":
            return self._go(parts[1:])
        if cmd == "stop":
            return []                                                          
        if cmd == "quit":
            return []

        raise ValueError(f"Unknown UCI command: {cmd!r}")

    def _new_game(self):
        self.board = chess.Board()
        self.acc = self.engine.new_accumulator(self.board)
        self.engine.tt.clear()
        self.engine.history.clear()
        self.last_info = []

    def _position(self, args):
        if not args:
            raise ValueError("position requires 'startpos' or 'fen ...'")

        if args[0] == "startpos":
            board = chess.Board()
            rest = args[1:]
        elif args[0] == "fen":
            if "moves" in args:
                moves_idx = args.index("moves")
                fen = " ".join(args[1:moves_idx])
                rest = args[moves_idx:]
            else:
                fen = " ".join(args[1:])
                rest = []
            board = chess.Board(fen)
        else:
            raise ValueError("position requires 'startpos' or 'fen ...'")

        moves = rest[1:] if rest and rest[0] == "moves" else []

        self.board = board
        self.acc = self.engine.new_accumulator(self.board)
        for move_uci in moves:
            move = chess.Move.from_uci(move_uci)
            if move not in self.board.legal_moves:
                raise ValueError(f"Illegal move in position command: {move_uci}")
            self.acc.push(self.board, move)
            self.board.push(move)

    def _go(self, args):
        depth = DEFAULT_DEPTH
        movetime_ms = None
        i = 0
        while i < len(args):
            token = args[i]
            if token == "depth" and i + 1 < len(args):
                depth = int(args[i + 1])
                i += 2
            elif token == "movetime" and i + 1 < len(args):
                movetime_ms = int(args[i + 1])
                i += 2
            elif token in ("wtime", "btime", "winc", "binc") and i + 1 < len(args):
                i += 2                                            
            else:
                i += 1

        time_limit = (movetime_ms / 1000.0) if movetime_ms is not None else None

        move, score, info = self.engine.search(
            self.board, self.acc, max_depth=depth, time_limit=time_limit
        )

        self.last_info = [{
            "move": move.uci() if move else None,
            "score": round(score, 4),
            "depth": info["depth"],
            "nodes": info["nodes"],
            "time": round(info["time"], 3),
            "nps": info["nps"],
        }]

        lines = [
            f"info depth {info['depth']} nodes {info['nodes']} "
            f"nps {info['nps']} time {int(info['time'] * 1000)} "
            f"score internal {score:.4f}",
            f"bestmove {move.uci() if move else '0000'}",
        ]
        return lines

    def push_move(self, move):
        """Advance the adapter's own board/accumulator by one move (e.g.
        after the human plays a move from the web UI) without needing a
        full `position ... moves ...` re-send from scratch each time."""
        if move not in self.board.legal_moves:
            raise ValueError(f"Illegal move: {move.uci()}")
        self.acc.push(self.board, move)
        self.board.push(move)


def moves_to_position_command(moves):
    """Builds a 'position startpos moves ...' string from a list of UCI
    move strings -- kept as a standalone helper (same name/shape as the
    old adapter had) since server.py's /api/uci passthrough route builds
    commands this way for the raw UCI text-command testing endpoint."""
    if not moves:
        return "position startpos"
    return "position startpos moves " + " ".join(moves)
