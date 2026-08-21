
import numpy as np

from halfkp_encoding import NUM_FEATURES, encode_perspective, compute_move_diff
import zobrist
import chess

class AccumulatorPair:
    def __init__(self, embedding_weights, board=None):
        
        self.weights = embedding_weights
        self.embedding_dim = embedding_weights.shape[1]
        self.white = np.zeros(self.embedding_dim, dtype=np.float32)
        self.black = np.zeros(self.embedding_dim, dtype=np.float32)
        self._undo_stack = []                                                                                         

        self.zobrist = 0
        self.castling_rights = 0

        if board is not None:
            self.refresh(board)

    def refresh(self, board):
        white_indices, black_indices = _encode_both(board)
        self.white = self.weights[white_indices].sum(axis=0) if white_indices\
            else np.zeros(self.embedding_dim, dtype=np.float32)
        self.black = self.weights[black_indices].sum(axis=0) if black_indices\
            else np.zeros(self.embedding_dim, dtype=np.float32)
                                                                         
                                                                      
        self.zobrist = zobrist.compute_full_zobrist(board)
        self.castling_rights = zobrist.castling_rights_from_board(board)

    def push(self, board, move):
        
        diff = compute_move_diff(board, move)
        prev_castling_rights = self.castling_rights
        zobrist_delta, new_castling_rights = zobrist.move_zobrist_update(
            board, move, self.castling_rights
        )

        prev_white = self.white.copy() if diff.white_refresh else None
        prev_black = self.black.copy() if diff.black_refresh else None

        if diff.white_refresh:
            board_after = board.copy(stack=False)
            board_after.push(move)
            white_indices = encode_perspective(board_after, chess.WHITE)
            self.white = self.weights[white_indices].sum(axis=0) if white_indices\
                else np.zeros(self.embedding_dim, dtype=np.float32)
        else:
            for kind, idx in diff.white_changes:
                if kind == "add":
                    self.white = self.white + self.weights[idx]
                else:
                    self.white = self.white - self.weights[idx]

        if diff.black_refresh:
            board_after = board.copy(stack=False)
            board_after.push(move)
            black_indices = encode_perspective(board_after, chess.BLACK)
            self.black = self.weights[black_indices].sum(axis=0) if black_indices\
                else np.zeros(self.embedding_dim, dtype=np.float32)
        else:
            for kind, idx in diff.black_changes:
                if kind == "add":
                    self.black = self.black + self.weights[idx]
                else:
                    self.black = self.black - self.weights[idx]

        self.zobrist ^= zobrist_delta
        self.castling_rights = new_castling_rights

        self._undo_stack.append((diff, prev_white, prev_black, zobrist_delta, prev_castling_rights))

    def push_null(self, board):
        
        ep_file = zobrist.ep_file_from_board(board)
        delta = zobrist.SIDE_KEY
        if ep_file is not None:
            delta ^= zobrist.EP_FILE_KEYS[ep_file]
        self.zobrist ^= delta
        self._undo_stack.append(("null", delta))

    def pop_null(self):
        
        marker, delta = self._undo_stack.pop()
        assert marker == "null", "pop_null() called but the top of the undo stack isn't a null move"
        self.zobrist ^= delta

    def pop(self):
        
        entry = self._undo_stack.pop()
        assert entry[0] != "null", "pop() called but the top of the undo stack is a null move -- use pop_null() instead"
        diff, prev_white, prev_black, zobrist_delta, prev_castling_rights = entry

        if diff.white_refresh:
            self.white = prev_white
        else:
            for kind, idx in diff.white_changes:
                if kind == "add":
                    self.white = self.white - self.weights[idx]
                else:
                    self.white = self.white + self.weights[idx]

        if diff.black_refresh:
            self.black = prev_black
        else:
            for kind, idx in diff.black_changes:
                if kind == "add":
                    self.black = self.black - self.weights[idx]
                else:
                    self.black = self.black + self.weights[idx]

                                                                        
                                                 
        self.zobrist ^= zobrist_delta
                                                                        
                                                                       
                                                         
        self.castling_rights = prev_castling_rights

    def stm_and_opponent(self, side_to_move_is_white):
        
        if side_to_move_is_white:
            return self.white, self.black
        return self.black, self.white


def _encode_both(board):
    from halfkp_encoding import encode_halfkp
    return encode_halfkp(board)