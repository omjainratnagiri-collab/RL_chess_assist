
import chess

_SEED = 0xC0FFEE_D00D5EED                                                                 

def _rng():
                             
    state = _SEED
    while True:
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        yield state


_gen = _rng()
                                  
                                                                               
PIECE_KEYS = [[[next(_gen) for _ in range(64)] for _ in range(7)] for _ in range(2)]

                                                                          
CASTLING_KEYS = [next(_gen) for _ in range(16)]

                                                            
EP_FILE_KEYS = [next(_gen) for _ in range(8)]

SIDE_KEY = next(_gen)

WK, WQ, BK, BQ = 1, 2, 4, 8

_ROOK_HOME_RIGHT = {
    (chess.WHITE, chess.H1): WK,
    (chess.WHITE, chess.A1): WQ,
    (chess.BLACK, chess.H8): BK,
    (chess.BLACK, chess.A8): BQ,
}


def castling_rights_from_board(board):
    rights = 0
    if board.has_kingside_castling_rights(chess.WHITE):
        rights |= WK
    if board.has_queenside_castling_rights(chess.WHITE):
        rights |= WQ
    if board.has_kingside_castling_rights(chess.BLACK):
        rights |= BK
    if board.has_queenside_castling_rights(chess.BLACK):
        rights |= BQ
    return rights


def ep_file_from_board(board):
    return chess.square_file(board.ep_square) if board.ep_square is not None else None


def compute_full_zobrist(board):
    
    h = 0
    for square, piece in board.piece_map().items():
        h ^= PIECE_KEYS[int(piece.color)][piece.piece_type][square]

    h ^= CASTLING_KEYS[castling_rights_from_board(board)]

    ep_file = ep_file_from_board(board)
    if ep_file is not None:
        h ^= EP_FILE_KEYS[ep_file]

    if board.turn == chess.BLACK:
        h ^= SIDE_KEY

    return h


def move_zobrist_update(board, move, castling_rights_before):
    
    delta = 0
    mover_color = board.turn
    moving_piece = board.piece_at(move.from_square)
    moving_type = moving_piece.piece_type

    is_ep = board.is_en_passant(move)
    if is_ep:
        captured_sq = move.to_square + (-8 if mover_color == chess.WHITE else 8)
        captured = board.piece_at(captured_sq)
    else:
        captured_sq = move.to_square
        captured = board.piece_at(captured_sq)

    if captured is not None:
        delta ^= PIECE_KEYS[int(captured.color)][captured.piece_type][captured_sq]

    delta ^= PIECE_KEYS[int(mover_color)][moving_type][move.from_square]
    final_type = move.promotion if move.promotion else moving_type
    delta ^= PIECE_KEYS[int(mover_color)][final_type][move.to_square]

    if board.is_castling(move):
        rank = chess.square_rank(move.from_square)
        if chess.square_file(move.to_square) > chess.square_file(move.from_square):
            rook_from, rook_to = chess.square(7, rank), chess.square(5, rank)
        else:
            rook_from, rook_to = chess.square(0, rank), chess.square(3, rank)
        delta ^= PIECE_KEYS[int(mover_color)][chess.ROOK][rook_from]
        delta ^= PIECE_KEYS[int(mover_color)][chess.ROOK][rook_to]

                                                                        
                                                                         
                                                                           
    new_rights = castling_rights_before
    if moving_type == chess.KING:
        new_rights &= ~(WK | WQ) if mover_color == chess.WHITE else ~(BK | BQ)
    if (mover_color, move.from_square) in _ROOK_HOME_RIGHT:
        new_rights &= ~_ROOK_HOME_RIGHT[(mover_color, move.from_square)]
    opponent_color = not mover_color
    if (opponent_color, move.to_square) in _ROOK_HOME_RIGHT:
        new_rights &= ~_ROOK_HOME_RIGHT[(opponent_color, move.to_square)]

    if new_rights != castling_rights_before:
        delta ^= CASTLING_KEYS[castling_rights_before] ^ CASTLING_KEYS[new_rights]

                                                                     
    old_ep_file = ep_file_from_board(board)
    new_ep_file = None
    if moving_type == chess.PAWN and abs(chess.square_rank(move.to_square) - chess.square_rank(move.from_square)) == 2:
        new_ep_file = chess.square_file(move.from_square)

    if old_ep_file is not None:
        delta ^= EP_FILE_KEYS[old_ep_file]
    if new_ep_file is not None:
        delta ^= EP_FILE_KEYS[new_ep_file]

    delta ^= SIDE_KEY                             

    return delta, new_rights