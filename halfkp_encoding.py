
import chess

NUM_KING_SQUARES = 64
PIECE_TYPES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]
NUM_PIECE_TYPES = len(PIECE_TYPES)                     
NUM_PIECE_PLANES = NUM_PIECE_TYPES * 2                    
NUM_PIECE_SQUARES = 64

NUM_FEATURES = NUM_KING_SQUARES * NUM_PIECE_PLANES * NUM_PIECE_SQUARES          

_PIECE_TYPE_INDEX = {pt: i for i, pt in enumerate(PIECE_TYPES)}


def orient(square, perspective_is_white):
    return square if perspective_is_white else chess.square_mirror(square)


def feature_index(king_square_oriented, piece_type, piece_square_oriented, is_own_piece):
    plane = _PIECE_TYPE_INDEX[piece_type] * 2 + (0 if is_own_piece else 1)
    return king_square_oriented * NUM_PIECE_PLANES * NUM_PIECE_SQUARES\
        + plane * NUM_PIECE_SQUARES + piece_square_oriented


def encode_perspective(board, perspective_color):
    """Full (non-incremental) feature list for one color's accumulator.
    Used for initial setup and for the full-refresh-on-king-move case."""
    perspective_is_white = (perspective_color == chess.WHITE)
    king_square = board.king(perspective_color)
    king_oriented = orient(king_square, perspective_is_white)

    indices = []
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is None or piece.piece_type == chess.KING:
            continue
        square_oriented = orient(square, perspective_is_white)
        is_own = (piece.color == perspective_color)
        indices.append(
            feature_index(king_oriented, piece.piece_type, square_oriented, is_own)
        )
    return indices


def encode_halfkp(board):
    """Full encode for both accumulators -- (white_indices, black_indices)."""
    return encode_perspective(board, chess.WHITE), encode_perspective(board, chess.BLACK)


def encode_stm(board):
    
    white_indices, black_indices = encode_halfkp(board)
    if board.turn == chess.WHITE:
        return white_indices, black_indices
    return black_indices, white_indices


class MoveDiff:
    
    __slots__ = ("white_refresh", "white_changes", "black_refresh", "black_changes")

    def __init__(self):
        self.white_refresh = False
        self.white_changes = []                                           
        self.black_refresh = False
        self.black_changes = []


def _piece_feature_both_colors(piece_type, square, piece_color):
    
    white_sq = orient(square, True)
    black_sq = orient(square, False)
    white_is_own = (piece_color == chess.WHITE)
    black_is_own = (piece_color == chess.BLACK)
                                                                       
                                                   
    return white_sq, black_sq, white_is_own, black_is_own


def compute_move_diff(board, move):
    
    diff = MoveDiff()

    moving_piece = board.piece_at(move.from_square)
    if moving_piece is None:
        raise ValueError("No piece at move.from_square -- board/move mismatch.")

    is_king_move = (moving_piece.piece_type == chess.KING)
    mover_color = moving_piece.color

    if is_king_move:
        if mover_color == chess.WHITE:
            diff.white_refresh = True
        else:
            diff.black_refresh = True
                                                                           
                                                                         
                                                                         
                                                                           
                                                                    

    white_king_sq = orient(board.king(chess.WHITE), True)
    black_king_sq = orient(board.king(chess.BLACK), False)

    def add_change(color_changes, kind, piece_type, square, piece_color, king_sq_oriented,
                    perspective_is_white):
        sq_oriented = orient(square, perspective_is_white)
        is_own = (piece_color == (chess.WHITE if perspective_is_white else chess.BLACK))
        idx = feature_index(king_sq_oriented, piece_type, sq_oriented, is_own)
        color_changes.append((kind, idx))

    def record_piece_move(piece_type, piece_color, from_sq, to_sq):
                           
        if not diff.white_refresh:
            add_change(diff.white_changes, "remove", piece_type, from_sq, piece_color,
                       white_king_sq, True)
            add_change(diff.white_changes, "add", piece_type, to_sq, piece_color,
                       white_king_sq, True)
                           
        if not diff.black_refresh:
            add_change(diff.black_changes, "remove", piece_type, from_sq, piece_color,
                       black_king_sq, False)
            add_change(diff.black_changes, "add", piece_type, to_sq, piece_color,
                       black_king_sq, False)

    def record_piece_removed(piece_type, piece_color, square):
        if not diff.white_refresh:
            add_change(diff.white_changes, "remove", piece_type, square, piece_color,
                       white_king_sq, True)
        if not diff.black_refresh:
            add_change(diff.black_changes, "remove", piece_type, square, piece_color,
                       black_king_sq, False)

    def record_piece_added(piece_type, piece_color, square):
        if not diff.white_refresh:
            add_change(diff.white_changes, "add", piece_type, square, piece_color,
                       white_king_sq, True)
        if not diff.black_refresh:
            add_change(diff.black_changes, "add", piece_type, square, piece_color,
                       black_king_sq, False)

                                                                              
                                                                               
    captured = board.piece_at(move.to_square)
    if captured is not None:
        record_piece_removed(captured.piece_type, captured.color, move.to_square)

                                                                                  
    if board.is_en_passant(move):
        ep_square = move.to_square + (-8 if mover_color == chess.WHITE else 8)
        ep_piece = board.piece_at(ep_square)
        if ep_piece is not None:
            record_piece_removed(ep_piece.piece_type, ep_piece.color, ep_square)

                                 
    if is_king_move:
                                                                               
                                                                             
                                                                              
        pass
    elif move.promotion is not None:
                                                                                
        record_piece_removed(chess.PAWN, mover_color, move.from_square)
        record_piece_added(move.promotion, mover_color, move.to_square)
    else:
        record_piece_move(moving_piece.piece_type, mover_color, move.from_square, move.to_square)

                                                                          
                                                                       
                            
    if board.is_castling(move):
        if board.is_kingside_castling(move):
            rook_from = chess.H1 if mover_color == chess.WHITE else chess.H8
            rook_to = chess.F1 if mover_color == chess.WHITE else chess.F8
        else:
            rook_from = chess.A1 if mover_color == chess.WHITE else chess.A8
            rook_to = chess.D1 if mover_color == chess.WHITE else chess.D8
        record_piece_move(chess.ROOK, mover_color, rook_from, rook_to)

    return diff
