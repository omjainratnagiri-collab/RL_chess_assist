import time
import sys
from pathlib import Path

import chess
import numpy as np
import torch

from accumulator import AccumulatorPair
from numpy_inference import NumpyValueHead

MATE_SCORE = 2.0                                                      
DRAW_SCORE = 0.0
                                  
MAX_QUIESCENCE_PLY = 24
                                   
TIME_CHECK_NODE_INTERVAL = 1024

TT_EXACT = 0
TT_LOWERBOUND = 1
TT_UPPERBOUND = 2

_PIECE_VALUES = {
    chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
    chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0,
}


class SearchTimeout(Exception):
    """Raised internally when the time budget is exceeded mid-recursion.
    Caught inside search() -- callers never see this; they just get back
    whatever the last fully-completed iterative-deepening depth found."""
    pass


def mate_score_for_ply(ply_from_root):
    """Mate scores get slightly reduced with distance from root so the
    search prefers a FASTER mate over a slower one, and a slower loss over
    a faster one, instead of being indifferent between them."""
    return MATE_SCORE - ply_from_root * 0.001


class TTEntry:
    __slots__ = ("depth", "score", "flag", "best_move")

    def __init__(self, depth, score, flag, best_move):
        self.depth = depth
        self.score = score
        self.flag = flag
        self.best_move = best_move


class SearchEngine:
    def __init__(self, model, device, tt_max_entries=2_000_000, use_numpy_eval=True,
                 use_null_move=True, use_lmr=True):
        self.model = model
        self.device = device
        self.tt_max_entries = tt_max_entries
        self.tt = {}
        self.history = {}                                                      
        self.nodes = 0
        self._deadline = None                                                                
        self.use_numpy_eval = use_numpy_eval
        self.numpy_value_head = NumpyValueHead() if use_numpy_eval else None                                               
        self.use_null_move = use_null_move
        self.use_lmr = use_lmr
        self.sync_weights()

    def sync_weights(self):
        
        self.weights = self.model.feature_transformer.embedding.weight.detach().cpu().numpy()
        if self.use_numpy_eval:
            self.numpy_value_head.sync(self.model)

    def new_accumulator(self, board):
        return AccumulatorPair(self.weights, board=board)

    def _check_time(self):
        
        if self._deadline is None:
            return
        if self.nodes % TIME_CHECK_NODE_INTERVAL == 0 and time.time() > self._deadline:
            raise SearchTimeout()


    def evaluate(self, board, acc):
        if board.is_checkmate():
            return -MATE_SCORE                               
        if board.is_stalemate() or board.is_insufficient_material():
            return DRAW_SCORE

        us_vec, them_vec = acc.stm_and_opponent(board.turn == chess.WHITE)

        if self.use_numpy_eval:
            return self.numpy_value_head.evaluate(us_vec, them_vec)

        with torch.no_grad():
            _, value = self.model.forward_from_accumulators(us_vec, them_vec, device=self.device)
        return float(value.item())

                                                                      
    def _mvv_lva_score(self, board, move):
        victim = board.piece_at(move.to_square)
        attacker = board.piece_at(move.from_square)
        victim_value = _PIECE_VALUES[victim.piece_type] if victim else 0
        attacker_value = _PIECE_VALUES[attacker.piece_type] if attacker else 0
        return victim_value * 10 - attacker_value

    def _has_non_pawn_material(self, board, color):
        """True if `color` has any piece besides pawns/king. Used to guard
        null-move pruning against zugzwang -- see the comment at its call
        site in negamax() for why this matters."""
        for piece_type in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
            if board.pieces(piece_type, color):
                return True
        return False

    def _ordered_moves(self, board, tt_move, captures_only=False):
        moves = list(board.legal_moves)
        if captures_only:
            moves = [m for m in moves if board.is_capture(m) or m.promotion is not None]

        def sort_key(move):
            if tt_move is not None and move == tt_move:
                return (3, 0)
            if board.is_capture(move):
                return (2, self._mvv_lva_score(board, move))
            if move.promotion is not None:
                return (2, _PIECE_VALUES.get(move.promotion, 0) * 10)
            return (1, self.history.get((move.from_square, move.to_square), 0))

        moves.sort(key=sort_key, reverse=True)
        return moves
                                                               

    def quiescence(self, board, acc, alpha, beta, ply, qply=0):
        self.nodes += 1
        self._check_time()
                                       
        if board.is_repetition(2) or board.halfmove_clock >= 100:
            return DRAW_SCORE                                          
                                                     
        if qply >= MAX_QUIESCENCE_PLY:
            return self.evaluate(board, acc)

        in_check = board.is_check()
        stand_pat = self.evaluate(board, acc)

        if not in_check:
            if stand_pat >= beta:
                return beta
            if stand_pat > alpha:
                alpha = stand_pat

        moves = self._ordered_moves(board, None, captures_only=not in_check)
        if in_check and not moves:
            return -mate_score_for_ply(ply)                           

        for move in moves:
            acc.push(board, move)
            board.push(move)
            try:
                score = -self.quiescence(board, acc, -beta, -alpha, ply + 1, qply + 1)
            finally:
                                            
                board.pop()
                acc.pop()

            if score >= beta:
                return beta
            if score > alpha:
                alpha = score

        return alpha

    def negamax(self, board, acc, depth, alpha, beta, ply):
        self.nodes += 1
        self._check_time()

        if board.is_repetition(2) or board.halfmove_clock >= 100:
            return DRAW_SCORE

        key = acc.zobrist
        tt_entry = self.tt.get(key)
        tt_move = tt_entry.best_move if tt_entry else None

        if tt_entry is not None and tt_entry.depth >= depth:
            if tt_entry.flag == TT_EXACT:
                return tt_entry.score
            if tt_entry.flag == TT_LOWERBOUND and tt_entry.score > alpha:
                alpha = tt_entry.score
            elif tt_entry.flag == TT_UPPERBOUND and tt_entry.score < beta:
                beta = tt_entry.score
            if alpha >= beta:
                return tt_entry.score

        if depth <= 0:
            return self.quiescence(board, acc, alpha, beta, ply)

        in_check_here = board.is_check()
             
        if (self.use_null_move and depth >= 3 and not in_check_here and beta < MATE_SCORE - 1
                and self._has_non_pawn_material(board, board.turn)):
            R = 2                                                         
            acc.push_null(board)
            board.push(chess.Move.null())
            try:
                null_score = -self.negamax(board, acc, depth - 1 - R, -beta, -beta + 1, ply + 1)
            finally:
                board.pop()
                acc.pop_null()
            if null_score >= beta:
                return beta

        moves = self._ordered_moves(board, tt_move)
        if not moves:
            if in_check_here:
                return -mate_score_for_ply(ply)
            return DRAW_SCORE

        original_alpha = alpha
        best_score = -MATE_SCORE - 1
        best_move = None
        move_count = 0

        for move in moves:
            move_count += 1
            is_capture_or_promo = board.is_capture(move) or move.promotion is not None
            gives_check = board.gives_check(move)
            is_tt_move = tt_move is not None and move == tt_move

            acc.push(board, move)
            board.push(move)
            try:
                                             
                use_reduction = (
                    self.use_lmr and depth >= 3 and move_count > 4
                    and not is_capture_or_promo and not gives_check
                    and not in_check_here and not is_tt_move
                )
                if use_reduction:
                    reduction = 2 if depth >= 6 else 1
                                                                      
                                                                      
                    score = -self.negamax(board, acc, depth - 1 - reduction, -alpha - 1, -alpha, ply + 1)
                    if score > alpha:
                                                      
                                                                      
                        score = -self.negamax(board, acc, depth - 1, -beta, -alpha, ply + 1)
                else:
                    score = -self.negamax(board, acc, depth - 1, -beta, -alpha, ply + 1)
            finally:
                                                     
                board.pop()
                acc.pop()

            if score > best_score:
                best_score = score
                best_move = move

            if best_score > alpha:
                alpha = best_score

            if alpha >= beta:
                if not board.is_capture(move):
                    key_hist = (move.from_square, move.to_square)
                    self.history[key_hist] = self.history.get(key_hist, 0) + depth * depth
                break

        flag = TT_EXACT
        if best_score <= original_alpha:
            flag = TT_UPPERBOUND
        elif best_score >= beta:
            flag = TT_LOWERBOUND

        if len(self.tt) < self.tt_max_entries:
            self.tt[key] = TTEntry(depth, best_score, flag, best_move)

        return best_score

    def _search_root(self, board, acc, depth, ply=0):
        
        key = acc.zobrist
        tt_entry = self.tt.get(key)
        tt_move = tt_entry.best_move if tt_entry else None
        moves = self._ordered_moves(board, tt_move)
        if not moves:
            return None, DRAW_SCORE, True

        alpha, beta = -MATE_SCORE - 1, MATE_SCORE + 1
        best_score = -MATE_SCORE - 1
        best_move = None                                                                     

        for move in moves:
            acc.push(board, move)
            board.push(move)
            try:
                score = -self.negamax(board, acc, depth - 1, -beta, -alpha, ply + 1)
            except SearchTimeout:
                board.pop()
                acc.pop()
                return best_move, best_score, False
            board.pop()
            acc.pop()

            if score > best_score:
                best_score = score
                best_move = move
            if best_score > alpha:
                alpha = best_score

        self.tt[key] = TTEntry(depth, best_score, TT_EXACT, best_move)
        return best_move, best_score, True

    def search(self, board, acc, max_depth=6, time_limit=None, min_depth=1):
        
        self.nodes = 0
        start = time.time()
        best_move = None
        best_score = 0.0
        depth_reached = 0

        for depth in range(1, max_depth + 1):
            if time_limit is not None and depth > min_depth and (time.time() - start) > time_limit:
                break

            self._deadline = (start + time_limit) if time_limit is not None else None

            move, score, completed = self._search_root(board, acc, depth)

            if move is not None:
                best_move = move
                best_score = score
                if completed:
                    depth_reached = depth

            if not completed:
                                                                           
                                                                  
                break

            if time_limit is not None and (time.time() - start) > time_limit:
                break

        self._deadline = None

        if best_move is None:
            legal = list(board.legal_moves)
            best_move = legal[0] if legal else None

        elapsed = time.time() - start
        info = {
            "depth": depth_reached,
            "nodes": self.nodes,
            "time": elapsed,
            "nps": int(self.nodes / elapsed) if elapsed > 0 else 0,
        }
        return best_move, best_score, info
