"""
Web server for playing against our alpha-beta + NNUE engine.
"""

import os
import sys
import time
import uuid
import json
import base64
import hashlib
import hmac
import urllib.request
import urllib.error
import threading
from pathlib import Path
from contextlib import asynccontextmanager

from dotenv import load_dotenv
import chess
import chess.pgn
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

HERE = Path(__file__).resolve().parent
load_dotenv(HERE / ".env")
for candidate in filter(None, [os.getenv("ENGINE_SRC_DIR"), str(HERE), str(HERE.parent)]):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from nnue import NNUE
from uci_adapter import UCIAdapter
from database import database

STATIC_DIR =  HERE/ "static"

# Define the lifespan context manager to replace the deprecated on_event pattern
@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- STARTUP LOGIC ---
    load_engine()
    yield
    # --- SHUTDOWN LOGIC ---
    # Optional cleanup hooks can be added here if needed in the future

app = FastAPI(title="RL Chess Engine", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
async def serve_frontend():
    """Serve the chess UI at the normal localhost root URL."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
async def health():
    try:
        backend = database.healthcheck()
    except Exception as exc:
        raise HTTPException(503, "Database connection failed") from exc
    # return {"status": "ok", "database": backend}
    return {
        "status": "ok",
        "database": backend,
        "google_client_configured": bool(os.getenv("GOOGLE_CLIENT_ID")),
        "auth_secret_configured": bool(os.getenv("AUTH_SECRET")),
        "google_client_id_debug": repr(os.getenv("GOOGLE_CLIENT_ID")),  # TEMP - remove after debugging
    }
games = {}

DEFAULT_CHECKPOINT_CANDIDATES = [
    "checkpoints/td_selfplay_best.pt",
    "checkpoints/best_model_halfkp_stockfish_zst.pt",
]

# Placeholder until you've run a real anchor match
ENGINE_ELO = 2000
DEFAULT_PLAYER_ELO = 800
ELO_K_FACTOR = 24

ENGINE_NAME = "RL Chess Python Engine"
ENGINE_ESTIMATED_ELO = 2000
ENGINE_DETAILS = (
    "Python alpha-beta/negamax | NNUE HalfKP | incremental accumulators | "
    "NumPy value inference | Zobrist transposition table | quiescence | "
    "iterative deepening"
)

DIFFICULTY_PRESETS = {
    "easy":   {"search_depth": 3, "move_time": 1.0},
    "medium": {"search_depth": 5, "move_time": 2.0},
    "hard":   {"search_depth": 7, "move_time": 3.5},
}

engine_status = "not loaded"
active_checkpoint_path = None
adapter = None  # set in load_engine()
engine_lock = threading.Lock()


class NewGameRequest(BaseModel):
    player_name: str = "Guest"
    player_color: str = "white"
    difficulty: str = "medium"
    search_depth: int | None = None
    move_time: float | None = None
    player_token: str | None = None


class MoveRequest(BaseModel):
    game_id: str
    move: str


class AnalyzeRequest(BaseModel):
    game_id: str


class ChatRequest(BaseModel):
    game_id: str
    message: str


class GoogleLoginRequest(BaseModel):
    credential: str


class ResignRequest(BaseModel):
    game_id: str


class HistoryRequest(BaseModel):
    game_id: str


class UCICommandRequest(BaseModel):
    game_id: str | None = None
    command: str
    search_depth: int = DIFFICULTY_PRESETS["medium"]["search_depth"]


def load_engine():
    global engine_status, active_checkpoint_path, adapter

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NNUE().to(device)

    checkpoint_override = os.getenv("RL_CHESS_CHECKPOINT")
    candidates = [checkpoint_override] if checkpoint_override else DEFAULT_CHECKPOINT_CANDIDATES
    checkpoint_path = None
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            checkpoint_path = Path(candidate)
            break

    if checkpoint_path is None:
        engine_status = (
            "no checkpoint found -- set RL_CHESS_CHECKPOINT or edit "
            "DEFAULT_CHECKPOINT_CANDIDATES in server.py. Running with "
            "randomly-initialized weights (will play very badly)."
        )
    else:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
        model.load_state_dict(state_dict)
        active_checkpoint_path = checkpoint_path
        engine_status = (
            f"{ENGINE_NAME} | approx. {ENGINE_ESTIMATED_ELO} Elo | "
            f"NNUE weights loaded ({checkpoint_path.name})"
        )

    model.eval()
    adapter = UCIAdapter(model, device)


def create_session_token(profile):
    secret = os.getenv("AUTH_SECRET")
    if not secret:
        raise HTTPException(503, "Google login is not configured. Set AUTH_SECRET on the server.")
    payload = {
        "sub": profile["sub"],
        "name": profile["name"],
        "email": profile.get("email", ""),
        "exp": int(time.time()) + 60 * 60 * 24 * 30,
    }
    encoded = base64.urlsafe_b64encode(json.dumps(payload, separators=(",", ":")).encode()).rstrip(b"=")
    signature = hmac.new(secret.encode(), encoded, hashlib.sha256).digest()
    signed = base64.urlsafe_b64encode(signature).rstrip(b"=")
    return encoded.decode() + "." + signed.decode()


def verify_session_token(token):
    secret = os.getenv("AUTH_SECRET")
    if not secret or not token or "." not in token:
        return None
    encoded, supplied_signature = token.split(".", 1)
    expected_signature = base64.urlsafe_b64encode(
        hmac.new(secret.encode(), encoded.encode(), hashlib.sha256).digest()
    ).rstrip(b"=").decode()
    if not hmac.compare_digest(supplied_signature, expected_signature):
        return None
    try:
        padded = encoded + "=" * (-len(encoded) % 4)
        payload = json.loads(base64.urlsafe_b64decode(padded).decode())
    except (ValueError, json.JSONDecodeError):
        return None
    if payload.get("exp", 0) < time.time() or not payload.get("sub"):
        return None
    payload["player_key"] = "google:" + payload["sub"]
    return payload


def verify_google_credential(credential):
    client_id = os.getenv("GOOGLE_CLIENT_ID")
    if not client_id:
        raise HTTPException(503, "Google login is not configured. Set GOOGLE_CLIENT_ID on the server.")
    try:
        from google.auth.transport import requests as google_requests
        from google.oauth2 import id_token
        profile = id_token.verify_oauth2_token(credential, google_requests.Request(), client_id)
    except (ValueError, ImportError) as exc:
        raise HTTPException(401, "Google sign-in verification failed") from exc
    if not profile.get("sub") or not profile.get("email_verified", False):
        raise HTTPException(401, "Google account email is not verified")
    return {
        "sub": profile["sub"],
        "name": profile.get("name") or profile.get("email", "Google Player"),
        "email": profile.get("email", ""),
    }


def get_or_create_player(player_name, key=None):
    key = key or " ".join(player_name.strip().lower().split()) or "guest"
    return database.player(key, player_name, DEFAULT_PLAYER_ELO)


def player_key(player_name):
    return " ".join(player_name.strip().lower().split()) or "guest"


def game_state(game):
    return {
        "game_id": game["game_id"],
        "fen": game["board"].fen(),
        "history": game["history"],
        "redo": game.get("redo", []),
        "player_name": game["player_name"],
        "player_key": game.get("player_key"),
        "player_color": "white" if game["player_color"] == chess.WHITE else "black",
        "difficulty": game["difficulty"],
        "result": game.get("result"),
        "status": game.get("status"),
        "last_move": game.get("last_move"),
        "engine_move": game.get("engine_move"),
        "analysis": game.get("analysis", []),
        "review": game.get("review"),
    }


def save_game(game):
    database.save_game(game["game_id"], game_state(game))


def restore_game(game_id, state):
    return {
        "game_id": game_id,
        "board": chess.Board(state["fen"]),
        "history": state.get("history", []),
        "redo": state.get("redo", []),
        "player_name": state.get("player_name", "Guest"),
        "player_key": state.get("player_key"),
        "player_color": chess.WHITE if state.get("player_color") == "white" else chess.BLACK,
        "difficulty": state.get("difficulty", "medium"),
        "result": state.get("result"),
        "status": state.get("status"),
        "last_move": state.get("last_move"),
        "engine_move": state.get("engine_move"),
        "analysis": state.get("analysis", []),
        "review": state.get("review"),
    }


def update_player_elo(old_elo, opponent_elo, score):
    """Standard Elo update: score is 1.0/0.5/0.0 for win/draw/loss from
    the player's perspective. K=24 is a common mid-range choice."""
    expected = 1.0 / (1.0 + 10 ** ((opponent_elo - old_elo) / 400.0))
    return round(old_elo + ELO_K_FACTOR * (score - expected))


def player_score_from_result(result, player_color):
    if result == "1/2-1/2":
        return 0.5
    if result == "1-0":
        return 1.0 if player_color == "white" else 0.0
    if result == "0-1":
        return 1.0 if player_color == "black" else 0.0
    return 0.5


def game_to_pgn(game):
    pgn_game = chess.pgn.Game()
    result = game.get("result") or game["board"].result()
    pgn_game.headers["Event"] = "RL Chess Engine"
    pgn_game.headers["White"] = game["player_name"] if game["player_color"] == "white" else "RL Chess Engine"
    pgn_game.headers["Black"] = "RL Chess Engine" if game["player_color"] == "white" else game["player_name"]
    pgn_game.headers["Result"] = result

    node = pgn_game
    for item in game["history"]:
        node = node.add_variation(chess.Move.from_uci(item["uci"]))

    return str(pgn_game)


def finish_game(game, result):
    """Completes the game loop, updates player record data, and computes 
    in-memory Elo swings against ENGINE_ELO."""
    if game.get("result"):
        return

    game["result"] = result
    player = get_or_create_player(game["player_name"], game.get("player_key"))
    
    score = player_score_from_result(result, game["player_color"])
    old_elo = player["elo"]
    new_elo = update_player_elo(old_elo, ENGINE_ELO, score)
    
    player["elo"] = new_elo
    player["games_played"] += 1
    database.update_player(game.get("player_key") or player_key(game["player_name"]), player["name"], new_elo, player["games_played"])
    
    game["elo_change"] = {
        "old_elo": old_elo,
        "new_elo": new_elo,
        "change": new_elo - old_elo
    }


def game_payload(game, status=None):
    board = game["board"]
    return {
        "game_id": game["game_id"], "fen": board.fen(),
        "legal_moves": [m.uci() for m in board.legal_moves],
        "history": game["history"], "last_move": game.get("last_move"),
        "engine_move": game.get("engine_move"), "game_over": board.is_game_over(),
        "result": game.get("result"), "can_move": not board.is_game_over() and board.turn == game["player_color"],
        "turn": "white" if board.turn else "black", "status": status or "Your turn",
        "engine_status": engine_status, "engine_details": ENGINE_DETAILS,
        "player": get_or_create_player(game["player_name"], game.get("player_key")),
        "account": {"name": game["player_name"], "authenticated": game.get("player_key", "").startswith("google:")},
        "can_undo": len(game["history"]) >= 2 and not board.is_game_over(),
        "can_redo": bool(game.get("redo")) and not board.is_game_over(),
        "analysis": game.get("analysis", []),
    }


def describe_game_status(game):
    board = game["board"]
    if game.get("result") and "resign" in game.get("status", "").lower():
        return game["status"]
    if board.is_checkmate():
        winner = "White" if board.turn == chess.BLACK else "Black"
        return f"Checkmate - {winner} wins."
        return f"Checkmate — {winner} wins."
    if board.is_stalemate():
        return "Draw - stalemate."
        return "Draw — stalemate."
    if board.is_insufficient_material():
        return "Draw - insufficient material."
        return "Draw — insufficient material."
    if board.is_check():
        return f"Check - {'White' if board.turn else 'Black'} to move."
        return f"Check — {'White' if board.turn else 'Black'} to move."
    return "Your turn"


@app.post("/api/auth/google")
async def google_login(request: GoogleLoginRequest):
    profile = verify_google_credential(request.credential)
    token = create_session_token(profile)
    account_key = "google:" + profile["sub"]
    player = get_or_create_player(profile["name"], account_key)
    return {
        "token": token,
        "user": {"name": profile["name"], "email": profile["email"]},
        "player": player,
    }


@app.post("/api/new-game")
async def new_game(request: NewGameRequest):
    profile = verify_session_token(request.player_token) if request.player_token else None
    if request.player_token and not profile:
        raise HTTPException(401, "Your Google session has expired. Please sign in again.")
    player_name = profile["name"] if profile else request.player_name
    account_key = profile["player_key"] if profile else player_key(player_name)
    color = chess.WHITE if request.player_color.lower() == "white" else chess.BLACK
    game = {"game_id": str(uuid.uuid4()), "board": chess.Board(), "history": [],
            "player_name": player_name, "player_key": account_key, "player_color": color,
            "difficulty": request.difficulty, "redo": []}
    games[game["game_id"]] = game
    # White moves first in chess. If the human chose Black, let the engine
    # make the opening move before returning the initial position.
    if color == chess.BLACK:
        if adapter is None:
            raise HTTPException(503, "Engine is not loaded")
        preset = DIFFICULTY_PRESETS.get(request.difficulty, DIFFICULTY_PRESETS["medium"])
        with engine_lock:
            adapter.command("position startpos")
            lines = adapter.command(f"go depth {preset['search_depth']} movetime {int(preset['move_time'] * 1000)}")
            game["analysis"] = list(adapter.last_info)
        bestmove = next((line.split()[1] for line in lines if line.startswith("bestmove ")), None)
        if bestmove and bestmove != "0000":
            opening_move = chess.Move.from_uci(bestmove)
            san = game["board"].san(opening_move)
            game["board"].push(opening_move)
            game["history"].append({"uci": bestmove, "san": san, "side": "engine"})
            game["last_move"] = bestmove
    save_game(game)
    return game_payload(game)


@app.post("/api/move")
async def make_move(request: MoveRequest):
    game = find_game(request.game_id)
    try:
        move = chess.Move.from_uci(request.move)
    except ValueError:
        raise HTTPException(400, "Invalid move")
    board = game["board"]
    if move not in board.legal_moves:
        raise HTTPException(400, "Illegal move")
    san = board.san(move)
    board.push(move)
    game["redo"] = []
    game["history"].append({"uci": move.uci(), "san": san, "side": "player"})
    game["last_move"] = move.uci()
    if not board.is_game_over():
        # Rebuild the adapter position from the complete game history before
        # searching.  The old placeholder selected the first legal move,
        # which made the UI engine repeat predictable moves.
        if adapter is None:
            raise HTTPException(503, "Engine is not loaded")
        preset = DIFFICULTY_PRESETS.get(game["difficulty"], DIFFICULTY_PRESETS["medium"])
        with engine_lock:
            adapter.command("position startpos moves " + " ".join(item["uci"] for item in game["history"]))
            lines = adapter.command(f"go depth {preset['search_depth']} movetime {int(preset['move_time'] * 1000)}")
            game["analysis"] = list(adapter.last_info)
        bestmove = next((line.split()[1] for line in lines if line.startswith("bestmove ")), None)
        engine_move = chess.Move.from_uci(bestmove) if bestmove and bestmove != "0000" else None
        if engine_move:
            engine_san = board.san(engine_move)
            board.push(engine_move)
            game["history"].append({"uci": engine_move.uci(), "san": engine_san, "side": "engine"})
            game["engine_move"] = engine_move.uci()
            game["last_move"] = engine_move.uci()
    if board.is_game_over():
        finish_game(game, board.result(claim_draw=True))
    save_game(game)
    return game_payload(game, describe_game_status(game) if board.is_game_over() else "Your turn")


def find_game(game_id):
    game = games.get(game_id)
    if not game:
        state = database.load_game(game_id)
        if state:
            game = restore_game(game_id, state)
            games[game_id] = game
    if not game:
        raise HTTPException(404, "Game not found")
    return game


def rebuild_game_board(game):
    board = chess.Board()
    for item in game["history"]:
        board.push(chess.Move.from_uci(item["uci"]))
    game["board"] = board


@app.post("/api/resign")
async def resign(request: ResignRequest):
    game = find_game(request.game_id)
    result = "0-1" if game["player_color"] == chess.WHITE else "1-0"
    opponent = "Black" if game["player_color"] == chess.WHITE else "White"
    game["status"] = f"You resigned. {opponent} wins by resignation."
    finish_game(game, result)
    save_game(game)
    return game_payload(game, game["status"])


@app.get("/api/game/{game_id}")
async def resume_game(game_id: str):
    game = find_game(game_id)
    status = describe_game_status(game) if game.get("result") or game["board"].is_game_over() else "Your turn"
    return game_payload(game, status)


@app.post("/api/undo")
async def undo(request: HistoryRequest):
    game = find_game(request.game_id)
    if len(game["history"]) < 2:
        raise HTTPException(400, "Nothing to undo")
    game["redo"] = game["history"][-2:] + game.get("redo", [])
    game["history"] = game["history"][:-2]
    rebuild_game_board(game)
    save_game(game)
    return game_payload(game, "Move undone")


@app.post("/api/redo")
async def redo(request: HistoryRequest):
    game = find_game(request.game_id)
    if not game.get("redo"):
        raise HTTPException(400, "Nothing to redo")
    game["history"].extend(game["redo"][:2])
    game["redo"] = game["redo"][2:]
    rebuild_game_board(game)
    save_game(game)
    return game_payload(game, "Move redone")


def review_phase(ply):
    if ply < 10:
        return "opening"
    if ply < 30:
        return "middlegame"
    return "endgame"


def personalized_coaching(game, current_review):
    player = get_or_create_player(game["player_name"], game.get("player_key"))
    previous_games = database.completed_games_for_player(game.get("player_key"), 50)
    previous_games = [item for item in previous_games if item.get("game_id") != game.get("game_id")]
    accuracies = [
        item["review"].get("accuracy")
        for item in previous_games
        if isinstance(item.get("review"), dict) and item["review"].get("accuracy") is not None
    ]
    average_accuracy = round(sum(accuracies) / len(accuracies)) if accuracies else None

    elo = player["elo"]
    if elo < 1100:
        difficulty, depth, move_time = "easy", 3, 1.5
        reason = "Build consistency with shorter searches and a clear tactical focus."
    elif elo < 1400:
        difficulty, depth, move_time = "medium", 4, 2.0
        reason = "Balance practical play with enough search time to spot tactics."
    elif elo < 1750:
        difficulty, depth, move_time = "medium", 5, 2.5
        reason = "Use a deeper medium search to challenge calculation without long waits."
    else:
        difficulty, depth, move_time = "hard", 6, 3.5
        reason = "Your rating supports deeper searches and more demanding positions."

    if average_accuracy is not None and average_accuracy < 65:
        difficulty, depth, move_time = "easy", min(depth, 4), min(move_time, 2.0)
        reason = "Recent accuracy is below 65%, so a shorter search gives you more time to learn from each position."
    elif average_accuracy is not None and average_accuracy >= 88 and elo >= 1400:
        depth = min(depth + 1, 8)
        move_time = min(move_time + 0.5, 5.0)
        reason = "Recent accuracy is strong, so the next step is a deeper search with a slightly larger time budget."

    prior_errors = {}
    phase_errors = {"opening": 0, "middlegame": 0, "endgame": 0}
    for previous in previous_games:
        review = previous.get("review") or {}
        for item in review.get("moves", []):
            classification = item.get("classification") or {}
            tone = classification.get("tone")
            if tone not in {"inaccuracy", "mistake", "blunder"}:
                continue
            phase = classification.get("phase") or item.get("phase") or "middlegame"
            played = classification.get("played_move") or item.get("played_move")
            if not played:
                continue
            phase_errors[phase] = phase_errors.get(phase, 0) + 1
            key = (phase, played)
            prior_errors.setdefault(key, {"count": 0, "best": set(), "san": item.get("san", played)})
            prior_errors[key]["count"] += 1
            best = classification.get("best_move") or item.get("best_move")
            if best:
                prior_errors[key]["best"].add(best)

    patterns = []
    for item in current_review.get("moves", []):
        classification = item.get("classification") or {}
        if classification.get("tone") not in {"inaccuracy", "mistake", "blunder"}:
            continue
        phase = classification.get("phase") or item.get("phase") or "middlegame"
        played = classification.get("played_move") or item.get("played_move")
        match = prior_errors.get((phase, played))
        if not match:
            continue
        best_moves = ", ".join(sorted(match["best"])) or "a stronger engine continuation"
        patterns.append({
            "title": f"Repeated {phase} decision",
            "detail": f"You played {item.get('san', played)} in the {phase} before. The engine repeatedly preferred {best_moves}.",
            "count": match["count"] + 1,
            "tone": classification.get("tone", "mistake"),
        })

    if not patterns and any(phase_errors.values()):
        phase, count = max(phase_errors.items(), key=lambda entry: entry[1])
        patterns.append({
            "title": f"Focus on the {phase}",
            "detail": f"Most tracked inaccuracies, mistakes, and blunders happened in the {phase} ({count} total).",
            "count": count,
            "tone": "inaccuracy",
        })

    if previous_games and not patterns:
        patterns.append({
            "title": "No repeated mistake detected",
            "detail": "Your latest game did not repeat a tracked mistake from the previous analyzed matches.",
            "count": 0,
            "tone": "good",
        })

    history_text = f"Based on {len(previous_games)} previous analyzed match{'es' if len(previous_games) != 1 else ''}."
    if average_accuracy is not None:
        history_text += f" Average accuracy: {average_accuracy}%."
    else:
        history_text = "This is your first analyzed match with stored personalization data."

    return {
        "history_text": history_text,
        "games_analyzed": len(previous_games),
        "average_accuracy": average_accuracy,
        "recommendation": {
            "difficulty": difficulty,
            "depth": depth,
            "move_time": move_time,
            "reason": reason,
        },
        "patterns": patterns[:4],
    }


@app.post("/api/analyze")
async def analyze(request: AnalyzeRequest):
    game = find_game(request.game_id)
    moves, counts, player_moves = [], {"excellent": 0, "good": 0, "inaccuracy": 0, "mistake": 0, "blunder": 0}, 0
    prior = []
    for item in game["history"]:
        if item["side"] == "player":
            player_moves += 1
            phase = review_phase(len(prior))
            best = None
            if adapter is not None:
                preset = DIFFICULTY_PRESETS.get(game["difficulty"], DIFFICULTY_PRESETS["medium"])
                with engine_lock:
                    adapter.command("position startpos" + (" moves " + " ".join(prior) if prior else ""))
                    lines = adapter.command(f"go depth {min(preset['search_depth'], 5)} movetime 400")
                    best = next((line.split()[1] for line in lines if line.startswith("bestmove ")), None)
            if best == item["uci"]:
                tone, reason = "excellent", "You played the engine's preferred move."
            elif best and item["uci"][0:2] == best[0:2]:
                tone, reason = "good", "A reasonable move, close to the engine choice."
            elif best:
                tone, reason = "inaccuracy", "The engine preferred a different continuation."
            else:
                tone, reason = "good", "Move recorded without a comparison search."
            counts[tone] += 1
            moves.append({"san": item.get("san", item["uci"]), "phase": phase, "classification": {"tone": tone, "symbol": {"excellent":"!!","good":"!","inaccuracy":"?!","mistake":"?","blunder":"??"}[tone], "label": tone.title(), "reason": reason, "best_move": best, "played_move": item["uci"], "phase": phase}})
        prior.append(item["uci"])
    accuracy = round(100 * (counts["excellent"] + 0.8 * counts["good"] + 0.5 * counts["inaccuracy"]) / max(player_moves, 1))
    review = {"accuracy": accuracy, "counts": counts, "moves": moves}
    game["review"] = review
    save_game(game)
    personalization = personalized_coaching(game, review)
    return {"accuracy": accuracy, "verdict": "Game review", "result": game.get("result") or game["board"].result(),
            "pgn": game_to_pgn(game), "counts": counts, "moves": moves,
            "player": get_or_create_player(game["player_name"], game.get("player_key")),
            "personalization": personalization}


@app.post("/api/chat")
async def chat(request: ChatRequest):
    game = find_game(request.game_id)
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(503, "Chatbot is not configured. Set GROQ_API_KEY on the server.")
    board = game["board"]
    recent = " ".join(item.get("san", item["uci"]) for item in game["history"][-20:]) or "No moves yet"
    context = f"FEN: {board.fen()}\nRecent moves: {recent}\nStatus: {describe_game_status(game)}"
    model_name = os.getenv("GROQ_MODEL", "qwen/qwen3.6-27b").strip()
    if model_name in {"llama-3.3-70b-versatile", "openai/gpt-oss-20b"}:
        model_name = "qwen/qwen3.6-27b"
    body = json.dumps({
        "model": model_name,
        "messages": [
            {"role": "system", "content": "You are RL Chess Assist. Explain chess clearly and briefly. Use only the supplied game position and moves; do not claim engine analysis you did not perform."},
            {"role": "user", "content": f"Game context:\n{context}\n\nPlayer question: {request.message}"},
        ],
        "max_tokens": 300,
    }).encode("utf-8")
    http_request = urllib.request.Request(
        "https://api.groq.com/openai/v1/chat/completions",
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "RLChessAssist/1.0 (+https://rl-chess-assist-1.onrender.com)",
        },
    )
    try:
        with urllib.request.urlopen(http_request, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        try:
            raw_body = exc.read().decode("utf-8", errors="replace")
        except Exception:
            raw_body = "<could not read response body>"
        raise HTTPException(502, f"Chatbot request failed: HTTP {exc.code} {exc.reason} - body: {raw_body[:500]}")
    except (urllib.error.URLError, TimeoutError) as exc:
        raise HTTPException(502, f"Chatbot connection failed: {exc}")
    choices = data.get("choices", [])
    text = choices[0].get("message", {}).get("content") if choices else None
    return {"text": text or "The chatbot returned no text."}
