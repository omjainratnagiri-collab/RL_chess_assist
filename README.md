# RL Chess Python Engine

RL Chess is a Python chess engine and browser application that combines supervised learning, reinforcement-learning fine-tuning, NNUE evaluation, and alpha-beta search.

Live demo: https://rl-chess-assist-1.onrender.com

## Training progression

The project was developed in two main learning stages:

1. **Supervised learning (SL).** The NNUE value and policy heads were trained on Lichess game positions evaluated by Stockfish. The supervised-learning model reached approximately **1300 Elo**.
2. **Reinforcement-learning fine-tuning (RL).** Starting from the SL checkpoint, the model was fine-tuned with TD-lambda self-play driven by the project's alpha-beta search. This improved the engine to approximately **1800 Elo**.

The Lichess evaluation data was used as the initial supervised-learning source. Elo estimates were produced by `estimate_elo_vs_stockfish.py`, which plays anchor matches against Stockfish at configured strength levels. The estimates are approximate engine-versus-engine measurements, not official human ratings, and depend on search depth, time limit, hardware, and match settings.

The application displays the engine as approximately 2000 Elo as a configured current-strength label. The documented Stockfish anchor evaluation is approximately 1800 Elo.

## Model architecture

- **HalfKP encoding:** each perspective uses own king square, piece type, piece color, and piece square features, giving 40,960 possible features per perspective.
- **Feature transformer:** a 40,960 x 160 embedding table. Active feature embeddings are summed for both perspectives and passed through ClippedReLU.
- **Search input:** the two 160-dimensional accumulator vectors are concatenated into a 320-dimensional representation.
- **Trunk:** Linear(320, 256), ClippedReLU, one residual block, and a final ClippedReLU.
- **Value head:** Linear(256, 128), ReLU, Linear(128, 1), and Tanh. The value is side-to-move relative and approximately bounded to [-1, 1].
- **Policy head:** an 8 x 8 x 73 move encoding with 4,672 outputs. It is trained during SL and reserved for possible future move ordering; the current alpha-beta search does not require policy priors.
- **Parameters:** approximately 8.1 million, with most parameters in the feature-transformer embedding table.

## Search engine

- Negamax alpha-beta search
- Iterative deepening with a time limit enforced during recursion
- Quiescence search for captures and check evasions
- Transposition table using incrementally maintained Zobrist hashing
- Transposition-table move ordering, MVV-LVA capture ordering, and a quiet-move history heuristic
- Null-move pruning with zugzwang protection
- Late-move reductions with a null-window probe and full re-search when needed
- Incremental NNUE accumulator push/pop during make/unmake
- Pure NumPy value-head inference at search leaves
- `python-chess` for legal moves, castling, en passant, promotion, check, and game termination
- Search information containing score, depth, nodes, elapsed time, and nodes per second

The reported score is an internal side-to-move value, not a centipawn score. Forced mates are represented outside the normal value range so they outrank ordinary positional evaluations.

## Reinforcement learning

The RL training loop uses TD-lambda self-play:

- TD-lambda bootstraps from future search values instead of assigning only the final game result to every position.
- Randomized opening plies provide exploration for the deterministic alpha-beta search.
- Candidate checkpoints are gated against the previous best checkpoint before promotion.
- Self-play and gate games run as independent CPU worker processes.
- Replay samples and checkpoints can be saved and resumed during training.

## Application

The FastAPI server provides:

- Browser chessboard playable as White or Black
- Easy, medium, and hard search presets
- Search-depth and move-time controls
- Resign, undo, redo, promotion, check, and game-over handling
- Manual post-game analysis after the match ends
- Move classifications such as excellent, good, inaccuracy, mistake, and blunder
- Current-game PGN generation in the analysis report
- Optional Groq chess assistant using the current FEN, recent SAN moves, and game status
- Optional Google login with verified account-based player Elo
- UCI-style commands through the Python adapter: `uci`, `isready`, `ucinewgame`, `position`, `go`, `stop`, and `quit`

## Multi-player behavior

Every new game receives a UUID. Active games are kept in the Python `games` dictionary cache, allowing multiple browsers to play independently. Player Elo records and chat context are available in memory while a process is running.

Engine searches are protected for correctness, so simultaneous engine turns may queue briefly without mixing player games. If no database URL is configured, restarting or redeploying the service clears active sessions.

When `DATABASE_URL` is configured, the application stores game state and player ratings in SQL. The browser stores only the active game UUID and uses it to restore the board after refresh. Google account IDs are used as stable player keys, so each signed-in player has a separate Elo record. Without `DATABASE_URL`, local development falls back to a small SQLite database.

## SQL deployment

For local development, leave `DATABASE_URL` unset and the app creates `rl_chess.sqlite3`. To use persistent cloud storage on Render:

1. Create a Render PostgreSQL database in the same region as the web service.
2. Copy its **Internal Database URL** into the web service environment variable `DATABASE_URL`.
3. Add `GOOGLE_CLIENT_ID`, `AUTH_SECRET`, `RL_CHESS_CHECKPOINT`, and any required `GROQ_API_KEY` values in the web service Environment settings.
4. Deploy with build command `pip install -r requirements.txt` and start command `uvicorn server:app --host 0.0.0.0 --port $PORT`.
5. Open `/api/health`. A working deployment returns `{"status":"ok","database":"postgres"}`.

The server creates the `players` and `games` tables automatically on startup. After deployment, start and finish a test game, then refresh the page. The active game and Elo should remain available because they are stored in PostgreSQL rather than the service filesystem.

## Main files

- `server.py` - FastAPI server and temporary game sessions
- `search.py` - alpha-beta search, transposition table, quiescence, null-move pruning, and LMR
- `accumulator.py` - incremental NNUE accumulators and Zobrist state
- `zobrist.py` - incremental hash updates
- `nnue.py` - HalfKP model, trunk, value head, and policy head
- `feature_transformer.py` - embedding and activation layers
- `numpy_inference.py` - NumPy search-time value inference
- `halfkp_encoding.py` - HalfKP feature encoding and move differences
- `move_encoder.py` - supervised-learning policy move encoding
- `uci_adapter.py` - bridge between the server and search engine
- `td_selfplay.py` - TD-lambda RL self-play and checkpoint gating
- `stockfish_zst.py` and `stockfish_zst_dataset.py` - supervised-learning data pipeline
- `estimate_elo_vs_stockfish.py` - Stockfish anchor-match Elo estimation
- `play_own_versions.py` - direct matches between project checkpoints
- `static/` - browser interface
- `checkpoints/td_selfplay_best.pt` - included NNUE weights

## Current checkpoint

The deployment copy contains one weights file:

```text
checkpoints/td_selfplay_best.pt
```
