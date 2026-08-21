import json
import os
import sqlite3
import threading


class Database:
    def __init__(self):
        self.url = os.getenv("DATABASE_URL")
        self.lock = threading.RLock()
        if self.url:
            if self.url.startswith("postgres://"):
                self.url = "postgresql://" + self.url[len("postgres://"):]
            try:
                import psycopg
            except ImportError as exc:
                raise RuntimeError("DATABASE_URL is set but psycopg is not installed") from exc
            self.psycopg = psycopg
            self.backend = "postgres"
            self.placeholder = "%s"
        else:
            self.backend = "sqlite"
            self.placeholder = "?"
            self.sqlite_path = os.getenv("SQLITE_PATH", "rl_chess.sqlite3")
        self.initialize()

    def connection(self):
        if self.backend == "postgres":
            return self.psycopg.connect(self.url)
        connection = sqlite3.connect(self.sqlite_path, timeout=30)
        connection.row_factory = sqlite3.Row
        return connection

    def initialize(self):
        with self.lock, self.connection() as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS games ("
                "game_id TEXT PRIMARY KEY, state TEXT NOT NULL, "
                "updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS players ("
                "player_key TEXT PRIMARY KEY, name TEXT NOT NULL, elo INTEGER NOT NULL, "
                "games_played INTEGER NOT NULL DEFAULT 0)"
            )
            connection.commit()

    def save_game(self, game_id, state):
        encoded = json.dumps(state)
        with self.lock, self.connection() as connection:
            if self.backend == "postgres":
                connection.execute(
                    "INSERT INTO games (game_id, state, updated_at) VALUES (%s, %s, CURRENT_TIMESTAMP) "
                    "ON CONFLICT (game_id) DO UPDATE SET state = EXCLUDED.state, updated_at = CURRENT_TIMESTAMP",
                    (game_id, encoded),
                )
            else:
                connection.execute(
                    "INSERT INTO games (game_id, state, updated_at) VALUES (?, ?, CURRENT_TIMESTAMP) "
                    "ON CONFLICT(game_id) DO UPDATE SET state = excluded.state, updated_at = CURRENT_TIMESTAMP",
                    (game_id, encoded),
                )
            connection.commit()

    def load_game(self, game_id):
        with self.lock, self.connection() as connection:
            row = connection.execute(
                f"SELECT state FROM games WHERE game_id = {self.placeholder}", (game_id,)
            ).fetchone()
        if not row:
            return None
        value = row[0] if not isinstance(row, sqlite3.Row) else row["state"]
        return json.loads(value)

    def player(self, player_key, name, default_elo):
        with self.lock, self.connection() as connection:
            row = connection.execute(
                f"SELECT player_key, name, elo, games_played FROM players WHERE player_key = {self.placeholder}",
                (player_key,),
            ).fetchone()
            if not row:
                connection.execute(
                    f"INSERT INTO players (player_key, name, elo, games_played) VALUES ({self.placeholder}, {self.placeholder}, {self.placeholder}, {self.placeholder})",
                    (player_key, name, default_elo, 0),
                )
                connection.commit()
                return {"name": name, "elo": default_elo, "games_played": 0}
            if isinstance(row, sqlite3.Row):
                return {"name": row["name"], "elo": row["elo"], "games_played": row["games_played"]}
            return {"name": row[1], "elo": row[2], "games_played": row[3]}

    def update_player(self, player_key, name, elo, games_played):
        with self.lock, self.connection() as connection:
            connection.execute(
                f"UPDATE players SET name = {self.placeholder}, elo = {self.placeholder}, games_played = {self.placeholder} WHERE player_key = {self.placeholder}",
                (name, elo, games_played, player_key),
            )
            connection.commit()

    def completed_games_for_player(self, player_key, limit=50):
        if not player_key:
            return []
        with self.lock, self.connection() as connection:
            rows = connection.execute(
                "SELECT state FROM games ORDER BY updated_at DESC"
            ).fetchall()
        games = []
        for row in rows:
            value = row[0] if not isinstance(row, sqlite3.Row) else row["state"]
            state = json.loads(value)
            if state.get("player_key") == player_key and state.get("result"):
                games.append(state)
                if len(games) >= limit:
                    break
        return games

    def healthcheck(self):
        with self.lock, self.connection() as connection:
            connection.execute("SELECT 1").fetchone()
        return self.backend


database = Database()
