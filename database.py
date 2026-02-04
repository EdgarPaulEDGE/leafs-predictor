"""
database.py – Datenbank für Tipps und Punkte (Multi-User)
==========================================================
Unterstützt SQLite (lokal) und PostgreSQL (Render).
Wenn DATABASE_URL gesetzt ist → PostgreSQL, sonst → SQLite.
"""

import os
import sqlite3

DATABASE_URL = os.environ.get("DATABASE_URL", "")

# PostgreSQL verwenden wenn DATABASE_URL gesetzt ist
USE_POSTGRES = bool(DATABASE_URL)

if USE_POSTGRES:
    import psycopg2
    import psycopg2.extras
    # Render gibt postgres:// aber psycopg2 braucht postgresql://
    if DATABASE_URL.startswith("postgres://"):
        DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

DB_PATH = "leafs_game.db"


def get_connection():
    """Erstellt eine Verbindung zur Datenbank."""
    if USE_POSTGRES:
        conn = psycopg2.connect(DATABASE_URL)
        return conn
    else:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        return conn


def _dict_from_row(row, cursor=None):
    """Konvertiert eine DB-Zeile in ein Dictionary."""
    if USE_POSTGRES:
        if row is None:
            return None
        columns = [desc[0] for desc in cursor.description]
        return dict(zip(columns, row))
    else:
        if row is None:
            return None
        return dict(row)


def _rows_to_dicts(cursor):
    """Konvertiert alle Zeilen in Dictionaries."""
    if USE_POSTGRES:
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]
    else:
        return [dict(row) for row in cursor.fetchall()]


def _placeholder():
    """Gibt den richtigen Platzhalter zurück: %s für Postgres, ? für SQLite."""
    return "%s" if USE_POSTGRES else "?"


def init_db():
    """Erstellt die Datenbank-Tabellen."""
    conn = get_connection()
    cursor = conn.cursor()

    if USE_POSTGRES:
        # Users-Tabelle
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # Predictions-Tabelle mit username
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id SERIAL PRIMARY KEY,
                username TEXT DEFAULT '',
                game_id INTEGER NOT NULL,
                game_date TEXT NOT NULL,
                opponent TEXT NOT NULL,
                is_home INTEGER NOT NULL,
                user_prediction TEXT NOT NULL,
                user_score_leafs INTEGER,
                user_score_opponent INTEGER,
                model_prediction TEXT NOT NULL,
                model_win_probability REAL,
                actual_result TEXT,
                actual_score_leafs INTEGER,
                actual_score_opponent INTEGER,
                user_points INTEGER DEFAULT 0,
                model_points INTEGER DEFAULT 0,
                is_resolved INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # Username-Spalte hinzufügen falls alte Tabelle existiert
        try:
            cursor.execute("""
                ALTER TABLE predictions ADD COLUMN username TEXT DEFAULT ''
            """)
        except Exception:
            pass  # Spalte existiert bereits
    else:
        # SQLite: Users-Tabelle
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # SQLite: Predictions-Tabelle mit username
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT DEFAULT '',
                game_id INTEGER NOT NULL,
                game_date TEXT NOT NULL,
                opponent TEXT NOT NULL,
                is_home INTEGER NOT NULL,
                user_prediction TEXT NOT NULL,
                user_score_leafs INTEGER,
                user_score_opponent INTEGER,
                model_prediction TEXT NOT NULL,
                model_win_probability REAL,
                actual_result TEXT,
                actual_score_leafs INTEGER,
                actual_score_opponent INTEGER,
                user_points INTEGER DEFAULT 0,
                model_points INTEGER DEFAULT 0,
                is_resolved INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

    conn.commit()
    conn.close()
    db_type = "PostgreSQL" if USE_POSTGRES else "SQLite"
    print(f"Datenbank initialisiert! ({db_type})")


# ---- User-Funktionen ----

def create_user(username: str) -> bool:
    """Erstellt einen neuen User. Gibt True zurück wenn erfolgreich."""
    conn = get_connection()
    cursor = conn.cursor()
    ph = _placeholder()
    try:
        cursor.execute(f"INSERT INTO users (username) VALUES ({ph})", (username,))
        conn.commit()
        conn.close()
        return True
    except Exception:
        conn.rollback()
        conn.close()
        return False  # Username existiert bereits


def user_exists(username: str) -> bool:
    """Prüft ob ein Username existiert."""
    conn = get_connection()
    cursor = conn.cursor()
    ph = _placeholder()
    cursor.execute(f"SELECT id FROM users WHERE username = {ph}", (username,))
    exists = cursor.fetchone() is not None
    conn.close()
    return exists


def get_or_create_user(username: str) -> str:
    """Holt oder erstellt einen User. Gibt den Username zurück."""
    if not user_exists(username):
        create_user(username)
    return username


# ---- Prediction-Funktionen ----

def add_prediction(
    username: str,
    game_id: int,
    game_date: str,
    opponent: str,
    is_home: int,
    user_prediction: str,
    user_score_leafs: int,
    user_score_opponent: int,
    model_prediction: str,
    model_win_probability: float,
) -> int:
    """Speichert einen neuen Tipp für einen User."""
    conn = get_connection()
    cursor = conn.cursor()
    ph = _placeholder()

    if USE_POSTGRES:
        cursor.execute(f"""
            INSERT INTO predictions
            (username, game_id, game_date, opponent, is_home,
             user_prediction, user_score_leafs, user_score_opponent,
             model_prediction, model_win_probability)
            VALUES ({ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph})
            RETURNING id
        """, (
            username, game_id, game_date, opponent, is_home,
            user_prediction, user_score_leafs, user_score_opponent,
            model_prediction, model_win_probability,
        ))
        prediction_id = cursor.fetchone()[0]
    else:
        cursor.execute(f"""
            INSERT INTO predictions
            (username, game_id, game_date, opponent, is_home,
             user_prediction, user_score_leafs, user_score_opponent,
             model_prediction, model_win_probability)
            VALUES ({ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph})
        """, (
            username, game_id, game_date, opponent, is_home,
            user_prediction, user_score_leafs, user_score_opponent,
            model_prediction, model_win_probability,
        ))
        prediction_id = cursor.lastrowid

    conn.commit()
    conn.close()
    return prediction_id


def resolve_prediction(
    prediction_id: int,
    actual_result: str,
    actual_score_leafs: int,
    actual_score_opponent: int,
):
    """Löst einen Tipp auf nachdem das Spiel vorbei ist."""
    conn = get_connection()
    cursor = conn.cursor()
    ph = _placeholder()

    cursor.execute(f"SELECT * FROM predictions WHERE id = {ph}", (prediction_id,))
    row = cursor.fetchone()
    pred = _dict_from_row(row, cursor)

    if not pred:
        conn.close()
        return

    # User-Punkte berechnen
    user_points = 0
    if pred["user_prediction"] == actual_result:
        user_points = 1
        if (pred["user_score_leafs"] == actual_score_leafs and
                pred["user_score_opponent"] == actual_score_opponent):
            user_points = 3

    # Modell-Punkte berechnen
    model_points = 0
    if pred["model_prediction"] == actual_result:
        model_points = 1

    cursor.execute(f"""
        UPDATE predictions
        SET actual_result = {ph},
            actual_score_leafs = {ph},
            actual_score_opponent = {ph},
            user_points = {ph},
            model_points = {ph},
            is_resolved = 1
        WHERE id = {ph}
    """, (
        actual_result, actual_score_leafs, actual_score_opponent,
        user_points, model_points, prediction_id,
    ))

    conn.commit()
    conn.close()


def get_pending_predictions(username: str = None) -> list:
    """Holt offene Tipps. Optional gefiltert nach User."""
    conn = get_connection()
    cursor = conn.cursor()
    ph = _placeholder()
    if username:
        cursor.execute(f"""
            SELECT * FROM predictions
            WHERE is_resolved = 0 AND username = {ph}
            ORDER BY game_date ASC
        """, (username,))
    else:
        cursor.execute("""
            SELECT * FROM predictions
            WHERE is_resolved = 0
            ORDER BY game_date ASC
        """)
    rows = _rows_to_dicts(cursor)
    conn.close()
    return rows


def get_resolved_predictions(username: str = None) -> list:
    """Holt aufgelöste Tipps. Optional gefiltert nach User."""
    conn = get_connection()
    cursor = conn.cursor()
    ph = _placeholder()
    if username:
        cursor.execute(f"""
            SELECT * FROM predictions
            WHERE is_resolved = 1 AND username = {ph}
            ORDER BY game_date DESC
        """, (username,))
    else:
        cursor.execute("""
            SELECT * FROM predictions
            WHERE is_resolved = 1
            ORDER BY game_date DESC
        """)
    rows = _rows_to_dicts(cursor)
    conn.close()
    return rows


def get_all_predictions(username: str = None) -> list:
    """Holt alle Tipps. Optional gefiltert nach User."""
    conn = get_connection()
    cursor = conn.cursor()
    ph = _placeholder()
    if username:
        cursor.execute(f"""
            SELECT * FROM predictions
            WHERE username = {ph}
            ORDER BY game_date DESC
        """, (username,))
    else:
        cursor.execute("SELECT * FROM predictions ORDER BY game_date DESC")
    rows = _rows_to_dicts(cursor)
    conn.close()
    return rows


def get_leaderboard() -> dict:
    """
    Berechnet Multi-User Leaderboard.

    Rückgabe: Dict mit 'users' (Liste aller User-Stats) und 'model' (ML-Stats).
    """
    conn = get_connection()
    cursor = conn.cursor()

    # Pro-User Stats
    cursor.execute("""
        SELECT
            username,
            COUNT(*) as total_games,
            SUM(user_points) as total_points,
            SUM(CASE WHEN user_points > 0 THEN 1 ELSE 0 END) as correct,
            SUM(CASE WHEN user_points = 3 THEN 1 ELSE 0 END) as exact
        FROM predictions
        WHERE is_resolved = 1 AND username != ''
        GROUP BY username
        ORDER BY total_points DESC
    """)
    user_rows = _rows_to_dicts(cursor)

    # ML-Modell Stats (aggregiert über alle Predictions)
    cursor.execute("""
        SELECT
            COUNT(*) as total_games,
            SUM(model_points) as total_points,
            SUM(CASE WHEN model_points > 0 THEN 1 ELSE 0 END) as correct
        FROM predictions
        WHERE is_resolved = 1
    """)
    model_row = cursor.fetchone()
    if USE_POSTGRES:
        columns = [desc[0] for desc in cursor.description]
        model_stats = dict(zip(columns, model_row)) if model_row else {}
    else:
        model_stats = dict(model_row) if model_row else {}
    conn.close()

    # User-Liste mit Accuracy berechnen
    users = []
    for i, u in enumerate(user_rows):
        total = u["total_games"] or 0
        users.append({
            "rank": i + 1,
            "username": u["username"],
            "total_games": total,
            "total_points": u["total_points"] or 0,
            "correct": u["correct"] or 0,
            "exact": u["exact"] or 0,
            "accuracy": round((u["correct"] or 0) / total * 100, 1) if total > 0 else 0,
        })

    # ML-Modell Stats
    model_total = model_stats.get("total_games", 0) or 0
    model = {
        "total_games": model_total,
        "total_points": model_stats.get("total_points", 0) or 0,
        "correct": model_stats.get("correct", 0) or 0,
        "accuracy": round((model_stats.get("correct", 0) or 0) / model_total * 100, 1) if model_total > 0 else 0,
    }

    return {"users": users, "model": model}


def prediction_exists(game_id: int, username: str = None) -> bool:
    """Prüft ob für ein Spiel schon ein Tipp existiert (pro User)."""
    conn = get_connection()
    cursor = conn.cursor()
    ph = _placeholder()
    if username:
        cursor.execute(
            f"SELECT id FROM predictions WHERE game_id = {ph} AND username = {ph}",
            (game_id, username),
        )
    else:
        cursor.execute(
            f"SELECT id FROM predictions WHERE game_id = {ph}",
            (game_id,),
        )
    exists = cursor.fetchone() is not None
    conn.close()
    return exists


# ---- Hauptprogramm ----
if __name__ == "__main__":
    init_db()
    db_type = "PostgreSQL" if USE_POSTGRES else "SQLite"
    print(f"Datenbank ({db_type}) ist bereit!")
    lb = get_leaderboard()
    print(f"\nLeaderboard: {lb}")
