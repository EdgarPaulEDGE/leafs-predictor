"""
database.py – Datenbank für Tipps und Punkte
==============================================
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


def init_db():
    """Erstellt die Datenbank-Tabellen."""
    conn = get_connection()
    cursor = conn.cursor()

    if USE_POSTGRES:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id SERIAL PRIMARY KEY,
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
    else:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
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


def _placeholder():
    """Gibt den richtigen Platzhalter zurück: %s für Postgres, ? für SQLite."""
    return "%s" if USE_POSTGRES else "?"


def add_prediction(
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
    """Speichert einen neuen Tipp."""
    conn = get_connection()
    cursor = conn.cursor()
    ph = _placeholder()

    if USE_POSTGRES:
        cursor.execute(f"""
            INSERT INTO predictions
            (game_id, game_date, opponent, is_home,
             user_prediction, user_score_leafs, user_score_opponent,
             model_prediction, model_win_probability)
            VALUES ({ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph})
            RETURNING id
        """, (
            game_id, game_date, opponent, is_home,
            user_prediction, user_score_leafs, user_score_opponent,
            model_prediction, model_win_probability,
        ))
        prediction_id = cursor.fetchone()[0]
    else:
        cursor.execute(f"""
            INSERT INTO predictions
            (game_id, game_date, opponent, is_home,
             user_prediction, user_score_leafs, user_score_opponent,
             model_prediction, model_win_probability)
            VALUES ({ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph})
        """, (
            game_id, game_date, opponent, is_home,
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


def get_pending_predictions() -> list:
    """Holt alle offenen Tipps."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM predictions
        WHERE is_resolved = 0
        ORDER BY game_date ASC
    """)
    if USE_POSTGRES:
        columns = [desc[0] for desc in cursor.description]
        rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
    else:
        rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


def get_resolved_predictions() -> list:
    """Holt alle aufgelösten Tipps."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM predictions
        WHERE is_resolved = 1
        ORDER BY game_date DESC
    """)
    if USE_POSTGRES:
        columns = [desc[0] for desc in cursor.description]
        rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
    else:
        rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


def get_all_predictions() -> list:
    """Holt alle Tipps."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM predictions ORDER BY game_date DESC")
    if USE_POSTGRES:
        columns = [desc[0] for desc in cursor.description]
        rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
    else:
        rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


def get_leaderboard() -> dict:
    """Berechnet den Punktestand: User vs. Modell."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            COUNT(*) as total_games,
            SUM(user_points) as user_total,
            SUM(model_points) as model_total,
            SUM(CASE WHEN user_points > 0 THEN 1 ELSE 0 END) as user_correct,
            SUM(CASE WHEN model_points > 0 THEN 1 ELSE 0 END) as model_correct
        FROM predictions
        WHERE is_resolved = 1
    """)

    row = cursor.fetchone()
    if USE_POSTGRES:
        result = _dict_from_row(row, cursor)
    else:
        result = dict(row) if row else None
    conn.close()

    if not result or result["total_games"] == 0:
        return {
            "total_games": 0,
            "user_total": 0,
            "model_total": 0,
            "user_correct": 0,
            "model_correct": 0,
            "user_accuracy": 0,
            "model_accuracy": 0,
        }

    total = result["total_games"]
    return {
        "total_games": total,
        "user_total": result["user_total"] or 0,
        "model_total": result["model_total"] or 0,
        "user_correct": result["user_correct"] or 0,
        "model_correct": result["model_correct"] or 0,
        "user_accuracy": round((result["user_correct"] or 0) / total * 100, 1),
        "model_accuracy": round((result["model_correct"] or 0) / total * 100, 1),
    }


def prediction_exists(game_id: int) -> bool:
    """Prüft ob für ein Spiel schon ein Tipp existiert."""
    conn = get_connection()
    cursor = conn.cursor()
    ph = _placeholder()
    cursor.execute(f"SELECT id FROM predictions WHERE game_id = {ph}", (game_id,))
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
