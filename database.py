"""
database.py – SQLite Datenbank für Tipps und Punkte
=====================================================
Lernziele:
- SQL Grundlagen (CREATE, INSERT, SELECT, UPDATE)
- SQLite mit Python nutzen
- Daten speichern und abfragen

SQLite ist eine dateibasierte Datenbank – kein Server nötig!
Die Datenbank wird als einzelne Datei gespeichert.
"""

import sqlite3
import os

DB_PATH = "leafs_game.db"


def get_connection() -> sqlite3.Connection:
    """Erstellt eine Verbindung zur Datenbank."""
    conn = sqlite3.connect(DB_PATH)
    # Row-Modus: Ergebnisse als Dictionary statt Tuple
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """
    Erstellt die Datenbank-Tabellen.

    Tabellen:
    - predictions: Alle Tipps (User + Modell)
    - scores: Punktestand-Übersicht
    """
    conn = get_connection()
    cursor = conn.cursor()

    # Tabelle für Tipps/Vorhersagen
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
    print("Datenbank initialisiert!")


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
    """
    Speichert einen neuen Tipp.

    Rückgabe: ID des neuen Eintrags
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO predictions
        (game_id, game_date, opponent, is_home,
         user_prediction, user_score_leafs, user_score_opponent,
         model_prediction, model_win_probability)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
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
    """
    Löst einen Tipp auf nachdem das Spiel vorbei ist.

    Punkte:
    - 3 Punkte: Exaktes Ergebnis richtig
    - 1 Punkt:  Richtige Tendenz (Win/Loss)
    - 0 Punkte: Falsch
    """
    conn = get_connection()
    cursor = conn.cursor()

    # Tipp laden
    cursor.execute("SELECT * FROM predictions WHERE id = ?", (prediction_id,))
    pred = cursor.fetchone()

    if not pred:
        conn.close()
        return

    # User-Punkte berechnen
    user_points = 0
    if pred["user_prediction"] == actual_result:
        user_points = 1  # Richtige Tendenz
        if (pred["user_score_leafs"] == actual_score_leafs and
                pred["user_score_opponent"] == actual_score_opponent):
            user_points = 3  # Exaktes Ergebnis!

    # Modell-Punkte berechnen
    model_points = 0
    if pred["model_prediction"] == actual_result:
        model_points = 1  # Richtige Tendenz

    # Update in Datenbank
    cursor.execute("""
        UPDATE predictions
        SET actual_result = ?,
            actual_score_leafs = ?,
            actual_score_opponent = ?,
            user_points = ?,
            model_points = ?,
            is_resolved = 1
        WHERE id = ?
    """, (
        actual_result, actual_score_leafs, actual_score_opponent,
        user_points, model_points, prediction_id,
    ))

    conn.commit()
    conn.close()


def get_pending_predictions() -> list[dict]:
    """Holt alle offenen Tipps (noch nicht aufgelöst)."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM predictions
        WHERE is_resolved = 0
        ORDER BY game_date ASC
    """)
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


def get_resolved_predictions() -> list[dict]:
    """Holt alle aufgelösten Tipps."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM predictions
        WHERE is_resolved = 1
        ORDER BY game_date DESC
    """)
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


def get_all_predictions() -> list[dict]:
    """Holt alle Tipps."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM predictions ORDER BY game_date DESC")
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


def get_leaderboard() -> dict:
    """
    Berechnet den Punktestand: User vs. Modell.

    Rückgabe:
        Dict mit total_games, user_points, model_points, etc.
    """
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
    conn.close()

    if not row or row["total_games"] == 0:
        return {
            "total_games": 0,
            "user_total": 0,
            "model_total": 0,
            "user_correct": 0,
            "model_correct": 0,
            "user_accuracy": 0,
            "model_accuracy": 0,
        }

    total = row["total_games"]
    return {
        "total_games": total,
        "user_total": row["user_total"] or 0,
        "model_total": row["model_total"] or 0,
        "user_correct": row["user_correct"] or 0,
        "model_correct": row["model_correct"] or 0,
        "user_accuracy": round((row["user_correct"] or 0) / total * 100, 1),
        "model_accuracy": round((row["model_correct"] or 0) / total * 100, 1),
    }


def prediction_exists(game_id: int) -> bool:
    """Prüft ob für ein Spiel schon ein Tipp existiert."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM predictions WHERE game_id = ?", (game_id,))
    exists = cursor.fetchone() is not None
    conn.close()
    return exists


# ---- Hauptprogramm ----
if __name__ == "__main__":
    init_db()
    print(f"Datenbank '{DB_PATH}' ist bereit!")

    # Test: Leaderboard anzeigen
    lb = get_leaderboard()
    print(f"\nLeaderboard: {lb}")
