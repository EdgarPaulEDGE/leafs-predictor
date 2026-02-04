"""
model.py – ML-Modell für Leafs Spielvorhersagen (V2 – Upgraded)
================================================================
Lernziele:
- Feature Engineering (viele Features aus Rohdaten ableiten)
- Mehrere ML-Algorithmen vergleichen
- Gradient Boosting (oft das beste für tabellarische Daten)
- Cross-Validation für robuste Bewertung

Modelle die wir testen:
1. Random Forest – Viele Entscheidungsbäume
2. Gradient Boosting – Bäume die voneinander lernen
3. Logistic Regression – Einfach aber solide Baseline
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

MODEL_PATH = "leafs_model.pkl"
DATA_PATH = "leafs_data.csv"

# Alle NHL-Teams mit Codes
NHL_TEAMS = [
    "ANA", "ARI", "BOS", "BUF", "CAR", "CBJ", "CGY", "CHI",
    "COL", "DAL", "DET", "EDM", "FLA", "LAK", "MIN", "MTL",
    "NJD", "NSH", "NYI", "NYR", "OTT", "PHI", "PIT", "SEA",
    "SJS", "STL", "TBL", "UTA", "VAN", "VGK", "WPG", "WSH",
]


def load_data() -> pd.DataFrame:
    """Lädt die CSV-Daten."""
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"'{DATA_PATH}' nicht gefunden! Führe zuerst 'python3 data.py' aus."
        )
    df = pd.read_csv(DATA_PATH)
    print(f"Daten geladen: {len(df)} Spiele")
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature Engineering V2 – deutlich mehr Features.

    Neue Features:
    - rest_days: Ruhetage seit letztem Spiel
    - opp_win_pct: Gegner-Gewinnrate (aus Standings)
    - opp_goals_per_game: Gegner-Tore pro Spiel
    - opp_goals_against_per_game: Gegner-Gegentore pro Spiel
    - opp_l10_wins: Gegner Wins in letzten 10
    - leafs_standing_points: Leafs Punkte in Standings
    - streak: Aktuelle Win/Loss-Streak
    - h2h_win_pct: Head-to-Head Bilanz gegen diesen Gegner
    - goal_diff_rolling: Tordifferenz der letzten Spiele
    - home_win_pct / away_win_pct: Heim/Auswärts-spezifische Winrate
    """
    df = df.copy()
    df = df.sort_values("date").reset_index(drop=True)

    # Ergebnis als Zahl
    df["win"] = (df["result"] == "W").astype(int)

    # Gegner als Nummer
    team_to_num = {team: i for i, team in enumerate(NHL_TEAMS)}
    df["opponent_encoded"] = df["opponent"].map(team_to_num).fillna(-1).astype(int)

    # --- Rolling Stats (Leafs Form) ---
    df["rolling_win_pct"] = df["win"].rolling(window=10, min_periods=5).mean()
    df["rolling_goals_for"] = df["leafs_score"].rolling(window=5, min_periods=3).mean()
    df["rolling_goals_against"] = df["opponent_score"].rolling(window=5, min_periods=3).mean()

    # Tordifferenz der letzten 10 Spiele
    df["goal_diff_rolling"] = (
        df["leafs_score"].rolling(window=10, min_periods=5).sum()
        - df["opponent_score"].rolling(window=10, min_periods=5).sum()
    )

    # --- Win/Loss Streak ---
    streak = []
    current_streak = 0
    for _, row in df.iterrows():
        streak.append(current_streak)
        if row["result"] == "W":
            current_streak = current_streak + 1 if current_streak > 0 else 1
        else:
            current_streak = current_streak - 1 if current_streak < 0 else -1
    df["streak"] = streak

    # --- Head-to-Head Bilanz ---
    h2h_records: dict[str, list[int]] = {}
    h2h_win_pcts = []
    for _, row in df.iterrows():
        opp = row["opponent"]
        if opp not in h2h_records:
            h2h_records[opp] = []
        # Bisherige H2H-Bilanz VOR diesem Spiel
        games = h2h_records[opp]
        if len(games) >= 2:
            h2h_win_pcts.append(sum(games) / len(games))
        else:
            h2h_win_pcts.append(0.5)  # Default wenn zu wenig Daten
        h2h_records[opp].append(row["win"])
    df["h2h_win_pct"] = h2h_win_pcts

    # --- Heim/Auswärts spezifische Winrate ---
    home_wins: list[int] = []
    away_wins: list[int] = []
    home_results: list[int] = []
    away_results: list[int] = []
    for _, row in df.iterrows():
        # Berechne VOR diesem Spiel
        if len(home_results) >= 5:
            home_wins.append(sum(home_results[-20:]) / len(home_results[-20:]))
        else:
            home_wins.append(0.5)
        if len(away_results) >= 5:
            away_wins.append(sum(away_results[-20:]) / len(away_results[-20:]))
        else:
            away_wins.append(0.5)

        if row["is_home"] == 1:
            home_results.append(row["win"])
        else:
            away_results.append(row["win"])

    df["home_win_pct"] = home_wins
    df["away_win_pct"] = away_wins

    # Rest days (falls schon in den Daten, sonst berechnen)
    if "rest_days" not in df.columns:
        df["rest_days"] = 1

    # Back-to-back? (Ruhetag = 1)
    df["is_back_to_back"] = (df["rest_days"] == 1).astype(int)

    # Gegner-Stats (falls aus data.py vorhanden, sonst Defaults)
    for col, default in [
        ("opp_win_pct", 0.5),
        ("opp_goals_per_game", 3.0),
        ("opp_goals_against_per_game", 3.0),
        ("opp_points", 50),
        ("opp_l10_wins", 5),
        ("leafs_standing_points", 50),
    ]:
        if col not in df.columns:
            df[col] = default

    # Punkte-Differenz (Leafs vs Gegner Standings)
    df["standing_diff"] = df["leafs_standing_points"] - df["opp_points"]

    # Advanced Stats Defaults (falls nicht in den Daten)
    adv_defaults = {
        "leafs_pp_pct": 0.20, "leafs_pk_pct": 0.80,
        "leafs_corsi_pct": 0.50, "leafs_fenwick_pct": 0.50,
        "leafs_pdo": 1.00, "leafs_shots_pg": 30.0,
        "leafs_shots_against_pg": 30.0, "leafs_faceoff_pct": 0.50,
        "leafs_save_pct": 0.91, "leafs_shooting_pct": 0.09,
        "leafs_zone_start_pct": 0.50,
        "opp_pp_pct": 0.20, "opp_pk_pct": 0.80,
        "opp_corsi_pct": 0.50, "opp_pdo": 1.00,
        "opp_save_pct": 0.91,
    }
    for col, default in adv_defaults.items():
        if col not in df.columns:
            df[col] = default

    # Abgeleitete Differenz-Features
    df["corsi_diff"] = df["leafs_corsi_pct"] - df["opp_corsi_pct"]
    df["special_teams_diff"] = (
        (df["leafs_pp_pct"] + df["leafs_pk_pct"])
        - (df["opp_pp_pct"] + df["opp_pk_pct"])
    )
    df["shots_diff"] = df["leafs_shots_pg"] - df["leafs_shots_against_pg"]

    # Zeilen ohne Rolling-Daten entfernen
    df = df.dropna(subset=["rolling_win_pct", "rolling_goals_for", "rolling_goals_against"])

    return df


def get_feature_columns() -> list[str]:
    """Liste der Features die das Modell nutzt."""
    return [
        # Basis
        "is_home",
        "opponent_encoded",
        # Leafs Form (Rolling Stats)
        "rolling_win_pct",
        "rolling_goals_for",
        "rolling_goals_against",
        "goal_diff_rolling",
        "streak",
        "h2h_win_pct",
        # Situational
        "rest_days",
        "is_back_to_back",
        # Gegner Standings
        "opp_win_pct",
        "opp_goals_per_game",
        "opp_goals_against_per_game",
        "opp_l10_wins",
        "standing_diff",
        # Advanced Stats (Leafs)
        "leafs_pp_pct",
        "leafs_pk_pct",
        "leafs_corsi_pct",
        "leafs_fenwick_pct",
        "leafs_pdo",
        "leafs_shots_pg",
        "leafs_shots_against_pg",
        "leafs_faceoff_pct",
        "leafs_save_pct",
        "leafs_shooting_pct",
        "leafs_zone_start_pct",
        # Advanced Stats (Gegner)
        "opp_pp_pct",
        "opp_pk_pct",
        "opp_corsi_pct",
        "opp_pdo",
        "opp_save_pct",
        # Differenzen (abgeleitet)
        "corsi_diff",
        "special_teams_diff",
        "shots_diff",
    ]


def train_all_models(df: pd.DataFrame) -> tuple:
    """
    Trainiert 3 verschiedene Modelle und wählt das beste.

    1. Gradient Boosting (oft am besten für tabellarische Daten)
    2. Random Forest
    3. Logistic Regression (Baseline)
    """
    features = get_feature_columns()
    X = df[features]
    y = df["win"]

    # 80/20 Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"\nTraining: {len(X_train)} Spiele | Test: {len(X_test)} Spiele")
    print(f"Features: {len(features)}")
    print("=" * 60)

    models = {
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42,
        ),
    }

    results = {}
    best_model = None
    best_accuracy = 0
    best_name = ""
    best_scaler = None

    # Scaler für Logistic Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_all_scaled = scaler.transform(X)

    for name, model in models.items():
        print(f"\n--- {name} ---")

        needs_scaling = name == "Logistic Regression"

        if needs_scaling:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            cv_scores = cross_val_score(model, X_all_scaled, y, cv=5)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            cv_scores = cross_val_score(model, X, y, cv=5)

        accuracy = accuracy_score(y_test, y_pred)
        cv_mean = cv_scores.mean()

        print(f"  Test-Genauigkeit:  {accuracy:.1%}")
        print(f"  Cross-Val (5-fold): {cv_mean:.1%} (+/- {cv_scores.std():.1%})")

        results[name] = {
            "accuracy": accuracy,
            "cv_mean": cv_mean,
            "model": model,
        }

        if cv_mean > best_accuracy and name != "Logistic Regression":
            # Logistic Regression wird ausgeschlossen weil sie
            # extrem overconfident Probabilities liefert (95%+),
            # was fuer ein Tipp-Spiel unrealistisch ist.
            # Baum-Modelle geben kalibriertere Wahrscheinlichkeiten.
            best_accuracy = cv_mean
            best_model = model
            best_name = name
            best_scaler = None  # Baum-Modelle brauchen keinen Scaler

    print(f"\n{'=' * 60}")
    print(f"BESTES MODELL: {best_name} (CV: {best_accuracy:.1%})")
    print(f"{'=' * 60}")

    # Detaillierter Report für bestes Modell
    if best_scaler is not None:
        y_pred = best_model.predict(X_test_scaled)
    else:
        y_pred = best_model.predict(X_test)
    print(f"\nDetailreport ({best_name}):")
    print(classification_report(y_test, y_pred, target_names=["Loss", "Win"]))

    # Feature Importance (nur für Baum-Modelle)
    if hasattr(best_model, "feature_importances_"):
        print("Feature Importance:")
        for feat, imp in sorted(
            zip(features, best_model.feature_importances_), key=lambda x: -x[1]
        ):
            bar = "#" * int(imp * 50)
            print(f"  {feat:30s} {imp:.3f} {bar}")

    return best_model, best_accuracy, best_name, best_scaler


def save_model(model, scaler, model_name: str = ""):
    """Speichert das trainierte Modell + Scaler."""
    joblib.dump({"model": model, "scaler": scaler}, MODEL_PATH)
    print(f"\nModell gespeichert in '{MODEL_PATH}' ({model_name})")


def load_model():
    """Lädt ein gespeichertes Modell + Scaler."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"'{MODEL_PATH}' nicht gefunden! Führe zuerst Training aus."
        )
    data = joblib.load(MODEL_PATH)
    if isinstance(data, dict):
        return data["model"], data["scaler"]
    # Fallback für alte Modelle ohne Scaler
    return data, None


def predict_game(
    model,
    opponent: str,
    is_home: bool,
    recent_games: pd.DataFrame,
    scaler=None,
) -> dict:
    """
    Macht eine Vorhersage für ein kommendes Spiel.
    Nutzt jetzt alle V2-Features.
    """
    team_to_num = {team: i for i, team in enumerate(NHL_TEAMS)}

    # Rolling Stats aus den letzten Spielen
    recent = recent_games.tail(10)
    wins = (recent["result"] == "W").sum()
    rolling_win_pct = wins / len(recent) if len(recent) > 0 else 0.5

    last5 = recent_games.tail(5)
    rolling_goals_for = last5["leafs_score"].mean() if len(last5) > 0 else 3.0
    rolling_goals_against = last5["opponent_score"].mean() if len(last5) > 0 else 3.0

    goal_diff = (
        recent["leafs_score"].sum() - recent["opponent_score"].sum()
    ) if len(recent) > 0 else 0

    # Streak berechnen
    streak = 0
    for r in recent_games.tail(20)["result"].values[::-1]:
        if r == "W" and streak >= 0:
            streak += 1
        elif r == "L" and streak <= 0:
            streak -= 1
        else:
            break

    # H2H gegen diesen Gegner
    h2h_games = recent_games[recent_games["opponent"] == opponent]
    if len(h2h_games) >= 2:
        h2h_win_pct = (h2h_games["result"] == "W").mean()
    else:
        h2h_win_pct = 0.5

    # Heim/Auswärts Winrate
    if is_home:
        venue_games = recent_games[recent_games["is_home"] == 1].tail(20)
    else:
        venue_games = recent_games[recent_games["is_home"] == 0].tail(20)

    # Rest days und Gegner-Stats aus den Daten
    rest_days = 2  # Default für Vorhersage
    opp_win_pct = 0.5
    opp_goals_per_game = 3.0
    opp_goals_against_per_game = 3.0
    opp_l10_wins = 5
    standing_diff = 0

    # Gegner-Stats + Advanced Stats aus den neuesten Daten
    if "opp_win_pct" in recent_games.columns:
        opp_recent = recent_games[recent_games["opponent"] == opponent].tail(1)
        if len(opp_recent) > 0:
            opp_win_pct = opp_recent["opp_win_pct"].values[0]
            opp_goals_per_game = opp_recent["opp_goals_per_game"].values[0]
            opp_goals_against_per_game = opp_recent["opp_goals_against_per_game"].values[0]
            if "opp_l10_wins" in opp_recent.columns:
                opp_l10_wins = opp_recent["opp_l10_wins"].values[0]
            if "leafs_standing_points" in opp_recent.columns and "opp_points" in opp_recent.columns:
                standing_diff = (
                    opp_recent["leafs_standing_points"].values[0]
                    - opp_recent["opp_points"].values[0]
                )

    # Advanced Stats aus den neuesten Daten
    last_game = recent_games.tail(1)
    def get_val(col, default):
        if col in recent_games.columns and len(last_game) > 0:
            return last_game[col].values[0]
        return default

    leafs_pp_pct = get_val("leafs_pp_pct", 0.20)
    leafs_pk_pct = get_val("leafs_pk_pct", 0.80)
    leafs_corsi_pct = get_val("leafs_corsi_pct", 0.50)
    leafs_fenwick_pct = get_val("leafs_fenwick_pct", 0.50)
    leafs_pdo = get_val("leafs_pdo", 1.00)
    leafs_shots_pg = get_val("leafs_shots_pg", 30.0)
    leafs_shots_against_pg = get_val("leafs_shots_against_pg", 30.0)
    leafs_faceoff_pct = get_val("leafs_faceoff_pct", 0.50)
    leafs_save_pct = get_val("leafs_save_pct", 0.91)
    leafs_shooting_pct = get_val("leafs_shooting_pct", 0.09)
    leafs_zone_start_pct = get_val("leafs_zone_start_pct", 0.50)

    # Gegner Advanced Stats
    opp_pp_pct = 0.20
    opp_pk_pct = 0.80
    opp_corsi_pct = 0.50
    opp_pdo_val = 1.00
    opp_save_pct_val = 0.91
    if "opp_pp_pct" in recent_games.columns:
        opp_recent = recent_games[recent_games["opponent"] == opponent].tail(1)
        if len(opp_recent) > 0:
            opp_pp_pct = opp_recent["opp_pp_pct"].values[0]
            opp_pk_pct = opp_recent["opp_pk_pct"].values[0]
            opp_corsi_pct = opp_recent["opp_corsi_pct"].values[0]
            opp_pdo_val = opp_recent["opp_pdo"].values[0]
            opp_save_pct_val = opp_recent["opp_save_pct"].values[0]

    # Abgeleitete Features
    corsi_diff = leafs_corsi_pct - opp_corsi_pct
    special_teams_diff = (leafs_pp_pct + leafs_pk_pct) - (opp_pp_pct + opp_pk_pct)
    shots_diff = leafs_shots_pg - leafs_shots_against_pg

    # Feature-Vektor bauen
    features = pd.DataFrame([{
        "is_home": 1 if is_home else 0,
        "opponent_encoded": team_to_num.get(opponent, -1),
        "rolling_win_pct": rolling_win_pct,
        "rolling_goals_for": rolling_goals_for,
        "rolling_goals_against": rolling_goals_against,
        "goal_diff_rolling": goal_diff,
        "streak": streak,
        "h2h_win_pct": h2h_win_pct,
        "rest_days": rest_days,
        "is_back_to_back": 0,
        "opp_win_pct": opp_win_pct,
        "opp_goals_per_game": opp_goals_per_game,
        "opp_goals_against_per_game": opp_goals_against_per_game,
        "opp_l10_wins": opp_l10_wins,
        "standing_diff": standing_diff,
        "leafs_pp_pct": leafs_pp_pct,
        "leafs_pk_pct": leafs_pk_pct,
        "leafs_corsi_pct": leafs_corsi_pct,
        "leafs_fenwick_pct": leafs_fenwick_pct,
        "leafs_pdo": leafs_pdo,
        "leafs_shots_pg": leafs_shots_pg,
        "leafs_shots_against_pg": leafs_shots_against_pg,
        "leafs_faceoff_pct": leafs_faceoff_pct,
        "leafs_save_pct": leafs_save_pct,
        "leafs_shooting_pct": leafs_shooting_pct,
        "leafs_zone_start_pct": leafs_zone_start_pct,
        "opp_pp_pct": opp_pp_pct,
        "opp_pk_pct": opp_pk_pct,
        "opp_corsi_pct": opp_corsi_pct,
        "opp_pdo": opp_pdo_val,
        "opp_save_pct": opp_save_pct_val,
        "corsi_diff": corsi_diff,
        "special_teams_diff": special_teams_diff,
        "shots_diff": shots_diff,
    }])

    # Skalieren wenn Scaler vorhanden
    if scaler is not None:
        features = pd.DataFrame(
            scaler.transform(features), columns=features.columns
        )

    # Vorhersage
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]

    win_prob = probabilities[1]
    loss_prob = probabilities[0]


    return {
        "prediction": "W" if prediction == 1 else "L",
        "win_probability": round(win_prob * 100, 1),
        "loss_probability": round(loss_prob * 100, 1),
        "confidence": round(max(win_prob, loss_prob) * 100, 1),
    }


# ---- Hauptprogramm ----
if __name__ == "__main__":
    # Daten laden
    df = load_data()

    # Features hinzufügen
    df = add_features(df)
    print(f"Features berechnet: {len(df)} Spiele mit {len(get_feature_columns())} Features")

    # Alle Modelle trainieren und bestes wählen
    model, accuracy, model_name, scaler = train_all_models(df)

    # Bestes Modell speichern
    save_model(model, scaler, model_name)

    # Beispiel-Vorhersagen
    print("\n--- Beispiel-Vorhersagen ---")
    for opp, home in [("BOS", True), ("MTL", False), ("FLA", True), ("EDM", False)]:
        pred = predict_game(model, opp, home, df, scaler)
        venue = "Heim" if home else "Ausw."
        print(f"  Leafs vs {opp} ({venue}): {pred['prediction']} "
              f"({pred['win_probability']}% Win, Confidence: {pred['confidence']}%)")
