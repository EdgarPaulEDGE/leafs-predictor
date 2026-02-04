"""
data.py – Historische Leafs-Spieldaten von der NHL API holen (V3)
==================================================================
Lernziele:
- API-Aufrufe mit 'requests'
- JSON-Daten verarbeiten
- Daten als CSV speichern mit 'pandas'
- Standings-Daten für Gegner-Stärke
- Advanced Stats: Corsi, Fenwick, PP%, PK%, PDO, Shots, Faceoffs

Zwei NHL APIs:
- api-web.nhle.com/v1     -> Spielplan, Standings, Boxscores
- api.nhle.com/stats/rest -> Advanced Stats (Corsi, PP%, PK%, etc.)
"""

import requests
import pandas as pd
import time
from datetime import datetime, timedelta

# NHL API Base URLs
BASE_URL = "https://api-web.nhle.com/v1"
STATS_URL = "https://api.nhle.com/stats/rest/en/team"

# Toronto Maple Leafs
TEAM = "TOR"
TEAM_FULL = "Toronto Maple Leafs"


def get_schedule(season: str) -> list[dict]:
    """Holt den Spielplan der Leafs für eine Saison."""
    url = f"{BASE_URL}/club-schedule-season/{TEAM}/{season}"
    print(f"Lade Spielplan für Saison {season}...")
    response = requests.get(url, timeout=15)
    response.raise_for_status()
    data = response.json()
    games = data.get("games", [])
    print(f"  -> {len(games)} Spiele gefunden")
    return games


def get_standings(date: str) -> dict[str, dict]:
    """Holt die Standings zu einem bestimmten Datum."""
    url = f"{BASE_URL}/standings/{date}"
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        data = response.json()

        standings = {}
        for entry in data.get("standings", []):
            abbrev = entry.get("teamAbbrev", {}).get("default", "")
            if not abbrev:
                continue
            standings[abbrev] = {
                "wins": entry.get("wins", 0),
                "losses": entry.get("losses", 0),
                "ot_losses": entry.get("otLosses", 0),
                "points": entry.get("points", 0),
                "games_played": entry.get("gamesPlayed", 1),
                "goals_for": entry.get("goalFor", 0),
                "goals_against": entry.get("goalAgainst", 0),
                "streak_count": entry.get("streakCount", 0),
                "streak_code": entry.get("streakCode", ""),
                "home_wins": entry.get("homeWins", 0),
                "home_losses": entry.get("homeLosses", 0),
                "road_wins": entry.get("roadWins", 0),
                "road_losses": entry.get("roadLosses", 0),
                "l10_wins": entry.get("l10Wins", 0),
                "l10_losses": entry.get("l10Losses", 0),
            }
        return standings
    except requests.RequestException:
        return {}


def get_advanced_stats(season: str) -> dict[str, dict]:
    """
    Holt Advanced Stats für ALLE Teams einer Saison.

    Stats die wir holen:
    - Team Summary: PP%, PK%, Shots/Game, Faceoff%
    - Team Percentages: Corsi (CF%), Fenwick (FF%), PDO, Save%

    Rückgabe:
        Dict von Team-Fullname -> Advanced Stats
    """
    season_id = season  # z.B. "20252026"
    advanced = {}

    # 1) Team Summary -> PP%, PK%, Shots, Faceoffs
    try:
        url = f"{STATS_URL}/summary?cayenneExp=seasonId={season_id}"
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        for team in resp.json().get("data", []):
            name = team.get("teamFullName", "")
            if not name:
                continue
            advanced[name] = {
                "pp_pct": team.get("powerPlayPct", 0.0),
                "pk_pct": team.get("penaltyKillPct", 0.0),
                "shots_per_game": team.get("shotsForPerGame", 30.0),
                "shots_against_per_game": team.get("shotsAgainstPerGame", 30.0),
                "faceoff_pct": team.get("faceoffWinPct", 0.5),
                "goals_for_per_game": team.get("goalsForPerGame", 3.0),
                "goals_against_per_game": team.get("goalsAgainstPerGame", 3.0),
            }
        time.sleep(0.3)
    except requests.RequestException as e:
        print(f"    Fehler bei Team Summary: {e}")

    # 2) Team Percentages -> Corsi, Fenwick, PDO, Save%
    try:
        url = f"{STATS_URL}/percentages?cayenneExp=seasonId={season_id}"
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        for team in resp.json().get("data", []):
            name = team.get("teamFullName", "")
            if name in advanced:
                advanced[name].update({
                    "corsi_pct": team.get("satPct", 0.5),       # CF%
                    "corsi_pct_close": team.get("satPctClose", 0.5),  # CF% bei knappem Spielstand
                    "fenwick_pct": team.get("usatPct", 0.5),    # FF%
                    "pdo": team.get("shootingPlusSavePct5v5", 1.0),
                    "save_pct_5v5": team.get("savePct5v5", 0.91),
                    "shooting_pct_5v5": team.get("shootingPct5v5", 0.09),
                    "zone_start_pct": team.get("zoneStartPct5v5", 0.5),
                })
    except requests.RequestException as e:
        print(f"    Fehler bei Team Percentages: {e}")

    return advanced


# Mapping von Team-Abbrev zu Full Name (für Advanced Stats Lookup)
TEAM_NAMES = {
    "ANA": "Anaheim Ducks", "ARI": "Arizona Coyotes", "BOS": "Boston Bruins",
    "BUF": "Buffalo Sabres", "CAR": "Carolina Hurricanes", "CBJ": "Columbus Blue Jackets",
    "CGY": "Calgary Flames", "CHI": "Chicago Blackhawks", "COL": "Colorado Avalanche",
    "DAL": "Dallas Stars", "DET": "Detroit Red Wings", "EDM": "Edmonton Oilers",
    "FLA": "Florida Panthers", "LAK": "Los Angeles Kings", "MIN": "Minnesota Wild",
    "MTL": "Montreal Canadiens", "NJD": "New Jersey Devils", "NSH": "Nashville Predators",
    "NYI": "New York Islanders", "NYR": "New York Rangers", "OTT": "Ottawa Senators",
    "PHI": "Philadelphia Flyers", "PIT": "Pittsburgh Penguins", "SEA": "Seattle Kraken",
    "SJS": "San Jose Sharks", "STL": "St. Louis Blues", "TBL": "Tampa Bay Lightning",
    "UTA": "Utah Hockey Club", "VAN": "Vancouver Canucks", "VGK": "Vegas Golden Knights",
    "WPG": "Winnipeg Jets", "WSH": "Washington Capitals",
}


def parse_games(games: list[dict], standings_cache: dict, adv_stats: dict) -> list[dict]:
    """
    Verarbeitet die rohen API-Daten in ein sauberes Format.
    Inkludiert Gegner-Stärke, Ruhetage und Advanced Stats.
    """
    parsed = []
    prev_date = None

    # Leafs Advanced Stats
    leafs_adv = adv_stats.get(TEAM_FULL, {})

    for game in games:
        state = game.get("gameState", "")
        if state not in ("OFF", "FINAL"):
            continue

        game_id = game.get("id", 0)
        game_date = game.get("gameDate", "")
        game_type = game.get("gameType", 0)

        if game_type != 2:
            continue

        home_team = game.get("homeTeam", {})
        away_team = game.get("awayTeam", {})
        home_abbrev = home_team.get("abbrev", "")
        away_abbrev = away_team.get("abbrev", "")
        home_score = home_team.get("score", 0)
        away_score = away_team.get("score", 0)

        is_home = home_abbrev == TEAM
        if is_home:
            leafs_score = home_score
            opponent_score = away_score
            opponent = away_abbrev
        else:
            leafs_score = away_score
            opponent_score = home_score
            opponent = home_abbrev

        result = "W" if leafs_score > opponent_score else "L"

        # Ruhetage
        rest_days = 1
        if prev_date and game_date:
            try:
                current = datetime.strptime(game_date, "%Y-%m-%d")
                previous = datetime.strptime(prev_date, "%Y-%m-%d")
                rest_days = (current - previous).days
            except ValueError:
                rest_days = 1
        prev_date = game_date

        # Gegner-Stärke aus Standings
        opp_win_pct = 0.5
        opp_goals_per_game = 3.0
        opp_goals_against_per_game = 3.0
        opp_points = 50
        opp_l10_wins = 5
        leafs_standing_points = 50

        if game_date in standings_cache:
            standings = standings_cache[game_date]
            opp_stats = standings.get(opponent, {})
            if opp_stats and opp_stats.get("games_played", 0) > 0:
                gp = opp_stats["games_played"]
                opp_win_pct = opp_stats["wins"] / gp
                opp_goals_per_game = opp_stats["goals_for"] / gp
                opp_goals_against_per_game = opp_stats["goals_against"] / gp
                opp_points = opp_stats["points"]
                opp_l10_wins = opp_stats.get("l10_wins", 5)
            leafs_stats = standings.get(TEAM, {})
            if leafs_stats:
                leafs_standing_points = leafs_stats.get("points", 50)

        # Advanced Stats (Leafs)
        leafs_pp = leafs_adv.get("pp_pct", 0.20)
        leafs_pk = leafs_adv.get("pk_pct", 0.80)
        leafs_corsi = leafs_adv.get("corsi_pct", 0.50)
        leafs_fenwick = leafs_adv.get("fenwick_pct", 0.50)
        leafs_pdo = leafs_adv.get("pdo", 1.00)
        leafs_shots_pg = leafs_adv.get("shots_per_game", 30.0)
        leafs_shots_against_pg = leafs_adv.get("shots_against_per_game", 30.0)
        leafs_faceoff = leafs_adv.get("faceoff_pct", 0.50)
        leafs_save_pct = leafs_adv.get("save_pct_5v5", 0.91)
        leafs_shooting_pct = leafs_adv.get("shooting_pct_5v5", 0.09)
        leafs_zone_start = leafs_adv.get("zone_start_pct", 0.50)

        # Advanced Stats (Gegner)
        opp_full_name = TEAM_NAMES.get(opponent, "")
        opp_adv = adv_stats.get(opp_full_name, {})
        opp_pp = opp_adv.get("pp_pct", 0.20)
        opp_pk = opp_adv.get("pk_pct", 0.80)
        opp_corsi = opp_adv.get("corsi_pct", 0.50)
        opp_pdo = opp_adv.get("pdo", 1.00)
        opp_save_pct = opp_adv.get("save_pct_5v5", 0.91)

        parsed.append({
            "game_id": game_id,
            "date": game_date,
            "opponent": opponent,
            "is_home": 1 if is_home else 0,
            "leafs_score": leafs_score,
            "opponent_score": opponent_score,
            "result": result,
            "total_goals": leafs_score + opponent_score,
            "rest_days": min(rest_days, 7),
            # Standings
            "opp_win_pct": round(opp_win_pct, 3),
            "opp_goals_per_game": round(opp_goals_per_game, 2),
            "opp_goals_against_per_game": round(opp_goals_against_per_game, 2),
            "opp_points": opp_points,
            "opp_l10_wins": opp_l10_wins,
            "leafs_standing_points": leafs_standing_points,
            # Leafs Advanced Stats
            "leafs_pp_pct": round(leafs_pp, 4),
            "leafs_pk_pct": round(leafs_pk, 4),
            "leafs_corsi_pct": round(leafs_corsi, 4),
            "leafs_fenwick_pct": round(leafs_fenwick, 4),
            "leafs_pdo": round(leafs_pdo, 4),
            "leafs_shots_pg": round(leafs_shots_pg, 2),
            "leafs_shots_against_pg": round(leafs_shots_against_pg, 2),
            "leafs_faceoff_pct": round(leafs_faceoff, 4),
            "leafs_save_pct": round(leafs_save_pct, 4),
            "leafs_shooting_pct": round(leafs_shooting_pct, 4),
            "leafs_zone_start_pct": round(leafs_zone_start, 4),
            # Gegner Advanced Stats
            "opp_pp_pct": round(opp_pp, 4),
            "opp_pk_pct": round(opp_pk, 4),
            "opp_corsi_pct": round(opp_corsi, 4),
            "opp_pdo": round(opp_pdo, 4),
            "opp_save_pct": round(opp_save_pct, 4),
        })

    return parsed


def collect_data(seasons: list[str]) -> pd.DataFrame:
    """Sammelt Daten über mehrere Saisons."""
    all_games = []
    standings_cache = {}
    sampled_dates = set()

    for season in seasons:
        try:
            games = get_schedule(season)

            # Advanced Stats für diese Saison laden
            print(f"  Lade Advanced Stats...")
            adv_stats = get_advanced_stats(season)
            if adv_stats:
                print(f"    -> {len(adv_stats)} Teams mit Advanced Stats")
            else:
                print(f"    -> Keine Advanced Stats verfügbar")

            # Standings Snapshots laden
            for game in games:
                gd = game.get("gameDate", "")
                if gd:
                    try:
                        d = datetime.strptime(gd, "%Y-%m-%d")
                        nearest = d.replace(day=1 if d.day <= 7 else 15)
                        key = nearest.strftime("%Y-%m-%d")
                        if key not in standings_cache:
                            sampled_dates.add(key)
                    except ValueError:
                        pass

            print(f"  Lade {len(sampled_dates)} Standings-Snapshots...")
            for sd in sorted(sampled_dates):
                if sd not in standings_cache:
                    standings = get_standings(sd)
                    if standings:
                        standings_cache[sd] = standings
                    time.sleep(0.3)

            # Nächstgelegene Standings zuweisen
            for game in games:
                gd = game.get("gameDate", "")
                if gd and gd not in standings_cache:
                    try:
                        game_d = datetime.strptime(gd, "%Y-%m-%d")
                        best_key = None
                        best_diff = 999
                        for cached_date in standings_cache:
                            cd = datetime.strptime(cached_date, "%Y-%m-%d")
                            diff = abs((game_d - cd).days)
                            if diff < best_diff:
                                best_diff = diff
                                best_key = cached_date
                        if best_key:
                            standings_cache[gd] = standings_cache[best_key]
                    except ValueError:
                        pass

            parsed = parse_games(games, standings_cache, adv_stats)
            all_games.extend(parsed)
            sampled_dates.clear()
            time.sleep(0.5)
        except requests.RequestException as e:
            print(f"  Fehler bei Saison {season}: {e}")

    df = pd.DataFrame(all_games)
    print(f"\nGesamt: {len(df)} Spiele gesammelt")
    return df


def save_data(df: pd.DataFrame, filename: str = "leafs_data.csv"):
    """Speichert die Daten als CSV-Datei."""
    df.to_csv(filename, index=False)
    print(f"Daten gespeichert in '{filename}'")


# ---- Hauptprogramm ----
if __name__ == "__main__":
    seasons = [
        "20172018",
        "20182019",
        "20192020",
        "20202021",
        "20212022",
        "20222023",
        "20232024",
        "20242025",
        "20252026",
    ]

    df = collect_data(seasons)

    if not df.empty:
        save_data(df)
        print("\n--- Vorschau der Daten ---")
        print(df.head(5).to_string())
        print(f"\nSpalten ({len(df.columns)}):")
        for col in df.columns:
            print(f"  {col}")
        print(f"\nLeafs Bilanz: {len(df[df['result'] == 'W'])}W - {len(df[df['result'] == 'L'])}L")
    else:
        print("Keine Daten gefunden!")
