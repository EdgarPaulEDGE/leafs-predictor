"""
app.py – Flask Web-App für das Leafs Prediction Game
======================================================
Lernziele:
- Web-Entwicklung mit Flask
- Routen und HTTP-Methoden (GET/POST)
- Templates mit Jinja2
- Formulare verarbeiten

Starten mit: python app.py
Dann im Browser: http://localhost:5000
"""

import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import threading
import requests
from flask import Flask, render_template, request, redirect, url_for, flash, session, g
from flask_compress import Compress
import pandas as pd
from datetime import datetime

from database import (
    init_db,
    add_prediction,
    get_pending_predictions,
    get_resolved_predictions,
    get_leaderboard,
    prediction_exists,
    resolve_prediction,
    get_all_predictions,
    get_or_create_user,
)

# Flask-App erstellen
app = Flask(__name__)
app.secret_key = "leafs-prediction-game-2026"  # Für Flash-Messages & Sessions

# Gzip-Kompression (66KB CSS → ~15KB, HTML ~70% kleiner)
Compress(app)
app.config["COMPRESS_MIMETYPES"] = [
    "text/html", "text/css", "text/javascript",
    "application/javascript", "application/json",
    "image/svg+xml",
]
app.config["COMPRESS_MIN_SIZE"] = 500  # Nur Dateien > 500 Bytes komprimieren

# Statische Assets 1 Tag cachen (CSS, SVG, JS)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 86400


# ---- User Session ----
@app.before_request
def load_current_user():
    """Lädt den aktuellen User in g.username für alle Templates."""
    g.username = session.get("username")
    # Automatisch zum Login weiterleiten wenn nicht eingeloggt
    # (außer für Login-Seite, statische Dateien und API-Endpunkte)
    open_routes = ("login", "static")
    if not g.username and request.endpoint and request.endpoint not in open_routes \
            and not request.path.startswith("/api/"):
        return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    """Login-Seite: Einfach Username eingeben und losspielen."""
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        if not username:
            flash("Bitte gib einen Namen ein!", "error")
            return redirect(url_for("login"))
        if len(username) > 20:
            flash("Name darf maximal 20 Zeichen lang sein!", "error")
            return redirect(url_for("login"))
        # User erstellen oder einloggen
        get_or_create_user(username)
        session["username"] = username
        flash(f"Willkommen, {username}! 🏒", "success")
        return redirect(url_for("index"))
    return render_template("login.html")


@app.route("/logout")
def logout():
    """Logout: Session löschen."""
    session.pop("username", None)
    flash("Abgemeldet! Bis bald! 👋", "info")
    return redirect(url_for("login"))


# NHL API
NHL_API = "https://api-web.nhle.com/v1"
STATS_API = "https://api.nhle.com/stats/rest/en/team"
TEAM = "TOR"
TEAM_FULL = "Toronto Maple Leafs"
SEASON = "20252026"

# ML-Modell (wird beim Start geladen)
ml_model = None
ml_scaler = None
game_data = None

# Cache für Team-Stats (mit Ablaufzeit)
team_stats_cache = {}
team_stats_cache_time = {}
CACHE_TTL = 1800  # 30 Minuten in Sekunden

# Letztes Update Tracking
last_data_update = None
update_lock = threading.Lock()

# Globaler Headshot-Cache: playerId → fallback-URL (nur für Spieler mit Default-Bild)
_headshot_cache = {}
_headshot_cache_time = 0


def _is_default_headshot_resp(resp):
    """Prüft ob eine HEAD-Response auf ein Default-Platzhalterbild zeigt.
    Checkt URL UND Dateigröße — manche Defaults haben normale URLs aber nur ~7-12KB."""
    url = resp.url
    if "default-skater" in url or url.endswith("default.jpg") or url.endswith("default.png"):
        return True
    # Echte Headshots sind 100-200KB+, Defaults sind <20KB
    size = int(resp.headers.get("content-length", 0))
    if 0 < size < 20000:
        return True
    return False


def _resolve_headshot(pid, team, all_teams=""):
    """Findet die beste verfügbare Headshot-URL für einen Spieler."""
    teams_list = [t.strip() for t in all_teams.split(",") if t.strip()] if all_teams else [team]
    if team and team not in teams_list:
        teams_list.insert(0, team)

    # 1) Aktuelle Saison bei allen Teams
    for t in teams_list:
        url = f"https://assets.nhle.com/mugs/nhl/20252026/{t}/{pid}.png"
        try:
            resp = requests.head(url, timeout=2, allow_redirects=True)
            if not _is_default_headshot_resp(resp):
                return url
        except Exception:
            pass

    # 2) Actionshot
    action_url = f"https://assets.nhle.com/mugs/actionshots/1296x729/{pid}.jpg"
    try:
        resp = requests.head(action_url, timeout=2, allow_redirects=True)
        if not _is_default_headshot_resp(resp):
            return action_url
    except Exception:
        pass

    # 3) Vorherige Saisons
    for season in ["20242025", "20232024"]:
        for t in teams_list:
            url = f"https://assets.nhle.com/mugs/nhl/{season}/{t}/{pid}.png"
            try:
                resp = requests.head(url, timeout=2, allow_redirects=True)
                if not _is_default_headshot_resp(resp):
                    return url
            except Exception:
                pass

    return None  # Kein Bild gefunden


def refresh_headshot_cache():
    """
    Prüft alle Spieler-Headshots und cached Fallback-URLs.
    Wird im Hintergrund ausgeführt, blockiert keine Requests.
    """
    global _headshot_cache, _headshot_cache_time
    if time.time() - _headshot_cache_time < CACHE_TTL * 2:
        return  # Cache noch frisch (1h TTL)

    print("[Headshot] Starte Headshot-Cache-Update...")

    # Alle Spieler-IDs + Teams sammeln (aus der Stats-API)
    player_info = {}  # pid → (team, allTeams)
    try:
        start = 0
        while True:
            resp = requests.get(
                f"https://api.nhle.com/stats/rest/en/skater/summary?isAggregate=false&isGame=false"
                f"&start={start}&limit=100"
                f"&cayenneExp=seasonId={SEASON}%20and%20gameTypeId=2%20and%20gamesPlayed%3E=10",
                timeout=15,
            )
            data = resp.json()
            for p in data.get("data", []):
                teams_str = p.get("teamAbbrevs", "")
                last_team = teams_str.split(",")[-1].strip() if teams_str else ""
                player_info[p["playerId"]] = (last_team, teams_str)
            if start + 100 >= data.get("total", 0):
                break
            start += 100
        # Goalies
        resp = requests.get(
            f"https://api.nhle.com/stats/rest/en/goalie/summary?isAggregate=false&isGame=false"
            f"&start=0&limit=100"
            f"&cayenneExp=seasonId={SEASON}%20and%20gameTypeId=2%20and%20gamesPlayed%3E=10",
            timeout=15,
        )
        for g in resp.json().get("data", []):
            teams_str = g.get("teamAbbrevs", "")
            last_team = teams_str.split(",")[-1].strip() if teams_str else ""
            player_info[g["playerId"]] = (last_team, teams_str)
    except Exception as e:
        print(f"[Headshot] Fehler beim Laden der Spieler: {e}")
        return

    # Schritt 1: Schneller Batch-Check welche Headshots fehlen (parallel)
    def _quick_check(pid_team):
        pid, (team, _) = pid_team
        url = f"https://assets.nhle.com/mugs/nhl/20252026/{team}/{pid}.png"
        try:
            resp = requests.head(url, timeout=2, allow_redirects=True)
            if _is_default_headshot_resp(resp):
                return pid  # Braucht Fallback
        except Exception:
            pass
        return None

    missing_pids = []
    with ThreadPoolExecutor(max_workers=30) as executor:
        for pid in executor.map(_quick_check, player_info.items()):
            if pid:
                missing_pids.append(pid)

    print(f"[Headshot] {len(missing_pids)} von {len(player_info)} Spielern brauchen Fallback")

    # Schritt 2: Nur für fehlende Spieler die Fallbacks suchen (wenige!)
    new_cache = {}
    def _find_fallback(pid):
        team, all_teams = player_info[pid]
        fallback = _resolve_headshot(pid, team, all_teams)
        if fallback:
            new_cache[pid] = fallback
        else:
            # Kein Bild verfügbar → Team-Logo als Platzhalter
            new_cache[pid] = f"https://assets.nhle.com/logos/nhl/svg/{team}_dark.svg"

    with ThreadPoolExecutor(max_workers=10) as executor:
        executor.map(_find_fallback, missing_pids)

    _headshot_cache = new_cache
    _headshot_cache_time = time.time()
    print(f"[Headshot] Cache aktualisiert: {len(new_cache)} Fallbacks")


# ---- Fun Stats System ----

# Alle möglichen Stats die wir aus der API ziehen können
SKATER_STAT_DEFS = [
    {
        "key": "faceoffWinPctg",
        "label": "Faceoff King",
        "desc": "Bester Faceoff%",
        "format": lambda v: f"{v*100:.1f}%",
        "filter": lambda p: p.get("faceoffWinPctg", 0) > 0 and p.get("positionCode") == "C",
        "icon": "&#x1F94F;",
    },
    {
        "key": "shootingPctg",
        "label": "Scharfschuetze",
        "desc": "Beste Shooting%",
        "format": lambda v: f"{v*100:.1f}%",
        "filter": lambda p: p.get("shots", 0) >= 30,
        "icon": "&#x1F3AF;",
    },
    {
        "key": "goals",
        "label": "Torjaeger",
        "desc": "Meiste Tore",
        "format": lambda v: f"{int(v)} Tore",
        "filter": lambda p: True,
        "icon": "&#x1F525;",
    },
    {
        "key": "assists",
        "label": "Vorlagenkoenig",
        "desc": "Meiste Assists",
        "format": lambda v: f"{int(v)} Assists",
        "filter": lambda p: True,
        "icon": "&#x1F91D;",
    },
    {
        "key": "points",
        "label": "Punktesammler",
        "desc": "Meiste Punkte",
        "format": lambda v: f"{int(v)} Pts",
        "filter": lambda p: True,
        "icon": "&#x2B50;",
    },
    {
        "key": "plusMinus",
        "label": "Plus/Minus Boss",
        "desc": "Beste +/- Bilanz",
        "format": lambda v: f"{int(v):+d}",
        "filter": lambda p: True,
        "icon": "&#x1F4CA;",
    },
    {
        "key": "powerPlayGoals",
        "label": "Powerplay Sniper",
        "desc": "Meiste PP Tore",
        "format": lambda v: f"{int(v)} PP Goals",
        "filter": lambda p: p.get("powerPlayGoals", 0) > 0,
        "icon": "&#x26A1;",
    },
    {
        "key": "gameWinningGoals",
        "label": "Clutch Player",
        "desc": "Meiste Game-Winner",
        "format": lambda v: f"{int(v)} GWG",
        "filter": lambda p: p.get("gameWinningGoals", 0) > 0,
        "icon": "&#x1F3C6;",
    },
    {
        "key": "shots",
        "label": "Schussmaschine",
        "desc": "Meiste Schuesse",
        "format": lambda v: f"{int(v)} Shots",
        "filter": lambda p: True,
        "icon": "&#x1F3D2;",
    },
    {
        "key": "penaltyMinutes",
        "label": "Boesewicht",
        "desc": "Meiste Strafminuten",
        "format": lambda v: f"{int(v)} PIM",
        "filter": lambda p: p.get("penaltyMinutes", 0) > 10,
        "icon": "&#x1F608;",
    },
    {
        "key": "avgTimeOnIcePerGame",
        "label": "Eiszeit-Monster",
        "desc": "Meiste Eiszeit/Spiel",
        "format": lambda v: f"{v/60:.1f} min",
        "filter": lambda p: True,
        "icon": "&#x23F1;",
    },
    {
        "key": "shorthandedGoals",
        "label": "Unterzahl-Held",
        "desc": "Meiste SH Tore",
        "format": lambda v: f"{int(v)} SHG",
        "filter": lambda p: p.get("shorthandedGoals", 0) > 0,
        "icon": "&#x1F9B8;",
    },
    {
        "key": "overtimeGoals",
        "label": "OT Hero",
        "desc": "Meiste OT Tore",
        "format": lambda v: f"{int(v)} OT Goals",
        "filter": lambda p: p.get("overtimeGoals", 0) > 0,
        "icon": "&#x1F4A5;",
    },
]

GOALIE_STAT_DEFS = [
    {
        "key": "savePercentage",
        "label": "Mauer",
        "desc": "Beste Save%",
        "format": lambda v: f"{v*100:.1f}%",
        "filter": lambda p: p.get("gamesPlayed", 0) >= 5,
        "icon": "&#x1F9F1;",
    },
    {
        "key": "wins",
        "label": "Sieg-Garant",
        "desc": "Meiste Siege",
        "format": lambda v: f"{int(v)} Wins",
        "filter": lambda p: True,
        "icon": "&#x1F451;",
    },
    {
        "key": "goalsAgainstAverage",
        "label": "Unschlagbar",
        "desc": "Bester GAA",
        "format": lambda v: f"{v:.2f} GAA",
        "filter": lambda p: p.get("gamesPlayed", 0) >= 5,
        "icon": "&#x1F6E1;",
        "reverse": True,  # niedrigerer Wert = besser
    },
    {
        "key": "shutouts",
        "label": "Shutout King",
        "desc": "Meiste Shutouts",
        "format": lambda v: f"{int(v)} SO",
        "filter": lambda p: p.get("shutouts", 0) > 0,
        "icon": "&#x1F6AB;",
    },
    {
        "key": "saves",
        "label": "Save-Maschine",
        "desc": "Meiste Saves",
        "format": lambda v: f"{int(v)} Saves",
        "filter": lambda p: True,
        "icon": "&#x1F9E4;",
    },
]


def get_team_stats(team: str) -> dict | None:
    """Holt Spieler-Stats für ein Team von der NHL API. Cached mit TTL."""
    now = time.time()

    # Cache prüfen (gültig für CACHE_TTL Sekunden)
    if team in team_stats_cache and team in team_stats_cache_time:
        age = now - team_stats_cache_time[team]
        if age < CACHE_TTL:
            return team_stats_cache[team]

    try:
        url = f"{NHL_API}/club-stats/{team}/{SEASON}/2"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        team_stats_cache[team] = data
        team_stats_cache_time[team] = now
        return data
    except requests.RequestException as e:
        print(f"Fehler beim Laden der Stats für {team}: {e}")
        # Falls alter Cache vorhanden, nutze ihn trotzdem
        if team in team_stats_cache:
            return team_stats_cache[team]
        return None


def get_fun_stat_for_game(game_index: int, opponent: str) -> dict | None:
    """
    Wählt eine zufällige Fun-Stat für ein Spiel.
    Nutzt game_index als Seed damit jedes Spiel eine andere Stat bekommt,
    aber bei Reload die gleiche bleibt.
    """
    # Beide Teams laden
    tor_stats = get_team_stats("TOR")
    opp_stats = get_team_stats(opponent)

    if not tor_stats or not opp_stats:
        return None

    # Seed basierend auf game_index + Opponent für Reproduzierbarkeit
    rng = random.Random(game_index + hash(opponent))

    # Alle Stats mischen (Skater + Goalie)
    all_defs = list(SKATER_STAT_DEFS) + list(GOALIE_STAT_DEFS)
    rng.shuffle(all_defs)

    for stat_def in all_defs:
        key = stat_def["key"]
        is_goalie = stat_def in GOALIE_STAT_DEFS
        reverse = stat_def.get("reverse", False)

        # Finde den Leader in BEIDEN Teams
        tor_leader = _find_leader(
            tor_stats, key, is_goalie, stat_def["filter"], reverse
        )
        opp_leader = _find_leader(
            opp_stats, key, is_goalie, stat_def["filter"], reverse
        )

        if tor_leader and opp_leader:
            return {
                "label": stat_def["label"],
                "desc": stat_def["desc"],
                "icon": stat_def["icon"],
                "tor": {
                    "name": tor_leader["name"],
                    "headshot": tor_leader["headshot"],
                    "value": stat_def["format"](tor_leader["value"]),
                    "raw": tor_leader["value"],
                },
                "opp": {
                    "name": opp_leader["name"],
                    "headshot": opp_leader["headshot"],
                    "value": stat_def["format"](opp_leader["value"]),
                    "raw": opp_leader["value"],
                    "team": opponent,
                },
                "reverse": reverse,
            }

    return None


def _find_leader(team_data: dict, key: str, is_goalie: bool,
                 filter_fn, reverse: bool = False) -> dict | None:
    """Findet den Leader für eine bestimmte Stat in einem Team."""
    pool = team_data.get("goalies" if is_goalie else "skaters", [])

    candidates = []
    for p in pool:
        if not filter_fn(p):
            continue
        val = p.get(key, 0)
        if val is None:
            continue
        candidates.append({
            "name": f"{p['firstName']['default']} {p['lastName']['default']}",
            "headshot": p.get("headshot", ""),
            "value": val,
        })

    if not candidates:
        return None

    # Sortieren: reverse=True heißt niedrigerer Wert ist besser (z.B. GAA)
    candidates.sort(key=lambda x: x["value"], reverse=not reverse)
    return candidates[0]


def get_team_names():
    """Team-Abkürzungen zu vollen Namen."""
    return {
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


# ---- Auto-Update System ----

def fetch_latest_results() -> list[dict]:
    """
    Holt die neuesten Spielergebnisse der Leafs von der NHL API.
    Gibt nur abgeschlossene Spiele zurück (OFF/FINAL).
    """
    try:
        url = f"{NHL_API}/club-schedule-season/{TEAM}/{SEASON}"
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        finished = []
        for game in data.get("games", []):
            state = game.get("gameState", "")
            if state not in ("OFF", "FINAL"):
                continue
            game_type = game.get("gameType", 0)
            if game_type != 2:  # Nur Regular Season
                continue

            home = game.get("homeTeam", {})
            away = game.get("awayTeam", {})
            is_home = home.get("abbrev", "") == TEAM

            finished.append({
                "game_id": game.get("id", 0),
                "date": game.get("gameDate", ""),
                "opponent": away.get("abbrev", "") if is_home else home.get("abbrev", ""),
                "is_home": 1 if is_home else 0,
                "leafs_score": home.get("score", 0) if is_home else away.get("score", 0),
                "opponent_score": away.get("score", 0) if is_home else home.get("score", 0),
            })

        return finished
    except requests.RequestException as e:
        print(f"[Auto-Update] Fehler beim Laden der Ergebnisse: {e}")
        return []


def fetch_standings_for_date(date_str: str) -> dict:
    """Holt Standings für ein bestimmtes Datum."""
    try:
        url = f"{NHL_API}/standings/{date_str}"
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        standings = {}
        for entry in data.get("standings", []):
            abbrev = entry.get("teamAbbrev", {}).get("default", "")
            if not abbrev:
                continue
            gp = entry.get("gamesPlayed", 1)
            standings[abbrev] = {
                "wins": entry.get("wins", 0),
                "losses": entry.get("losses", 0),
                "points": entry.get("points", 0),
                "games_played": gp,
                "goals_for": entry.get("goalFor", 0),
                "goals_against": entry.get("goalAgainst", 0),
                "l10_wins": entry.get("l10Wins", 5),
                "win_pct": entry.get("wins", 0) / max(gp, 1),
                "goals_per_game": entry.get("goalFor", 0) / max(gp, 1),
                "goals_against_per_game": entry.get("goalAgainst", 0) / max(gp, 1),
            }
        return standings
    except requests.RequestException:
        return {}


def fetch_advanced_stats() -> dict:
    """Holt aktuelle Advanced Stats für alle Teams."""
    advanced = {}
    try:
        url = f"{STATS_API}/summary?cayenneExp=seasonId={SEASON}"
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        for team in resp.json().get("data", []):
            name = team.get("teamFullName", "")
            if not name:
                continue
            advanced[name] = {
                "pp_pct": team.get("powerPlayPct", 0.20),
                "pk_pct": team.get("penaltyKillPct", 0.80),
                "shots_per_game": team.get("shotsForPerGame", 30.0),
                "shots_against_per_game": team.get("shotsAgainstPerGame", 30.0),
                "faceoff_pct": team.get("faceoffWinPct", 0.50),
            }
        time.sleep(0.3)
    except requests.RequestException as e:
        print(f"[Auto-Update] Fehler bei Team Summary: {e}")

    try:
        url = f"{STATS_API}/percentages?cayenneExp=seasonId={SEASON}"
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        for team in resp.json().get("data", []):
            name = team.get("teamFullName", "")
            if name in advanced:
                advanced[name].update({
                    "corsi_pct": team.get("satPct", 0.50),
                    "fenwick_pct": team.get("usatPct", 0.50),
                    "pdo": team.get("shootingPlusSavePct5v5", 1.00),
                    "save_pct_5v5": team.get("savePct5v5", 0.91),
                    "shooting_pct_5v5": team.get("shootingPct5v5", 0.09),
                    "zone_start_pct": team.get("zoneStartPct5v5", 0.50),
                })
    except requests.RequestException as e:
        print(f"[Auto-Update] Fehler bei Team Percentages: {e}")

    return advanced


def update_game_data():
    """
    Hauptfunktion: Aktualisiert leafs_data.csv mit neuen Spielen.
    Wird automatisch aufgerufen wenn neue Ergebnisse erkannt werden.
    """
    global game_data, last_data_update

    with update_lock:
        print("[Auto-Update] Prüfe auf neue Spieldaten...")

        # Aktuelle CSV laden
        try:
            existing_df = pd.read_csv("leafs_data.csv")
            existing_ids = set(existing_df["game_id"].values)
            print(f"[Auto-Update] CSV hat {len(existing_df)} Spiele")
        except FileNotFoundError:
            existing_df = pd.DataFrame()
            existing_ids = set()

        # Neueste Ergebnisse von der API
        all_results = fetch_latest_results()
        new_games = [g for g in all_results if g["game_id"] not in existing_ids]

        if not new_games:
            print("[Auto-Update] Keine neuen Spiele gefunden.")
            last_data_update = datetime.now()
            return False

        print(f"[Auto-Update] {len(new_games)} neue Spiele gefunden!")

        # Standings und Advanced Stats holen
        today = datetime.now().strftime("%Y-%m-%d")
        standings = fetch_standings_for_date(today)
        adv_stats = fetch_advanced_stats()
        team_names = get_team_names()

        # Leafs Advanced Stats
        leafs_adv = adv_stats.get(TEAM_FULL, {})

        # Neue Spiele parsen
        new_rows = []
        for game in new_games:
            opponent = game["opponent"]
            leafs_score = game["leafs_score"]
            opponent_score = game["opponent_score"]
            result = "W" if leafs_score > opponent_score else "L"

            # Berechne rest_days vom vorherigen Spiel
            rest_days = 2  # Default
            if not existing_df.empty:
                last_date = existing_df["date"].iloc[-1]
                try:
                    current = datetime.strptime(game["date"], "%Y-%m-%d")
                    previous = datetime.strptime(last_date, "%Y-%m-%d")
                    rest_days = min((current - previous).days, 7)
                except ValueError:
                    rest_days = 2

            # Gegner-Stärke aus Standings
            opp_st = standings.get(opponent, {})
            opp_win_pct = opp_st.get("win_pct", 0.5)
            opp_goals_pg = opp_st.get("goals_per_game", 3.0)
            opp_goals_ag = opp_st.get("goals_against_per_game", 3.0)
            opp_points = opp_st.get("points", 50)
            opp_l10 = opp_st.get("l10_wins", 5)

            leafs_st = standings.get(TEAM, {})
            leafs_points = leafs_st.get("points", 50)

            # Advanced Stats
            opp_full = team_names.get(opponent, "")
            opp_adv = adv_stats.get(opp_full, {})

            new_rows.append({
                "game_id": game["game_id"],
                "date": game["date"],
                "opponent": opponent,
                "is_home": game["is_home"],
                "leafs_score": leafs_score,
                "opponent_score": opponent_score,
                "result": result,
                "total_goals": leafs_score + opponent_score,
                "rest_days": rest_days,
                "opp_win_pct": round(opp_win_pct, 3),
                "opp_goals_per_game": round(opp_goals_pg, 2),
                "opp_goals_against_per_game": round(opp_goals_ag, 2),
                "opp_points": opp_points,
                "opp_l10_wins": opp_l10,
                "leafs_standing_points": leafs_points,
                # Leafs Advanced
                "leafs_pp_pct": round(leafs_adv.get("pp_pct", 0.20), 4),
                "leafs_pk_pct": round(leafs_adv.get("pk_pct", 0.80), 4),
                "leafs_corsi_pct": round(leafs_adv.get("corsi_pct", 0.50), 4),
                "leafs_fenwick_pct": round(leafs_adv.get("fenwick_pct", 0.50), 4),
                "leafs_pdo": round(leafs_adv.get("pdo", 1.00), 4),
                "leafs_shots_pg": round(leafs_adv.get("shots_per_game", 30.0), 2),
                "leafs_shots_against_pg": round(leafs_adv.get("shots_against_per_game", 30.0), 2),
                "leafs_faceoff_pct": round(leafs_adv.get("faceoff_pct", 0.50), 4),
                "leafs_save_pct": round(leafs_adv.get("save_pct_5v5", 0.91), 4),
                "leafs_shooting_pct": round(leafs_adv.get("shooting_pct_5v5", 0.09), 4),
                "leafs_zone_start_pct": round(leafs_adv.get("zone_start_pct", 0.50), 4),
                # Gegner Advanced
                "opp_pp_pct": round(opp_adv.get("pp_pct", 0.20), 4),
                "opp_pk_pct": round(opp_adv.get("pk_pct", 0.80), 4),
                "opp_corsi_pct": round(opp_adv.get("corsi_pct", 0.50), 4),
                "opp_pdo": round(opp_adv.get("pdo", 1.00), 4),
                "opp_save_pct": round(opp_adv.get("save_pct_5v5", 0.91), 4),
            })

        # An CSV anhängen
        new_df = pd.DataFrame(new_rows)
        if not existing_df.empty:
            updated_df = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            updated_df = new_df

        updated_df.to_csv("leafs_data.csv", index=False)
        print(f"[Auto-Update] CSV aktualisiert: {len(updated_df)} Spiele (+{len(new_rows)} neu)")

        # game_data aktualisieren
        game_data = updated_df
        last_data_update = datetime.now()

        return True


def retrain_model():
    """Trainiert das ML-Modell automatisch neu mit den aktualisierten Daten."""
    global ml_model, ml_scaler
    try:
        print("[Auto-Update] Starte Modell-Retraining...")
        from model import load_data, add_features, train_all_models, save_model

        df = load_data()
        df = add_features(df)
        model, accuracy, model_name, scaler = train_all_models(df)
        save_model(model, scaler, model_name)

        ml_model = model
        ml_scaler = scaler
        print(f"[Auto-Update] Modell neu trainiert! ({model_name}, {accuracy:.1%})")
        return True
    except Exception as e:
        print(f"[Auto-Update] Fehler beim Retraining: {e}")
        return False


def auto_update_cycle():
    """
    Kompletter Update-Zyklus:
    1. Neue Spieldaten holen
    2. CSV aktualisieren
    3. Offene Tipps auflösen
    4. Modell neu trainieren (wenn neue Daten)
    5. Team-Stats Cache leeren
    """
    has_new = update_game_data()

    if has_new:
        # Modell neu trainieren
        retrain_model()

        # Team-Stats Cache leeren (damit Fun Stats frisch geladen werden)
        team_stats_cache.clear()
        team_stats_cache_time.clear()
        print("[Auto-Update] Team-Stats Cache geleert.")

    # Offene Tipps auflösen
    check_and_resolve_games()

    # Headshot-Cache auffrischen (im Hintergrund, blockiert nicht)
    try:
        refresh_headshot_cache()
    except Exception as e:
        print(f"[Auto-Update] Headshot-Cache Fehler: {e}")

    print(f"[Auto-Update] Zyklus abgeschlossen um {datetime.now().strftime('%H:%M:%S')}")


def start_scheduler():
    """Startet den Background-Scheduler der alle 30 Minuten Updates prüft."""
    def run():
        while True:
            try:
                auto_update_cycle()
            except Exception as e:
                print(f"[Scheduler] Fehler: {e}")
            # 30 Minuten warten
            time.sleep(CACHE_TTL)

    thread = threading.Thread(target=run, daemon=True, name="AutoUpdateScheduler")
    thread.start()
    print("[Scheduler] Background-Updates alle 30 Minuten gestartet!")


def load_ml_model():
    """Lädt das trainierte ML-Modell."""
    global ml_model, ml_scaler, game_data
    try:
        from model import load_model
        ml_model, ml_scaler = load_model()
        print("ML-Modell geladen!")
    except FileNotFoundError:
        print("Kein ML-Modell gefunden. Starte ohne Vorhersagen.")
        print("Trainiere eins mit: python3 data.py && python3 model.py")
        ml_model = None
        ml_scaler = None

    # Historische Daten für Rolling Stats
    try:
        game_data = pd.read_csv("leafs_data.csv")
        print(f"Spieldaten geladen: {len(game_data)} Spiele")
    except FileNotFoundError:
        game_data = None


def get_upcoming_games() -> list[dict]:
    """Holt kommende Leafs-Spiele von der NHL API."""
    try:
        url = f"{NHL_API}/club-schedule-season/{TEAM}/20252026"
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        upcoming = []
        for game in data.get("games", []):
            state = game.get("gameState", "")
            # FUT = Future, PRE = Pre-game
            if state not in ("FUT", "PRE"):
                continue

            home = game.get("homeTeam", {})
            away = game.get("awayTeam", {})
            is_home = home.get("abbrev", "") == TEAM

            game_id = game.get("id", 0)
            opponent = away.get("abbrev", "") if is_home else home.get("abbrev", "")

            upcoming.append({
                "game_id": game_id,
                "date": game.get("gameDate", ""),
                "opponent": opponent,
                "is_home": is_home,
                "already_predicted": prediction_exists(game_id, session.get("username")),
            })

        return upcoming[:10]  # Maximal 10 Spiele anzeigen

    except requests.RequestException as e:
        print(f"Fehler beim Laden der Spiele: {e}")
        return []


def get_model_prediction(opponent: str, is_home: bool) -> dict | None:
    """Macht eine ML-Vorhersage für ein Spiel."""
    if ml_model is None or game_data is None:
        return None

    try:
        from model import predict_game
        return predict_game(
            model=ml_model,
            opponent=opponent,
            is_home=is_home,
            recent_games=game_data,
            scaler=ml_scaler,
        )
    except Exception as e:
        print(f"Vorhersage-Fehler: {e}")
        return None


def check_and_resolve_games():
    """Prüft offene Tipps und löst sie auf wenn das Spiel vorbei ist."""
    pending = get_pending_predictions()
    if not pending:
        return

    # Schedule nur EINMAL laden (statt pro Prediction)
    try:
        url = f"{NHL_API}/club-schedule-season/{TEAM}/20252026"
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        all_games = {g["id"]: g for g in resp.json().get("games", [])}
    except requests.RequestException:
        return

    for pred in pending:
        game_id = pred["game_id"]
        game = all_games.get(game_id)
        if not game:
            continue

        state = game.get("gameState", "")
        if state not in ("OFF", "FINAL"):
            continue

        home = game.get("homeTeam", {})
        away = game.get("awayTeam", {})
        is_home = home.get("abbrev", "") == TEAM

        if is_home:
            leafs_score = home.get("score", 0)
            opp_score = away.get("score", 0)
        else:
            leafs_score = away.get("score", 0)
            opp_score = home.get("score", 0)

        result = "W" if leafs_score > opp_score else "L"

        resolve_prediction(
            pred["id"], result, leafs_score, opp_score
        )
        print(f"Spiel {game_id} aufgelöst: {result} ({leafs_score}:{opp_score})")


# ---- NHL Scoreboard System ----

# Cache für Scoreboard-Daten
scoreboard_cache = {}
scoreboard_cache_time = 0

SKATER_LEADERS_API = f"{NHL_API}/skater-stats-leaders/{SEASON}/2"
GOALIE_LEADERS_API = f"{NHL_API}/goalie-stats-leaders/{SEASON}/2"
SKATER_STATS_REST = "https://api.nhle.com/stats/rest/en/skater/summary"
GOALIE_STATS_REST = "https://api.nhle.com/stats/rest/en/goalie/summary"
TEAM_STATS_REST = "https://api.nhle.com/stats/rest/en/team"


def fetch_scoreboard_data() -> dict:
    """
    Holt alle NHL Scoreboard-Daten.
    Cached für CACHE_TTL Sekunden.
    """
    global scoreboard_cache, scoreboard_cache_time
    now = time.time()

    if scoreboard_cache and (now - scoreboard_cache_time) < CACHE_TTL:
        return scoreboard_cache

    data = {}

    # Parallel: Leaders, Goalie Leaders, Team Standings + Goalies gleichzeitig laden
    def _fetch_skater_leaders():
        try:
            resp = requests.get(
                f"{SKATER_LEADERS_API}?categories=goals,assists,points,plusMinus,toi,penaltyMins&limit=10",
                timeout=15,
            )
            resp.raise_for_status()
            return ("skater_leaders", resp.json())
        except requests.RequestException as e:
            print(f"[Scoreboard] Fehler bei Skater Leaders: {e}")
            return ("skater_leaders", {})

    def _fetch_goalie_leaders():
        try:
            resp = requests.get(
                f"{GOALIE_LEADERS_API}?categories=wins,savePctg,goalsAgainstAverage,shutouts&limit=10",
                timeout=15,
            )
            resp.raise_for_status()
            return ("goalie_leaders", resp.json())
        except requests.RequestException as e:
            print(f"[Scoreboard] Fehler bei Goalie Leaders: {e}")
            return ("goalie_leaders", {})

    def _fetch_team_standings():
        try:
            resp = requests.get(
                f"{TEAM_STATS_REST}/summary?cayenneExp=seasonId={SEASON}%20and%20gameTypeId=2"
                "&sort=%5B%7B%22property%22:%22points%22,%22direction%22:%22DESC%22%7D%5D",
                timeout=15,
            )
            resp.raise_for_status()
            return ("team_standings", resp.json().get("data", []))
        except requests.RequestException as e:
            print(f"[Scoreboard] Fehler bei Team Standings: {e}")
            return ("team_standings", [])

    def _fetch_goalies():
        try:
            resp = requests.get(
                f"{GOALIE_STATS_REST}?isAggregate=false&isGame=false"
                f"&sort=%5B%7B%22property%22:%22wins%22,%22direction%22:%22DESC%22%7D%5D"
                f"&start=0&limit=100"
                f"&cayenneExp=seasonId={SEASON}%20and%20gameTypeId=2"
                f"%20and%20gamesPlayed%3E=10",
                timeout=15,
            )
            resp.raise_for_status()
            goalies = resp.json().get("data", [])
            print(f"[Scoreboard] {len(goalies)} Goalies geladen (min. 10 GP)")
            return ("goalies", goalies)
        except requests.RequestException as e:
            print(f"[Scoreboard] Fehler bei Goalies: {e}")
            return ("goalies", [])

    # Alle 4 Calls parallel + Skater-Pages sequentiell (da paginiert)
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(_fetch_skater_leaders),
            executor.submit(_fetch_goalie_leaders),
            executor.submit(_fetch_team_standings),
            executor.submit(_fetch_goalies),
        ]
        for f in as_completed(futures):
            key, val = f.result()
            if key == "skater_leaders":
                data.update(val)
            elif key == "goalie_leaders":
                data.update(val)
            elif key == "team_standings":
                data["teamStandings"] = val
            elif key == "goalies":
                data["topGoalies"] = val

    # Skaters paginiert laden (muss sequentiell wegen Paginierung)
    all_skaters = []
    start = 0
    page_size = 100
    while True:
        try:
            resp = requests.get(
                f"{SKATER_STATS_REST}?isAggregate=false&isGame=false"
                f"&sort=%5B%7B%22property%22:%22points%22,%22direction%22:%22DESC%22%7D%5D"
                f"&start={start}&limit={page_size}"
                f"&cayenneExp=seasonId={SEASON}%20and%20gameTypeId=2"
                f"%20and%20gamesPlayed%3E=10",
                timeout=15,
            )
            resp.raise_for_status()
            page_data = resp.json()
            entries = page_data.get("data", [])
            all_skaters.extend(entries)
            total = page_data.get("total", 0)

            if start + page_size >= total or not entries:
                break
            start += page_size
        except requests.RequestException as e:
            print(f"[Scoreboard] Fehler bei Skaters (start={start}): {e}")
            break

    data["topSkaters"] = all_skaters
    print(f"[Scoreboard] {len(all_skaters)} Skater geladen (min. 10 GP)")

    scoreboard_cache = data
    scoreboard_cache_time = now
    print(f"[Scoreboard] Daten geladen: {len(data)} Kategorien")
    return data


def _safe(val, default=0):
    """Gibt default zurück wenn val None ist (API gibt manchmal None statt fehlend)."""
    return val if val is not None else default


def format_scoreboard(raw: dict) -> dict:
    """Formatiert die rohen API-Daten für das Template."""
    result = {"categories": [], "teamStandings": [], "topSkaters": [], "topGoalies": []}

    # Kategorien mit Headshots (Leaders API)
    category_config = [
        ("goals", "Tore", "&#x1F525;", lambda v: f"{int(v)}"),
        ("assists", "Assists", "&#x1F91D;", lambda v: f"{int(v)}"),
        ("points", "Punkte", "&#x2B50;", lambda v: f"{int(v)}"),
        ("plusMinus", "+/-", "&#x1F4CA;", lambda v: f"{int(v):+d}"),
        ("toi", "Eiszeit/Spiel", "&#x23F1;", lambda v: f"{v/60:.1f} min"),
        ("penaltyMins", "Strafminuten", "&#x1F608;", lambda v: f"{int(v)}"),
        ("wins", "Siege (Goalie)", "&#x1F451;", lambda v: f"{int(v)}"),
        ("savePctg", "Save%", "&#x1F9F1;", lambda v: f"{v*100:.1f}%"),
        ("goalsAgainstAverage", "GAA", "&#x1F6E1;", lambda v: f"{v:.2f}"),
        ("shutouts", "Shutouts", "&#x1F6AB;", lambda v: f"{int(v)}"),
    ]

    for key, label, icon, fmt in category_config:
        entries = raw.get(key, [])
        if not entries:
            continue

        players = []
        for i, p in enumerate(entries[:10]):
            pid = p.get("id", 0)
            players.append({
                "rank": i + 1,
                "playerId": pid,
                "name": f"{p['firstName']['default']} {p['lastName']['default']}",
                "team": p.get("teamAbbrev", ""),
                "headshot": p.get("headshot", ""),
                "teamLogo": p.get("teamLogo", ""),
                "position": p.get("position", ""),
                "number": p.get("sweaterNumber", ""),
                "value": fmt(p["value"]),
                "raw_value": p["value"],
            })

        result["categories"].append({
            "key": key,
            "label": label,
            "icon": icon,
            "players": players,
        })

    # Team Standings
    for team in raw.get("teamStandings", []):
        result["teamStandings"].append({
            "name": _safe(team.get("teamFullName"), ""),
            "gp": _safe(team.get("gamesPlayed")),
            "wins": _safe(team.get("wins")),
            "losses": _safe(team.get("losses")),
            "otl": _safe(team.get("otLosses")),
            "points": _safe(team.get("points")),
            "ppPct": round(_safe(team.get("powerPlayPct")) * 100, 1),
            "pkPct": round(_safe(team.get("penaltyKillPct")) * 100, 1),
            "gf": round(_safe(team.get("goalsForPerGame")), 2),
            "ga": round(_safe(team.get("goalsAgainstPerGame")), 2),
            "foPct": round(_safe(team.get("faceoffWinPct")) * 100, 1),
            "shotsPg": round(_safe(team.get("shotsForPerGame")), 1),
            "shotsAgPg": round(_safe(team.get("shotsAgainstPerGame")), 1),
        })

    # Top Skaters (vollständig)
    for p in raw.get("topSkaters", []):
        toi_raw = _safe(p.get("timeOnIcePerGame"))
        teams_str = _safe(p.get("teamAbbrevs"), "")
        last_team = teams_str.split(",")[-1].strip() if teams_str else ""
        pid = _safe(p.get("playerId"))
        # Headshot: alle Teams durchprobieren (Spieler wie Zamula haben Bild beim alten Team)
        headshot = ""
        all_teams = [t.strip() for t in teams_str.split(",") if t.strip()] if teams_str else []
        if last_team and last_team not in all_teams:
            all_teams.append(last_team)
        for t in reversed(all_teams):  # letztes Team zuerst, dann ältere
            headshot = f"https://assets.nhle.com/mugs/nhl/20252026/{t}/{pid}.png"
            break  # Default erstmal auf aktuelles Team setzen
        result["topSkaters"].append({
            "name": _safe(p.get("skaterFullName"), ""),
            "team": last_team,
            "allTeams": teams_str,
            "pos": _safe(p.get("positionCode"), ""),
            "gp": _safe(p.get("gamesPlayed")),
            "goals": _safe(p.get("goals")),
            "assists": _safe(p.get("assists")),
            "points": _safe(p.get("points")),
            "plusMinus": _safe(p.get("plusMinus")),
            "ppGoals": _safe(p.get("ppGoals")),
            "shGoals": _safe(p.get("shGoals")),
            "gwg": _safe(p.get("gameWinningGoals")),
            "shots": _safe(p.get("shots")),
            "shootPct": round(_safe(p.get("shootingPct")) * 100, 1),
            "foPct": round(_safe(p.get("faceoffWinPct")) * 100, 1),
            "toi": round(toi_raw / 60, 1) if toi_raw > 5 else toi_raw,
            "pim": _safe(p.get("penaltyMinutes")),
            "ppg": round(_safe(p.get("pointsPerGame")), 2),
            "playerId": pid,
            "headshot": headshot,
        })

    # Top Goalies
    for g in raw.get("topGoalies", []):
        g_teams_str = _safe(g.get("teamAbbrevs"), "")
        g_last_team = g_teams_str.split(",")[-1].strip() if g_teams_str else ""
        g_pid = _safe(g.get("playerId"))
        g_headshot = f"https://assets.nhle.com/mugs/nhl/20252026/{g_last_team}/{g_pid}.png" if g_last_team else ""
        result["topGoalies"].append({
            "name": _safe(g.get("goalieFullName"), ""),
            "team": g_last_team,
            "allTeams": g_teams_str,
            "gp": _safe(g.get("gamesPlayed")),
            "gs": _safe(g.get("gamesStarted")),
            "wins": _safe(g.get("wins")),
            "losses": _safe(g.get("losses")),
            "otl": _safe(g.get("otLosses")),
            "savePct": round(_safe(g.get("savePct")), 3),
            "gaa": round(_safe(g.get("goalsAgainstAverage")), 2),
            "shutouts": _safe(g.get("shutouts")),
            "saves": _safe(g.get("saves")),
            "shotsAgainst": _safe(g.get("shotsAgainst")),
            "playerId": g_pid,
            "headshot": g_headshot,
        })

    # Headshot-Fix: Cache-Treffer ersetzen + verdächtige Spieler live prüfen
    all_players = result["topSkaters"] + result["topGoalies"]
    for cat in result["categories"]:
        all_players.extend(cat.get("players", []))

    # Schritt 1: Cache-Treffer sofort ersetzen, verdächtige sammeln
    needs_check = []  # Spieler ohne Cache-Treffer die Multi-Team sind
    seen_pids = set()
    for player in all_players:
        pid = player.get("playerId")
        if not pid:
            continue
        if pid in _headshot_cache:
            player["headshot"] = _headshot_cache[pid]
        elif pid not in seen_pids:
            seen_pids.add(pid)
            # Standard-Headshot setzen falls nicht vorhanden
            team = player.get("team", "")
            if not player.get("headshot") and team:
                player["headshot"] = f"https://assets.nhle.com/mugs/nhl/20252026/{team}/{pid}.png"
            # Nur Multi-Team-Spieler live prüfen (max ~30 Spieler, schnell!)
            all_teams = player.get("allTeams", "")
            if "," in all_teams:
                needs_check.append(player)

    # Schritt 2: Multi-Team-Spieler parallel prüfen (~30 HEAD-Requests, ~3s)
    if needs_check:
        def _check_multi_team(player):
            pid = player.get("playerId")
            hs = player.get("headshot", "")
            if not hs or not pid:
                return
            try:
                resp = requests.head(hs, timeout=3, allow_redirects=True)
                if _is_default_headshot_resp(resp):
                    team = player.get("team", "")
                    all_teams = player.get("allTeams", "")
                    fallback = _resolve_headshot(pid, team, all_teams)
                    if fallback:
                        player["headshot"] = fallback
                        _headshot_cache[pid] = fallback
                    else:
                        logo = f"https://assets.nhle.com/logos/nhl/svg/{team}_dark.svg"
                        player["headshot"] = logo
                        _headshot_cache[pid] = logo
            except Exception:
                pass

        with ThreadPoolExecutor(max_workers=15) as executor:
            executor.map(_check_multi_team, needs_check)

    # Schritt 3: Doppelte Einträge (gleiche pid in mehreren Listen) updaten
    for player in all_players:
        pid = player.get("playerId")
        if pid and pid in _headshot_cache:
            player["headshot"] = _headshot_cache[pid]

    return result


# ---- Playoff Bracket Daten ----

playoff_cache = {}
playoff_cache_time = 0

def fetch_playoff_data():
    """Holt Playoff-Bracket und Standings von der NHL API."""
    global playoff_cache, playoff_cache_time

    if time.time() - playoff_cache_time < CACHE_TTL and playoff_cache:
        return playoff_cache

    result = {"bracket_title": "", "bracket_subtitle": "", "eastern": {}, "western": {}}

    # 1. Bracket-Daten holen
    try:
        resp = requests.get(
            f"https://api-web.nhle.com/v1/playoff-bracket/{SEASON[4:]}",
            timeout=15,
        )
        resp.raise_for_status()
        bracket = resp.json()
        result["bracket_title"] = bracket.get("bracketTitle", {}).get("default", "Playoff Bracket")
        result["bracket_subtitle"] = bracket.get("bracketSubTitle", {}).get("default", "")

        # Serien nach Runde und Conference sortieren
        series_list = bracket.get("series", [])

        # Serien in Eastern (A-D, I, J, M) und Western (E-H, K, L, N) aufteilen
        # R1: A-D = Eastern, E-H = Western
        # R2: I,J = Eastern, K,L = Western
        # R3: M = Eastern CF, N = Western CF
        # R4: O = Stanley Cup Final
        east_letters = {"A", "B", "C", "D", "I", "J", "M"}
        west_letters = {"E", "F", "G", "H", "K", "L", "N"}

        east_series = []
        west_series = []
        cup_final = None

        for s in series_list:
            letter = s.get("seriesLetter", "")
            series_data = _format_series(s)

            if letter == "O":
                cup_final = series_data
            elif letter in east_letters:
                east_series.append(series_data)
            elif letter in west_letters:
                west_series.append(series_data)

        result["eastern"]["series"] = east_series
        result["western"]["series"] = west_series
        result["cup_final"] = cup_final

    except requests.RequestException as e:
        print(f"[Playoffs] Fehler beim Bracket: {e}")

    # 2. Standings holen für Wildcard-Ansicht
    try:
        resp = requests.get(
            "https://api-web.nhle.com/v1/standings/now",
            timeout=15,
            allow_redirects=True,
        )
        resp.raise_for_status()
        standings = resp.json().get("standings", [])

        for conf_name in ["Eastern", "Western"]:
            conf_key = conf_name.lower()
            conf_teams = [t for t in standings if t["conferenceName"] == conf_name]
            conf_teams.sort(key=lambda t: t["conferenceSequence"])

            # Division-Teams (wildcardSequence == 0)
            divisions = {}
            wildcard = []

            for t in conf_teams:
                team_data = {
                    "abbrev": t["teamAbbrev"]["default"],
                    "name": t["teamName"]["default"],
                    "logo": t.get("teamLogo", ""),
                    "points": t["points"],
                    "wins": t["wins"],
                    "losses": t["losses"],
                    "otl": t["otLosses"],
                    "gp": t["gamesPlayed"],
                    "gd": t.get("goalDifferential", 0),
                    "streak": f"{t.get('streakCode', '')}{t.get('streakCount', '')}",
                    "l10": f"{t.get('l10Wins', 0)}-{t.get('l10Losses', 0)}-{t.get('l10OtLosses', 0)}",
                    "divName": t["divisionName"],
                    "divRank": t["divisionSequence"],
                    "wcRank": t["wildcardSequence"],
                    "confRank": t["conferenceSequence"],
                    "regWins": t.get("regulationWins", 0),
                    "ppct": round(t.get("pointPctg", 0) * 100, 1),
                }

                if t["wildcardSequence"] == 0:
                    div = t["divisionName"]
                    if div not in divisions:
                        divisions[div] = []
                    divisions[div].append(team_data)
                else:
                    wildcard.append(team_data)

            # Sortiere Division-Teams nach Rang
            for div in divisions:
                divisions[div].sort(key=lambda x: x["divRank"])

            # Sortiere Wildcard nach Rang
            wildcard.sort(key=lambda x: x["wcRank"])

            result[conf_key]["divisions"] = divisions
            result[conf_key]["wildcard"] = wildcard

    except requests.RequestException as e:
        print(f"[Playoffs] Fehler bei Standings: {e}")

    playoff_cache = result
    playoff_cache_time = time.time()
    return result


# ---- Trade Board Daten ----

trade_cache = {}
trade_cache_time = 0

# NHL Team-Abkürzungen zu Player-ID Mapping
TEAM_ABBREVS = {
    "ANA": "Anaheim Ducks", "ARI": "Arizona Coyotes", "BOS": "Boston Bruins",
    "BUF": "Buffalo Sabres", "CGY": "Calgary Flames", "CAR": "Carolina Hurricanes",
    "CHI": "Chicago Blackhawks", "COL": "Colorado Avalanche", "CBJ": "Columbus Blue Jackets",
    "DAL": "Dallas Stars", "DET": "Detroit Red Wings", "EDM": "Edmonton Oilers",
    "FLA": "Florida Panthers", "LAK": "Los Angeles Kings", "MIN": "Minnesota Wild",
    "MTL": "Montreal Canadiens", "NSH": "Nashville Predators", "NJD": "New Jersey Devils",
    "NYI": "New York Islanders", "NYR": "New York Rangers", "OTT": "Ottawa Senators",
    "PHI": "Philadelphia Flyers", "PIT": "Pittsburgh Penguins", "SJS": "San Jose Sharks",
    "SEA": "Seattle Kraken", "STL": "St. Louis Blues", "TBL": "Tampa Bay Lightning",
    "TOR": "Toronto Maple Leafs", "UTA": "Utah Hockey Club", "VAN": "Vancouver Canucks",
    "VGK": "Vegas Golden Knights", "WSH": "Washington Capitals", "WPG": "Winnipeg Jets",
}


def fetch_trade_data():
    """Holt Trade Board Daten: RSS-Feed + strukturierte Insider-Infos."""
    global trade_cache, trade_cache_time

    if time.time() - trade_cache_time < CACHE_TTL and trade_cache:
        return trade_cache

    result = {"players": [], "news": [], "last_update": ""}

    # 1. Trade Board Spieler (basierend auf Insider-Berichten)
    trade_candidates = _get_trade_candidates()

    # 2. NHL API: Spieler-Headshots und Live-Stats PARALLEL holen
    def _fetch_player_stats(player):
        """Holt Stats für einen Spieler (wird parallel aufgerufen)."""
        pid = player.get("playerId")
        # Headshot direkt aus bekannter URL-Struktur (kein API-Call nötig)
        if pid:
            player["headshot"] = f"https://assets.nhle.com/mugs/nhl/20252026/{player['team']}/{pid}.png"
        try:
            if pid:
                resp = requests.get(
                    f"https://api-web.nhle.com/v1/player/{pid}/landing",
                    timeout=8,
                )
                if resp.status_code == 200:
                    pdata = resp.json()
                    # Offizielles Headshot überschreiben falls vorhanden
                    if pdata.get("headshot"):
                        player["headshot"] = pdata["headshot"]
                    player["heroImage"] = pdata.get("heroImage", "")
                    stats = pdata.get("featuredStats", {}).get("regularSeason", {}).get("subSeason", {})
                    if stats:
                        player["liveStats"] = {
                            "gp": stats.get("gamesPlayed", 0),
                            "goals": stats.get("goals", 0),
                            "assists": stats.get("assists", 0),
                            "points": stats.get("points", 0),
                            "plusMinus": stats.get("plusMinus", 0),
                            "wins": stats.get("wins", 0),
                            "gaa": round(stats.get("goalsAgainstAvg", 0) or 0, 2),
                            "savePct": round((stats.get("savePctg", 0) or 0), 3),
                        }
        except Exception as e:
            print(f"[TradeBoard] Fehler bei Spieler {pid}: {e}")
        # Headshot-Fix: Nutze globalen Cache oder prüfe on-the-fly
        if pid and pid in _headshot_cache:
            player["headshot"] = _headshot_cache[pid]
        elif pid and player.get("headshot"):
            try:
                head_resp = requests.head(player["headshot"], timeout=2, allow_redirects=True)
                if _is_default_headshot_resp(head_resp):
                    fallback = _resolve_headshot(pid, player["team"])
                    if fallback:
                        player["headshot"] = fallback
                        _headshot_cache[pid] = fallback
            except Exception:
                pass
        return player

    # Parallel: bis zu 10 gleichzeitige API-Calls
    with ThreadPoolExecutor(max_workers=10) as executor:
        result["players"] = list(executor.map(_fetch_player_stats, trade_candidates))

    # 3. RSS-Feeds: Neueste Trade-Artikel von allen Quellen
    import re as _re
    import html as _html

    trade_words = ["trade", "acqui", "deal", "deadline", "rumou", "swap",
                   "move", "sign", "waiv", "buyer", "seller"]

    rss_feeds = [
        ("Sportsnet", "https://www.sportsnet.ca/hockey/nhl/feed/"),
        ("ESPN", "https://www.espn.com/espn/rss/nhl/news"),
        ("DailyFaceoff", "https://www.dailyfaceoff.com/feed/"),
        ("ProHockeyRumors", "https://www.prohockeyrumors.com/feed"),
        ("Yahoo Sports", "https://sports.yahoo.com/nhl/rss.xml"),
        ("NY Post", "https://nypost.com/tag/nhl/feed/"),
        ("CBS Sports", "https://www.cbssports.com/rss/headlines/nhl/"),
    ]

    def _fetch_rss(feed_info):
        """Holt Trade-News aus einem einzelnen RSS-Feed."""
        source, url = feed_info
        articles = []
        try:
            resp = requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
            if resp.status_code != 200:
                return articles
            titles = _re.findall(r"<title>(.*?)</title>", resp.text)
            links = _re.findall(r"<link>(.*?)</link>", resp.text)
            dates = _re.findall(r"<pubDate>(.*?)</pubDate>", resp.text)
            for i, t in enumerate(titles):
                t_clean = _html.unescape(_re.sub(r"<!\[CDATA\[\s*|\s*\]\]>", "", t))
                if any(w in t_clean.lower() for w in trade_words):
                    link = ""
                    if i < len(links):
                        link = _re.sub(r"<!\[CDATA\[\s*|\s*\]\]>", "", links[i]).strip()
                    date = dates[i - 1] if i - 1 < len(dates) and i > 0 else ""
                    articles.append({
                        "title": t_clean,
                        "link": link,
                        "date": date,
                        "source": source,
                    })
        except Exception as e:
            print(f"[TradeBoard] Fehler bei {source} RSS: {e}")
        return articles

    # Alle Feeds parallel abfragen
    with ThreadPoolExecutor(max_workers=7) as executor:
        for articles in executor.map(_fetch_rss, rss_feeds):
            result["news"].extend(articles)

    # Duplikate entfernen (gleicher Titel)
    seen_titles = set()
    unique_news = []
    for item in result["news"]:
        title_key = item["title"].lower().strip()
        if title_key not in seen_titles:
            seen_titles.add(title_key)
            unique_news.append(item)
    result["news"] = unique_news

    result["last_update"] = datetime.now().strftime("%d.%m.%Y %H:%M")
    trade_cache = result
    trade_cache_time = time.time()
    return result


def _get_trade_candidates():
    """Strukturierte Trade-Kandidaten basierend auf aktuellen Insider-Berichten."""
    # Quelle: Sportsnet Kyper Trade Board 3.0 (02.02.2026) + diverse Insider
    # likelihood: 1-5 (1=unwahrscheinlich, 5=sehr wahrscheinlich)
    return [
        {
            "name": "Bobby McMann", "playerId": 8482259, "team": "TOR", "pos": "LW",
            "cap": "$1.35M", "contract": "UFA 2026",
            "likelihood": 5, "tier": "hot",
            "destinations": ["COL", "FLA", "ANA", "OTT"],
            "summary": "Meistgefragter Leafs-Spieler. Karrierejahr mit 30 Punkten. Leicht einzupassen dank niedrigem Cap Hit.",
            "source": "Kyper Trade Board 3.0",
        },
        {
            "name": "Anthony Stolarz", "playerId": 8476932, "team": "TOR", "pos": "G",
            "cap": "$2.5M", "contract": "UFA 2027",
            "likelihood": 3, "tier": "warm",
            "destinations": [],
            "summary": "Leafs fühlen sich im Tor gut aufgestellt mit Woll & Hildeby. Stolarz muss besser spielen um Wert zu steigern.",
            "source": "Kyper Trade Board 3.0",
        },
        {
            "name": "Artemi Panarin", "playerId": 8478550, "team": "NYR", "pos": "LW",
            "cap": "$11.64M", "contract": "UFA 2026",
            "likelihood": 4, "tier": "hot",
            "destinations": ["WSH", "FLA", "LAK"],
            "summary": "NMC - hat volle Kontrolle. Will Extension (~$50M). Von NYR freigestellt bis Trade. Washington sehr interessiert.",
            "source": "Kyper Trade Board 3.0",
        },
        {
            "name": "Robert Thomas", "playerId": 8480023, "team": "STL", "pos": "C",
            "cap": "$8.125M", "contract": "UFA 2030",
            "likelihood": 2, "tier": "warm",
            "destinations": [],
            "summary": "Wird aktiv angeboten. Preis astronomisch hoch (~3 Top-15 Picks). Verletzt bis nach Olympia.",
            "source": "Kyper Trade Board 3.0",
        },
        {
            "name": "Jordan Kyrou", "playerId": 8479385, "team": "STL", "pos": "RW",
            "cap": "$8.125M", "contract": "UFA 2030",
            "likelihood": 3, "tier": "warm",
            "destinations": [],
            "summary": "Wahrscheinlicher Trade als Thomas. Blues 11 Punkte vom Playoff entfernt. Armstrongs letztes Deadline als GM.",
            "source": "Kyper Trade Board 3.0",
        },
        {
            "name": "Ryan O'Reilly", "playerId": 8475158, "team": "NSH", "pos": "C",
            "cap": "$4.5M", "contract": "UFA 2027",
            "likelihood": 4, "tier": "hot",
            "destinations": [],
            "summary": "Meistbeachteter Predator. Center sind Premium. Keine Trade-Protection. Wert war nie höher.",
            "source": "Kyper Trade Board 3.0",
        },
        {
            "name": "Elias Pettersson", "playerId": 8480012, "team": "VAN", "pos": "C",
            "cap": "$11.6M", "contract": "UFA 2032",
            "likelihood": 3, "tier": "warm",
            "destinations": ["CAR"],
            "summary": "Canucks im Rebuild - alles steht zum Verkauf. Hoher Cap Hit + 6 Jahre Vertrag machen Trade komplex.",
            "source": "Kyper Trade Board 3.0",
        },
        {
            "name": "Vincent Trocheck", "playerId": 8476389, "team": "NYR", "pos": "C",
            "cap": "$5.625M", "contract": "UFA 2030",
            "likelihood": 4, "tier": "hot",
            "destinations": ["MIN"],
            "summary": "Rangers wissen sie können ihn nicht halten. Minnesota schaut sich ihn bei Olympia genau an.",
            "source": "Kyper Trade Board 3.0",
        },
        {
            "name": "Nazem Kadri", "playerId": 8475172, "team": "CGY", "pos": "C",
            "cap": "$7M", "contract": "UFA 2029",
            "likelihood": 3, "tier": "warm",
            "destinations": [],
            "summary": "Hat Calgary informiert dass er gehen möchte um Cup zu jagen. Noch 3 Jahre Vertrag bremst den Markt.",
            "source": "Kyper Trade Board 3.0",
        },
        {
            "name": "Brayden Schenn", "playerId": 8475170, "team": "STL", "pos": "C",
            "cap": "$5.375M", "contract": "UFA 2027",
            "likelihood": 4, "tier": "hot",
            "destinations": [],
            "summary": "Armstrong nimmt Anrufe entgegen. Schenn und Bruder Luke wollen zusammenspielen. Günstigere Center-Option.",
            "source": "Kyper Trade Board 3.0",
        },
        {
            "name": "Blake Coleman", "playerId": 8476399, "team": "CGY", "pos": "LW",
            "cap": "$4.9M", "contract": "UFA 2027",
            "likelihood": 4, "tier": "hot",
            "destinations": [],
            "summary": "Meistgefragter Flame. Zweifacher Cup-Sieger. Verletzt bis nach Olympia. Noch ein Jahr Vertrag.",
            "source": "Kyper Trade Board 3.0",
        },
        {
            "name": "Steven Stamkos", "playerId": 8474564, "team": "NSH", "pos": "RW",
            "cap": "$8M", "contract": "UFA 2028",
            "likelihood": 2, "tier": "cold",
            "destinations": [],
            "summary": "NMC - unwahrscheinlich vor Deadline. Würde nur zu Contender gehen. Eher Sommer-Thema.",
            "source": "Kyper Trade Board 3.0",
        },
        {
            "name": "Dougie Hamilton", "playerId": 8476462, "team": "NJD", "pos": "D",
            "cap": "$9M", "contract": "UFA 2028",
            "likelihood": 2, "tier": "cold",
            "destinations": [],
            "summary": "Devils nur 7 Punkte hinter Playoff. Hughes verletzt - Hamilton wird gebraucht. Trade hängt vom Rennen ab.",
            "source": "Kyper Trade Board 3.0",
        },
        {
            "name": "Jesper Wallstedt", "playerId": 8482661, "team": "MIN", "pos": "G",
            "cap": "$0.863M", "contract": "RFA 2026",
            "likelihood": 3, "tier": "warm",
            "destinations": [],
            "summary": "Calder-Kandidat. Luxus mit Gustavsson als #1 bis 2031. Könnte Top-6 Scoring einbringen.",
            "source": "Kyper Trade Board 3.0",
        },
        {
            "name": "Shane Wright", "playerId": 8483524, "team": "SEA", "pos": "C",
            "cap": "$0.916M", "contract": "RFA 2026",
            "likelihood": 3, "tier": "warm",
            "destinations": [],
            "summary": "4. Pick 2022 - hat sich nicht durchgesetzt (20 Punkte). Kraken brauchen Scoring-Upgrade.",
            "source": "Kyper Trade Board 3.0",
        },
        {
            "name": "Matty Beniers", "playerId": 8482665, "team": "SEA", "pos": "C",
            "cap": "$0.950M", "contract": "RFA 2026",
            "likelihood": 2, "tier": "cold",
            "destinations": [],
            "summary": "2. Pick 2021. Auf Liste weil Kraken für Blockbuster-Return zuhören würden.",
            "source": "Kyper Trade Board 3.0",
        },
        {
            "name": "Evander Kane", "playerId": 8475169, "team": "VAN", "pos": "LW",
            "cap": "$7M", "contract": "UFA 2027",
            "likelihood": 4, "tier": "hot",
            "destinations": ["LAK"],
            "summary": "Agent hat Trade-Erlaubnis. Canucks nutzen Salary Retention. LA hat Interesse.",
            "source": "Kyper Trade Board 3.0",
        },
        {
            "name": "Jake DeBrusk", "playerId": 8478498, "team": "VAN", "pos": "RW",
            "cap": "$5.5M", "contract": "UFA 2030",
            "likelihood": 3, "tier": "warm",
            "destinations": [],
            "summary": "Schwaches 2. Jahr in Vancouver. Erst letztes Jahr 28 Tore. Sollte noch komplementäres Scoring liefern können.",
            "source": "Kyper Trade Board 3.0",
        },
        {
            "name": "Patrik Laine", "playerId": 8479339, "team": "MTL", "pos": "LW",
            "cap": "$8.7M", "contract": "UFA 2026",
            "likelihood": 4, "tier": "hot",
            "destinations": [],
            "summary": "Seit Oktober verletzt. MTL retainet bis 50%. Expiring Contract = Freebie mit Upside.",
            "source": "Kyper Trade Board 3.0",
        },
        {
            "name": "Braden Schneider", "playerId": 8482073, "team": "NYR", "pos": "D",
            "cap": "$2.2M", "contract": "RFA 2026",
            "likelihood": 3, "tier": "warm",
            "destinations": [],
            "summary": "24 Jahre, Top-Pair Minuten, Rechtsschuss. Rangers nehmen Anrufe entgegen. Breites Interesse.",
            "source": "Kyper Trade Board 3.0",
        },
        {
            "name": "Jesperi Kotkaniemi", "playerId": 8480829, "team": "CAR", "pos": "C",
            "cap": "$4.82M", "contract": "UFA 2030",
            "likelihood": 3, "tier": "warm",
            "destinations": [],
            "summary": "Carolina will upgraden. KK hat Erwartungen nicht erfüllt, aber erst 25 und Center. Teams könnten auf Rebound wetten.",
            "source": "Kyper Trade Board 3.0",
        },
        # ---- Kyper: weitere Spieler ----
        {
            "name": "Jonathan Marchessault", "playerId": 8476539, "team": "NSH", "pos": "RW",
            "cap": "$5.5M", "contract": "UFA 2029",
            "likelihood": 2, "tier": "cold",
            "destinations": [],
            "summary": "NMC wie Stamkos. Spielt nicht gut. Predators im Umbruch aber Marchessault muss Trade zustimmen.",
            "source": "Kyper Trade Board 3.0",
        },
        {
            "name": "Andrew Mangiapane", "playerId": 8478233, "team": "EDM", "pos": "LW",
            "cap": "$5.8M", "contract": "UFA 2027",
            "likelihood": 4, "tier": "hot",
            "destinations": ["MTL"],
            "summary": "Hat bei Edmonton nicht funktioniert. Oilers suchen aktiv Tradepartner. NTC, aber bereit zu waiven. MTL-EDM Match möglich.",
            "source": "Kyper Trade Board 3.0",
        },
        {
            "name": "Ryan Strome", "playerId": 8476458, "team": "ANA", "pos": "C",
            "cap": "$5M", "contract": "UFA 2026",
            "likelihood": 4, "tier": "hot",
            "destinations": [],
            "summary": "Anaheim will Geld loswerden. Günstigste Akquisitionskosten unter den verfügbaren Centern. Vor Olympia möglich.",
            "source": "Kyper Trade Board 3.0",
        },
        # ---- Friedman / TSN / DailyFaceoff Quellen ----
        {
            "name": "Morgan Rielly", "playerId": 8476853, "team": "TOR", "pos": "D",
            "cap": "$7.5M", "contract": "UFA 2030",
            "likelihood": 3, "tier": "warm",
            "destinations": ["EDM", "VAN"],
            "summary": "Leafs sind 'open for business'. Spiel stark abgefallen (-17). Volle NMC + verletzt bis nach Olympia. Komplexer Trade.",
            "source": "Friedman / TSN",
        },
        {
            "name": "Oliver Ekman-Larsson", "playerId": 8475171, "team": "TOR", "pos": "D",
            "cap": "$3.5M", "contract": "UFA 2028",
            "likelihood": 2, "tier": "cold",
            "destinations": [],
            "summary": "Bester Leafs-Verteidiger dieses Jahr. Trade erst möglich wenn Leafs endgültig aus dem Rennen sind. 16-Team NTC.",
            "source": "Kyper / Friedman",
        },
        {
            "name": "Scott Laughton", "playerId": 8476872, "team": "TOR", "pos": "C",
            "cap": "$3M", "contract": "UFA 2026",
            "likelihood": 4, "tier": "hot",
            "destinations": [],
            "summary": "Pending UFA. Depth-Center Option für Contender. Leafs werden ihn als Rental verkaufen wenn sie Seller werden.",
            "source": "TSN / Friedman",
        },
        {
            "name": "Calle Jarnkrok", "playerId": 8475714, "team": "TOR", "pos": "C",
            "cap": "$2.1M", "contract": "UFA 2026",
            "likelihood": 3, "tier": "warm",
            "destinations": [],
            "summary": "Pending UFA. Vielseitiger Forward. Hat eine NTC die Sache kompliziert.",
            "source": "TSN",
        },
        {
            "name": "Brandon Carlo", "playerId": 8478443, "team": "TOR", "pos": "D",
            "cap": "$3.485M", "contract": "UFA 2027",
            "likelihood": 3, "tier": "warm",
            "destinations": [],
            "summary": "Rechtsschuss Shutdown-D. Noch ein Jahr Vertrag. 8-Team NTC. Leafs erkunden seinen Marktwert.",
            "source": "TSN / Friedman",
        },
        {
            "name": "Troy Stecher", "playerId": 8479442, "team": "TOR", "pos": "D",
            "cap": "$1.1M", "contract": "UFA 2026",
            "likelihood": 4, "tier": "hot",
            "destinations": [],
            "summary": "Pending UFA, niedriger Cap Hit. Solider Depth-D den jeder Contender gebrauchen kann. Leicht zu traden.",
            "source": "TSN",
        },
        {
            "name": "Travis Konecny", "playerId": 8478439, "team": "PHI", "pos": "RW",
            "cap": "$5.5M", "contract": "UFA 2027",
            "likelihood": 2, "tier": "cold",
            "destinations": ["TOR", "LAK", "CAR", "NJD"],
            "summary": "Flyers bester Spieler. Nur wenn Philly voll in den Seller-Modus geht. Würde 1st-Round Pick + Elite-Prospects bringen.",
            "source": "TSN / HockeyBuzz",
        },
        {
            "name": "Rasmus Ristolainen", "playerId": 8477499, "team": "PHI", "pos": "D",
            "cap": "$5.1M", "contract": "UFA 2027",
            "likelihood": 3, "tier": "warm",
            "destinations": ["EDM", "TOR", "TBL", "FLA"],
            "summary": "Grosser, physischer Rechtsschuss-D. Genau was Contender suchen. Philly offen für richtige Angebote.",
            "source": "HockeyBuzz / DailyFaceoff",
        },
        {
            "name": "Justin Faulk", "playerId": 8475753, "team": "STL", "pos": "D",
            "cap": "$6.5M", "contract": "UFA 2027",
            "likelihood": 3, "tier": "warm",
            "destinations": ["FLA", "UTA", "TOR"],
            "summary": "33 Jahre, Rechtsschuss, 22 Min/Spiel. 11 Tore, auf Karrierehoch-Kurs. 15-Team NTL. Armstrongs letzter Deadline.",
            "source": "DailyFaceoff",
        },
        {
            "name": "Jordan Binnington", "playerId": 8476412, "team": "STL", "pos": "G",
            "cap": "$6M", "contract": "UFA 2027",
            "likelihood": 2, "tier": "cold",
            "destinations": [],
            "summary": "Schlechte Saison, aber Goalie-Markt ist dünn. Starke Olympia-Performance könnte Markt beleben. 10-Team NTL.",
            "source": "Kyper / DailyFaceoff",
        },
        {
            "name": "Boone Jenner", "playerId": 8476432, "team": "CBJ", "pos": "C",
            "cap": "$3.75M", "contract": "UFA 2026",
            "likelihood": 4, "tier": "hot",
            "destinations": [],
            "summary": "Pending UFA. Physisch, Top-6 Scoring. Extension-Gespräche in Olympia-Pause - wenn kein Deal, wird er getradet.",
            "source": "Friedman / DailyFaceoff",
        },
        {
            "name": "Jared McCann", "playerId": 8477955, "team": "SEA", "pos": "C",
            "cap": "$5M", "contract": "UFA 2028",
            "likelihood": 2, "tier": "cold",
            "destinations": [],
            "summary": "31 Tore/82 Spiele Durchschnitt. Kraken in Playoff-Nähe - Trade nur wenn SEA endgültig Seller wird.",
            "source": "NHL Rumors",
        },
        {
            "name": "Michael Bunting", "playerId": 8478047, "team": "NSH", "pos": "LW",
            "cap": "$4.5M", "contract": "UFA 2026",
            "likelihood": 4, "tier": "hot",
            "destinations": [],
            "summary": "Pending UFA. 12G, 17A in 52 Spielen. Physisch + torgefährlich. Nashville shoppt ihn aktiv.",
            "source": "Predlines / TSN",
        },
        {
            "name": "Luke Schenn", "playerId": 8474568, "team": "WPG", "pos": "D",
            "cap": "$2.5M", "contract": "UFA 2026",
            "likelihood": 5, "tier": "hot",
            "destinations": ["DET"],
            "summary": "Pending UFA. Veteran Shutdown-D. Detroit und Winnipeg arbeiten laut Friedman bereits an einem Deal.",
            "source": "Friedman",
        },
        {
            "name": "Juuse Saros", "playerId": 8477424, "team": "NSH", "pos": "G",
            "cap": "$7.74M", "contract": "UFA 2030",
            "likelihood": 1, "tier": "cold",
            "destinations": ["EDM"],
            "summary": "Langschuss wegen grossem Vertrag + NMC. Eher Sommer-Trade. Edmonton laut Marek interessiert.",
            "source": "TSN / Marek",
        },
    ]


def _format_series(s):
    """Formatiert ein einzelnes Serien-Objekt."""
    top_team = s.get("topSeedTeam") or {}
    bot_team = s.get("bottomSeedTeam") or {}

    return {
        "letter": s.get("seriesLetter", ""),
        "round": s.get("playoffRound", 0),
        "title": s.get("seriesTitle", ""),
        "abbrev": s.get("seriesAbbrev", ""),
        "topSeed": {
            "abbrev": top_team.get("abbrev", "TBD"),
            "name": top_team.get("commonName", {}).get("default", "TBD"),
            "logo": top_team.get("darkLogo", ""),
            "rank": s.get("topSeedRank", 0),
            "rankAbbrev": s.get("topSeedRankAbbrev", ""),
            "wins": s.get("topSeedWins", 0),
        },
        "bottomSeed": {
            "abbrev": bot_team.get("abbrev", "TBD"),
            "name": bot_team.get("commonName", {}).get("default", "TBD"),
            "logo": bot_team.get("darkLogo", ""),
            "rank": s.get("bottomSeedRank", 0),
            "rankAbbrev": s.get("bottomSeedRankAbbrev", ""),
            "wins": s.get("bottomSeedWins", 0),
        },
        "winnerTeamId": s.get("winningTeamId"),
        "topTeamId": top_team.get("id"),
        "botTeamId": bot_team.get("id"),
    }


# ---- Salary Cap System (Spotrac Scraping) ----

SPOTRAC_TEAMS = {
    "ANA": ("anaheim-ducks", "Anaheim Ducks"),
    "BOS": ("boston-bruins", "Boston Bruins"),
    "BUF": ("buffalo-sabres", "Buffalo Sabres"),
    "CGY": ("calgary-flames", "Calgary Flames"),
    "CAR": ("carolina-hurricanes", "Carolina Hurricanes"),
    "CHI": ("chicago-blackhawks", "Chicago Blackhawks"),
    "COL": ("colorado-avalanche", "Colorado Avalanche"),
    "CBJ": ("columbus-blue-jackets", "Columbus Blue Jackets"),
    "DAL": ("dallas-stars", "Dallas Stars"),
    "DET": ("detroit-red-wings", "Detroit Red Wings"),
    "EDM": ("edmonton-oilers", "Edmonton Oilers"),
    "FLA": ("florida-panthers", "Florida Panthers"),
    "LAK": ("los-angeles-kings", "Los Angeles Kings"),
    "MIN": ("minnesota-wild", "Minnesota Wild"),
    "MTL": ("montreal-canadiens", "Montreal Canadiens"),
    "NSH": ("nashville-predators", "Nashville Predators"),
    "NJD": ("new-jersey-devils", "New Jersey Devils"),
    "NYI": ("new-york-islanders", "New York Islanders"),
    "NYR": ("new-york-rangers", "New York Rangers"),
    "OTT": ("ottawa-senators", "Ottawa Senators"),
    "PHI": ("philadelphia-flyers", "Philadelphia Flyers"),
    "PIT": ("pittsburgh-penguins", "Pittsburgh Penguins"),
    "SJS": ("san-jose-sharks", "San Jose Sharks"),
    "SEA": ("seattle-kraken", "Seattle Kraken"),
    "STL": ("st-louis-blues", "St. Louis Blues"),
    "TBL": ("tampa-bay-lightning", "Tampa Bay Lightning"),
    "TOR": ("toronto-maple-leafs", "Toronto Maple Leafs"),
    "UTA": ("utah-mammoth", "Utah Mammoth"),
    "VAN": ("vancouver-canucks", "Vancouver Canucks"),
    "VGK": ("vegas-golden-knights", "Vegas Golden Knights"),
    "WSH": ("washington-capitals", "Washington Capitals"),
    "WPG": ("winnipeg-jets", "Winnipeg Jets"),
}

salary_cap_cache = {}
salary_cap_cache_time = {}

SPOTRAC_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9",
    "Accept-Language": "en-US,en;q=0.5",
    "Referer": "https://www.google.com/",
}


def _parse_dollar(s):
    """Wandelt '$13,250,000' in 13250000 (int) um."""
    s = s.strip().replace("$", "").replace(",", "").replace("-", "0")
    try:
        return int(s)
    except ValueError:
        return 0


def _parse_pct(s):
    """Wandelt '13.87%' in 13.87 (float) um."""
    s = s.strip().replace("%", "")
    try:
        return float(s)
    except ValueError:
        return 0.0


def fetch_salary_data(team_abbr="TOR"):
    """Scraped Spotrac für Salary Cap Daten eines NHL-Teams."""
    import re as _re

    if team_abbr not in SPOTRAC_TEAMS:
        team_abbr = "TOR"

    # Cache prüfen
    if team_abbr in salary_cap_cache and \
       (time.time() - salary_cap_cache_time.get(team_abbr, 0)) < CACHE_TTL:
        return salary_cap_cache[team_abbr]

    slug, full_name = SPOTRAC_TEAMS[team_abbr]
    url = f"https://www.spotrac.com/nhl/{slug}/cap/_/year/2025"

    result = {
        "team": team_abbr,
        "team_name": full_name,
        "active_roster": [],
        "minor_league": [],
        "summary": {},
        "error": None,
    }

    try:
        resp = requests.get(url, headers=SPOTRAC_HEADERS, timeout=15)
        if resp.status_code != 200:
            result["error"] = f"Spotrac HTTP {resp.status_code}"
            return result

        html = resp.text
        # Tabellen einzeln parsen (Active, LTIR, Buyout, Summary, Minor)
        tables = _re.findall(r"<table[^>]*>(.*?)</table>", html, _re.DOTALL)
        # Alle <tr> aus allen Tabellen, mit Tabellen-Index
        tr_blocks = []
        for t_idx, table_html in enumerate(tables):
            for tr_html in _re.findall(r"<tr[^>]*>(.*?)</tr>", table_html, _re.DOTALL):
                tr_blocks.append((t_idx, tr_html))

        # ---- Headshot-Matching: NHL Roster laden ----
        roster_map = {}  # nachname_lower → headshot_url
        try:
            roster_resp = requests.get(
                f"https://api-web.nhle.com/v1/roster/{team_abbr}/20252026",
                timeout=10,
            )
            if roster_resp.status_code == 200:
                roster_data = roster_resp.json()
                for group in ["forwards", "defensemen", "goalies"]:
                    for p in roster_data.get(group, []):
                        last = p.get("lastName", {}).get("default", "").lower()
                        first = p.get("firstName", {}).get("default", "").lower()
                        full = f"{first} {last}"
                        pid = p.get("id", 0)
                        hs = p.get("headshot", "")
                        roster_map[last] = {"headshot": hs, "playerId": pid}
                        roster_map[full] = {"headshot": hs, "playerId": pid}
        except Exception:
            pass

        # ---- Spieler-Rows parsen ----
        # Tabellen: 0=Active, 1=LTIR, 2=Buyout, 3=Summary, 4=Minor
        summary_keys = {
            "Salary Cap Maximum": "cap_ceiling",
            "Adjustment": "adjustment",
            "Adjusted Salary Cap Maximum": "adjusted_cap",
            "Active Roster": "active_total",
            "Long-Term Injured Reserve": "ltir",
            "Buyout": "buyout",
            "Minor": "minor_total",
            "Total Allocations": "total_allocations",
            "Cap Space": "cap_space",
            "Potential Bonuses": "potential_bonuses",
        }

        for t_idx, tr in tr_blocks:
            tds = _re.findall(r"<td[^>]*>(.*?)</td>", tr, _re.DOTALL)
            if not tds:
                continue

            # Zellen bereinigen (Text-only)
            cells = []
            for td in tds:
                clean = _re.sub(r"<[^>]+>", "", td).strip()
                clean = _re.sub(r"\s+", " ", clean)
                cells.append(clean)

            # Summary-Rows erkennen (2-3 Zellen, kein Positions-Kürzel)
            if len(cells) in (2, 3) and cells[0]:
                for key, field in summary_keys.items():
                    if key in cells[0]:
                        val_str = cells[1] if len(cells) > 1 else "0"
                        rank_match = _re.search(r"(\d+)(?:st|nd|rd|th)", val_str)
                        val = _parse_dollar(val_str.split("/")[0].strip())
                        result["summary"][field] = val
                        if rank_match:
                            result["summary"][field + "_rank"] = rank_match.group(0)
                        break
                continue

            # Spieler-Rows: 7-8 Zellen
            if len(cells) < 6:
                continue

            # Name aus <a>-Tag im rohen HTML holen (sauberer als text-strip)
            first_td_raw = tds[0]
            a_match = _re.search(r"<a[^>]*>([^<]+)</a>", first_td_raw)
            if a_match:
                display_name = a_match.group(1).strip()
            else:
                # Minor-League/Reserve: kein <a>-Tag, Format "1 Henry Thrun"
                name_raw = cells[0]
                num_match = _re.match(r"^\d+\s+", name_raw)
                if num_match:
                    name_raw = name_raw[num_match.end():]
                display_name = name_raw.strip()

            # Vor-/Nachname splitten
            name_parts = display_name.split()
            if len(name_parts) >= 2:
                first_name = name_parts[0]
                last_name = " ".join(name_parts[1:])  # "Van Riemsdyk" etc.
            else:
                first_name = display_name
                last_name = display_name

            pos = cells[1].strip() if len(cells) > 1 else ""
            # Position validieren
            if pos not in ("C", "LW", "RW", "D", "G", "F", "L", "R", "W"):
                continue  # Keine Spieler-Row

            cap_hit = _parse_dollar(cells[2]) if len(cells) > 2 else 0
            cash = _parse_dollar(cells[3]) if len(cells) > 3 else 0
            cap_pct = _parse_pct(cells[4]) if len(cells) > 4 else 0.0
            bonus = _parse_dollar(cells[5]) if len(cells) > 5 else 0
            aav = _parse_dollar(cells[6]) if len(cells) > 6 else 0

            # Headshot matchen
            headshot = ""
            player_id = 0
            ln = last_name.lower()
            fn = first_name.lower()
            full_lower = f"{fn} {ln}"
            if full_lower in roster_map:
                headshot = roster_map[full_lower]["headshot"]
                player_id = roster_map[full_lower]["playerId"]
            elif ln in roster_map:
                headshot = roster_map[ln]["headshot"]
                player_id = roster_map[ln]["playerId"]

            player = {
                "name": display_name,
                "pos": pos,
                "capHit": cap_hit,
                "cash": cash,
                "capPct": cap_pct,
                "bonus": bonus,
                "aav": aav,
                "headshot": headshot,
                "playerId": player_id,
                "team": team_abbr,
            }

            # Tabelle 0=Active, 1=LTIR, 4=Minor, Rest ignorieren
            if t_idx == 4:  # Minor League
                result["minor_league"].append(player)
            elif t_idx in (0, 1):  # Active Roster + LTIR
                result["active_roster"].append(player)

    except Exception as e:
        result["error"] = str(e)

    salary_cap_cache[team_abbr] = result
    salary_cap_cache_time[team_abbr] = time.time()
    return result


# ---- Live Scores ----
_live_scores_cache = []
_live_scores_time = 0

def fetch_live_scores():
    """Holt alle heutigen NHL-Spiele mit Live-Scores."""
    global _live_scores_cache, _live_scores_time
    now = time.time()
    if now - _live_scores_time < 60:  # 60s Cache
        return _live_scores_cache

    try:
        today = datetime.now().strftime("%Y-%m-%d")
        resp = requests.get(f"https://api-web.nhle.com/v1/score/{today}", timeout=8)
        if resp.status_code != 200:
            return _live_scores_cache

        raw_games = resp.json().get("games", [])
        scores = []
        for g in raw_games:
            away = g.get("awayTeam", {})
            home = g.get("homeTeam", {})
            state = g.get("gameState", "FUT")
            period = g.get("periodDescriptor", {})
            clock = g.get("clock", {})
            start_utc = g.get("startTimeUTC", "")

            # Startzeit in lokale Zeit konvertieren
            start_display = ""
            if start_utc:
                try:
                    from datetime import timezone, timedelta
                    utc_dt = datetime.fromisoformat(start_utc.replace("Z", "+00:00"))
                    # EST = UTC-5
                    est_dt = utc_dt.replace(tzinfo=timezone.utc).astimezone(
                        timezone(timedelta(hours=-5))
                    )
                    start_display = est_dt.strftime("%-I:%M %p ET")
                except Exception:
                    start_display = start_utc[11:16] if len(start_utc) > 16 else ""

            # Status-Text
            if state in ("LIVE", "CRIT"):
                p_num = period.get("number", 0)
                p_type = period.get("periodType", "REG")
                time_left = clock.get("timeRemaining", "")
                in_int = clock.get("inIntermission", False)
                if in_int:
                    status = f"INT {p_num}"
                elif p_type == "OT":
                    status = f"OT {time_left}"
                elif p_type == "SO":
                    status = "SO"
                else:
                    status = f"P{p_num} {time_left}"
                status_class = "live"
            elif state in ("OFF", "FINAL"):
                p_num = period.get("number", 3)
                p_type = period.get("periodType", "REG")
                if p_type == "OT":
                    status = "Final/OT"
                elif p_type == "SO":
                    status = "Final/SO"
                elif p_num > 3:
                    status = f"Final/{p_num - 3}OT"
                else:
                    status = "Final"
                status_class = "final"
            else:
                status = start_display or "TBD"
                status_class = "scheduled"

            scores.append({
                "away": away.get("abbrev", "?"),
                "home": home.get("abbrev", "?"),
                "awayScore": away.get("score", 0),
                "homeScore": home.get("score", 0),
                "status": status,
                "statusClass": status_class,
                "gameId": g.get("id", 0),
            })

        _live_scores_cache = scores
        _live_scores_time = now
        return scores
    except Exception:
        return _live_scores_cache


# ---- Routen ----

@app.route("/")
def index():
    """Startseite – Zeigt kommende Spiele."""
    # Zuerst offene Tipps checken
    check_and_resolve_games()

    games = get_upcoming_games()
    live_scores = fetch_live_scores()

    # Team-Stats für alle Gegner + TOR parallel vorladen (Cache füllen)
    opponents = list(set(g["opponent"] for g in games))
    teams_to_load = [t for t in ["TOR"] + opponents if t not in team_stats_cache
                     or (time.time() - team_stats_cache_time.get(t, 0)) >= CACHE_TTL]
    if teams_to_load:
        with ThreadPoolExecutor(max_workers=10) as executor:
            executor.map(get_team_stats, teams_to_load)

    # Fun Stats (jetzt aus Cache, kein Warten)
    for i, game in enumerate(games):
        game["fun_stat"] = get_fun_stat_for_game(i, game["opponent"])

    # User-Stats Dashboard
    user_stats = None
    if g.username:
        resolved = get_resolved_predictions(g.username)
        pending = get_pending_predictions(g.username)
        if resolved or pending:
            total = len(resolved)
            correct = sum(1 for r in resolved if r.get("user_points", 0) > 0)
            exact = sum(1 for r in resolved if r.get("user_points", 0) == 3)
            points = sum(r.get("user_points", 0) for r in resolved)
            model_points = sum(r.get("model_points", 0) for r in resolved)

            # Streak berechnen (aktuelle Serie richtiger Tipps)
            streak = 0
            for r in resolved:
                if r.get("user_points", 0) > 0:
                    streak += 1
                else:
                    break

            # Badges berechnen
            badges = []
            if streak >= 3:
                badges.append({"icon": "🔥", "name": "Hot Streak", "desc": f"{streak} richtige in Folge"})
            if streak >= 5:
                badges.append({"icon": "⚡", "name": "On Fire", "desc": "5+ Streak"})
            if exact >= 1:
                badges.append({"icon": "🎯", "name": "Sniper", "desc": f"{exact}x exakter Tipp"})
            if exact >= 5:
                badges.append({"icon": "💎", "name": "Diamond", "desc": "5+ exakte Tipps"})
            if points > model_points and total >= 3:
                badges.append({"icon": "🏆", "name": "ML Slayer", "desc": "Besser als das Modell"})
            if total >= 10:
                badges.append({"icon": "🎮", "name": "Veteran", "desc": "10+ Tipps"})
            if total >= 25:
                badges.append({"icon": "👑", "name": "Legend", "desc": "25+ Tipps"})
            acc = round(correct / total * 100, 1) if total > 0 else 0
            if acc >= 70 and total >= 5:
                badges.append({"icon": "⭐", "name": "All-Star", "desc": "70%+ Genauigkeit"})

            user_stats = {
                "total": total,
                "pending": len(pending),
                "correct": correct,
                "exact": exact,
                "points": points,
                "model_points": model_points,
                "accuracy": acc,
                "streak": streak,
                "beating_model": points > model_points,
                "badges": badges,
            }

    # Tipp-Reminder: Gibt es Spiele ohne Tipp?
    untipped = [g for g in games if not g.get("already_predicted", False)]
    has_reminder = len(untipped) > 0 and g.username

    return render_template("index.html", games=games, active_page="index",
                           live_scores=live_scores, user_stats=user_stats,
                           has_reminder=has_reminder, untipped_count=len(untipped))


@app.route("/predict/<int:game_id>")
def predict(game_id):
    """Tipp-Seite für ein bestimmtes Spiel."""
    if "username" not in session:
        flash("Bitte erst einloggen!", "error")
        return redirect(url_for("login"))
    if prediction_exists(game_id, session["username"]):
        flash("Du hast für dieses Spiel schon getippt!", "error")
        return redirect(url_for("index"))

    # Spiel-Info von der API holen
    games = get_upcoming_games()
    game = None
    for g in games:
        if g["game_id"] == game_id:
            game = g
            break

    if not game:
        flash("Spiel nicht gefunden!", "error")
        return redirect(url_for("index"))

    # ML-Vorhersage
    model_pred = get_model_prediction(game["opponent"], game["is_home"])

    return render_template(
        "predict.html",
        game=game,
        model_prediction=model_pred,
        active_page="predict",
    )


@app.route("/submit_prediction", methods=["POST"])
def submit_prediction():
    """Verarbeitet einen abgegebenen Tipp."""
    if "username" not in session:
        flash("Bitte erst einloggen!", "error")
        return redirect(url_for("login"))
    username = session["username"]
    game_id = int(request.form["game_id"])
    game_date = request.form["game_date"]
    opponent = request.form["opponent"]
    is_home = 1 if request.form["is_home"] in ("1", "True") else 0
    user_prediction = request.form["user_prediction"]
    user_score_leafs = int(request.form["user_score_leafs"])
    user_score_opponent = int(request.form["user_score_opponent"])

    # ML-Vorhersage
    model_pred = get_model_prediction(opponent, bool(is_home))
    model_prediction = model_pred["prediction"] if model_pred else "W"
    model_win_prob = model_pred["win_probability"] if model_pred else 50.0

    # In Datenbank speichern
    add_prediction(
        username=username,
        game_id=game_id,
        game_date=game_date,
        opponent=opponent,
        is_home=is_home,
        user_prediction=user_prediction,
        user_score_leafs=user_score_leafs,
        user_score_opponent=user_score_opponent,
        model_prediction=model_prediction,
        model_win_probability=model_win_prob,
    )

    flash(f"Tipp gespeichert! TOR {user_prediction} ({user_score_leafs}:{user_score_opponent})", "success")
    return redirect(url_for("results"))


@app.route("/results")
def results():
    """Zeigt alle Tipps und Ergebnisse des eingeloggten Users."""
    username = session.get("username")
    pending = get_pending_predictions(username)
    resolved = get_resolved_predictions(username)

    # Chart data: kumulierte Punkte über Zeit (chronologisch)
    chart_data = {"labels": [], "user_pts": [], "model_pts": [], "accuracy": []}
    if resolved:
        sorted_res = list(reversed(resolved))  # chronologisch
        cum_user = 0
        cum_model = 0
        correct = 0
        for i, r in enumerate(sorted_res):
            cum_user += r.get("user_points", 0)
            cum_model += r.get("model_points", 0)
            if r.get("user_points", 0) > 0:
                correct += 1
            chart_data["labels"].append(r.get("game_date", f"Spiel {i+1}"))
            chart_data["user_pts"].append(cum_user)
            chart_data["model_pts"].append(cum_model)
            chart_data["accuracy"].append(round(correct / (i + 1) * 100, 1))

    return render_template(
        "results.html",
        pending=pending,
        resolved=resolved,
        chart_data=chart_data,
        active_page="results",
    )


@app.route("/leaderboard")
def leaderboard():
    """Zeigt das Leaderboard: User vs. Modell."""
    lb = get_leaderboard()
    return render_template(
        "leaderboard.html",
        lb=lb,
        active_page="leaderboard",
    )


_h2h_cache = {}
_h2h_cache_time = 0


@app.route("/head-to-head")
def head_to_head():
    """Head-to-Head Records – TOR vs. alle Teams."""
    global _h2h_cache, _h2h_cache_time
    now = time.time()

    if now - _h2h_cache_time < 600 and _h2h_cache:
        records = _h2h_cache
    else:
        try:
            resp = requests.get(
                f"{NHL_API}/club-schedule-season/TOR/{SEASON}", timeout=12
            )
            resp.raise_for_status()
            games = resp.json().get("games", [])
        except Exception as e:
            print(f"[H2H] Fehler: {e}")
            games = []

        records = {}
        for g in games:
            state = g.get("gameState", "")
            if state not in ("OFF", "FINAL"):
                continue
            away = g.get("awayTeam", {})
            home = g.get("homeTeam", {})
            if away.get("abbrev") == "TOR":
                opp = home.get("abbrev", "")
                tor_score = away.get("score", 0)
                opp_score = home.get("score", 0)
                is_home = False
            else:
                opp = away.get("abbrev", "")
                tor_score = home.get("score", 0)
                opp_score = away.get("score", 0)
                is_home = True
            if not opp:
                continue
            if opp not in records:
                records[opp] = {"w": 0, "l": 0, "otl": 0, "gf": 0, "ga": 0, "games": []}
            records[opp]["gf"] += tor_score
            records[opp]["ga"] += opp_score
            period = g.get("periodDescriptor", {})
            if tor_score > opp_score:
                records[opp]["w"] += 1
                result = "W"
            elif period.get("periodType", "REG") in ("OT", "SO"):
                records[opp]["otl"] += 1
                result = "OTL"
            else:
                records[opp]["l"] += 1
                result = "L"
            records[opp]["games"].append({
                "date": g.get("gameDate", ""),
                "tor": tor_score, "opp": opp_score,
                "result": result, "home": is_home,
                "gameId": g.get("id", 0),
            })

        _h2h_cache = records
        _h2h_cache_time = now

    # Sortieren: meiste Spiele zuerst
    sorted_records = sorted(records.items(), key=lambda x: -(x[1]["w"] + x[1]["l"] + x[1]["otl"]))

    return render_template(
        "head_to_head.html",
        records=sorted_records,
        active_page="h2h",
    )


_news_cache = []
_news_cache_time = 0


@app.route("/news")
def news():
    """NHL News Feed – Aktuelle Headlines."""
    global _news_cache, _news_cache_time
    now = time.time()

    if now - _news_cache_time < 300 and _news_cache:
        articles = _news_cache
    else:
        try:
            resp = requests.get(
                "https://forge-dapi.d3.nhle.com/v2/content/en-us/stories"
                "?context.slug=nhl&$skip=0&$top=20",
                timeout=10,
            )
            resp.raise_for_status()
            items = resp.json().get("items", [])
            articles = []
            for it in items:
                thumb_url = it.get("thumbnail", {}).get("thumbnailUrl", "")
                # Resize thumbnail
                if thumb_url and "t_ratio" in thumb_url:
                    thumb_url = thumb_url.replace("t_ratio1_1-size20", "t_ratio16_9-size40")
                articles.append({
                    "title": it.get("headline", ""),
                    "summary": it.get("summary", ""),
                    "date": it.get("contentDate", "")[:10],
                    "slug": it.get("slug", ""),
                    "thumbnail": thumb_url,
                    "url": f"https://www.nhl.com/news/{it.get('slug', '')}",
                })
            _news_cache = articles
            _news_cache_time = now
        except Exception as e:
            print(f"[News] Fehler: {e}")
            articles = _news_cache

    return render_template(
        "news.html",
        articles=articles,
        active_page="news",
    )


@app.route("/playoffs")
def playoffs():
    """Playoff Bracket – Beide Conferences."""
    data = fetch_playoff_data()
    return render_template(
        "playoffs.html",
        data=data,
        active_page="playoffs",
    )


@app.route("/trades")
def trades():
    """Trade Board – Aktuelle Trade-Gerüchte und Wahrscheinlichkeiten."""
    data = fetch_trade_data()
    return render_template(
        "trades.html",
        data=data,
        active_page="trades",
    )


@app.route("/salary")
@app.route("/salary/<team>")
def salary(team="TOR"):
    """Salary Cap Übersicht für ein NHL-Team."""
    team = team.upper()
    if team not in SPOTRAC_TEAMS:
        team = "TOR"
    data = fetch_salary_data(team)
    return render_template(
        "salary.html",
        data=data,
        active_page="salary",
        teams=SPOTRAC_TEAMS,
        current_team=team,
    )


@app.route("/compare")
def compare():
    """Zwei NHL-Spieler vergleichen."""
    p1_id = request.args.get("p1", "")
    p2_id = request.args.get("p2", "")
    query = request.args.get("q", "")

    # Spieler-Suche
    search_results = []
    if query:
        try:
            sr = requests.get(
                f"https://search.d3.nhle.com/api/v1/search/player?culture=en-us&limit=8&q={query}",
                timeout=8,
            )
            if sr.status_code == 200:
                for p in sr.json():
                    if p.get("active") and p.get("lastSeasonId") == "20252026":
                        search_results.append({
                            "id": p["playerId"],
                            "name": p["name"],
                            "pos": p.get("positionCode", ""),
                            "team": p.get("teamAbbrev", ""),
                            "number": p.get("sweaterNumber", ""),
                        })
        except Exception:
            pass

    # Spieler-Daten laden
    players = []
    for pid in [p1_id, p2_id]:
        if not pid:
            continue
        try:
            r = requests.get(
                f"https://api-web.nhle.com/v1/player/{pid}/landing",
                timeout=10,
            )
            if r.status_code != 200:
                continue
            d = r.json()
            season = d.get("featuredStats", {}).get("regularSeason", {}).get("subSeason", {})
            career = d.get("featuredStats", {}).get("regularSeason", {}).get("career", {})
            draft = d.get("draftDetails") or {}
            is_goalie = d.get("position") == "G"

            player_data = {
                "id": pid,
                "name": f"{d.get('firstName',{}).get('default','')} {d.get('lastName',{}).get('default','')}",
                "pos": d.get("position", ""),
                "team": d.get("currentTeamAbbrev", ""),
                "number": d.get("sweaterNumber", ""),
                "headshot": d.get("headshot", ""),
                "heroImage": d.get("heroImage", ""),
                "height": d.get("heightInCentimeters", 0),
                "weight": d.get("weightInKilograms", 0),
                "age": 0,
                "birthDate": d.get("birthDate", ""),
                "birthCity": d.get("birthCity", {}).get("default", "") if isinstance(d.get("birthCity"), dict) else d.get("birthCity", ""),
                "birthCountry": d.get("birthCountry", ""),
                "shoots": d.get("shootsCatches", ""),
                "draftYear": draft.get("year", ""),
                "draftPick": f"Runde {draft.get('round', '?')}, Pick {draft.get('overallPick', '?')}" if draft else "Undrafted",
                "isGoalie": is_goalie,
                "season": season,
                "career": career,
            }

            # Alter berechnen
            if player_data["birthDate"]:
                try:
                    bd = datetime.strptime(player_data["birthDate"], "%Y-%m-%d")
                    player_data["age"] = (datetime.now() - bd).days // 365
                except Exception:
                    pass

            players.append(player_data)
        except Exception:
            pass

    return render_template(
        "compare.html",
        players=players,
        search_results=search_results,
        query=query,
        p1=p1_id,
        p2=p2_id,
        active_page="compare",
    )


# ---- NHL Standings ----
_standings_cache = {}
_standings_cache_time = 0


def fetch_standings_data():
    """Holt aktuelle NHL Standings von der API. Cache: 10 min."""
    global _standings_cache, _standings_cache_time
    now = time.time()
    if now - _standings_cache_time < 600 and _standings_cache:
        return _standings_cache

    try:
        resp = requests.get(f"{NHL_API}/standings/now", timeout=15)
        resp.raise_for_status()
        raw = resp.json().get("standings", [])
    except Exception as e:
        print(f"[Standings] API-Fehler: {e}")
        return _standings_cache or {"divisions": {}, "conferences": {}, "wildcard": {}}

    # Nach Divisionen gruppieren
    divisions = {}
    conferences = {}  # Für Wildcard-Berechnung

    for team in raw:
        div_name = team.get("divisionName", "Unknown")
        conf_name = team.get("conferenceName", "Unknown")

        entry = {
            "abbr": team.get("teamAbbrev", {}).get("default", ""),
            "name": team.get("teamName", {}).get("default", ""),
            "logo": team.get("teamLogo", ""),
            "gp": team.get("gamesPlayed", 0),
            "w": team.get("wins", 0),
            "l": team.get("losses", 0),
            "otl": team.get("otLosses", 0),
            "pts": team.get("points", 0),
            "ptsPct": round(team.get("pointPctg", 0), 3),
            "rw": team.get("regulationWins", 0),
            "row": team.get("regulationPlusOtWins", 0),
            "gf": team.get("goalFor", 0),
            "ga": team.get("goalAgainst", 0),
            "diff": team.get("goalDifferential", 0),
            "streak": f"{team.get('streakCode', '')}{team.get('streakCount', '')}",
            "l10w": team.get("l10Wins", 0),
            "l10l": team.get("l10Losses", 0),
            "l10otl": team.get("l10OtLosses", 0),
            "homeW": team.get("homeWins", 0),
            "homeL": team.get("homeLosses", 0),
            "homeOtl": team.get("homeOtLosses", 0),
            "awayW": team.get("roadWins", 0),
            "awayL": team.get("roadLosses", 0),
            "awayOtl": team.get("roadOtLosses", 0),
            "divisionSequence": team.get("divisionSequence", 99),
            "wildcardSequence": team.get("wildcardSequence", 99),
            "conferenceSequence": team.get("conferenceSequence", 99),
            "clinchIndicator": team.get("clinchIndicator", ""),
        }

        if div_name not in divisions:
            divisions[div_name] = []
        divisions[div_name].append(entry)

        if conf_name not in conferences:
            conferences[conf_name] = []
        conferences[conf_name].append(entry)

    # Jede Division nach divisionSequence sortieren
    for div in divisions:
        divisions[div].sort(key=lambda x: x["divisionSequence"])

    # Wildcard Race: Teams auf Platz 4+ in jeder Division, nach Conf sortiert
    wildcard = {}
    for conf_name, conf_teams in conferences.items():
        # Teams die nicht in den Top 3 ihrer Division sind
        wc_teams = [t for t in conf_teams if t["divisionSequence"] > 3]
        wc_teams.sort(key=lambda x: x["wildcardSequence"])
        wildcard[conf_name] = wc_teams

    result = {
        "divisions": divisions,
        "wildcard": wildcard,
        "conferences": conferences,
    }

    _standings_cache = result
    _standings_cache_time = now
    return result


@app.route("/standings")
def standings():
    """NHL Standings – Divisionen + Wildcard Race."""
    data = fetch_standings_data()
    return render_template(
        "standings.html",
        data=data,
        active_page="standings",
    )


_leaders_cache = {}
_leaders_cache_time = 0


@app.route("/leaders")
def leaders():
    """NHL Stats Leaders – Torschützen, Assists, Punkte, Goalies."""
    global _leaders_cache, _leaders_cache_time
    now = time.time()
    if now - _leaders_cache_time < 600 and _leaders_cache:
        data = _leaders_cache
    else:
        data = {}
        for cat in ["goals", "assists", "points", "plusMinus"]:
            try:
                resp = requests.get(
                    f"{NHL_API}/skater-stats-leaders/current"
                    f"?categories={cat}&limit=10", timeout=8
                )
                if resp.status_code == 200:
                    data[cat] = resp.json().get(cat, [])
            except Exception:
                data[cat] = []
        for cat in ["wins", "savePctg", "goalsAgainstAverage"]:
            try:
                resp = requests.get(
                    f"{NHL_API}/goalie-stats-leaders/current"
                    f"?categories={cat}&limit=10", timeout=8
                )
                if resp.status_code == 200:
                    data[cat] = resp.json().get(cat, [])
            except Exception:
                data[cat] = []
        _leaders_cache = data
        _leaders_cache_time = now

    return render_template("leaders.html", data=data, active_page="leaders")


@app.route("/team/<abbr>")
def team_page(abbr):
    """Team Detail Page – Roster, Stats, Schedule."""
    abbr = abbr.upper()
    try:
        # Roster
        roster_resp = requests.get(f"{NHL_API}/roster/{abbr}/current", timeout=10)
        roster_resp.raise_for_status()
        roster = roster_resp.json()

        # Club Stats (Spieler-Statistiken)
        stats_resp = requests.get(f"{NHL_API}/club-stats/{abbr}/now", timeout=10)
        stats_data = stats_resp.json() if stats_resp.status_code == 200 else {}
        skaters = stats_data.get("skaters", [])
        goalie_stats = stats_data.get("goalies", [])

        # Nächste Spiele
        schedule_resp = requests.get(
            f"{NHL_API}/club-schedule-season/{abbr}/{SEASON}", timeout=10
        )
        schedule = schedule_resp.json().get("games", []) if schedule_resp.status_code == 200 else []

        # Nur zukünftige + letzte 5 Spiele
        upcoming = [g for g in schedule if g.get("gameState") in ("FUT", "PRE")][:5]
        recent = [g for g in schedule
                  if g.get("gameState") in ("OFF", "FINAL")][-5:]
        recent.reverse()

        # Team Record aus Standings
        standings = fetch_standings_data()
        team_record = None
        for div_teams in standings.get("divisions", {}).values():
            for t in div_teams:
                if t["abbr"] == abbr:
                    team_record = t
                    break

        # Skater stats sortiert nach Punkten
        skaters.sort(key=lambda x: x.get("points", 0), reverse=True)

        return render_template(
            "team_page.html",
            abbr=abbr,
            roster=roster,
            skaters=skaters[:15],
            goalie_stats=goalie_stats,
            upcoming=upcoming,
            recent=recent,
            record=team_record,
            active_page="team",
        )
    except Exception as e:
        print(f"[Team] Fehler: {e}")
        flash(f"Team {abbr} konnte nicht geladen werden.", "error")
        return redirect(url_for("standings"))


@app.route("/calendar")
@app.route("/calendar/<team>")
def calendar(team="TOR"):
    """Season Schedule Calendar – visuelle Monatsübersicht."""
    team = team.upper()
    try:
        resp = requests.get(
            f"{NHL_API}/club-schedule-season/{team}/{SEASON}", timeout=10
        )
        resp.raise_for_status()
        games = resp.json().get("games", [])
    except Exception:
        games = []

    # Spiele nach Monat gruppieren
    months = {}
    for g in games:
        date_str = g.get("gameDate", "")
        if not date_str:
            continue
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
        except Exception:
            continue
        month_key = dt.strftime("%Y-%m")
        month_name = dt.strftime("%B %Y")
        if month_key not in months:
            months[month_key] = {"name": month_name, "days": {}, "year": dt.year, "month": dt.month}

        away = g.get("awayTeam", {})
        home = g.get("homeTeam", {})
        is_home = home.get("abbrev", "") == team
        opp = away.get("abbrev", "") if is_home else home.get("abbrev", "")
        state = g.get("gameState", "FUT")

        day_data = {
            "day": dt.day,
            "opp": opp,
            "is_home": is_home,
            "game_id": g.get("id", 0),
            "state": state,
        }

        if state in ("OFF", "FINAL"):
            team_score = home.get("score", 0) if is_home else away.get("score", 0)
            opp_score = away.get("score", 0) if is_home else home.get("score", 0)
            day_data["score"] = f"{team_score}-{opp_score}"
            day_data["result"] = "W" if team_score > opp_score else ("L" if team_score < opp_score else "T")
        else:
            day_data["score"] = ""
            day_data["result"] = ""

        months[month_key]["days"][dt.day] = day_data

    # Calendar-Grid pro Monat (Wochentage)
    import calendar as cal_mod
    cal_months = []
    for key in sorted(months.keys()):
        m = months[key]
        first_weekday, num_days = cal_mod.monthrange(m["year"], m["month"])
        # first_weekday: 0=Mon, 6=Sun. We want Mon-Sun layout.
        grid = []
        # Leere Zellen am Anfang
        for _ in range(first_weekday):
            grid.append(None)
        for day in range(1, num_days + 1):
            grid.append(m["days"].get(day, {"day": day, "opp": None}))
        cal_months.append({"name": m["name"], "grid": grid})

    # Team-Liste für Dropdown
    all_teams = [
        "ANA","ARI","BOS","BUF","CGY","CAR","CHI","COL","CBJ","DAL","DET",
        "EDM","FLA","LAK","MIN","MTL","NSH","NJD","NYI","NYR","OTT","PHI",
        "PIT","SJS","SEA","STL","TBL","TOR","UTA","VAN","VGK","WSH","WPG"
    ]

    return render_template(
        "calendar.html",
        cal_months=cal_months,
        current_team=team,
        all_teams=all_teams,
        active_page="calendar",
    )


@app.route("/power-rankings")
def power_rankings():
    """Power Rankings basierend auf aktuellem Record + Trend."""
    standings = fetch_standings_data()
    all_teams = []
    for div_teams in standings.get("divisions", {}).values():
        all_teams.extend(div_teams)

    # Sortieren nach Punkte-Prozent (pts / (gp * 2))
    for t in all_teams:
        gp = t.get("gp", 1) or 1
        t["pts_pct"] = round(t.get("pts", 0) / (gp * 2) * 100, 1)
        t["gpg"] = round(t.get("gf", 0) / gp, 2)
        t["gapg"] = round(t.get("ga", 0) / gp, 2)
        t["diff_pg"] = round(t.get("diff", 0) / gp, 2)
        # Streak-Faktor: L10-Wins als Trend-Indikator
        l10 = t.get("l10", "")
        try:
            l10_w = int(l10.split("-")[0]) if "-" in l10 else 5
        except Exception:
            l10_w = 5
        t["l10_wins"] = l10_w
        # Power Score = 60% Punkte% + 25% L10 + 15% Diff/G
        t["power_score"] = round(
            t["pts_pct"] * 0.6 +
            l10_w * 10 * 0.25 +
            (t["diff_pg"] + 2) * 15 * 0.15,  # normalize diff_pg around 0
            1
        )

    all_teams.sort(key=lambda x: x["power_score"], reverse=True)

    # Ranking zuweisen
    for i, t in enumerate(all_teams):
        t["rank"] = i + 1
        # Tier-Einteilung
        if i < 8:
            t["tier"] = "elite"
        elif i < 16:
            t["tier"] = "contender"
        elif i < 24:
            t["tier"] = "bubble"
        else:
            t["tier"] = "rebuild"

    return render_template(
        "power_rankings.html",
        teams=all_teams,
        active_page="power-rankings",
    )


@app.route("/api/export-stats")
def export_stats():
    """Exportiert User-Tipps als CSV."""
    import csv
    import io

    username = session.get("username")
    if not username:
        return {"error": "Nicht eingeloggt"}, 401

    resolved = get_resolved_predictions(username)

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Datum", "Gegner", "Heim/Aus", "Tipp", "Ergebnis", "Punkte", "ML-Punkte", "Exakt"])
    for r in resolved:
        writer.writerow([
            r.get("game_date", ""),
            r.get("opponent", ""),
            "Heim" if r.get("is_home") else "Auswärts",
            f"{r.get('predicted_home',0)}-{r.get('predicted_away',0)}",
            f"{r.get('actual_home','-')}-{r.get('actual_away','-')}",
            r.get("user_points", 0),
            r.get("model_points", 0),
            "Ja" if r.get("exact_match") else "Nein",
        ])

    from flask import Response
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename=edge-nhl-stats-{username}.csv"}
    )


@app.route("/draft-lottery")
def draft_lottery():
    """Draft Lottery Simulator – Interaktive Lotterie-Simulation."""
    # Aktuelle Standings holen um Bottom-16 Teams zu ermitteln
    standings_data = fetch_standings_data()

    # Alle Teams nach Punkten sortieren (aufsteigend = schlechteste zuerst)
    all_teams = []
    for div_teams in standings_data.get("divisions", {}).values():
        all_teams.extend(div_teams)
    all_teams.sort(key=lambda x: (x["pts"], x["diff"]))

    # Bottom 16 = Lottery Teams
    lottery_teams = all_teams[:16]

    # Offizielle NHL Draft Lottery Odds (2024/25 Regeln)
    # Position 1-16 → Prozentsatz für 1st Overall Pick
    lottery_odds = [
        18.5, 13.5, 11.5, 9.5, 8.5, 7.5, 6.5, 6.0,
        5.0, 3.5, 3.0, 2.5, 2.0, 1.5, 0.5, 0.5
    ]

    for i, team in enumerate(lottery_teams):
        team["seed"] = i + 1
        team["odds"] = lottery_odds[i] if i < len(lottery_odds) else 0.5

    return render_template(
        "draft_lottery.html",
        teams=lottery_teams,
        active_page="draft-lottery",
    )


@app.route("/scoreboard")
def scoreboard():
    """NHL Scoreboard – Aktuelle Liga-Statistiken."""
    # Headshot-Cache parallel laden während Scoreboard-Daten geholt werden
    hs_future = None
    if not _headshot_cache and _headshot_cache_time == 0:
        hs_executor = ThreadPoolExecutor(max_workers=1)
        hs_future = hs_executor.submit(refresh_headshot_cache)

    raw = fetch_scoreboard_data()

    # Warten bis Headshot-Cache fertig ist (max 45s, normalerweise ~20s)
    if hs_future:
        try:
            hs_future.result(timeout=45)
        except Exception:
            pass

    data = format_scoreboard(raw)
    return render_template(
        "scoreboard.html",
        data=data,
        active_page="scoreboard",
    )


@app.route("/game/<int:game_id>")
def game_detail(game_id):
    """Game Detail – Boxscore, Scoring, Stats."""
    try:
        # Landing für Scoring + Three Stars + Penalties
        landing_resp = requests.get(
            f"{NHL_API}/gamecenter/{game_id}/landing", timeout=12
        )
        landing_resp.raise_for_status()
        landing = landing_resp.json()

        # Boxscore für Player Stats
        box_resp = requests.get(
            f"{NHL_API}/gamecenter/{game_id}/boxscore", timeout=12
        )
        box_resp.raise_for_status()
        boxscore = box_resp.json()

        away = landing.get("awayTeam", {})
        home = landing.get("homeTeam", {})
        state = landing.get("gameState", "FUT")

        # Summary data
        summary = landing.get("summary", {})
        scoring = summary.get("scoring", [])
        three_stars = summary.get("threeStars", [])
        penalties = summary.get("penalties", [])

        # Player stats from boxscore
        player_stats = boxscore.get("playerByGameStats", {})

        # Team game stats berechnen aus Player-Boxscore
        def calc_team_stats(team_data):
            stats = {"sog": 0, "hits": 0, "blocks": 0, "pim": 0,
                     "giveaways": 0, "takeaways": 0, "faceoffPct": 0}
            all_players = (team_data.get("forwards", []) +
                          team_data.get("defense", []))
            fo_total = 0
            for p in all_players:
                stats["sog"] += p.get("sog", 0)
                stats["hits"] += p.get("hits", 0)
                stats["blocks"] += p.get("blockedShots", 0)
                stats["pim"] += p.get("pim", 0)
                stats["giveaways"] += p.get("giveaways", 0)
                stats["takeaways"] += p.get("takeaways", 0)
                fo_pct = p.get("faceoffWinningPctg", 0)
                if fo_pct and fo_pct > 0:
                    fo_total += 1
            return stats

        away_stats = calc_team_stats(player_stats.get("awayTeam", {}))
        home_stats = calc_team_stats(player_stats.get("homeTeam", {}))

        # Period/Clock info
        period = landing.get("periodDescriptor", {})
        clock = landing.get("clock", {})

        game_data = {
            "id": game_id,
            "state": state,
            "date": landing.get("gameDate", ""),
            "venue": landing.get("venue", {}).get("default", ""),
            "away": {
                "abbr": away.get("abbrev", ""),
                "name": away.get("commonName", {}).get("default", away.get("abbrev", "")),
                "score": away.get("score", 0),
                "sog": away.get("sog", away_stats["sog"]),
                "logo": f"https://assets.nhle.com/logos/nhl/svg/{away.get('abbrev', '')}_dark.svg",
            },
            "home": {
                "abbr": home.get("abbrev", ""),
                "name": home.get("commonName", {}).get("default", home.get("abbrev", "")),
                "score": home.get("score", 0),
                "sog": home.get("sog", home_stats["sog"]),
                "logo": f"https://assets.nhle.com/logos/nhl/svg/{home.get('abbrev', '')}_dark.svg",
            },
            "away_stats": away_stats,
            "home_stats": home_stats,
            "scoring": scoring,
            "three_stars": three_stars,
            "penalties": penalties,
            "players": player_stats,
            "period": period,
            "clock": clock,
        }

        return render_template(
            "game_detail.html",
            game=game_data,
            active_page="game",
        )
    except Exception as e:
        print(f"[Game Detail] Fehler: {e}")
        flash("Spiel konnte nicht geladen werden.", "error")
        return redirect(url_for("index"))


@app.route("/api/live-scores")
def api_live_scores():
    """API-Endpunkt: Live-Scores als JSON für AJAX-Polling."""
    scores = fetch_live_scores()
    has_live = any(s["statusClass"] == "live" for s in scores)
    return {"scores": scores, "hasLive": has_live}


@app.route("/api/status")
def api_status():
    """API-Endpunkt: Zeigt den Status des Auto-Update Systems."""
    return {
        "status": "running",
        "last_update": last_data_update.strftime("%Y-%m-%d %H:%M:%S") if last_data_update else "nie",
        "games_in_csv": len(game_data) if game_data is not None else 0,
        "model_loaded": ml_model is not None,
        "cached_teams": list(team_stats_cache.keys()),
        "cache_ttl_minutes": CACHE_TTL // 60,
    }


@app.route("/api/force-update")
def force_update():
    """Manuelles Update auslösen."""
    auto_update_cycle()
    return {
        "message": "Update-Zyklus abgeschlossen!",
        "games_in_csv": len(game_data) if game_data is not None else 0,
        "last_update": last_data_update.strftime("%Y-%m-%d %H:%M:%S") if last_data_update else "nie",
    }


# ---- Start ----
# Initialisierung (wird sowohl von gunicorn als auch direkt aufgerufen)
def _init_app():
    init_db()
    load_ml_model()
    # Ersten Update im Hintergrund starten (blockiert nicht den Server-Start)
    import threading
    def _bg_init():
        try:
            print("\n[Start] Fuehre ersten Auto-Update Zyklus aus...")
            auto_update_cycle()
            # Caches vorladen damit erste Seitenaufrufe instant sind
            print("[Start] Lade Caches vor...")
            # Headshot-Cache ZUERST (wird vom Scoreboard gebraucht)
            try:
                refresh_headshot_cache()
                print("[Start] Headshot-Cache geladen.")
            except Exception:
                pass
            # Scoreboard vorladen
            try:
                fetch_scoreboard_data()
                print("[Start] Scoreboard-Cache geladen.")
            except Exception:
                pass
            # Team-Stats für kommende Gegner vorladen
            try:
                games = get_upcoming_games()
                teams = list(set(["TOR"] + [g["opponent"] for g in games]))
                with ThreadPoolExecutor(max_workers=10) as executor:
                    executor.map(get_team_stats, teams)
                print(f"[Start] Team-Stats für {len(teams)} Teams vorgeladen.")
            except Exception:
                pass
        except Exception as e:
            print(f"[Start] Fehler beim ersten Update: {e}")
    threading.Thread(target=_bg_init, daemon=True).start()
    start_scheduler()

# Immer initialisieren (gunicorn + direkt)
_init_app()

if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("  EDGE NHL gestartet!")
    print("  Oeffne: http://localhost:8080")
    print("  Auto-Update: alle 30 Minuten")
    print("  Status:  http://localhost:8080/api/status")
    print("  Manuell: http://localhost:8080/api/force-update")
    print("=" * 50 + "\n")

    port = int(os.environ.get("PORT", 8080))
    debug = os.environ.get("RENDER") is None  # Lokal: debug an, Render: aus
    app.run(debug=debug, host="0.0.0.0", port=port, use_reloader=False)
