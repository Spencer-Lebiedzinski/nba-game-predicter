import json
import os
import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import pandas as pd
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Resolve project root so we can import sibling modules & load model artifacts
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# from live_feed_reader import LiveMarketFeed

# ---------------------------------------------------------------------------
# Live market feed (reads JSONL produced by market agents)
# ---------------------------------------------------------------------------
class LiveMarketFeed:
    def __init__(self, jsonl_file: str):
        self.jsonl_file = jsonl_file
        self.latest_data: Dict = {}

    def load_latest(self, max_age_seconds: int = 300) -> Dict:
        """Load latest data from JSONL file within max_age"""
        path = Path(self.jsonl_file)
        if not path.exists():
            return {}

        try:
            from datetime import datetime, timedelta
            cutoff_time = datetime.utcnow() - timedelta(seconds=max_age_seconds)

            with open(self.jsonl_file, 'r') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        game_key = data.get('game_key')
                        if not game_key:
                            continue

                        # Check if data is recent enough
                        last_updated = data.get('last_updated')
                        if last_updated:
                            try:
                                update_time = datetime.fromisoformat(last_updated)
                                if update_time < cutoff_time:
                                    continue
                            except:
                                pass
                        
                        self.latest_data[game_key] = data
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"Error loading live data: {e}")

        return self.latest_data

    def get_game_odds(self, game_key: str) -> Optional[Dict]:
        """Get latest odds for a specific game"""
        return self.latest_data.get(game_key)

    def get_all_games(self) -> Dict:
        """Get all games with latest odds"""
        return self.latest_data

# ---------------------------------------------------------------------------
# Load trained ML model artifacts
# ---------------------------------------------------------------------------
MODEL_DIR = PROJECT_ROOT / "models"

def load_artifacts():
    try:
        model = joblib.load(MODEL_DIR / "nba_predictor_model.pkl")
        teams_list = joblib.load(MODEL_DIR / "teams.pkl")
        team_features = joblib.load(MODEL_DIR / "team_features.pkl")
        feature_cols = joblib.load(MODEL_DIR / "feature_cols.pkl")
        print("✅ Model artifacts loaded successfully.")
        return model, teams_list, team_features, feature_cols
    except FileNotFoundError as e:
        print(f"❌ Error: Model artifact missing: {e.filename}")
        print("Please run 'python models/train_model.py' to generate artifacts.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error loading model artifacts: {e}")
        sys.exit(1)

model, teams_list, team_features, feature_cols = load_artifacts()

# ---------------------------------------------------------------------------
# Live market feed (reads JSONL produced by market agents)
# ---------------------------------------------------------------------------
live_feed = LiveMarketFeed(str(PROJECT_ROOT / "data" / "live_probs.jsonl"))


def _refresh_live_data():
    """Background thread that polls the JSONL file every 10 s."""
    while True:
        live_feed.load_latest()
        threading.Event().wait(10)


# ---------------------------------------------------------------------------
# Team metadata (full names + NBA CDN logos)
# ---------------------------------------------------------------------------
TEAM_INFO: Dict[str, dict] = {
    "ATL": {"name": "Atlanta Hawks",          "logo": "https://cdn.nba.com/logos/nba/1610612737/primary/L/logo.svg"},
    "BOS": {"name": "Boston Celtics",         "logo": "https://cdn.nba.com/logos/nba/1610612738/primary/L/logo.svg"},
    "BKN": {"name": "Brooklyn Nets",          "logo": "https://cdn.nba.com/logos/nba/1610612751/primary/L/logo.svg"},
    "CHA": {"name": "Charlotte Hornets",      "logo": "https://cdn.nba.com/logos/nba/1610612766/primary/L/logo.svg"},
    "CHI": {"name": "Chicago Bulls",          "logo": "https://cdn.nba.com/logos/nba/1610612741/primary/L/logo.svg"},
    "CLE": {"name": "Cleveland Cavaliers",    "logo": "https://cdn.nba.com/logos/nba/1610612739/primary/L/logo.svg"},
    "DAL": {"name": "Dallas Mavericks",       "logo": "https://cdn.nba.com/logos/nba/1610612742/primary/L/logo.svg"},
    "DEN": {"name": "Denver Nuggets",         "logo": "https://cdn.nba.com/logos/nba/1610612743/primary/L/logo.svg"},
    "DET": {"name": "Detroit Pistons",        "logo": "https://cdn.nba.com/logos/nba/1610612765/primary/L/logo.svg"},
    "GSW": {"name": "Golden State Warriors",  "logo": "https://cdn.nba.com/logos/nba/1610612744/primary/L/logo.svg"},
    "HOU": {"name": "Houston Rockets",        "logo": "https://cdn.nba.com/logos/nba/1610612745/primary/L/logo.svg"},
    "IND": {"name": "Indiana Pacers",         "logo": "https://cdn.nba.com/logos/nba/1610612754/primary/L/logo.svg"},
    "LAC": {"name": "LA Clippers",            "logo": "https://cdn.nba.com/logos/nba/1610612746/primary/L/logo.svg"},
    "LAL": {"name": "Los Angeles Lakers",     "logo": "https://cdn.nba.com/logos/nba/1610612747/primary/L/logo.svg"},
    "MEM": {"name": "Memphis Grizzlies",      "logo": "https://cdn.nba.com/logos/nba/1610612763/primary/L/logo.svg"},
    "MIA": {"name": "Miami Heat",             "logo": "https://cdn.nba.com/logos/nba/1610612748/primary/L/logo.svg"},
    "MIL": {"name": "Milwaukee Bucks",        "logo": "https://cdn.nba.com/logos/nba/1610612749/primary/L/logo.svg"},
    "MIN": {"name": "Minnesota Timberwolves", "logo": "https://cdn.nba.com/logos/nba/1610612750/primary/L/logo.svg"},
    "NOP": {"name": "New Orleans Pelicans",   "logo": "https://cdn.nba.com/logos/nba/1610612740/primary/L/logo.svg"},
    "NYK": {"name": "New York Knicks",        "logo": "https://cdn.nba.com/logos/nba/1610612752/primary/L/logo.svg"},
    "OKC": {"name": "Oklahoma City Thunder",  "logo": "https://cdn.nba.com/logos/nba/1610612760/primary/L/logo.svg"},
    "ORL": {"name": "Orlando Magic",          "logo": "https://cdn.nba.com/logos/nba/1610612753/primary/L/logo.svg"},
    "PHI": {"name": "Philadelphia 76ers",     "logo": "https://cdn.nba.com/logos/nba/1610612755/primary/L/logo.svg"},
    "PHX": {"name": "Phoenix Suns",           "logo": "https://cdn.nba.com/logos/nba/1610612756/primary/L/logo.svg"},
    "POR": {"name": "Portland Trail Blazers", "logo": "https://cdn.nba.com/logos/nba/1610612757/primary/L/logo.svg"},
    "SAC": {"name": "Sacramento Kings",       "logo": "https://cdn.nba.com/logos/nba/1610612758/primary/L/logo.svg"},
    "SAS": {"name": "San Antonio Spurs",      "logo": "https://cdn.nba.com/logos/nba/1610612759/primary/L/logo.svg"},
    "TOR": {"name": "Toronto Raptors",        "logo": "https://cdn.nba.com/logos/nba/1610612761/primary/L/logo.svg"},
    "UTA": {"name": "Utah Jazz",              "logo": "https://cdn.nba.com/logos/nba/1610612762/primary/L/logo.svg"},
    "WAS": {"name": "Washington Wizards",     "logo": "https://cdn.nba.com/logos/nba/1610612764/primary/L/logo.svg"},
}


# ===================== Pydantic Schemas =====================

class TeamOut(BaseModel):
    id: str
    name: str
    logo_url: str


class PredictRequest(BaseModel):
    home_team: str
    away_team: str


class TeamDetail(BaseModel):
    id: str
    name: str
    logo_url: str
    rating: float


class GamePrediction(BaseModel):
    id: str
    home_team: TeamDetail
    away_team: TeamDetail
    prob_home_win: float
    prob_away_win: float
    predicted_score_home: int
    predicted_score_away: int
    home_market_implied: float
    away_market_implied: float
    confidence: str
    value_edge_home: float
    value_edge_away: float
    features: dict


class PredictResponse(BaseModel):
    home_team: str
    away_team: str
    home_name: str
    away_name: str
    home_logo: str
    away_logo: str
    prob_home_win: float
    prob_away_win: float
    winner: str
    confidence: float


# ===================== Helpers =====================

def _predict_game(home_abbr: str, away_abbr: str) -> float:
    """Run the ML model and return P(home win)."""
    features = []
    for col in feature_cols:
        if col.startswith("HOME_"):
            feat_name = col[5:]
            features.append(team_features.get(home_abbr, {}).get(feat_name, 0))
        else:
            feat_name = col[5:]
            features.append(team_features.get(away_abbr, {}).get(feat_name, 0))
    df = pd.DataFrame([features], columns=feature_cols)
    return float(model.predict_proba(df)[0][1])


def _get_upcoming_games() -> list:
    """Fetch today's NBA scoreboard from ESPN."""
    try:
        url = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
        resp = requests.get(url, timeout=5)
        data = resp.json()
        games = []
        for event in data.get("events", []):
            status = event["status"]["type"]["name"]
            comps = event["competitions"][0]

            home_abbr = comps["competitors"][0]["team"]["abbreviation"]
            away_abbr = comps["competitors"][1]["team"]["abbreviation"]

            # ESPN sometimes swaps home/away – "competitors[0]" has homeAway field
            for c in comps["competitors"]:
                if c.get("homeAway") == "home":
                    home_abbr = c["team"]["abbreviation"]
                elif c.get("homeAway") == "away":
                    away_abbr = c["team"]["abbreviation"]

            if home_abbr in TEAM_INFO and away_abbr in TEAM_INFO:
                games.append({
                    "home": home_abbr,
                    "away": away_abbr,
                    "status": status,
                })
        return games[:12]
    except Exception as exc:
        print(f"ESPN API error: {exc}")
        return []


def _confidence_label(prob_home: float, prob_away: float) -> str:
    diff = abs(prob_home - prob_away) * 100
    if diff > 10:
        return "High"
    elif diff > 5:
        return "Medium"
    return "Low"


# ===================== App lifecycle =====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start background market data refresh
    t = threading.Thread(target=_refresh_live_data, daemon=True)
    t.start()
    yield


# ===================== FastAPI App =====================

app = FastAPI(title="AstroHoops Analytics API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------- Routes --------------------

@app.get("/")
def root():
    return {"status": "ok", "message": "AstroHoops API is running"}


@app.get("/api/teams", response_model=List[TeamOut])
def get_teams():
    """Return all 30 NBA teams for dropdown selectors."""
    return [
        TeamOut(id=abbr, name=info["name"], logo_url=info["logo"])
        for abbr, info in sorted(TEAM_INFO.items(), key=lambda x: x[1]["name"])
    ]


@app.get("/api/games/today", response_model=List[GamePrediction])
def get_todays_games():
    """Fetch today's games from ESPN, run ML predictions, attach market data."""
    espn_games = _get_upcoming_games()
    results: List[GamePrediction] = []

    for g in espn_games:
        home = g["home"]
        away = g["away"]
        prob_home = _predict_game(home, away)
        prob_away = 1.0 - prob_home

        # Market data from live feed
        game_key = f"{away.upper()}_{home.upper()}"
        market = live_feed.get_game_odds(game_key) or {}
        home_market = 0.50
        away_market = 0.50
        if market:
            poly = market.get("polymarket", {})
            kalshi = market.get("kalshi", {})
            if poly and poly.get("home"):
                home_market = poly["home"].get("midpoint_pct", 0.5)
                away_market = poly["away"].get("midpoint_pct", 0.5)
            elif kalshi:
                home_market = kalshi.get("midpoint_pct", 0.5)
                away_market = 1.0 - home_market

        # Rough score estimate (baseline ~105 adjusted by prob)
        base = 105
        score_home = int(base + (prob_home - 0.5) * 20)
        score_away = int(base + (prob_away - 0.5) * 20)

        home_info = TEAM_INFO[home]
        away_info = TEAM_INFO[away]

        # Build a small feature explanation dict
        h_feats = team_features.get(home, {})
        a_feats = team_features.get(away, {})
        explanation: dict = {}
        if "W_PCT" in h_feats and "W_PCT" in a_feats:
            explanation["home_win_pct"] = f"{h_feats['W_PCT']:.3f}"
            explanation["away_win_pct"] = f"{a_feats['W_PCT']:.3f}"
        if "OFF_RATING" in h_feats and "OFF_RATING" in a_feats:
            explanation["off_rating_diff"] = f"{h_feats['OFF_RATING'] - a_feats['OFF_RATING']:+.1f}"
        if "DEF_RATING" in h_feats and "DEF_RATING" in a_feats:
            explanation["def_rating_diff"] = f"{h_feats['DEF_RATING'] - a_feats['DEF_RATING']:+.1f}"

        results.append(GamePrediction(
            id=f"{away}_{home}",
            home_team=TeamDetail(id=home, name=home_info["name"], logo_url=home_info["logo"],
                                 rating=h_feats.get("OFF_RATING", 0)),
            away_team=TeamDetail(id=away, name=away_info["name"], logo_url=away_info["logo"],
                                 rating=a_feats.get("OFF_RATING", 0)),
            prob_home_win=round(prob_home, 4),
            prob_away_win=round(prob_away, 4),
            predicted_score_home=score_home,
            predicted_score_away=score_away,
            home_market_implied=round(home_market, 4),
            away_market_implied=round(away_market, 4),
            confidence=_confidence_label(prob_home, prob_away),
            value_edge_home=round(prob_home - home_market, 4),
            value_edge_away=round(prob_away - away_market, 4),
            features=explanation,
        ))

    return results


@app.post("/api/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """Manual team-vs-team prediction."""
    home = req.home_team.upper()
    away = req.away_team.upper()

    if home not in TEAM_INFO:
        raise HTTPException(status_code=400, detail=f"Unknown home team: {home}")
    if away not in TEAM_INFO:
        raise HTTPException(status_code=400, detail=f"Unknown away team: {away}")
    if home == away:
        raise HTTPException(status_code=400, detail="Teams must be different")

    prob_home = _predict_game(home, away)
    prob_away = 1.0 - prob_home
    winner = home if prob_home > prob_away else away

    return PredictResponse(
        home_team=home,
        away_team=away,
        home_name=TEAM_INFO[home]["name"],
        away_name=TEAM_INFO[away]["name"],
        home_logo=TEAM_INFO[home]["logo"],
        away_logo=TEAM_INFO[away]["logo"],
        prob_home_win=round(prob_home, 4),
        prob_away_win=round(prob_away, 4),
        winner=winner,
        confidence=round(max(prob_home, prob_away) * 100, 1),
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
