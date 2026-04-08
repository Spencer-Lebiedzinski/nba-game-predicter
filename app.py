from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import requests
from datetime import datetime
import threading
from live_feed_reader import LiveMarketFeed

app = Flask(__name__)

# ── Load model artifacts ─────────────────────────────────────────────────────
model        = joblib.load('nba_predictor_model.pkl')
teams        = joblib.load('teams.pkl')
team_features = joblib.load('team_features.pkl')
feature_cols = joblib.load('feature_cols.pkl')

team_recent = {}
try:
    team_recent = joblib.load('team_recent.pkl')
except FileNotFoundError:
    pass

# ── Live market feed ─────────────────────────────────────────────────────────
live_feed = LiveMarketFeed()

def refresh_live_data():
    while True:
        live_feed.load_latest()
        threading.Event().wait(10)

threading.Thread(target=refresh_live_data, daemon=True).start()

# ── Team metadata ─────────────────────────────────────────────────────────────
team_info = {
    'ATL': {'name': 'Atlanta Hawks',           'logo': 'https://cdn.nba.com/logos/nba/1610612737/primary/L/logo.svg'},
    'BOS': {'name': 'Boston Celtics',           'logo': 'https://cdn.nba.com/logos/nba/1610612738/primary/L/logo.svg'},
    'BKN': {'name': 'Brooklyn Nets',            'logo': 'https://cdn.nba.com/logos/nba/1610612751/primary/L/logo.svg'},
    'CHA': {'name': 'Charlotte Hornets',        'logo': 'https://cdn.nba.com/logos/nba/1610612766/primary/L/logo.svg'},
    'CHI': {'name': 'Chicago Bulls',            'logo': 'https://cdn.nba.com/logos/nba/1610612741/primary/L/logo.svg'},
    'CLE': {'name': 'Cleveland Cavaliers',      'logo': 'https://cdn.nba.com/logos/nba/1610612739/primary/L/logo.svg'},
    'DAL': {'name': 'Dallas Mavericks',         'logo': 'https://cdn.nba.com/logos/nba/1610612742/primary/L/logo.svg'},
    'DEN': {'name': 'Denver Nuggets',           'logo': 'https://cdn.nba.com/logos/nba/1610612743/primary/L/logo.svg'},
    'DET': {'name': 'Detroit Pistons',          'logo': 'https://cdn.nba.com/logos/nba/1610612765/primary/L/logo.svg'},
    'GSW': {'name': 'Golden State Warriors',    'logo': 'https://cdn.nba.com/logos/nba/1610612744/primary/L/logo.svg'},
    'HOU': {'name': 'Houston Rockets',          'logo': 'https://cdn.nba.com/logos/nba/1610612745/primary/L/logo.svg'},
    'IND': {'name': 'Indiana Pacers',           'logo': 'https://cdn.nba.com/logos/nba/1610612754/primary/L/logo.svg'},
    'LAC': {'name': 'LA Clippers',              'logo': 'https://cdn.nba.com/logos/nba/1610612746/primary/L/logo.svg'},
    'LAL': {'name': 'Los Angeles Lakers',       'logo': 'https://cdn.nba.com/logos/nba/1610612747/primary/L/logo.svg'},
    'MEM': {'name': 'Memphis Grizzlies',        'logo': 'https://cdn.nba.com/logos/nba/1610612763/primary/L/logo.svg'},
    'MIA': {'name': 'Miami Heat',               'logo': 'https://cdn.nba.com/logos/nba/1610612748/primary/L/logo.svg'},
    'MIL': {'name': 'Milwaukee Bucks',          'logo': 'https://cdn.nba.com/logos/nba/1610612749/primary/L/logo.svg'},
    'MIN': {'name': 'Minnesota Timberwolves',   'logo': 'https://cdn.nba.com/logos/nba/1610612750/primary/L/logo.svg'},
    'NOP': {'name': 'New Orleans Pelicans',     'logo': 'https://cdn.nba.com/logos/nba/1610612740/primary/L/logo.svg'},
    'NYK': {'name': 'New York Knicks',          'logo': 'https://cdn.nba.com/logos/nba/1610612752/primary/L/logo.svg'},
    'OKC': {'name': 'Oklahoma City Thunder',    'logo': 'https://cdn.nba.com/logos/nba/1610612760/primary/L/logo.svg'},
    'ORL': {'name': 'Orlando Magic',            'logo': 'https://cdn.nba.com/logos/nba/1610612753/primary/L/logo.svg'},
    'PHI': {'name': 'Philadelphia 76ers',       'logo': 'https://cdn.nba.com/logos/nba/1610612755/primary/L/logo.svg'},
    'PHX': {'name': 'Phoenix Suns',             'logo': 'https://cdn.nba.com/logos/nba/1610612756/primary/L/logo.svg'},
    'POR': {'name': 'Portland Trail Blazers',   'logo': 'https://cdn.nba.com/logos/nba/1610612757/primary/L/logo.svg'},
    'SAC': {'name': 'Sacramento Kings',         'logo': 'https://cdn.nba.com/logos/nba/1610612758/primary/L/logo.svg'},
    'SAS': {'name': 'San Antonio Spurs',        'logo': 'https://cdn.nba.com/logos/nba/1610612759/primary/L/logo.svg'},
    'TOR': {'name': 'Toronto Raptors',          'logo': 'https://cdn.nba.com/logos/nba/1610612761/primary/L/logo.svg'},
    'UTA': {'name': 'Utah Jazz',                'logo': 'https://cdn.nba.com/logos/nba/1610612762/primary/L/logo.svg'},
    'WAS': {'name': 'Washington Wizards',       'logo': 'https://cdn.nba.com/logos/nba/1610612764/primary/L/logo.svg'},
}

# ── Helper functions ──────────────────────────────────────────────────────────

def get_team_display_stats(abbrev, game_date=None):
    """Return display-ready stats for a team. All values safe for templates."""
    f    = dict(team_features.get(abbrev, {}))
    recent = team_recent.get(abbrev, [])

    rest = float(f.get('DAYS_REST', 2) or 2)
    b2b  = int(f.get('B2B', 0) or 0)

    if game_date and f.get('LAST_GAME_DATE'):
        try:
            days = (pd.to_datetime(game_date).normalize()
                    - pd.to_datetime(f['LAST_GAME_DATE'])).days
            if days >= 0:
                rest = min(days, 14)
                b2b  = 1 if days <= 1 else 0
        except Exception:
            pass

    streak     = int(f.get('STREAK', 0) or 0)
    net_rating = round(float(f.get('NET_RATING_L10', 0) or 0), 1)
    form       = [g['result'] for g in recent[-5:]] if recent else []

    return {
        'elo':        round(float(f.get('ELO', 1500) or 1500)),
        'net_rating': net_rating,
        'win_rate':   round(float(f.get('WIN_L10', 0.5) or 0.5) * 100, 1),
        'streak':     streak,
        'streak_str': f'+{streak}' if streak > 0 else str(streak),
        'rest':       int(rest),
        'b2b':        b2b,
        'efg_pct':    round(float(f.get('EFG_PCT_L10', 0) or 0) * 100, 1),
        'tov_pct':    round(float(f.get('TOV_PCT_L10', 0) or 0) * 100, 1),
        'oreb_pct':   round(float(f.get('OREB_PCT_L10', 0) or 0) * 100, 1),
        'form':       form,
    }


def get_market_prob(market_data, side='home'):
    """Extract market implied probability (0–100) for home or away side."""
    if not market_data:
        return None
    poly   = market_data.get('polymarket', {})
    kalshi = market_data.get('kalshi', {})
    try:
        if poly:
            key = 'home' if side == 'home' else 'away'
            if poly.get(key):
                return round(poly[key].get('midpoint_pct', 0) * 100, 1)
        elif kalshi:
            mp = kalshi.get('midpoint_pct', 0) * 100
            return round(mp if side == 'home' else 100 - mp, 1)
    except Exception:
        pass
    return None


def prob_to_american(prob_pct):
    """Convert a win probability % to American odds string (e.g. -185, +160)."""
    try:
        p = float(prob_pct) / 100
        if p <= 0.01 or p >= 0.99:
            return 'N/A'
        if p >= 0.5:
            return f'-{int(round(100 * p / (1 - p)))}'
        else:
            return f'+{int(round(100 * (1 - p) / p))}'
    except Exception:
        return 'N/A'


def get_bet_recommendation(model_prob, market_prob):
    """
    Return bet recommendation dict based on edge (model - market).
      >= 8%  → STRONG BET, 3 units
      4–8%   → LEAN,       1 unit
      0–4%   → SMALL EDGE, 0 units (watch)
      < 0%   → PASS
    Returns None if no market data available.
    """
    if market_prob is None:
        return None
    edge = model_prob - market_prob
    if edge >= 8:
        return {'label': 'STRONG BET', 'units': 3, 'edge': round(edge, 1), 'color': 'green'}
    elif edge >= 4:
        return {'label': 'LEAN',       'units': 1, 'edge': round(edge, 1), 'color': 'amber'}
    elif edge >= 0:
        return {'label': 'SMALL EDGE', 'units': 0, 'edge': round(edge, 1), 'color': 'blue'}
    else:
        return {'label': 'PASS',       'units': 0, 'edge': round(edge, 1), 'color': 'gray'}


# ── Core prediction ───────────────────────────────────────────────────────────

def predict_game(home_team, away_team, game_date=None):
    """
    Predict home win probability. game_date triggers dynamic B2B/rest
    computation from each team's LAST_GAME_DATE stored in team_features.
    """
    home_f = dict(team_features.get(home_team, {}))
    away_f = dict(team_features.get(away_team, {}))

    if game_date:
        game_dt = pd.to_datetime(game_date).normalize()
        for f_dict in [home_f, away_f]:
            last = f_dict.get('LAST_GAME_DATE', '')
            if last:
                days = (game_dt - pd.to_datetime(last)).days
                f_dict['DAYS_REST'] = min(max(days, 0), 14)
                f_dict['B2B']       = 1 if days <= 1 else 0

    row = {}
    for feature in feature_cols:
        if feature.startswith('HOME_'):
            row[feature] = home_f.get(feature[5:], 0) or 0
        else:
            row[feature] = away_f.get(feature[5:], 0) or 0

    return model.predict_proba(pd.DataFrame([row], columns=feature_cols))[0][1]


# ── ESPN schedule ─────────────────────────────────────────────────────────────

def get_upcoming_games():
    try:
        resp = requests.get(
            'https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard',
            timeout=5)
        data = resp.json()
        upcoming = []
        for event in data.get('events', []):
            status    = event['status']['type']['name']
            comp      = event['competitions'][0]
            home_abbr = comp['home']['team']['abbreviation']
            away_abbr = comp['away']['team']['abbreviation']
            if status != 'STATUS_SCHEDULED':
                continue
            if home_abbr not in team_info or away_abbr not in team_info:
                continue
            upcoming.append({
                'home': home_abbr,
                'away': away_abbr,
                'date': event['date'],
                'time': comp.get('status', {}).get('type', {}).get('shortDetail', 'TBD'),
            })
        return upcoming[:10]
    except Exception:
        return []


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html', team_info=team_info)


@app.route('/upcoming')
def upcoming():
    games      = get_upcoming_games()
    last_updated = datetime.now().strftime('%I:%M %p')
    predictions = []

    for game in games:
        home      = game['home']
        away      = game['away']
        game_date = game.get('date')

        prob_home     = predict_game(home, away, game_date=game_date)
        prob_home_pct = round(prob_home * 100, 1)
        prob_away_pct = round((1 - prob_home) * 100, 1)

        game_key    = f"{away.upper()}_{home.upper()}"
        market_data = live_feed.get_game_odds(game_key) or {}

        mkt_home = get_market_prob(market_data, 'home')
        mkt_away = get_market_prob(market_data, 'away')

        # Bet recommendation targets the model's predicted winner
        if prob_home_pct >= prob_away_pct:
            bet      = get_bet_recommendation(prob_home_pct, mkt_home)
            bet_team = home
        else:
            bet      = get_bet_recommendation(prob_away_pct, mkt_away)
            bet_team = away

        predictions.append({
            'home':       home,
            'away':       away,
            'home_name':  team_info[home]['name'],
            'away_name':  team_info[away]['name'],
            'home_logo':  team_info[home]['logo'],
            'away_logo':  team_info[away]['logo'],
            'prob_home':  prob_home_pct,
            'prob_away':  prob_away_pct,
            'date':       game_date,
            'time':       game['time'],
            # Market
            'mkt_home':       mkt_home,
            'mkt_away':       mkt_away,
            'mkt_home_odds':  prob_to_american(mkt_home) if mkt_home else None,
            'mkt_away_odds':  prob_to_american(mkt_away) if mkt_away else None,
            # Bet
            'bet':      bet,
            'bet_team': bet_team,
            # Team stats for display
            'home_stats': get_team_display_stats(home, game_date),
            'away_stats': get_team_display_stats(away, game_date),
        })

    best_bets = sum(1 for g in predictions
                    if g['bet'] and g['bet']['color'] in ('green', 'amber'))

    return render_template('upcoming.html',
                           games=predictions,
                           team_info=team_info,
                           last_updated=last_updated,
                           best_bets=best_bets)


@app.route('/predict', methods=['POST'])
def predict():
    home_team = request.form['home_team']
    away_team = request.form['away_team']

    prob_home = predict_game(home_team, away_team)
    prob_home_pct = round(prob_home * 100, 1)
    prob_away_pct = round((1 - prob_home) * 100, 1)

    winner     = home_team if prob_home >= 0.5 else away_team
    confidence = max(prob_home_pct, prob_away_pct)

    home_stats = get_team_display_stats(home_team)
    away_stats = get_team_display_stats(away_team)

    # Key stat comparison rows for result page
    stat_comparison = [
        {'label': 'Elo Rating',     'home': home_stats['elo'],        'away': away_stats['elo'],        'higher_is_better': True,  'fmt': ''},
        {'label': 'Net Rating L10', 'home': home_stats['net_rating'], 'away': away_stats['net_rating'], 'higher_is_better': True,  'fmt': '+.1f'},
        {'label': 'Win Rate L10',   'home': home_stats['win_rate'],   'away': away_stats['win_rate'],   'higher_is_better': True,  'fmt': '.1f%'},
        {'label': 'eFG% L10',       'home': home_stats['efg_pct'],    'away': away_stats['efg_pct'],    'higher_is_better': True,  'fmt': '.1f%'},
        {'label': 'TOV% L10',       'home': home_stats['tov_pct'],    'away': away_stats['tov_pct'],    'higher_is_better': False, 'fmt': '.1f%'},
        {'label': 'OREB% L10',      'home': home_stats['oreb_pct'],   'away': away_stats['oreb_pct'],   'higher_is_better': True,  'fmt': '.1f%'},
        {'label': 'Streak',         'home': home_stats['streak'],     'away': away_stats['streak'],     'higher_is_better': True,  'fmt': ''},
        {'label': 'Rest Days',      'home': home_stats['rest'],       'away': away_stats['rest'],       'higher_is_better': True,  'fmt': ''},
    ]

    return render_template('result.html',
                           home_team=home_team,
                           away_team=away_team,
                           winner=winner,
                           confidence=confidence,
                           home_prob=prob_home_pct,
                           away_prob=prob_away_pct,
                           home_stats=home_stats,
                           away_stats=away_stats,
                           stat_comparison=stat_comparison,
                           team_info=team_info)


@app.route('/api/upcoming')
def api_upcoming():
    """JSON endpoint for client-side refresh without full page reload."""
    games = get_upcoming_games()
    out   = []
    for game in games:
        home      = game['home']
        away      = game['away']
        game_date = game.get('date')
        prob_home = predict_game(home, away, game_date=game_date)
        out.append({
            'home':      home,
            'away':      away,
            'prob_home': round(prob_home * 100, 1),
            'prob_away': round((1 - prob_home) * 100, 1),
            'time':      game['time'],
        })
    return jsonify(out)


if __name__ == '__main__':
    app.run(debug=True, port=5001)
