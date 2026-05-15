from flask import Flask, render_template, request, jsonify, make_response, redirect, url_for, abort
import joblib
import json
import numpy as np
import os
import pandas as pd
import requests
from datetime import datetime
from pathlib import Path
import threading
from live_feed_reader import LiveMarketFeed

app = Flask(__name__)
# In development we never want stale CSS/JS to mask a change.
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# ── Backend selection ────────────────────────────────────────────────────────
# Both backends load eagerly if their artifacts are present. The default
# backend comes from MODEL_BACKEND env var; per-session selection comes from
# the `backend` cookie set by the UI toggle. predict_game() resolves the
# active backend per request via _resolve_backend().
DEFAULT_BACKEND = os.environ.get('MODEL_BACKEND', 'gbm').lower()

# ── Artifact directory resolution ───────────────────────────────────────────
# Training scripts save .pkl/.pt files to wherever they're run from. To keep
# `python app.py` working regardless of cwd, we look in ./models/ first and
# fall back to cwd. Also add the chosen dir to sys.path so the transformer's
# helper modules import cleanly.
_HERE = Path(__file__).resolve().parent
if (_HERE / 'models' / 'nba_predictor_model.pkl').exists():
    MODELS_DIR = _HERE / 'models'
else:
    MODELS_DIR = _HERE
import sys
sys.path.insert(0, str(MODELS_DIR))


def _artifact(name: str) -> str:
    """Return absolute path string for an artifact file."""
    return str(MODELS_DIR / name)


# ── GBM artifacts (v2) ──────────────────────────────────────────────────────
model         = joblib.load(_artifact('nba_predictor_model.pkl'))
teams         = joblib.load(_artifact('teams.pkl'))
team_features = joblib.load(_artifact('team_features.pkl'))
feature_cols  = joblib.load(_artifact('feature_cols.pkl'))

team_recent = {}
try:
    team_recent = joblib.load(_artifact('team_recent.pkl'))
except FileNotFoundError:
    pass

# ── Transformer artifacts (v3, optional) ────────────────────────────────────
transformer_ready  = False
transformer_model  = None
transformer_config = None
transformer_norm   = None
team_sequences     = None
transformer_summary = None  # contents of transformer_run_summary.json if present

try:
    import torch
    from transformer_model import NBATransformer
    from transformer_data import NormStats

    transformer_config = joblib.load(_artifact('transformer_config.pkl'))
    transformer_norm   = NormStats.from_dict(joblib.load(_artifact('transformer_norm_stats.pkl')))
    team_sequences     = joblib.load(_artifact('team_sequences.pkl'))

    cfg = transformer_config
    transformer_model = NBATransformer(
        n_token_features=cfg['n_token_features'],
        n_ctx_features=cfg['n_ctx_features'],
        seq_len=cfg['seq_len'],
        d_model=cfg['d_model'], n_heads=cfg['n_heads'],
        n_layers=cfg['n_layers'], d_ff=cfg['d_ff'],
        dropout=cfg['dropout'], head_hidden=cfg['head_hidden'],
    )
    transformer_model.load_state_dict(torch.load(_artifact('transformer_model.pt'), map_location='cpu'))
    transformer_model.eval()
    transformer_ready = True
    summary_path = Path(_artifact('transformer_run_summary.json'))
    if summary_path.exists():
        with open(summary_path) as _f:
            transformer_summary = json.load(_f)
    print('[backend] Transformer loaded ({} params)'.format(
        sum(p.numel() for p in transformer_model.parameters())
    ))
except Exception as exc:
    print(f'[backend] Transformer artifacts not available ({exc.__class__.__name__}); GBM-only mode.')


def _resolve_backend():
    """
    Pick the backend for this request: cookie value if set + transformer is
    available, otherwise the env-var default, otherwise GBM.
    """
    pref = request.cookies.get('backend') or DEFAULT_BACKEND
    if pref == 'transformer' and not transformer_ready:
        return 'gbm'
    return pref if pref in ('gbm', 'transformer') else 'gbm'

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

def _resolve_rest_b2b(team_abbrev, game_date):
    """Compute (rest_days, b2b) for `team_abbrev` given an upcoming game_date.
    Falls back to the team's stored DAYS_REST if game_date or LAST_GAME_DATE
    is missing."""
    f = team_features.get(team_abbrev, {}) or {}
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
    return rest, b2b


def _predict_game_gbm(home_team, away_team, game_date=None):
    """Predict home win probability via the v2 GBM."""
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

    return float(model.predict_proba(pd.DataFrame([row], columns=feature_cols))[0][1])


def _predict_game_transformer(home_team, away_team, game_date=None):
    """
    Predict via the v3 Transformer. Returns a dict with home_win_prob plus the
    auxiliary score predictions (useful for spread/total surfaces in the UI).
    Callers that only need the probability can ignore the extra keys.
    """
    import torch  # local import keeps app startup fast when GBM backend is in use

    h_seq = team_sequences.get(home_team)
    a_seq = team_sequences.get(away_team)
    if h_seq is None or a_seq is None:
        # Missing sequence data — fall back to GBM rather than break the UI
        return {'home_win_prob': _predict_game_gbm(home_team, away_team, game_date),
                'home_score': None, 'away_score': None, 'backend': 'gbm-fallback'}

    cfg = transformer_config
    seq_len = cfg['seq_len']

    home_tokens = h_seq['tokens'].astype(np.float32).copy()
    away_tokens = a_seq['tokens'].astype(np.float32).copy()
    home_mask = np.zeros(seq_len, dtype=bool)
    away_mask = np.zeros(seq_len, dtype=bool)
    home_mask[: seq_len - h_seq['n_valid']] = True
    away_mask[: seq_len - a_seq['n_valid']] = True

    # Normalize tokens with the saved training-set stats
    home_tokens = (home_tokens - transformer_norm.token_mean) / transformer_norm.token_std
    away_tokens = (away_tokens - transformer_norm.token_mean) / transformer_norm.token_std

    home_rest, home_b2b = _resolve_rest_b2b(home_team, game_date)
    away_rest, away_b2b = _resolve_rest_b2b(away_team, game_date)
    home_elo = float((team_features.get(home_team, {}) or {}).get('ELO', 1500) or 1500)
    away_elo = float((team_features.get(away_team, {}) or {}).get('ELO', 1500) or 1500)

    ctx = np.array([
        home_elo, away_elo, home_elo - away_elo,
        home_rest, away_rest, home_rest - away_rest,
        float(home_b2b), float(away_b2b),
    ], dtype=np.float32)
    ctx = (ctx - transformer_norm.ctx_mean) / transformer_norm.ctx_std

    with torch.no_grad():
        out = transformer_model(
            torch.from_numpy(home_tokens).unsqueeze(0),
            torch.from_numpy(home_mask).unsqueeze(0),
            torch.from_numpy(away_tokens).unsqueeze(0),
            torch.from_numpy(away_mask).unsqueeze(0),
            torch.from_numpy(ctx).unsqueeze(0),
        )
        prob = torch.sigmoid(out['logit']).item()
        home_score = out['home_score'].item()
        away_score = out['away_score'].item()

    return {
        'home_win_prob': float(prob),
        'home_score':    round(home_score, 1),
        'away_score':    round(away_score, 1),
        'backend':       'transformer',
    }


def predict_game(home_team, away_team, game_date=None, backend=None):
    """
    Public prediction entry point. Returns home win probability (float).

    Dispatches to the active backend (per-request cookie, or env default).
    Pass `backend='gbm'` / `backend='transformer'` to force one.
    """
    if backend is None:
        backend = _resolve_backend()
    if backend == 'transformer' and transformer_ready:
        return _predict_game_transformer(home_team, away_team, game_date)['home_win_prob']
    return _predict_game_gbm(home_team, away_team, game_date)


def predict_game_full(home_team, away_team, game_date=None, backend=None):
    """
    Like predict_game but always returns a dict including auxiliary outputs
    when the transformer is in use (home_score, away_score, predicted spread,
    predicted total). For GBM, only home_win_prob is populated.
    """
    if backend is None:
        backend = _resolve_backend()
    if backend == 'transformer' and transformer_ready:
        return _predict_game_transformer(home_team, away_team, game_date)
    prob = _predict_game_gbm(home_team, away_team, game_date)
    return {
        'home_win_prob': prob,
        'home_score':    None,
        'away_score':    None,
        'backend':       'gbm',
    }


# ── Template globals ─────────────────────────────────────────────────────────
@app.context_processor
def inject_globals():
    """Make backend state + team_info available to every template."""
    backend = _resolve_backend()
    # Cache-bust static assets with their mtime so the browser always sees
    # the latest CSS/JS without a manual hard-refresh.
    def _mtime(rel):
        p = _HERE / 'static' / rel
        try:
            return int(p.stat().st_mtime)
        except OSError:
            return 0
    return {
        'team_info': team_info,
        'backend':   backend,
        'transformer_available': transformer_ready,
        'transformer_summary':   transformer_summary,
        'active_tab': None,  # individual routes override this
        'asset_v_css': _mtime('css/app.css'),
        'asset_v_js':  _mtime('js/app.js'),
    }


# ── ESPN schedule ─────────────────────────────────────────────────────────────

# ESPN uses slightly different team abbreviations than nba_api. Map ESPN -> our keys.
ESPN_ABBR_ALIASES = {
    'SA':   'SAS',   # San Antonio Spurs
    'NO':   'NOP',   # New Orleans Pelicans
    'GS':   'GSW',   # Golden State Warriors
    'NY':   'NYK',   # New York Knicks
    'UTAH': 'UTA',   # Utah Jazz
    'WSH':  'WAS',   # Washington Wizards
}

def _normalize_abbr(a):
    return ESPN_ABBR_ALIASES.get(a, a)


def get_upcoming_games():
    """
    Pull today's NBA games from ESPN. Includes scheduled and in-progress;
    final games are skipped (the model predicts pre-game state). When a game
    is part of a playoff series, ESPN tags it in `competition.notes[0].headline`
    e.g. "East Semifinals - Game 6" — surfaced as `series` in the payload.
    """
    try:
        resp = requests.get(
            'https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard',
            timeout=8)
        data = resp.json()
        games = []
        for event in data.get('events', []):
            status = event['status']['type']['name']
            if status == 'STATUS_FINAL':
                continue

            comp = event['competitions'][0]
            home_abbr = away_abbr = None
            home_score = away_score = None
            for c in comp.get('competitors', []):
                abbr = _normalize_abbr(c['team']['abbreviation'])
                if c.get('homeAway') == 'home':
                    home_abbr, home_score = abbr, c.get('score')
                elif c.get('homeAway') == 'away':
                    away_abbr, away_score = abbr, c.get('score')

            if not home_abbr or not away_abbr:
                continue
            if home_abbr not in team_info or away_abbr not in team_info:
                continue

            series_headline = None
            for note in comp.get('notes', []) or []:
                if note.get('headline'):
                    series_headline = note['headline']
                    break

            games.append({
                'home':   home_abbr,
                'away':   away_abbr,
                'date':   event['date'],
                'time':   comp.get('status', {}).get('type', {}).get('shortDetail', 'TBD'),
                'status': status,
                'is_live': status == 'STATUS_IN_PROGRESS',
                'series': series_headline,
                'live_score': {
                    'home': home_score,
                    'away': away_score,
                } if status == 'STATUS_IN_PROGRESS' else None,
            })
        return games[:12]
    except Exception as exc:
        print(f'[scoreboard] ESPN fetch failed: {exc}')
        return []


# ── Dashboard data ────────────────────────────────────────────────────────────

def _build_game_payload(game):
    """Common per-game enrichment for dashboard + upcoming routes."""
    home      = game['home']
    away      = game['away']
    game_date = game.get('date')

    full = predict_game_full(home, away, game_date=game_date)
    prob_home = float(full['home_win_prob'])
    prob_home_pct = round(prob_home * 100, 1)
    prob_away_pct = round((1 - prob_home) * 100, 1)

    game_key    = f"{away.upper()}_{home.upper()}"
    market_data = live_feed.get_game_odds(game_key) or {}
    mkt_home = get_market_prob(market_data, 'home')
    mkt_away = get_market_prob(market_data, 'away')

    if prob_home_pct >= prob_away_pct:
        bet      = get_bet_recommendation(prob_home_pct, mkt_home)
        bet_team = home
    else:
        bet      = get_bet_recommendation(prob_away_pct, mkt_away)
        bet_team = away

    # Build optional score prediction sub-payload (transformer only)
    score_pred = None
    if full.get('home_score') is not None and full.get('away_score') is not None:
        hs, as_ = full['home_score'], full['away_score']
        score_pred = {
            'home_score': hs,
            'away_score': as_,
            'spread':     round(hs - as_, 1),
            'total':      round(hs + as_, 1),
        }

    return {
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
        'series':     game.get('series'),
        'is_live':    game.get('is_live', False),
        'live_score': game.get('live_score'),
        'mkt_home':       mkt_home,
        'mkt_away':       mkt_away,
        'mkt_home_odds':  prob_to_american(mkt_home) if mkt_home else None,
        'mkt_away_odds':  prob_to_american(mkt_away) if mkt_away else None,
        'bet':      bet,
        'bet_team': bet_team,
        'home_stats': get_team_display_stats(home, game_date),
        'away_stats': get_team_display_stats(away, game_date),
        'score_pred': score_pred,
    }


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    """Dashboard — KPI strip, featured game, today's slate."""
    games_raw = get_upcoming_games()
    predictions = [_build_game_payload(g) for g in games_raw]

    best_bets = [g for g in predictions if g['bet'] and g['bet']['color'] in ('green', 'amber')]
    # Featured = largest absolute edge if any bets, else first game
    featured = None
    if best_bets:
        featured = max(best_bets, key=lambda g: g['bet']['edge'] if g['bet'] else 0)
    elif predictions:
        featured = predictions[0]

    biggest_edge = max((g['bet']['edge'] for g in predictions if g['bet']), default=0)
    teams_tracked = len(team_features)

    kpis = {
        'games_today':   len(predictions),
        'best_bets':     len(best_bets),
        'biggest_edge':  biggest_edge,
        'teams_tracked': teams_tracked,
    }

    return render_template('dashboard.html',
                           active_tab='dashboard',
                           predictions=predictions,
                           featured=featured,
                           kpis=kpis,
                           last_updated=datetime.now().strftime('%I:%M %p'))


@app.route('/upcoming')
def upcoming():
    games        = get_upcoming_games()
    last_updated = datetime.now().strftime('%I:%M %p')
    predictions  = [_build_game_payload(g) for g in games]
    best_bets    = sum(1 for g in predictions
                       if g['bet'] and g['bet']['color'] in ('green', 'amber'))
    return render_template('upcoming.html',
                           active_tab='upcoming',
                           games=predictions,
                           last_updated=last_updated,
                           best_bets=best_bets)


@app.route('/predict', methods=['GET'])
def predict_form():
    return render_template('predict.html', active_tab='predict')


@app.route('/predict', methods=['POST'])
def predict():
    home_team = request.form['home_team']
    away_team = request.form['away_team']

    full = predict_game_full(home_team, away_team)
    prob_home = float(full['home_win_prob'])
    prob_home_pct = round(prob_home * 100, 1)
    prob_away_pct = round((1 - prob_home) * 100, 1)

    score_pred = None
    if full.get('home_score') is not None and full.get('away_score') is not None:
        hs, as_ = full['home_score'], full['away_score']
        score_pred = {
            'home_score': hs,
            'away_score': as_,
            'spread':     round(hs - as_, 1),
            'total':      round(hs + as_, 1),
        }

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
                           active_tab='predict',
                           home_team=home_team,
                           away_team=away_team,
                           winner=winner,
                           confidence=confidence,
                           home_prob=prob_home_pct,
                           away_prob=prob_away_pct,
                           home_stats=home_stats,
                           away_stats=away_stats,
                           stat_comparison=stat_comparison,
                           score_pred=score_pred,
                           backend_used=full.get('backend', 'gbm'))


@app.route('/team/<abbr>')
def team_detail(abbr):
    """Per-team detail page — recent form, key stats, sparkline data."""
    abbr = abbr.upper()
    if abbr not in team_info:
        abort(404)

    info = team_info[abbr]
    feats = team_features.get(abbr, {}) or {}
    recent = team_recent.get(abbr, [])

    # Pull a longer history from team_sequences (v3) if available, else fall
    # back to the 5-game team_recent data.
    history = []
    last10_avgs = {}  # used to backfill display stats when team_features is sparse
    if team_sequences and abbr in team_sequences:
        seq = team_sequences[abbr]
        tokens = seq['tokens']
        from transformer_data import TOKEN_FEATURES
        idx = {n: i for i, n in enumerate(TOKEN_FEATURES)}
        n_valid = seq['n_valid']
        valid = tokens[-n_valid:] if n_valid > 0 else tokens
        for row in valid:
            history.append({
                'pts':       float(row[idx['PTS']]),
                'opp_pts':   float(row[idx['OPP_PTS']]),
                'win':       bool(row[idx['WIN']] > 0.5),
                'margin':    float(row[idx['MARGIN']]),
                'net_rating': float(row[idx['NET_RATING']]),
            })
        last10 = valid[-10:]
        if len(last10):
            last10_avgs = {
                'efg':    float(last10[:, idx['EFG_PCT']].mean()),
                'tov':    float(last10[:, idx['TOV_PCT']].mean()),
                'oreb':   float(last10[:, idx['OREB_PCT']].mean()),
                'pace':   float(last10[:, idx['PACE']].mean()),
                'netrtg': float(last10[:, idx['NET_RATING']].mean()),
                'win':    float(last10[:, idx['WIN']].mean()),
                'elo':    float(last10[-1, idx['TEAM_ELO_PRE']]),
            }

    def _stat(key, multiplier=1.0, default=0):
        """team_features first, fall back to last10 transformer-derived average."""
        v = feats.get(key)
        if v is None or v == 0 or v == 0.0:
            return last10_avgs.get({
                'EFG_PCT_L10': 'efg', 'TOV_PCT_L10': 'tov',
                'OREB_PCT_L10': 'oreb', 'PACE_L10': 'pace',
                'NET_RATING_L10': 'netrtg', 'WIN_L10': 'win',
                'ELO': 'elo',
            }.get(key, ''), default) * multiplier
        return float(v) * multiplier

    headline = {
        'elo':            int(_stat('ELO', default=1500) or 1500),
        'net_rating_l10': round(_stat('NET_RATING_L10'), 1),
        'win_rate_l10':   round(_stat('WIN_L10', 100), 1),
        'efg_pct_l10':    round(_stat('EFG_PCT_L10', 100), 1),
        'tov_pct_l10':    round(_stat('TOV_PCT_L10', 100), 1),
        'oreb_pct_l10':   round(_stat('OREB_PCT_L10', 100), 1),
        'pace_l10':       round(_stat('PACE_L10'), 1),
        'streak':         int(feats.get('STREAK', 0) or 0),
        'rest':           int(float(feats.get('DAYS_REST', 2) or 2)),
        'b2b':            int(feats.get('B2B', 0) or 0),
        'last_game_date': feats.get('LAST_GAME_DATE', ''),
        'home_win_rate':  round(float(feats.get('HOME_WIN_RATE_L15', 0) or 0) * 100, 1),
        'road_win_rate':  round(float(feats.get('ROAD_WIN_RATE_L15', 0) or 0) * 100, 1),
    }

    # Recent form for the dots strip
    form = [g['result'] for g in recent] if recent else (
        ['W' if h['win'] else 'L' for h in history[-5:]] if history else []
    )

    # Sparkline data: last 10-20 net ratings as comma-separated string for the JS
    pts_spark = ','.join(f"{h['pts']:.0f}" for h in history[-15:])
    margin_spark = ','.join(f"{h['margin']:.0f}" for h in history[-15:])
    netrtg_spark = ','.join(f"{h['net_rating']:.1f}" for h in history[-15:])

    return render_template('team_detail.html',
                           active_tab=None,
                           abbr=abbr,
                           name=info['name'],
                           logo=info['logo'],
                           headline=headline,
                           form=form,
                           history=history,
                           pts_spark=pts_spark,
                           margin_spark=margin_spark,
                           netrtg_spark=netrtg_spark)


@app.route('/models')
def models_page():
    """Side-by-side metrics + architecture cards for v2 GBM and v3 Transformer."""
    gbm_card = {
        'name':         'GBM (v2)',
        'algorithm':    'XGBoost / sklearn GradientBoosting',
        'features':     len(feature_cols),
        'training_games': 'about 12,000',
        'target_acc':   '~66-68%',
        'highlights':   [
            'Four Factors (Dean Oliver framework)',
            'FiveThirtyEight-style Elo (MOV multiplier, season regression)',
            'Pace-normalized Net Rating',
            'Home/road split rolling windows',
            'Schedule fatigue (B2B, GAMES_L7, rest)',
        ],
    }

    tx_card = None
    if transformer_summary:
        cfg = transformer_summary['config']
        tx_card = {
            'name':         'Transformer (v3)',
            'algorithm':    'Two-tower shared-weight encoder',
            'params':       transformer_summary['n_params'],
            'seq_len':      cfg['seq_len'],
            'n_token_feats': cfg['n_token_features'],
            'n_ctx_feats':   cfg['n_ctx_features'],
            'd_model':      cfg['d_model'],
            'n_layers':     cfg['n_layers'],
            'n_heads':      cfg['n_heads'],
            'cv_summary':   transformer_summary.get('cv_summary', []),
            'final':        transformer_summary.get('final_metrics', {}),
            'final_season': transformer_summary.get('final_eval_season'),
        }
    elif transformer_ready:
        # Loaded weights but no summary file
        tx_card = {
            'name':         'Transformer (v3)',
            'algorithm':    'Two-tower shared-weight encoder',
            'params':       sum(p.numel() for p in transformer_model.parameters()),
            'note':         'Loaded from saved weights — run training to populate metrics summary',
        }

    return render_template('models.html',
                           active_tab='models',
                           gbm_card=gbm_card,
                           tx_card=tx_card,
                           transformer_training=not transformer_ready)


# ── JSON APIs ────────────────────────────────────────────────────────────────

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


@app.route('/api/predict')
def api_predict():
    """One-off predict for the command palette / external callers."""
    home  = (request.args.get('home') or '').upper()
    away  = (request.args.get('away') or '').upper()
    if home not in team_info or away not in team_info:
        return jsonify({'error': 'unknown team'}), 400
    full = predict_game_full(home, away)
    return jsonify(full)


@app.route('/api/backend', methods=['GET', 'POST'])
def api_backend():
    """Get or switch the active prediction backend for this session."""
    if request.method == 'GET':
        return jsonify({
            'backend':              _resolve_backend(),
            'default':              DEFAULT_BACKEND,
            'transformer_available': transformer_ready,
        })

    payload = request.get_json(silent=True) or {}
    target = (payload.get('backend') or '').lower()
    if target not in ('gbm', 'transformer'):
        return jsonify({'error': 'invalid backend'}), 400
    if target == 'transformer' and not transformer_ready:
        return jsonify({'error': 'transformer not available'}), 409
    resp = jsonify({'backend': target, 'switched': True})
    resp.set_cookie('backend', target, max_age=60 * 60 * 24 * 30, samesite='Lax')
    return resp


if __name__ == '__main__':
    app.run(debug=True, port=5001)
