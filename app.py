from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import requests
from datetime import datetime, timedelta
import json
import threading
from live_feed_reader import LiveMarketFeed

app = Flask(__name__)

# Load the trained model and teams
model = joblib.load('nba_predictor_model.pkl')
teams = joblib.load('teams.pkl')
team_features = joblib.load('team_features.pkl')
feature_cols = joblib.load('feature_cols.pkl')

# Initialize live market feed reader
live_feed = LiveMarketFeed()

# Load live data in background
def refresh_live_data():
    """Refresh live market data periodically"""
    while True:
        live_feed.load_latest()
        threading.Event().wait(10)  # Refresh every 10 seconds

threading.Thread(target=refresh_live_data, daemon=True).start()

# Team info with full names and logos
team_info = {
    'ATL': {'name': 'Atlanta Hawks', 'logo': 'https://cdn.nba.com/logos/nba/1610612737/primary/L/logo.svg'},
    'BOS': {'name': 'Boston Celtics', 'logo': 'https://cdn.nba.com/logos/nba/1610612738/primary/L/logo.svg'},
    'BKN': {'name': 'Brooklyn Nets', 'logo': 'https://cdn.nba.com/logos/nba/1610612751/primary/L/logo.svg'},
    'CHA': {'name': 'Charlotte Hornets', 'logo': 'https://cdn.nba.com/logos/nba/1610612766/primary/L/logo.svg'},
    'CHI': {'name': 'Chicago Bulls', 'logo': 'https://cdn.nba.com/logos/nba/1610612741/primary/L/logo.svg'},
    'CLE': {'name': 'Cleveland Cavaliers', 'logo': 'https://cdn.nba.com/logos/nba/1610612739/primary/L/logo.svg'},
    'DAL': {'name': 'Dallas Mavericks', 'logo': 'https://cdn.nba.com/logos/nba/1610612742/primary/L/logo.svg'},
    'DEN': {'name': 'Denver Nuggets', 'logo': 'https://cdn.nba.com/logos/nba/1610612743/primary/L/logo.svg'},
    'DET': {'name': 'Detroit Pistons', 'logo': 'https://cdn.nba.com/logos/nba/1610612765/primary/L/logo.svg'},
    'GSW': {'name': 'Golden State Warriors', 'logo': 'https://cdn.nba.com/logos/nba/1610612744/primary/L/logo.svg'},
    'HOU': {'name': 'Houston Rockets', 'logo': 'https://cdn.nba.com/logos/nba/1610612745/primary/L/logo.svg'},
    'IND': {'name': 'Indiana Pacers', 'logo': 'https://cdn.nba.com/logos/nba/1610612754/primary/L/logo.svg'},
    'LAC': {'name': 'LA Clippers', 'logo': 'https://cdn.nba.com/logos/nba/1610612746/primary/L/logo.svg'},
    'LAL': {'name': 'Los Angeles Lakers', 'logo': 'https://cdn.nba.com/logos/nba/1610612747/primary/L/logo.svg'},
    'MEM': {'name': 'Memphis Grizzlies', 'logo': 'https://cdn.nba.com/logos/nba/1610612763/primary/L/logo.svg'},
    'MIA': {'name': 'Miami Heat', 'logo': 'https://cdn.nba.com/logos/nba/1610612748/primary/L/logo.svg'},
    'MIL': {'name': 'Milwaukee Bucks', 'logo': 'https://cdn.nba.com/logos/nba/1610612749/primary/L/logo.svg'},
    'MIN': {'name': 'Minnesota Timberwolves', 'logo': 'https://cdn.nba.com/logos/nba/1610612750/primary/L/logo.svg'},
    'NOP': {'name': 'New Orleans Pelicans', 'logo': 'https://cdn.nba.com/logos/nba/1610612740/primary/L/logo.svg'},
    'NYK': {'name': 'New York Knicks', 'logo': 'https://cdn.nba.com/logos/nba/1610612752/primary/L/logo.svg'},
    'OKC': {'name': 'Oklahoma City Thunder', 'logo': 'https://cdn.nba.com/logos/nba/1610612760/primary/L/logo.svg'},
    'ORL': {'name': 'Orlando Magic', 'logo': 'https://cdn.nba.com/logos/nba/1610612753/primary/L/logo.svg'},
    'PHI': {'name': 'Philadelphia 76ers', 'logo': 'https://cdn.nba.com/logos/nba/1610612755/primary/L/logo.svg'},
    'PHX': {'name': 'Phoenix Suns', 'logo': 'https://cdn.nba.com/logos/nba/1610612756/primary/L/logo.svg'},
    'POR': {'name': 'Portland Trail Blazers', 'logo': 'https://cdn.nba.com/logos/nba/1610612757/primary/L/logo.svg'},
    'SAC': {'name': 'Sacramento Kings', 'logo': 'https://cdn.nba.com/logos/nba/1610612758/primary/L/logo.svg'},
    'SAS': {'name': 'San Antonio Spurs', 'logo': 'https://cdn.nba.com/logos/nba/1610612759/primary/L/logo.svg'},
    'TOR': {'name': 'Toronto Raptors', 'logo': 'https://cdn.nba.com/logos/nba/1610612761/primary/L/logo.svg'},
    'UTA': {'name': 'Utah Jazz', 'logo': 'https://cdn.nba.com/logos/nba/1610612762/primary/L/logo.svg'},
    'WAS': {'name': 'Washington Wizards', 'logo': 'https://cdn.nba.com/logos/nba/1610612764/primary/L/logo.svg'},
}

def get_upcoming_games():
    """Fetch upcoming NBA games from ESPN API"""
    try:
        url = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
        response = requests.get(url, timeout=5)
        data = response.json()
        
        upcoming = []
        for event in data.get('events', []):
            date = event['date']
            status = event['status']['type']['name']
            
            if status != 'STATUS_SCHEDULED':
                continue
                
            home_team = event['competitions'][0]['home']['team']['abbreviation']
            away_team = event['competitions'][0]['away']['team']['abbreviation']
            
            if home_team in team_info and away_team in team_info:
                upcoming.append({
                    'home': home_team,
                    'away': away_team,
                    'date': date,
                    'time': event['competitions'][0].get('status', {}).get('type', {}).get('shortDetail', 'TBD')
                })
        
        return upcoming[:10]  # Return next 10 games
    except:
        return []

def predict_game(home_team, away_team, game_date=None):
    """Predict a game outcome with probability.

    game_date (str or datetime, optional): when provided, B2B and DAYS_REST
    are computed dynamically from each team's LAST_GAME_DATE so upcoming-game
    predictions reflect actual schedule fatigue.
    """
    home_f = dict(team_features.get(home_team, {}))
    away_f = dict(team_features.get(away_team, {}))

    # Dynamically compute B2B / rest days when game date is known
    if game_date:
        game_dt = pd.to_datetime(game_date).normalize()
        for f_dict in [home_f, away_f]:
            last = f_dict.get("LAST_GAME_DATE", "")
            if last:
                days = (game_dt - pd.to_datetime(last)).days
                f_dict["DAYS_REST"] = min(max(days, 0), 14)
                f_dict["B2B"] = 1 if days <= 1 else 0

    row = {}
    for feature in feature_cols:
        if feature.startswith("HOME_"):
            row[feature] = home_f.get(feature[5:], 0) or 0
        else:
            row[feature] = away_f.get(feature[5:], 0) or 0

    features_df = pd.DataFrame([row], columns=feature_cols)
    return model.predict_proba(features_df)[0][1]

@app.route('/')
def index():
    return render_template('index.html', team_info=team_info)

@app.route('/upcoming')
def upcoming():
    games = get_upcoming_games()
    game_predictions = []
    
    for game in games:
        home = game['home']
        away = game['away']
        prob_home = predict_game(home, away, game_date=game.get('date'))
        prob_away = 1 - prob_home
        
        game_key = f"{away.upper()}_{home.upper()}"
        market_data = live_feed.get_game_odds(game_key)
        
        game_predictions.append({
            'home': home,
            'away': away,
            'home_name': team_info[home]['name'],
            'away_name': team_info[away]['name'],
            'home_logo': team_info[home]['logo'],
            'away_logo': team_info[away]['logo'],
            'prob_home': round(prob_home * 100, 1),
            'prob_away': round(prob_away * 100, 1),
            'date': game['date'],
            'time': game['time'],
            'market_data': market_data or {}
        })
    
    return render_template('upcoming.html', games=game_predictions, team_info=team_info)

@app.route('/predict', methods=['POST'])
def predict():
    home_team = request.form['home_team']
    away_team = request.form['away_team']
    
    prob_home_win = predict_game(home_team, away_team)
    prob_away_win = 1 - prob_home_win
    
    winner = home_team if prob_home_win > prob_away_win else away_team
    confidence = max(prob_home_win, prob_away_win) * 100

    return render_template('result.html',
                          home_team=home_team,
                          away_team=away_team,
                          winner=winner,
                          confidence=f"{confidence:.1f}",
                          home_prob=f"{prob_home_win*100:.1f}",
                          away_prob=f"{prob_away_win*100:.1f}",
                          team_info=team_info)

if __name__ == '__main__':
    app.run(debug=True, port=5001)
