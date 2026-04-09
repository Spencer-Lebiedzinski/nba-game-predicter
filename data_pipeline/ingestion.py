import requests
from datetime import datetime
from database.db import SessionLocal
from database.models import Game, Team

def fetch_today_nba_games():
    """Fetch games from ESPN API (used as a reliable real-time source)"""
    url = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        games = []
        for event in data.get('events', []):
            home_team = event['competitions'][0]['home']['team']['abbreviation']
            away_team = event['competitions'][0]['away']['team']['abbreviation']
            games.append({
                'game_id': event['id'],
                'date': event['date'],
                'home_team': home_team,
                'away_team': away_team,
                'status': event['status']['type']['name']
            })
        return games
    except Exception as e:
        print(f"Error fetching games: {e}")
        return []

def store_games_to_db(games):
    db = SessionLocal()
    for game_data in games:
        # Simplistic ingestion
        existing = db.query(Game).filter(Game.id == game_data['game_id']).first()
        if not existing:
            new_game = Game(
                id=game_data['game_id'],
                date=datetime.strptime(game_data['date'], "%Y-%m-%dT%H:%MZ"),
                home_team_id=game_data['home_team'],
                away_team_id=game_data['away_team'],
                status=game_data['status']
            )
            db.add(new_game)
    db.commit()
    db.close()

if __name__ == "__main__":
    print("Running Data Pipeline...")
    games = fetch_today_nba_games()
    print(f"Ingested {len(games)} games.")
