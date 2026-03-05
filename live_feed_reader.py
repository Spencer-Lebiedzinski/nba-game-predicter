"""
Live market data reader - reads JSONL stream and serves via Flask
"""
import json
from pathlib import Path
from typing import Dict, Optional
from collections import defaultdict
from datetime import datetime, timedelta

class LiveMarketFeed:
    def __init__(self, jsonl_file: str = "data/live_probs.jsonl"):
        self.jsonl_file = jsonl_file
        self.latest_data: Dict = {}
    
    def load_latest(self, max_age_seconds: int = 300) -> Dict:
        """Load latest data from JSONL file within max_age"""
        if not Path(self.jsonl_file).exists():
            return {}
        
        try:
            cutoff_time = datetime.utcnow() - timedelta(seconds=max_age_seconds)
            
            with open(self.jsonl_file, 'r') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        game_key = data.get('game_key')
                        
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
