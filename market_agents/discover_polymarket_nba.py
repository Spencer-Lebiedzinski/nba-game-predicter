"""
Discover Polymarket NBA markets and fetch active events with CLOB asset IDs.
"""
import requests
import json
from typing import List, Dict, Optional
import re

POLYMARKET_API = "https://gamma-api.polymarket.com"

def get_nba_tag_id() -> Optional[str]:
    """Fetch the NBA tag_id from Polymarket /sports endpoint"""
    try:
        url = f"{POLYMARKET_API}/sports"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        sports = response.json()
        
        for sport in sports:
            if sport.get('label', '').upper() == 'NBA':
                tag_id = sport.get('id')
                print(f"✓ Found NBA tag_id: {tag_id}")
                return tag_id
        
        print("✗ NBA tag not found in sports list")
        return None
    except Exception as e:
        print(f"✗ Error fetching NBA tag_id: {e}")
        return None

def get_nba_events(tag_id: str, active: bool = True, closed: bool = False) -> List[Dict]:
    """Fetch active NBA events with their CLOB asset IDs"""
    try:
        # Polymarket uses a /events endpoint with filters
        url = f"{POLYMARKET_API}/events"
        params = {
            "tag_id": tag_id,
            "active": str(active).lower(),
            "closed": str(closed).lower(),
            "limit": 1000
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        events = response.json()
        
        print(f"✓ Fetched {len(events)} NBA events")
        return events
    except Exception as e:
        print(f"✗ Error fetching NBA events: {e}")
        return []

def normalize_matchups(events: List[Dict]) -> List[Dict]:
    """
    Normalize events into matchup format with asset_ids for each side
    Returns list of dicts with structure:
    {
        'game_id': str,
        'home_team': str,
        'away_team': str,
        'created_at': str,
        'assets': [
            {'side': 'home', 'asset_id': str, 'ticker': str},
            {'side': 'away', 'asset_id': str, 'ticker': str}
        ]
    }
    """
    matchups = []
    
    for event in events:
        try:
            title = event.get('title', '')
            
            # Simple parsing: "Team1 vs Team2" or similar
            if ' vs ' in title:
                parts = title.split(' vs ')
                away_team = parts[0].strip()
                home_team = parts[1].strip()
            elif ' @ ' in title:
                parts = title.split(' @ ')
                away_team = parts[0].strip()
                home_team = parts[1].strip()
            else:
                continue
            
            event_id = event.get('id')
            created_at = event.get('created_at', '')
            
            # Extract CLOB conditions (assets)
            outcomes = event.get('outcomes', [])
            
            if len(outcomes) < 2:
                continue
            
            assets = []
            for outcome in outcomes:
                asset_id = outcome.get('id')
                ticker = outcome.get('ticker', '')
                
                # Determine which side this is
                label = outcome.get('label', '').lower()
                if 'yes' in label or home_team.lower() in label:
                    side = 'home'
                else:
                    side = 'away'
                
                assets.append({
                    'side': side,
                    'asset_id': asset_id,
                    'ticker': ticker
                })
            
            matchups.append({
                'game_id': event_id,
                'home_team': home_team,
                'away_team': away_team,
                'created_at': created_at,
                'assets': assets
            })
        
        except Exception as e:
            print(f"⚠ Error parsing event {event.get('id')}: {e}")
            continue
    
    return matchups

def discover_nba_markets() -> List[Dict]:
    """Main function: discover all active NBA markets"""
    print("🔍 Discovering Polymarket NBA markets...")
    
    tag_id = get_nba_tag_id()
    if not tag_id:
        return []
    
    events = get_nba_events(tag_id)
    if not events:
        return []
    
    matchups = normalize_matchups(events)
    print(f"✓ Normalized {len(matchups)} matchups\n")
    
    return matchups

if __name__ == "__main__":
    markets = discover_nba_markets()
    print(json.dumps(markets, indent=2))
