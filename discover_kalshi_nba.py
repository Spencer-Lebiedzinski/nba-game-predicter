"""
Discover Kalshi NBA markets and fetch open game events.
"""
import requests
import json
from typing import List, Dict, Optional
from datetime import datetime

KALSHI_API = "https://api.kalshi.com/trade-api/v2"

def get_nba_filters() -> Optional[Dict]:
    """Fetch filters from /search/filters_by_sport to locate NBA"""
    try:
        url = f"{KALSHI_API}/search/filters_by_sport"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        filters = response.json()
        
        # Look for NBA in sports
        for item in filters.get('data', []):
            if item.get('ticker_symbol', '').upper() == 'NBASPORTS':
                print(f"✓ Found NBA filters")
                return item
        
        print("✗ NBA filters not found")
        return None
    except Exception as e:
        print(f"✗ Error fetching filters: {e}")
        return None

def search_nba_markets(market_type: str = "gameline") -> List[Dict]:
    """
    Search for open NBA game markets.
    market_type can be 'gameline', 'spread', 'over_under', etc.
    """
    try:
        url = f"{KALSHI_API}/search/markets"
        params = {
            "query": "NBA",
            "series_ticker_prefix": "NBASPORTS",
            "limit": 100,
            "sort_by": "closing_date"
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        markets = data.get('markets', [])
        print(f"✓ Fetched {len(markets)} NBA markets from Kalshi")
        return markets
    except Exception as e:
        print(f"✗ Error searching NBA markets: {e}")
        return []

def get_market_details(market_ticker: str) -> Optional[Dict]:
    """Fetch detailed market info including orderbook"""
    try:
        url = f"{KALSHI_API}/markets/{market_ticker}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"⚠ Error fetching market {market_ticker}: {e}")
        return None

def normalize_kalshi_markets(markets: List[Dict]) -> List[Dict]:
    """
    Normalize Kalshi markets to unified format.
    Returns list of dicts with structure:
    {
        'game_id': str,
        'home_team': str,
        'away_team': str,
        'market_ticker': str,
        'market_type': str,
        'closing_date': str,
        'orderbook': {
            'home': {'bid': float, 'ask': float},
            'away': {'bid': float, 'ask': float}
        }
    }
    """
    normalized = []
    
    for market in markets:
        try:
            ticker = market.get('ticker', '')
            title = market.get('title', '')
            
            # Simple parsing of title
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
            
            # Fetch orderbook for this market
            details = get_market_details(ticker)
            if not details:
                continue
            
            orderbook = details.get('orderbook', {})
            
            # Extract yes/no prices
            home_bid = orderbook.get('yes', {}).get('bid', 0)
            home_ask = orderbook.get('yes', {}).get('ask', 0)
            away_bid = orderbook.get('no', {}).get('bid', 0)
            away_ask = orderbook.get('no', {}).get('ask', 0)
            
            normalized.append({
                'game_id': ticker,
                'home_team': home_team,
                'away_team': away_team,
                'market_ticker': ticker,
                'market_type': 'gameline',
                'closing_date': market.get('closing_date', ''),
                'orderbook': {
                    'home': {'bid': home_bid, 'ask': home_ask},
                    'away': {'bid': away_bid, 'ask': away_ask}
                }
            })
        except Exception as e:
            print(f"⚠ Error normalizing market {market.get('ticker')}: {e}")
            continue
    
    return normalized

def discover_nba_markets() -> List[Dict]:
    """Main function: discover all open NBA markets on Kalshi"""
    print("🔍 Discovering Kalshi NBA markets...")
    
    markets = search_nba_markets()
    if not markets:
        return []
    
    normalized = normalize_kalshi_markets(markets)
    print(f"✓ Normalized {len(normalized)} Kalshi markets\n")
    
    return normalized

if __name__ == "__main__":
    markets = discover_nba_markets()
    print(json.dumps(markets, indent=2))
