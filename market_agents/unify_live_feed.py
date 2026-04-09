"""
Unified live feed orchestrator.
Runs both Polymarket and Kalshi streamers concurrently,
maps markets to unified game_key, and writes combined JSONL stream.
"""
import asyncio
import json
from datetime import datetime
from typing import Dict, List, Optional
import logging
import os
from pathlib import Path

from discover_polymarket_nba import discover_nba_markets as discover_polymarket
from discover_kalshi_nba import discover_nba_markets as discover_kalshi
from stream_polymarket import stream_polymarket_prices
from stream_kalshi import stream_kalshi_prices

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("unified_feed")

TEAM_NAME_TO_ABBR = {
    'ATLANTA': 'ATL', 'HAWKS': 'ATL', 'ATLANTA HAWKS': 'ATL',
    'BOSTON': 'BOS', 'CELTICS': 'BOS', 'BOSTON CELTICS': 'BOS',
    'BROOKLYN': 'BKN', 'NETS': 'BKN', 'BROOKLYN NETS': 'BKN',
    'CHARLOTTE': 'CHA', 'HORNETS': 'CHA', 'CHARLOTTE HORNETS': 'CHA',
    'CHICAGO': 'CHI', 'BULLS': 'CHI', 'CHICAGO BULLS': 'CHI',
    'CLEVELAND': 'CLE', 'CAVALIERS': 'CLE', 'CLEVELAND CAVALIERS': 'CLE',
    'DALLAS': 'DAL', 'MAVERICKS': 'DAL', 'DALLAS MAVERICKS': 'DAL',
    'DENVER': 'DEN', 'NUGGETS': 'DEN', 'DENVER NUGGETS': 'DEN',
    'DETROIT': 'DET', 'PISTONS': 'DET', 'DETROIT PISTONS': 'DET',
    'GOLDEN STATE': 'GSW', 'WARRIORS': 'GSW', 'GOLDEN STATE WARRIORS': 'GSW',
    'HOUSTON': 'HOU', 'ROCKETS': 'HOU', 'HOUSTON ROCKETS': 'HOU',
    'INDIANA': 'IND', 'PACERS': 'IND', 'INDIANA PACERS': 'IND',
    'LA': 'LAC', 'LAC': 'LAC', 'CLIPPERS': 'LAC', 'LA CLIPPERS': 'LAC', 'LOS ANGELES CLIPPERS': 'LAC',
    'LAL': 'LAL', 'LAKERS': 'LAL', 'LOS ANGELES LAKERS': 'LAL',
    'MEMPHIS': 'MEM', 'GRIZZLIES': 'MEM', 'MEMPHIS GRIZZLIES': 'MEM',
    'MIAMI': 'MIA', 'HEAT': 'MIA', 'MIAMI HEAT': 'MIA',
    'MILWAUKEE': 'MIL', 'BUCKS': 'MIL', 'MILWAUKEE BUCKS': 'MIL',
    'MINNESOTA': 'MIN', 'TIMBERWOLVES': 'MIN', 'MINNESOTA TIMBERWOLVES': 'MIN',
    'NEW ORLEANS': 'NOP', 'PELICANS': 'NOP', 'NEW ORLEANS PELICANS': 'NOP',
    'NEW YORK': 'NYK', 'KNICKS': 'NYK', 'NEW YORK KNICKS': 'NYK',
    'OKLAHOMA CITY': 'OKC', 'THUNDER': 'OKC', 'OKLAHOMA CITY THUNDER': 'OKC',
    'ORLANDO': 'ORL', 'MAGIC': 'ORL', 'ORLANDO MAGIC': 'ORL',
    'PHILADELPHIA': 'PHI', '76ERS': 'PHI', 'PHILADELPHIA 76ERS': 'PHI', 'SIXERS': 'PHI',
    'PHOENIX': 'PHX', 'SUNS': 'PHX', 'PHOENIX SUNS': 'PHX',
    'PORTLAND': 'POR', 'TRAIL BLAZERS': 'POR', 'PORTLAND TRAIL BLAZERS': 'POR',
    'SACRAMENTO': 'SAC', 'KINGS': 'SAC', 'SACRAMENTO KINGS': 'SAC',
    'SAN ANTONIO': 'SAS', 'SPURS': 'SAS', 'SAN ANTONIO SPURS': 'SAS',
    'TORONTO': 'TOR', 'RAPTORS': 'TOR', 'TORONTO RAPTORS': 'TOR',
    'UTAH': 'UTA', 'JAZZ': 'UTA', 'UTAH JAZZ': 'UTA',
    'WASHINGTON': 'WAS', 'WIZARDS': 'WAS', 'WASHINGTON WIZARDS': 'WAS'
}

def get_team_abbr(team_name: str) -> str:
    """Safely convert any team name/mascot to a standard abbreviation"""
    clean_name = team_name.strip().upper()
    return TEAM_NAME_TO_ABBR.get(clean_name, clean_name)

class UnifiedFeed:
    def __init__(self, output_file: str = "data/live_probs.jsonl"):
        self.output_file = output_file
        self.polymarket_markets: Dict = {}
        self.kalshi_markets: Dict = {}
        self.game_mappings: Dict[str, Dict] = {}  # game_key -> {polymarket, kalshi}
        self.live_data: Dict[str, Dict] = {}  # game_key -> latest data
        
        # Create output directory
        Path(self.output_file).parent.mkdir(parents=True, exist_ok=True)
    
    def discover_markets(self):
        """Discover markets from both exchanges"""
        logger.info("📊 Discovering markets...")
        
        try:
            self.polymarket_markets = {m['game_id']: m for m in discover_polymarket()}
            logger.info(f"✓ Found {len(self.polymarket_markets)} Polymarket markets")
        except Exception as e:
            logger.warning(f"⚠ Error discovering Polymarket: {e}")
        
        try:
            self.kalshi_markets = {m['game_id']: m for m in discover_kalshi()}
            logger.info(f"✓ Found {len(self.kalshi_markets)} Kalshi markets")
        except Exception as e:
            logger.warning(f"⚠ Error discovering Kalshi: {e}")
    
    def map_games(self):
        """Create unified game_key mappings"""
        logger.info("🔗 Mapping games across exchanges...")
        
        # Use Polymarket as primary source
        for poly_id, poly_market in self.polymarket_markets.items():
            poly_away_abbr = get_team_abbr(poly_market['away_team'])
            poly_home_abbr = get_team_abbr(poly_market['home_team'])
            game_key = f"{poly_away_abbr}_{poly_home_abbr}"
            
            self.game_mappings[game_key] = {
                'game_key': game_key,
                'away_team': poly_market['away_team'],
                'home_team': poly_market['home_team'],
                'polymarket': poly_market,
                'kalshi': None
            }
        
        # Match Kalshi markets
        for kalshi_id, kalshi_market in self.kalshi_markets.items():
            kalshi_away_abbr = get_team_abbr(kalshi_market['away_team'])
            kalshi_home_abbr = get_team_abbr(kalshi_market['home_team'])
            game_key = f"{kalshi_away_abbr}_{kalshi_home_abbr}"
            
            if game_key in self.game_mappings:
                self.game_mappings[game_key]['kalshi'] = kalshi_market
            else:
                # Create new entry for Kalshi-only games
                self.game_mappings[game_key] = {
                    'game_key': game_key,
                    'away_team': kalshi_market['away_team'],
                    'home_team': kalshi_market['home_team'],
                    'polymarket': None,
                    'kalshi': kalshi_market
                }
        
        logger.info(f"✓ Mapped {len(self.game_mappings)} unique games")
    
    async def handle_polymarket_update(self, update: Dict):
        """Handle Polymarket price update"""
        asset_id = update.get('asset_id')
        
        # Find game for this asset
        for game_key, mapping in self.game_mappings.items():
            if not mapping['polymarket']:
                continue
            
            for asset in mapping['polymarket'].get('assets', []):
                if asset['asset_id'] == asset_id:
                    if game_key not in self.live_data:
                        self.live_data[game_key] = {
                            'game_key': game_key,
                            'away_team': mapping['away_team'],
                            'home_team': mapping['home_team'],
                            'polymarket': {},
                            'kalshi': {},
                            'last_updated': None
                        }
                    
                    side = asset['side']
                    self.live_data[game_key]['polymarket'][side] = {
                        'asset_id': asset_id,
                        'midpoint_pct': update['midpoint_pct'],
                        'bid': update['bid'],
                        'ask': update['ask'],
                        'timestamp': update['timestamp']
                    }
                    
                    self.live_data[game_key]['last_updated'] = datetime.utcnow().isoformat()
                    await self.write_update(game_key)
                    return
    
    async def handle_kalshi_update(self, update: Dict):
        """Handle Kalshi price update"""
        market_ticker = update.get('market_ticker')
        
        # Find game for this market
        for game_key, mapping in self.game_mappings.items():
            if not mapping['kalshi']:
                continue
            
            if mapping['kalshi']['market_ticker'] == market_ticker:
                if game_key not in self.live_data:
                    self.live_data[game_key] = {
                        'game_key': game_key,
                        'away_team': mapping['away_team'],
                        'home_team': mapping['home_team'],
                        'polymarket': {},
                        'kalshi': {},
                        'last_updated': None
                    }
                
                self.live_data[game_key]['kalshi'] = {
                    'market_ticker': market_ticker,
                    'midpoint_pct': update['midpoint_pct'],
                    'timestamp': update['timestamp']
                }
                
                self.live_data[game_key]['last_updated'] = datetime.utcnow().isoformat()
                await self.write_update(game_key)
                return
    
    async def write_update(self, game_key: str):
        """Write update to JSONL file"""
        try:
            with open(self.output_file, 'a') as f:
                f.write(json.dumps(self.live_data[game_key]) + '\n')
        except Exception as e:
            logger.error(f"✗ Error writing update: {e}")
    
    async def run_stream(self):
        """Run both streamers concurrently"""
        self.discover_markets()
        self.map_games()
        
        if not self.game_mappings:
            logger.error("✗ No games to stream")
            return
        
        logger.info("🚀 Starting live feed streams...")
        
        # Collect asset IDs and market tickers
        poly_assets = []
        kalshi_tickers = []
        
        for mapping in self.game_mappings.values():
            if mapping['polymarket']:
                for asset in mapping['polymarket'].get('assets', []):
                    poly_assets.append(asset['asset_id'])
            
            if mapping['kalshi']:
                kalshi_tickers.append(mapping['kalshi']['market_ticker'])
        
        # Run both streamers
        tasks = []
        
        if poly_assets:
            tasks.append(
                stream_polymarket_prices(poly_assets, self.handle_polymarket_update)
            )
        
        if kalshi_tickers:
            tasks.append(
                stream_kalshi_prices(kalshi_tickers, self.handle_kalshi_update)
            )
        
        if tasks:
            await asyncio.gather(*tasks)
        else:
            logger.warning("⚠ No streams to run")

async def main():
    """Main entry point"""
    feed = UnifiedFeed()
    await feed.run_stream()

if __name__ == "__main__":
    asyncio.run(main())
