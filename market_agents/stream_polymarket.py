"""
Stream Polymarket prices via WebSocket and emit midpoint percentages.
Connects to Polymarket's WebSocket, subscribes to asset_ids, and maintains best bid/ask.
"""
import asyncio
import websockets
import json
from typing import Dict, Set, List, Optional
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("polymarket_stream")

POLYMARKET_WS = "wss://ws-spreads.polymarket.com/ws"

class PolymarketStreamer:
    def __init__(self):
        self.asset_ids: Set[str] = set()
        self.orderbooks: Dict[str, Dict] = {}  # asset_id -> {best_bid, best_ask}
        self.ws = None
    
    def add_assets(self, asset_ids: List[str]):
        """Add asset IDs to subscribe to"""
        self.asset_ids.update(asset_ids)
    
    async def connect(self):
        """Connect to Polymarket WebSocket"""
        try:
            self.ws = await websockets.connect(POLYMARKET_WS)
            logger.info(f"✓ Connected to Polymarket WS")
            return True
        except Exception as e:
            logger.error(f"✗ Failed to connect: {e}")
            return False
    
    async def subscribe(self):
        """Subscribe to asset updates"""
        try:
            for asset_id in self.asset_ids:
                subscribe_msg = {
                    "type": "subscribe",
                    "channel": "spreads",
                    "asset_id": asset_id
                }
                await self.ws.send(json.dumps(subscribe_msg))
            logger.info(f"✓ Subscribed to {len(self.asset_ids)} assets")
        except Exception as e:
            logger.error(f"✗ Subscription error: {e}")
    
    async def handle_message(self, msg: str) -> Optional[Dict]:
        """Parse incoming message and update orderbook"""
        try:
            data = json.loads(msg)
            
            if data.get('type') != 'spreads':
                return None
            
            asset_id = data.get('asset_id')
            if not asset_id:
                return None
            
            # Extract best bid/ask from spread
            spread = data.get('spread', {})
            best_bid = spread.get('bid', 0)
            best_ask = spread.get('ask', 0)
            
            self.orderbooks[asset_id] = {
                'bid': best_bid,
                'ask': best_ask,
                'midpoint': (best_bid + best_ask) / 2 if (best_bid + best_ask) > 0 else 0.5,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            return {
                'source': 'polymarket',
                'asset_id': asset_id,
                'bid': best_bid,
                'ask': best_ask,
                'midpoint_pct': self.orderbooks[asset_id]['midpoint'],
                'timestamp': self.orderbooks[asset_id]['timestamp']
            }
        except json.JSONDecodeError:
            return None
        except Exception as e:
            logger.warning(f"⚠ Error handling message: {e}")
            return None
    
    async def stream(self, callback=None):
        """Stream incoming updates"""
        try:
            while True:
                msg = await self.ws.recv()
                update = await self.handle_message(msg)
                if update and callback:
                    await callback(update)
        except websockets.exceptions.ConnectionClosed:
            logger.warning("⚠ WebSocket connection closed")
        except Exception as e:
            logger.error(f"✗ Stream error: {e}")
    
    async def close(self):
        """Close WebSocket connection"""
        if self.ws:
            await self.ws.close()

async def stream_polymarket_prices(asset_ids: List[str], callback=None):
    """
    Main entry point to stream Polymarket prices.
    asset_ids: List of asset IDs to subscribe to
    callback: Async function called with each update
    """
    streamer = PolymarketStreamer()
    streamer.add_assets(asset_ids)
    
    if not await streamer.connect():
        return
    
    await streamer.subscribe()
    await streamer.stream(callback)

async def demo():
    """Demo: stream a few sample assets"""
    async def on_update(update):
        print(f"[{update['timestamp']}] {update['asset_id']}: "
              f"bid={update['bid']:.4f} ask={update['ask']:.4f} "
              f"mid={update['midpoint_pct']:.4f}")
    
    # Example asset IDs (replace with real ones from discovery)
    sample_assets = ["example_asset_1", "example_asset_2"]
    
    await stream_polymarket_prices(sample_assets, on_update)

if __name__ == "__main__":
    asyncio.run(demo())
