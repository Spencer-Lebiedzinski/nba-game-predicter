"""
Stream Kalshi prices via WebSocket with authentication.
Maintains in-memory orderbook from snapshot + deltas and emits midpoint %.
"""
import asyncio
import websockets
import json
from typing import Dict, Set, List, Optional
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("kalshi_stream")

KALSHI_WS = "wss://api.kalshi.com/trade-api/v2/ws"

class KalshiOrderbook:
    """Maintain orderbook from snapshot and deltas"""
    def __init__(self):
        self.bids: Dict[float, float] = {}  # price -> quantity
        self.asks: Dict[float, float] = {}  # price -> quantity
    
    def apply_snapshot(self, bids: List, asks: List):
        """Apply orderbook snapshot"""
        self.bids = {float(b[0]): float(b[1]) for b in bids}
        self.asks = {float(a[0]): float(a[1]) for a in asks}
    
    def apply_delta(self, side: str, price: float, quantity: float):
        """Apply orderbook delta (update or delete)"""
        price = float(price)
        quantity = float(quantity)
        
        if side == 'bid':
            if quantity == 0:
                self.bids.pop(price, None)
            else:
                self.bids[price] = quantity
        elif side == 'ask':
            if quantity == 0:
                self.asks.pop(price, None)
            else:
                self.asks[price] = quantity
    
    def get_best_bid(self) -> Optional[float]:
        return max(self.bids.keys()) if self.bids else None
    
    def get_best_ask(self) -> Optional[float]:
        return min(self.asks.keys()) if self.asks else None
    
    def get_midpoint(self) -> Optional[float]:
        bid = self.get_best_bid()
        ask = self.get_best_ask()
        if bid and ask:
            return (bid + ask) / 2
        return None

class KalshiStreamer:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.market_tickers: Set[str] = set()
        self.orderbooks: Dict[str, KalshiOrderbook] = {}
        self.ws = None
    
    def add_markets(self, market_tickers: List[str]):
        """Add market tickers to subscribe to"""
        self.market_tickers.update(market_tickers)
    
    async def connect(self):
        """Connect to Kalshi WebSocket with auth"""
        try:
            headers = {}
            if self.api_key:
                headers['Authorization'] = f"Bearer {self.api_key}"
            
            self.ws = await websockets.connect(KALSHI_WS, subprotocols=["chat"])
            logger.info(f"✓ Connected to Kalshi WS")
            return True
        except Exception as e:
            logger.error(f"✗ Failed to connect: {e}")
            return False
    
    async def subscribe(self):
        """Subscribe to orderbook updates"""
        try:
            for ticker in self.market_tickers:
                # Initialize orderbook
                self.orderbooks[ticker] = KalshiOrderbook()
                
                # Subscribe to orderbook channel
                subscribe_msg = {
                    "type": "subscribe",
                    "channel": f"orderbook_v2",
                    "market_ticker": ticker
                }
                await self.ws.send(json.dumps(subscribe_msg))
            
            logger.info(f"✓ Subscribed to {len(self.market_tickers)} markets")
        except Exception as e:
            logger.error(f"✗ Subscription error: {e}")
    
    async def handle_message(self, msg: str) -> Optional[Dict]:
        """Parse incoming message and update orderbook"""
        try:
            data = json.loads(msg)
            msg_type = data.get('type', '')
            
            if msg_type == 'snapshot':
                ticker = data.get('market_ticker', '')
                if ticker not in self.orderbooks:
                    self.orderbooks[ticker] = KalshiOrderbook()
                
                self.orderbooks[ticker].apply_snapshot(
                    data.get('bids', []),
                    data.get('asks', [])
                )
                
                mid = self.orderbooks[ticker].get_midpoint()
                if mid:
                    return {
                        'source': 'kalshi',
                        'market_ticker': ticker,
                        'message_type': 'snapshot',
                        'midpoint_pct': mid,
                        'timestamp': datetime.utcnow().isoformat()
                    }
            
            elif msg_type == 'delta':
                ticker = data.get('market_ticker', '')
                if ticker not in self.orderbooks:
                    self.orderbooks[ticker] = KalshiOrderbook()
                
                # Apply bid deltas
                for bid_delta in data.get('bid_deltas', []):
                    self.orderbooks[ticker].apply_delta('bid', bid_delta[0], bid_delta[1])
                
                # Apply ask deltas
                for ask_delta in data.get('ask_deltas', []):
                    self.orderbooks[ticker].apply_delta('ask', ask_delta[0], ask_delta[1])
                
                mid = self.orderbooks[ticker].get_midpoint()
                if mid:
                    return {
                        'source': 'kalshi',
                        'market_ticker': ticker,
                        'message_type': 'delta',
                        'midpoint_pct': mid,
                        'timestamp': datetime.utcnow().isoformat()
                    }
            
            return None
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

async def stream_kalshi_prices(market_tickers: List[str], callback=None, api_key: Optional[str] = None):
    """
    Main entry point to stream Kalshi prices.
    market_tickers: List of market tickers to subscribe to
    callback: Async function called with each update
    api_key: Optional API key for authenticated access
    """
    streamer = KalshiStreamer(api_key=api_key)
    streamer.add_markets(market_tickers)
    
    if not await streamer.connect():
        return
    
    await streamer.subscribe()
    await streamer.stream(callback)

async def demo():
    """Demo: stream a few sample markets"""
    async def on_update(update):
        print(f"[{update['timestamp']}] {update['market_ticker']}: "
              f"mid={update['midpoint_pct']:.4f}")
    
    # Example market tickers (replace with real ones from discovery)
    sample_markets = ["EXAMPLE-1", "EXAMPLE-2"]
    
    await stream_kalshi_prices(sample_markets, on_update)

if __name__ == "__main__":
    asyncio.run(demo())
