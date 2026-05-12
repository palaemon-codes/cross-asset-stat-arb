# broker/publisher.py
# ZeroMQ price publisher
#
# In a real system this would connect to exchange WebSocket feeds and
# forward price updates to internal consumers.
# Here we simulate it by replaying historical data at configurable speed.
#
# Pattern: PUB/SUB
#   publisher (this file)  →  [ZMQ socket]  →  subscriber (subscriber.py)
#
# The publisher runs as an async task so it doesn't block the main thread.

import asyncio
import json
import zmq
import zmq.asyncio
import pandas as pd
import numpy as np
import logging
import time
from config import ZMQ_PUB_ADDRESS

logger = logging.getLogger(__name__)


class PricePublisher:
    """
    Replays historical prices over ZeroMQ PUB socket.
    
    Sends JSON messages like:
        {"ticker": "XOM", "price": 98.34, "timestamp": "2024-01-15", "seq": 42}
    """

    def __init__(self, prices_df: pd.DataFrame, delay_ms: int = 100):
        """
        Args:
            prices_df : DataFrame with dates as index, tickers as columns
            delay_ms  : milliseconds between ticks (100ms simulates ~10 ticks/sec)
        """
        self.prices_df = prices_df
        self.delay     = delay_ms / 1000.0
        self.context   = zmq.asyncio.Context()
        self.socket    = None
        self._running  = False
        self._seq      = 0

    async def start(self):
        """Bind the socket and start publishing."""
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind(ZMQ_PUB_ADDRESS)

        # small sleep so subscribers can connect before we start sending
        await asyncio.sleep(0.5)

        self._running = True
        logger.info(f"[publisher] Bound to {ZMQ_PUB_ADDRESS}")
        print(f"[publisher] Starting replay of {len(self.prices_df)} days of data")

        await self._replay_loop()

    async def _replay_loop(self):
        """Iterate over historical rows and publish each as a tick."""
        for date, row in self.prices_df.iterrows():
            if not self._running:
                break

            for ticker, price in row.items():
                if pd.isna(price):
                    continue

                msg = {
                    "ticker"    : str(ticker),
                    "price"     : round(float(price), 4),
                    "timestamp" : str(date.date()),
                    "seq"       : self._seq
                }
                # ZMQ topic filter: send on topic = ticker symbol
                topic   = ticker.encode()
                payload = json.dumps(msg).encode()
                await self.socket.send_multipart([topic, payload])
                self._seq += 1

            await asyncio.sleep(self.delay)

        print(f"[publisher] Replay complete. Sent {self._seq} messages.")
        self._running = False

    def stop(self):
        self._running = False
        if self.socket:
            self.socket.close()
        self.context.term()
        logger.info("[publisher] Stopped")


class LiveTickPublisher:
    """
    Minimal live publisher for connecting to actual exchange WebSockets.
    Stub — not implemented, just shows the interface.
    """

    def __init__(self, tickers, source='alpaca'):
        self.tickers = tickers
        self.source  = source
        self.context = zmq.asyncio.Context()
        self.socket  = None

    async def start(self):
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind(ZMQ_PUB_ADDRESS)
        await asyncio.sleep(0.3)

        if self.source == 'alpaca':
            await self._stream_alpaca()
        else:
            raise NotImplementedError(f"Source '{self.source}' not implemented yet")

    async def _stream_alpaca(self):
        # TODO: implement actual Alpaca websocket connection
        # would use alpaca-trade-api or websockets library
        # leaving as stub for now
        logger.warning("[publisher] Alpaca live streaming not implemented — use PricePublisher for replay")
        raise NotImplementedError

    def stop(self):
        if self.socket:
            self.socket.close()
        self.context.term()
