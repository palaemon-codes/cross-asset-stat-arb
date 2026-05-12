# broker/subscriber.py
# ZeroMQ subscriber — receives price ticks, updates the price buffer,
# and triggers the signal engine when a full bar arrives.
#
# Key design choice: the subscriber runs in its own async task. The math
# engine is called synchronously (it's fast enough for daily bars).
# For tick-level HFT you'd want to push the math to a separate process.

import asyncio
import json
import zmq
import zmq.asyncio
import pandas as pd
import numpy as np
import logging
from collections import defaultdict
from datetime import datetime
from config import ZMQ_SUB_ADDRESS

logger = logging.getLogger(__name__)


class PriceBuffer:
    """
    Accumulates incoming ticks per timestamp.
    Emits a complete bar (all tickers filled) once all assets have checked in.
    """

    def __init__(self, tickers):
        self.tickers     = set(tickers)
        self._buffer     = {}   # timestamp -> {ticker: price}
        self._bar_ready  = asyncio.Queue()

    def add_tick(self, ticker, price, timestamp):
        if timestamp not in self._buffer:
            self._buffer[timestamp] = {}
        self._buffer[timestamp][ticker] = price

        # check if all assets have reported for this timestamp
        if self.tickers.issubset(self._buffer[timestamp].keys()):
            bar = self._buffer.pop(timestamp)
            row = {t: bar[t] for t in self.tickers}
            self._bar_ready.put_nowait((timestamp, row))

    async def get_bar(self):
        return await self._bar_ready.get()


class PriceSubscriber:
    """
    Subscribes to all ticker topics on the ZMQ PUB socket.
    Feeds ticks into PriceBuffer and calls a callback when bars are complete.
    """

    def __init__(self, tickers, on_bar_callback):
        """
        Args:
            tickers          : list of ticker symbols to subscribe to
            on_bar_callback  : async function(timestamp, prices_dict) called on each bar
        """
        self.tickers     = tickers
        self.on_bar      = on_bar_callback
        self.context     = zmq.asyncio.Context()
        self.socket      = None
        self.buffer      = PriceBuffer(tickers)
        self._running    = False
        self._recv_count = 0

    async def start(self):
        self.socket = self.context.socket(zmq.SUB)
        self.socket.connect(ZMQ_SUB_ADDRESS)

        # subscribe to each ticker topic
        for ticker in self.tickers:
            self.socket.setsockopt(zmq.SUBSCRIBE, ticker.encode())

        self._running = True
        logger.info(f"[subscriber] Connected to {ZMQ_SUB_ADDRESS}, topics: {self.tickers}")
        print(f"[subscriber] Listening for {len(self.tickers)} tickers")

        # run recv loop and bar processor concurrently
        await asyncio.gather(
            self._recv_loop(),
            self._bar_dispatch_loop()
        )

    async def _recv_loop(self):
        """Non-blocking receive loop."""
        while self._running:
            try:
                parts = await asyncio.wait_for(
                    self.socket.recv_multipart(),
                    timeout=2.0
                )
                topic   = parts[0].decode()
                payload = json.loads(parts[1].decode())

                ticker    = payload['ticker']
                price     = payload['price']
                timestamp = payload['timestamp']

                self.buffer.add_tick(ticker, price, timestamp)
                self._recv_count += 1

            except asyncio.TimeoutError:
                # no data in 2s, probably replay finished
                continue
            except Exception as e:
                logger.error(f"[subscriber] recv error: {e}")
                break

    async def _bar_dispatch_loop(self):
        """Pull completed bars from the buffer and call callback."""
        while self._running:
            try:
                timestamp, prices_dict = await asyncio.wait_for(
                    self.buffer.get_bar(),
                    timeout=3.0
                )
                await self.on_bar(timestamp, prices_dict)
            except asyncio.TimeoutError:
                continue

    def stop(self):
        self._running = False
        if self.socket:
            self.socket.close()
        self.context.term()
        print(f"[subscriber] Stopped. Processed {self._recv_count} ticks.")


class InMemoryBroker:
    """
    Bypass ZMQ entirely for backtesting — directly calls the callback
    with each historical bar. Much faster than going through sockets.

    This lets us reuse the same signal pipeline for both live and backtest.
    """

    def __init__(self, prices_df, on_bar_callback):
        self.prices_df = prices_df
        self.on_bar    = on_bar_callback

    async def run(self):
        print(f"[broker] Running in-memory replay of {len(self.prices_df)} bars")
        for timestamp, row in self.prices_df.iterrows():
            prices_dict = row.to_dict()
            await self.on_bar(str(timestamp.date()), prices_dict)
        print("[broker] Replay complete")
