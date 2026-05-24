# main.py
# Live trading orchestrator
#
# This wires up the ZeroMQ publisher (simulating live data) + subscriber
# (signal engine) in a proper async event loop.
#
# For actual live trading you'd replace PricePublisher with LiveTickPublisher
# connected to your broker's WebSocket feed.
#
# Run:  python main.py [--replay | --live]

import asyncio
import argparse
import logging
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

from config import ASSETS, START_DATE, END_DATE, INITIAL_CAPITAL, LOOKBACK_WINDOW
from data.fetcher import fetch_prices_and_volume
from engine.johansen import run_johansen
from engine.vecm import compute_spread_from_beta, compute_zscore_from_spread
from engine.signals import generate_signals, get_signal_stats
from risk.hedging import RiskMonitor
from broker.publisher import PricePublisher
from broker.subscriber import PriceSubscriber


class LiveArbitrageEngine:
    """
    Main engine that runs in the async event loop.
    Receives price bars → updates z-score → generates signal → logs trades.
    """

    def __init__(self, warmup_prices: pd.DataFrame, capital: float = INITIAL_CAPITAL):
        self.capital  = capital
        self.tickers  = list(warmup_prices.columns)
        self.k        = len(self.tickers)

        # fit initial Johansen on warmup data
        logger.info("Fitting initial Johansen on warmup data...")
        joh = run_johansen(warmup_prices)
        rank = joh.cointegration_rank()
        joh.print_summary()

        if rank == 0:
            logger.warning("No cointegration in warmup data — engine will monitor but not trade")
            self.beta = None
        else:
            self.beta = joh.get_hedge_ratios(r=1)[:, 0]
            logger.info(f"Cointegration rank={rank}, beta={np.round(self.beta, 4)}")

        self.risk_monitor  = RiskMonitor(self.tickers, self.beta if self.beta is not None else np.zeros(self.k), capital)
        self.price_history = warmup_prices.copy()
        self.current_signal = 0
        self.position       = {t: 0.0 for t in self.tickers}
        self.pnl            = 0.0
        self.bar_count      = 0
        self.trade_count    = 0

    async def on_bar(self, timestamp: str, prices_dict: dict):
        """
        Called by subscriber when a complete bar arrives.
        This is the hot path — keep it fast.
        """
        self.bar_count += 1

        # build a price row
        try:
            row = pd.Series({t: prices_dict[t] for t in self.tickers if t in prices_dict})
        except Exception as e:
            logger.warning(f"Bad price bar at {timestamp}: {e}")
            return

        if row.isna().any():
            return

        # update price history
        new_row_df = pd.DataFrame([row], index=[pd.Timestamp(timestamp)])
        self.price_history = pd.concat([self.price_history, new_row_df]).tail(LOOKBACK_WINDOW * 2)

        # risk monitor may update beta if an asset is halted
        active_beta = self.risk_monitor.update(prices_dict)

        if active_beta is None or np.all(active_beta == 0):
            return

        # compute current z-score
        if len(self.price_history) < 30:
            return

        spread = compute_spread_from_beta(self.price_history, active_beta)
        zscore = compute_zscore_from_spread(spread, window=min(LOOKBACK_WINDOW, len(spread)))
        current_z = zscore.iloc[-1]

        if np.isnan(current_z):
            return

        # update signal
        prev_signal    = self.current_signal
        new_signal     = self._update_signal(current_z)
        self.current_signal = new_signal

        # log on signal change
        if new_signal != prev_signal:
            self.trade_count += 1
            action = {1: "LONG SPREAD", -1: "SHORT SPREAD", 0: "FLAT"}.get(new_signal, "?")
            logger.info(
                f"[{timestamp}]  Signal: {action:<14}  z={current_z:+.3f}  "
                f"trades={self.trade_count}"
            )

        # every 50 bars print a status update
        if self.bar_count % 50 == 0:
            logger.info(
                f"  bar #{self.bar_count:>5} | z={current_z:+.3f} | "
                f"signal={self.current_signal:+d} | PnL=${self.pnl:,.0f}"
            )

    def _update_signal(self, z):
        from config import ENTRY_Z_SCORE, EXIT_Z_SCORE
        sig = self.current_signal
        if sig == 0:
            if z > ENTRY_Z_SCORE:
                return -1
            elif z < -ENTRY_Z_SCORE:
                return 1
        elif sig == 1:
            if z > -EXIT_Z_SCORE:
                return 0
        elif sig == -1:
            if z < EXIT_Z_SCORE:
                return 0
        return sig


async def run_replay_mode(prices: pd.DataFrame):
    """
    Replay mode: publisher replays historical prices, subscriber runs signal engine.
    Demonstrates the full distributed architecture working end-to-end.
    """
    warmup = prices.iloc[:LOOKBACK_WINDOW]
    replay = prices.iloc[LOOKBACK_WINDOW:]

    engine     = LiveArbitrageEngine(warmup_prices=warmup)
    publisher  = PricePublisher(replay, delay_ms=5)   # fast replay
    subscriber = PriceSubscriber(ASSETS, on_bar_callback=engine.on_bar)

    logger.info("Starting ZMQ publisher and subscriber...")

    pub_task = asyncio.create_task(publisher.start())
    sub_task = asyncio.create_task(subscriber.start())

    try:
        await asyncio.wait_for(pub_task, timeout=300)
    except asyncio.TimeoutError:
        logger.info("Replay timeout reached")
    finally:
        publisher.stop()
        subscriber.stop()
        pub_task.cancel()
        sub_task.cancel()

    logger.info(f"Replay done. Total bars: {engine.bar_count}, trades: {engine.trade_count}")


def main():
    parser = argparse.ArgumentParser(description='Cross-Asset Stat Arb Engine')
    parser.add_argument('--mode', choices=['replay', 'backtest'], default='replay',
                        help='replay = ZMQ live simulation, backtest = run_backtest.py')
    args = parser.parse_args()

    print("\n" + "=" * 55)
    print("  CROSS-ASSET STATISTICAL ARBITRAGE ENGINE")
    print(f"  Mode: {args.mode.upper()}")
    print("=" * 55)

    print("\nFetching data...")
    prices, volume = fetch_prices_and_volume(ASSETS, START_DATE, END_DATE)

    if args.mode == 'replay':
        print(f"Running ZMQ replay of {len(prices)} bars...")
        asyncio.run(run_replay_mode(prices))
    else:
        print("For backtest mode, run: python run_backtest.py")


if __name__ == '__main__':
    main()
