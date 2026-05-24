# run_backtest.py
# Run the full backtest pipeline and show results.
# This is the main script to run if you just want to see performance numbers.
#
# Usage:
#   python run_backtest.py
#
# Takes about 2-5 minutes depending on your machine (mostly the rolling Johansen).

import sys
import logging
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')   # statsmodels throws a bunch of convergence warnings
logging.basicConfig(level=logging.WARNING)  # set to INFO for more detail

from config import ASSETS, START_DATE, END_DATE, INITIAL_CAPITAL
from data.fetcher import fetch_prices_and_volume, check_stationarity_summary, split_train_test
from backtest.engine import MicrostructureBacktester, capacity_analysis


def main():
    print("\n" + "=" * 60)
    print("  CROSS-ASSET STAT ARB ENGINE — BACKTEST")
    print(f"  Assets : {ASSETS}")
    print(f"  Period : {START_DATE} → {END_DATE}")
    print("=" * 60)

    # ---- 1. load data ----
    print("\n[step 1/5] Loading data...")
    prices, volume = fetch_prices_and_volume(ASSETS, START_DATE, END_DATE)

    print(f"\n  Price data shape : {prices.shape}")
    print(f"  Date range       : {prices.index[0].date()} → {prices.index[-1].date()}")
    print(f"\n  Price snapshot (last 3 rows):")
    print(prices.tail(3).to_string())

    # ---- 2. stationarity check ----
    print("\n[step 2/5] Checking stationarity (ADF test)...")
    adf_stats = check_stationarity_summary(prices)
    print("\n  ADF Test Summary (is each series I(1)?)")
    print(adf_stats[['adf_level_pval', 'adf_diff_pval', 'is_I1']].to_string())
    
    i1_count = adf_stats['is_I1'].sum()
    print(f"\n  => {i1_count}/{len(ASSETS)} assets confirmed I(1) at 5% significance")
    if i1_count < len(ASSETS) // 2:
        print("  [WARNING] Many assets not I(1) — Johansen results may be unreliable")

    # ---- 3. train/test split ----
    print("\n[step 3/5] Splitting train/test...")
    train_prices, test_prices = split_train_test(prices, train_end='2023-01-01')
    train_vol, test_vol = split_train_test(volume, train_end='2023-01-01')

    # ---- 4. run backtest on full data ----
    print("\n[step 4/5] Running backtest (this takes a few minutes)...")
    bt = MicrostructureBacktester(prices, volume_df=volume, capital=INITIAL_CAPITAL)
    result = bt.run()

    # ---- 5. results ----
    print("\n[step 5/5] Results:")
    result.print_tearsheet()

    # capacity analysis
    if len(result.trades) > 0:
        first_beta = list(bt.hedge_betas.values())[0] if bt.hedge_betas else None
        if first_beta is not None:
            print("  Capacity Analysis (how AUM affects alpha):")
            cap_df = capacity_analysis(result, prices, first_beta)
            print(cap_df.to_string(index=False))
            print()

    # save equity curve
    equity_path = 'equity_curve.csv'
    result.equity_curve.to_csv(equity_path, header=True)
    print(f"  Equity curve saved to: {equity_path}")

    if len(result.trades) > 0:
        trades_path = 'trades.csv'
        result.trades.to_csv(trades_path, index=False)
        print(f"  Trade log saved to: {trades_path}")

    return result


if __name__ == '__main__':
    result = main()
