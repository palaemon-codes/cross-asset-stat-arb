# run_backtest.py
# Run the full backtest pipeline, print results, and save everything to results/.
#
# Usage:
#   python run_backtest.py
#
# Takes about 2-5 minutes depending on your machine (rolling Johansen is the slow part).
# Output files go to the results/ folder.

import os
import sys
import logging
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.WARNING)

from config import ASSETS, START_DATE, END_DATE, INITIAL_CAPITAL
from data.fetcher import fetch_prices_and_volume, check_stationarity_summary, split_train_test
from backtest.engine import MicrostructureBacktester, capacity_analysis

RESULTS_DIR = 'results'


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

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

    # ---- 4. run backtest ----
    print("\n[step 4/5] Running backtest (this takes a few minutes)...")
    bt = MicrostructureBacktester(prices, volume_df=volume, capital=INITIAL_CAPITAL)
    result = bt.run()

    # ---- 5. results ----
    print("\n[step 5/5] Results:")
    result.print_tearsheet()

    cap_df = None
    if len(result.trades) > 0:
        first_beta = list(bt.hedge_betas.values())[0] if bt.hedge_betas else None
        if first_beta is not None:
            print("  Capacity Analysis:")
            cap_df = capacity_analysis(result, prices, first_beta)
            print(cap_df.to_string(index=False))
            print()

    # ---- save outputs to results/ ----
    _save_metrics(result, cap_df, adf_stats)
    _plot_tearsheet(result)
    _plot_spread_analysis(bt, prices)

    if len(result.trades) > 0:
        result.trades.to_csv(f'{RESULTS_DIR}/trades.csv', index=False)
        print(f"  Trade log  → {RESULTS_DIR}/trades.csv")

    return result


def _save_metrics(result, cap_df, adf_stats):
    """Write performance metrics and capacity table to a text file."""
    path = f'{RESULTS_DIR}/metrics.txt'
    with open(path, 'w') as f:
        f.write("PERFORMANCE METRICS\n")
        f.write("=" * 45 + "\n")
        f.write(f"Assets          : {ASSETS}\n")
        f.write(f"Period          : {START_DATE} to {END_DATE}\n")
        f.write(f"Initial Capital : ${INITIAL_CAPITAL:,.0f}\n\n")

        returns = result.equity_curve.pct_change().dropna()
        ann_ret = returns.mean() * 252 * 100
        ann_vol = returns.std() * (252 ** 0.5) * 100
        sharpe  = ann_ret / ann_vol if ann_vol > 0 else 0
        roll_max = result.equity_curve.cummax()
        max_dd   = ((result.equity_curve - roll_max) / roll_max).min() * 100
        tot_ret  = (result.equity_curve.iloc[-1] / result.equity_curve.iloc[0] - 1) * 100

        f.write(f"Total Return        : {tot_ret:.2f}%\n")
        f.write(f"Annualised Return   : {ann_ret:.2f}%\n")
        f.write(f"Annualised Vol      : {ann_vol:.2f}%\n")
        f.write(f"Sharpe Ratio        : {sharpe:.3f}\n")
        f.write(f"Max Drawdown        : {max_dd:.2f}%\n")
        f.write(f"Number of Trades    : {len(result.trades)}\n\n")

        f.write("COST BREAKDOWN\n")
        f.write("-" * 45 + "\n")
        for k, v in result.cost_breakdown.items():
            f.write(f"  {k:<25}: ${v:,.2f}\n")

        if cap_df is not None:
            f.write("\nCAPACITY ANALYSIS\n")
            f.write("-" * 45 + "\n")
            f.write(cap_df.to_string(index=False))
            f.write("\n")

        f.write("\nSTATIONARITY CHECK (ADF)\n")
        f.write("-" * 45 + "\n")
        f.write(adf_stats[['adf_level_pval', 'adf_diff_pval', 'is_I1']].to_string())
        f.write("\n")

    print(f"  Metrics    → {path}")


def _plot_tearsheet(result):
    """
    3-panel tearsheet: equity curve, drawdown, rolling Sharpe.
    Saved to results/tearsheet.png.
    """
    returns  = result.equity_curve.pct_change().dropna()
    roll_max = result.equity_curve.cummax()
    drawdown = (result.equity_curve - roll_max) / roll_max * 100
    roll_sharpe = (returns.rolling(63).mean() / returns.rolling(63).std()) * (252 ** 0.5)

    fig = plt.figure(figsize=(13, 9))
    gs  = gridspec.GridSpec(3, 1, figure=fig, hspace=0.4)

    ax1 = fig.add_subplot(gs[0])
    ax1.plot(result.equity_curve.index, result.equity_curve / 1e6, linewidth=1.3)
    ax1.axhline(result.equity_curve.iloc[0] / 1e6, color='gray', linestyle='--', linewidth=0.8)
    ax1.set_ylabel('Portfolio Value ($M)')
    ax1.set_title(f'Equity Curve — Energy Sector Stat Arb ({START_DATE} to {END_DATE})')
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:.2f}M'))
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.fill_between(drawdown.index, drawdown.values, 0, color='red', alpha=0.45)
    ax2.set_ylabel('Drawdown (%)')
    ax2.set_title('Drawdown')
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.plot(roll_sharpe.index, roll_sharpe.values, linewidth=1.1, color='green')
    ax3.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax3.set_ylabel('Sharpe')
    ax3.set_title('Rolling 63-day Sharpe')
    ax3.grid(True, alpha=0.3)

    # annotate final metrics
    ann_ret = returns.mean() * 252 * 100
    ann_vol = returns.std() * (252 ** 0.5) * 100
    sharpe  = ann_ret / ann_vol if ann_vol > 0 else 0
    max_dd  = drawdown.min()
    fig.text(
        0.13, 0.97,
        f"Ann. Return: {ann_ret:.1f}%   Ann. Vol: {ann_vol:.1f}%   "
        f"Sharpe: {sharpe:.2f}   Max DD: {max_dd:.1f}%",
        fontsize=9, color='#333333'
    )

    path = f'{RESULTS_DIR}/tearsheet.png'
    plt.savefig(path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  Tearsheet  → {path}")


def _plot_spread_analysis(bt, prices):
    """
    Plot the spread z-score over time using the first available beta.
    Shows how the z-score triggers entries/exits.
    """
    if not bt.hedge_betas:
        return

    from engine.vecm import compute_spread_from_beta, compute_zscore_from_spread
    from config import ENTRY_Z_SCORE, EXIT_Z_SCORE, LOOKBACK_WINDOW

    first_beta = list(bt.hedge_betas.values())[0]
    spread = compute_spread_from_beta(prices, first_beta)
    zscore = compute_zscore_from_spread(spread, window=LOOKBACK_WINDOW).dropna()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
    fig.suptitle('Spread and Z-score (first cointegrating vector)', fontsize=11)

    ax1.plot(spread.index, spread.values, linewidth=0.9, color='steelblue')
    ax1.set_ylabel('Spread level')
    ax1.set_title('Cointegrating Spread (β\'X_t)')
    ax1.grid(True, alpha=0.3)

    ax2.plot(zscore.index, zscore.values, linewidth=0.9, color='steelblue')
    ax2.axhline(ENTRY_Z_SCORE,  color='red',   linestyle='--', linewidth=0.9, label=f'+{ENTRY_Z_SCORE} entry')
    ax2.axhline(-ENTRY_Z_SCORE, color='green', linestyle='--', linewidth=0.9, label=f'-{ENTRY_Z_SCORE} entry')
    ax2.axhline(0, color='gray', linestyle='-', linewidth=0.6)
    ax2.set_ylabel('Z-score')
    ax2.set_title('Spread Z-score')
    ax2.legend(fontsize=8, loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = f'{RESULTS_DIR}/spread_zscore.png'
    plt.savefig(path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  Spread     → {path}")


if __name__ == '__main__':
    result = main()

