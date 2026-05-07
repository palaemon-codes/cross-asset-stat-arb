# engine/signals.py
# trading signal generation from the VECM spread
#
# logic is simple:
#   - z > +ENTRY_Z  => spread is too high => short the spread
#   - z < -ENTRY_Z  => spread is too low  => long the spread
#   - |z| < EXIT_Z  => close position
#
# the "spread" here is β' * X_t
# to trade it, we need to decompose the bet across individual assets
# using the beta hedge ratios

import numpy as np
import pandas as pd
from config import ENTRY_Z_SCORE, EXIT_Z_SCORE
import logging

logger = logging.getLogger(__name__)


def generate_signals(zscore_series):
    """
    Convert z-scores to position signals: +1 (long spread), -1 (short spread), 0 (flat).

    Uses a simple state machine — once in a position, stay until exit threshold.
    This avoids thrashing in and out near the entry level.
    """
    signals = pd.Series(0, index=zscore_series.index, dtype=float)
    position = 0

    for i, z in enumerate(zscore_series):
        if np.isnan(z):
            signals.iloc[i] = 0
            continue

        if position == 0:
            if z > ENTRY_Z_SCORE:
                position = -1   # short the spread (spread will revert down)
            elif z < -ENTRY_Z_SCORE:
                position = 1    # long the spread (spread will revert up)
        elif position == 1:
            if z > -EXIT_Z_SCORE:
                position = 0    # close
        elif position == -1:
            if z < EXIT_Z_SCORE:
                position = 0    # close

        signals.iloc[i] = position

    return signals


def signals_to_asset_positions(signals, beta_vec, prices_df, capital=1_000_000):
    """
    Convert scalar spread signals to dollar positions in each asset.

    When signal = +1 (long spread): buy assets with positive beta, short negative
    When signal = -1 (short spread): the reverse

    Dollar allocation is proportional to beta weights, scaled to capital.
    """
    k = len(beta_vec)
    tickers = list(prices_df.columns)

    # normalize beta so dollar weights sum to 1 on each side
    pos_beta = np.maximum(beta_vec, 0)
    neg_beta = np.minimum(beta_vec, 0)

    pos_sum = pos_beta.sum()
    neg_sum = abs(neg_beta.sum())

    if pos_sum > 0:
        pos_beta = pos_beta / pos_sum
    if neg_sum > 0:
        neg_beta = neg_beta / neg_sum

    positions = pd.DataFrame(0.0, index=prices_df.index, columns=tickers)

    for date, sig in signals.items():
        if sig == 1:    # long spread: buy positives, short negatives
            for j, t in enumerate(tickers):
                positions.loc[date, t] = pos_beta[j] * capital - neg_beta[j] * capital
        elif sig == -1:  # short spread: reverse
            for j, t in enumerate(tickers):
                positions.loc[date, t] = neg_beta[j] * capital - pos_beta[j] * capital
        # sig == 0 stays 0

    return positions


def compute_position_changes(positions):
    """
    Day-over-day changes in positions.
    Used to identify trade events for cost calculation.
    """
    return positions.diff().fillna(positions)


def compute_trade_notional(position_changes, prices_df):
    """
    Dollar notional traded on each day.
    |Δposition| gives us the traded shares * price = dollar value.
    """
    return position_changes.abs()


def get_signal_stats(signals, zscore):
    """Quick diagnostic stats on signals — useful for sanity checking."""
    n_trades = (signals.diff().abs() > 0).sum()
    long_pct  = (signals == 1).mean() * 100
    short_pct = (signals == -1).mean() * 100
    flat_pct  = (signals == 0).mean() * 100

    print(f"\n  Signal Statistics:")
    print(f"    Total signal changes : {n_trades}")
    print(f"    Time long            : {long_pct:.1f}%")
    print(f"    Time short           : {short_pct:.1f}%")
    print(f"    Time flat            : {flat_pct:.1f}%")
    print(f"    Z-score range        : [{zscore.min():.2f}, {zscore.max():.2f}]")
