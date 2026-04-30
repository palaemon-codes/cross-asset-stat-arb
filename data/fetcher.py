# data/fetcher.py
# handles all data downloading and cleaning
# uses yfinance for historical data (free and easy)

import yfinance as yf
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def fetch_prices(tickers, start, end):
    """
    Download adjusted close prices from Yahoo Finance.

    Returns a clean DataFrame (dates x tickers). Drops days where
    more than 20% of assets have missing data, forward-fills small gaps.
    """
    print(f"[data] Fetching {len(tickers)} tickers from {start} to {end} ...")

    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=True
    )

    # yfinance nests columns when fetching multiple tickers
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"].copy()
    else:
        # single ticker edge case
        prices = raw[["Close"]].copy()
        prices.columns = tickers

    # reorder to match the tickers list
    prices = prices[[t for t in tickers if t in prices.columns]]

    # drop rows missing too many assets
    thresh = max(1, int(0.8 * len(tickers)))
    prices = prices.dropna(thresh=thresh)

    # forward fill small gaps (public holidays, early closes)
    prices = prices.ffill(limit=3).dropna()

    print(f"[data] Clean dataset: {len(prices)} days × {prices.shape[1]} assets")
    return prices


def fetch_prices_and_volume(tickers, start, end):
    """
    Fetch both adjusted close and volume.
    Volume is needed for the square-root market impact model.
    """
    print(f"[data] Fetching prices + volume for {tickers} ...")

    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=True
    )

    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"].copy()
        volume = raw["Volume"].copy()
    else:
        prices = raw[["Close"]].copy()
        volume = raw[["Volume"]].copy()
        prices.columns = tickers
        volume.columns = tickers

    prices = prices[[t for t in tickers if t in prices.columns]]
    volume = volume[[t for t in tickers if t in volume.columns]]

    thresh = max(1, int(0.8 * len(tickers)))
    prices = prices.dropna(thresh=thresh).ffill(limit=3).dropna()
    volume = volume.reindex(prices.index).ffill(limit=3).fillna(0)

    return prices, volume


def compute_dollar_adv(prices, volume, window=20):
    """
    Dollar Average Daily Volume = price * share volume, smoothed over window.
    Used in the square-root market impact model.
    """
    dollar_vol = prices * volume
    adv = dollar_vol.rolling(window=window, min_periods=5).mean()
    return adv


def compute_log_returns(prices):
    """Log returns - stabler than simple returns for the VECM math."""
    return np.log(prices / prices.shift(1)).dropna()


def check_stationarity_summary(prices):
    """
    Quick ADF test on each price series to confirm they're I(1).
    The Johansen test requires all series to be integrated of order 1.
    Returns a DataFrame with ADF stats and a verdict.
    """
    from statsmodels.tsa.stattools import adfuller

    results = {}
    for col in prices.columns:
        adf_level  = adfuller(prices[col].dropna(), autolag='AIC')
        adf_diff   = adfuller(prices[col].diff().dropna(), autolag='AIC')
        results[col] = {
            'adf_level_stat'  : round(adf_level[0], 4),
            'adf_level_pval'  : round(adf_level[1], 4),
            'adf_diff_stat'   : round(adf_diff[0], 4),
            'adf_diff_pval'   : round(adf_diff[1], 4),
            'is_I1'           : adf_level[1] > 0.05 and adf_diff[1] < 0.05
        }

    df = pd.DataFrame(results).T
    return df


def split_train_test(prices, train_end='2023-01-01'):
    """Simple chronological train/test split."""
    train = prices[prices.index < train_end]
    test  = prices[prices.index >= train_end]
    print(f"[data] Train: {len(train)} days | Test: {len(test)} days")
    return train, test
