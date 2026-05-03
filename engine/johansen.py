# engine/johansen.py
# Johansen cointegration test wrapper
#
# The core idea: if k price series are cointegrated, there exist linear
# combinations (the cointegrating vectors) that are stationary even though
# the individual series are non-stationary (random walks).
#
# Johansen gives us:
#   - rank r  : how many independent cointegrating relationships exist
#   - beta    : the r cointegrating vectors (eigenvectors of Pi matrix)
#   - alpha   : the adjustment speed matrix (how fast prices return to eq.)
#
# Reference: Johansen (1988), "Statistical analysis of cointegration vectors"

import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import logging

logger = logging.getLogger(__name__)


class JohansenResult:
    """Wraps the statsmodels output so its easier to pass around."""

    def __init__(self, raw_result, tickers, det_order, k_ar_diff):
        self.raw        = raw_result
        self.tickers    = tickers
        self.det_order  = det_order
        self.k_ar_diff  = k_ar_diff

        # trace and max-eigenvalue test statistics
        self.trace_stat    = raw_result.lr1   # trace statistics
        self.trace_cv      = raw_result.cvt   # critical values [90%, 95%, 99%]
        self.max_eig_stat  = raw_result.lr2
        self.max_eig_cv    = raw_result.cvm

        # eigenvectors = cointegrating vectors (columns)
        # shape: (k, r) where k = # assets, r = # cointegrating relations
        self.evec = raw_result.evec

        # eigenvalues (sorted descending)
        self.eval = raw_result.eig

    def cointegration_rank(self, significance=0.05):
        """
        Determine rank r using the trace test at given significance level.
        cv_idx: 0 = 90%, 1 = 95%, 2 = 99%
        """
        cv_idx = {0.10: 0, 0.05: 1, 0.01: 2}.get(significance, 1)
        r = 0
        for i in range(len(self.trace_stat)):
            if self.trace_stat[i] > self.trace_cv[i, cv_idx]:
                r += 1
            else:
                break   # sequential test stops at first non-rejection
        return r

    def get_hedge_ratios(self, r=None):
        """
        Return the first r cointegrating vectors as hedge ratios.
        Normalized so that the first asset's coefficient is 1.0.

        If r is None, use the trace test at 5% to determine rank.
        """
        if r is None:
            r = self.cointegration_rank()

        if r == 0:
            logger.warning("No cointegration found at 5% level")
            return None

        # take the first r eigenvectors
        beta = self.evec[:, :r]   # shape: (k, r)

        # normalize: divide each vector by its first element
        # so the first asset acts as the "numeraire"
        beta_norm = np.zeros_like(beta)
        for j in range(r):
            beta_norm[:, j] = beta[:, j] / beta[0, j]

        return beta_norm

    def print_summary(self):
        k = len(self.tickers)
        print("\n" + "=" * 55)
        print("  JOHANSEN COINTEGRATION TEST RESULTS")
        print("=" * 55)
        print(f"  Assets : {self.tickers}")
        print(f"  Lags   : {self.k_ar_diff}  |  Det. order: {self.det_order}")
        print("-" * 55)
        print(f"  {'H0: rank ≤ r':<15} {'Trace stat':>12} {'CV 95%':>10} {'Reject?':>10}")
        print("-" * 55)
        for i in range(k):
            reject = "YES" if self.trace_stat[i] > self.trace_cv[i, 1] else "no"
            print(f"  r ≤ {i:<11} {self.trace_stat[i]:>12.4f} {self.trace_cv[i, 1]:>10.4f} {reject:>10}")
        r = self.cointegration_rank()
        print("-" * 55)
        print(f"  => Estimated cointegration rank: r = {r}")
        print("=" * 55 + "\n")


def run_johansen(prices_df, det_order=-1, k_ar_diff=2):
    """
    Run the Johansen test on a DataFrame of price levels.

    Args:
        prices_df   : pd.DataFrame, shape (T, k), price levels
        det_order   : deterministic term (-1, 0, 1)
        k_ar_diff   : number of lagged differences in the test VAR

    Returns:
        JohansenResult object
    """
    tickers = list(prices_df.columns)
    data    = prices_df.values

    print(f"[johansen] Running test on {len(tickers)} assets, {len(data)} observations")

    raw = coint_johansen(data, det_order=det_order, k_ar_diff=k_ar_diff)
    result = JohansenResult(raw, tickers, det_order, k_ar_diff)

    return result


def rolling_johansen(prices_df, window=252, step=21, det_order=-1, k_ar_diff=2):
    """
    Run Johansen test over a rolling window.
    Returns a list of (date, rank, beta) tuples for each window.

    step=21 means we re-run roughly every month (not every day,
    that would be too slow and also kind of unnecessary).
    """
    results = []
    dates = prices_df.index
    T = len(dates)

    for start_idx in range(0, T - window, step):
        end_idx    = start_idx + window
        window_df  = prices_df.iloc[start_idx:end_idx]
        window_end = dates[end_idx - 1]

        try:
            res = run_johansen(window_df, det_order=det_order, k_ar_diff=k_ar_diff)
            r   = res.cointegration_rank()
            beta = res.get_hedge_ratios(r=max(r, 1))  # always get at least 1 vector
            results.append({
                'date'  : window_end,
                'rank'  : r,
                'beta'  : beta,
                'result': res
            })
        except Exception as e:
            logger.warning(f"Johansen failed for window ending {window_end}: {e}")
            results.append({'date': window_end, 'rank': 0, 'beta': None, 'result': None})

    print(f"[johansen] Rolling test done — {len(results)} windows")
    return results
