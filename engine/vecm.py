# engine/vecm.py
# Vector Error Correction Model
#
# The VECM is basically a VAR in differences but with an error correction term.
# Written out:
#
#   ΔX_t = α * β' * X_{t-1}  +  Σ Γ_i * ΔX_{t-i}  +  ε_t
#
# where:
#   β  = cointegrating vectors (from Johansen) — the long-run equilibrium
#   α  = adjustment speed matrix — how fast each asset corrects back to eq.
#   Γ_i = short-run dynamics (lagged differences)
#
# The spread we trade is  s_t = β' * X_t
# When s_t is far from its mean, prices are "out of equilibrium" and
# the α coefficients tell us they should revert.

import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import VECM
import logging

logger = logging.getLogger(__name__)


class VECMResult:
    """Stores VECM fit results with helper methods for spread calculation."""

    def __init__(self, fitted_model, prices_df, beta_override=None):
        self.model      = fitted_model
        self.tickers    = list(prices_df.columns)
        self.n_assets   = len(self.tickers)

        # If Johansen gave us beta, use that. Otherwise pull from VECM fit.
        if beta_override is not None:
            self.beta = beta_override
        else:
            self.beta = fitted_model.beta   # shape (k, r)

        # alpha: speed of adjustment, shape (k, r)
        self.alpha = fitted_model.alpha

        # half-life of mean reversion for the first cointegrating relation
        self.half_life = self._estimate_half_life()

    def _estimate_half_life(self):
        """
        Rough half-life estimate from the first alpha coefficient.
        If α_1 is the adjustment speed for the spread, then:
            half_life ≈ -log(2) / log(1 + α_1)
        
        This is an approximation — proper way is to simulate the OUP.
        """
        try:
            # use the most negative alpha from the first cointegrating vector
            # (negative means mean-reverting)
            alpha_vals = self.alpha[:, 0]
            most_neg   = alpha_vals[np.argmin(alpha_vals)]
            if most_neg >= 0:
                return np.inf   # not mean reverting
            hl = -np.log(2) / np.log(1 + most_neg)
            return round(hl, 2)
        except Exception:
            return np.nan

    def compute_spread(self, prices_df, use_first_only=True):
        """
        Compute the spread (cointegrating combination) given prices.
        
        spread_t = β' * X_t  (using the first cointegrating vector only)
        
        If use_first_only=False, returns all r spreads.
        """
        X = prices_df.values   # shape (T, k)

        if use_first_only:
            beta_vec = self.beta[:, 0]   # first cointegrating vector, shape (k,)
            spread   = X @ beta_vec
            return pd.Series(spread, index=prices_df.index, name='spread')
        else:
            spread = X @ self.beta   # shape (T, r)
            cols   = [f'spread_{i}' for i in range(spread.shape[1])]
            return pd.DataFrame(spread, index=prices_df.index, columns=cols)

    def compute_zscore(self, prices_df, window=None):
        """
        Z-score of the spread: (spread - mean) / std
        If window is given, use rolling mean/std.
        """
        spread = self.compute_spread(prices_df)
        if window:
            mean = spread.rolling(window).mean()
            std  = spread.rolling(window).std()
        else:
            mean = spread.mean()
            std  = spread.std()
        z = (spread - mean) / std
        return z

    def print_summary(self):
        print("\n" + "=" * 55)
        print("  VECM FIT SUMMARY")
        print("=" * 55)
        print(f"  Assets     : {self.tickers}")
        print(f"  Beta (cointegrating vectors):")
        for i, ticker in enumerate(self.tickers):
            vals = "  ".join([f"{self.beta[i, j]:.4f}" for j in range(self.beta.shape[1])])
            print(f"    {ticker:<6}: {vals}")
        print(f"\n  Alpha (adjustment speeds):")
        for i, ticker in enumerate(self.tickers):
            vals = "  ".join([f"{self.alpha[i, j]:.4f}" for j in range(self.alpha.shape[1])])
            print(f"    {ticker:<6}: {vals}")
        print(f"\n  Approx. half-life of reversion: {self.half_life} days")
        print("=" * 55 + "\n")


def fit_vecm(prices_df, coint_rank, det_order='co', k_ar_diff=2, beta_init=None):
    """
    Fit a VECM to the price data.

    Args:
        prices_df   : pd.DataFrame of price levels (T, k)
        coint_rank  : number of cointegrating vectors (from Johansen)
        det_order   : 'n'=none, 'co'=restricted const, 'ci'=unrestricted const
        k_ar_diff   : lag order
        beta_init   : optional, fix beta from Johansen (improves estimates)

    Returns:
        VECMResult object
    """
    k = prices_df.shape[1]
    if coint_rank < 1:
        raise ValueError("Cointegration rank must be >= 1 to fit VECM")
    if coint_rank >= k:
        coint_rank = k - 1  # rank can't equal k

    print(f"[vecm] Fitting VECM (rank={coint_rank}, lags={k_ar_diff}) on {len(prices_df)} obs")

    model  = VECM(prices_df.values, k_ar_diff=k_ar_diff, coint_rank=coint_rank,
                  deterministic=det_order)
    fitted = model.fit()

    result = VECMResult(fitted, prices_df, beta_override=beta_init)
    return result


def compute_spread_from_beta(prices_df, beta_vec):
    """
    Compute spread directly from a beta vector (no need for fitted VECM).
    Useful in the rolling backtest where we update beta every month.

    spread_t = X_t @ beta_vec
    """
    spread = prices_df.values @ beta_vec
    return pd.Series(spread, index=prices_df.index, name='spread')


def compute_zscore_from_spread(spread, window=None):
    """Z-score a spread series, optionally rolling."""
    if window:
        mean = spread.rolling(window, min_periods=20).mean()
        std  = spread.rolling(window, min_periods=20).std()
    else:
        mean = spread.expanding(min_periods=20).mean()
        std  = spread.expanding(min_periods=20).std()

    z = (spread - mean) / std.replace(0, np.nan)
    return z
