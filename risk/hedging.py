# risk/hedging.py
# Dynamic hedge ratio re-optimization using CVXPY
#
# Problem: one of our N assets suddenly becomes unavailable
# (trading halt, hard-to-borrow, circuit breaker, whatever).
# We need to re-find hedge ratios for the remaining N-1 assets
# that preserve as much of the original trade as possible.
#
# Formulation (convex optimization):
#
#   minimize   ||w_new - w_old||^2   (stay close to original weights)
#
#   subject to:
#     sum(w_new) = 0              (market neutral — dollar net zero)
#     |w_new_i| ≤ max_weight      (position limits)
#     w_new_i = 0 for all i ∉ available  (can't trade unavailable assets)
#
# This is a standard QP — CVXPY solves it in milliseconds.

import numpy as np
import pandas as pd
import cvxpy as cp
import logging
from config import MAX_POSITION_PCT

logger = logging.getLogger(__name__)


def reoptimize_hedge_ratios(
    beta_full,
    available_mask,
    price_cov=None,
    max_weight=MAX_POSITION_PCT
):
    """
    Re-optimize hedge ratios given that some assets are unavailable.

    Args:
        beta_full       : np.array of shape (k,), original hedge ratios
        available_mask  : np.array of bool, shape (k,), True = asset available
        price_cov       : optional covariance matrix (k x k) for risk-aware opt
        max_weight      : max absolute weight for any asset

    Returns:
        np.array of shape (k,), new hedge ratios (zeros for unavailable assets)
    """
    k = len(beta_full)
    n_avail = available_mask.sum()

    if n_avail == 0:
        logger.error("[risk] No available assets — cannot re-optimize")
        return np.zeros(k)

    if n_avail == k:
        return beta_full.copy()   # nothing changed, return as-is

    unavailable = ~available_mask
    unavail_tickers_idx = np.where(unavailable)[0]
    logger.warning(f"[risk] Assets at indices {unavail_tickers_idx} unavailable, re-optimizing {n_avail}/{k} assets")

    # work in the subspace of available assets
    avail_idx  = np.where(available_mask)[0]
    beta_avail = beta_full[avail_idx]

    w = cp.Variable(n_avail)

    # objective: minimize tracking error to original weights
    objective = cp.Minimize(cp.sum_squares(w - beta_avail))

    constraints = [
        cp.sum(w) == 0.0,          # market neutral
        cp.abs(w) <= max_weight    # position limits
    ]

    # optional: add risk-weighted objective using covariance
    if price_cov is not None:
        cov_sub = price_cov[np.ix_(avail_idx, avail_idx)]
        portfolio_var = cp.quad_form(w, cov_sub)
        # blend tracking error and risk
        objective = cp.Minimize(cp.sum_squares(w - beta_avail) + 0.5 * portfolio_var)

    prob = cp.Problem(objective, constraints)

    try:
        prob.solve(solver=cp.OSQP, warm_start=True)
    except cp.SolverError:
        prob.solve(solver=cp.SCS)

    if prob.status not in ['optimal', 'optimal_inaccurate']:
        logger.warning(f"[risk] Optimization status: {prob.status} — falling back to scaled original")
        # fallback: just zero out unavailable assets and rescale
        w_fallback = beta_full.copy()
        w_fallback[unavailable] = 0.0
        total = np.abs(w_fallback).sum()
        if total > 0:
            w_fallback = w_fallback / total * np.abs(beta_full).sum()
        return w_fallback

    # reconstruct full k-vector (unavailable assets = 0)
    w_new = np.zeros(k)
    w_new[avail_idx] = w.value
    return w_new


def check_asset_availability(prices_row, halt_threshold=0.0):
    """
    Simple halt detection: if a price is 0 or NaN, mark as unavailable.
    In production this would hook into exchange halt feeds.

    Returns a bool mask: True = available, False = halted/unavailable.
    """
    available = np.array([
        (not np.isnan(p)) and (p > halt_threshold)
        for p in prices_row
    ])
    return available


def compute_portfolio_var(weights, cov_matrix):
    """Portfolio variance: w' Σ w"""
    return float(weights @ cov_matrix @ weights)


def compute_dollar_neutral_scale(weights, prices, target_notional):
    """
    Scale weights so that the long leg equals target_notional.
    Returns (shares_to_buy, shares_to_short) for each asset.
    """
    pos_weights = np.maximum(weights, 0)
    neg_weights = np.minimum(weights, 0)

    pos_total = pos_weights.sum()
    neg_total = abs(neg_weights.sum())

    if pos_total == 0 or neg_total == 0:
        return np.zeros_like(weights)

    # scale so longs = target notional
    scale_factor = target_notional / pos_total
    dollar_positions = weights * scale_factor

    # shares = dollar_position / price
    shares = dollar_positions / prices
    return shares


class RiskMonitor:
    """
    Tracks live risk metrics and triggers re-hedging when needed.
    In the main loop, this gets called after every new bar.
    """

    def __init__(self, tickers, beta_vec, capital):
        self.tickers     = tickers
        self.beta        = beta_vec.copy()
        self.capital     = capital
        self.active_beta = beta_vec.copy()   # may differ from beta if re-optimized

        # track recent prices for covariance estimation
        self._price_history = []
        self.n_halted       = 0

    def update(self, prices_row):
        """Call this on every new bar. Returns updated weights."""
        prices_arr = np.array([prices_row.get(t, np.nan) for t in self.tickers])
        self._price_history.append(prices_arr)

        available = check_asset_availability(prices_arr)

        if not available.all():
            newly_halted = (~available).sum()
            if newly_halted != self.n_halted:
                self.n_halted = newly_halted
                print(f"[risk] {newly_halted} asset(s) halted — re-optimizing hedge ratios")

                cov = None
                if len(self._price_history) >= 20:
                    hist = np.array(self._price_history[-60:])
                    returns = np.diff(np.log(np.where(hist > 0, hist, np.nan)), axis=0)
                    # handle nans in cov estimation
                    valid = ~np.isnan(returns).any(axis=0)
                    if valid.sum() > 1:
                        r = returns[:, valid]
                        cov_sub = np.cov(r.T)
                        cov = np.zeros((len(self.tickers), len(self.tickers)))
                        cov[np.ix_(np.where(valid)[0], np.where(valid)[0])] = cov_sub

                self.active_beta = reoptimize_hedge_ratios(
                    self.beta, available, price_cov=cov
                )
        else:
            if self.n_halted > 0:
                print(f"[risk] All assets available again — restoring original hedge ratios")
                self.n_halted    = 0
                self.active_beta = self.beta.copy()

        return self.active_beta
