# backtest/engine.py
# Microstructure-aware backtesting engine
#
# Cost model:
#   1. Taker fee: flat bps on gross traded notional
#   2. Short borrow: daily accrual on short dollar exposure = (annual_rate / 252)
#   3. Market impact: Almgren-Chriss square-root model
#       impact_cost = k * sigma_daily * sqrt(trade_dollar / daily_ADV) * trade_dollar
#
# The backtester runs in two phases:
#   Phase 1 (warmup): run rolling Johansen to estimate beta, no trading
#   Phase 2 (live):   generate signals, execute, track PnL
#
# Cointegration breakdown protocol:
#   If rolling Johansen rank drops to 0, signal = 0 and we liquidate.

import numpy as np
import pandas as pd
import logging
from config import (
    LOOKBACK_WINDOW, VECM_LAG_ORDER, JOHANSEN_DET_ORDER,
    ENTRY_Z_SCORE, EXIT_Z_SCORE,
    TAKER_FEE_BPS, ANNUAL_BORROW_RATE, MARKET_IMPACT_K,
    INITIAL_CAPITAL, REVALIDATION_FREQ_DAYS, MIN_COINTEGRATION_RANK
)
from engine.johansen import run_johansen
from engine.vecm import compute_spread_from_beta, compute_zscore_from_spread
from engine.signals import generate_signals

logger = logging.getLogger(__name__)


class BacktestResult:
    """Stores and summarizes backtest output."""

    def __init__(self, equity_curve, trades_df, daily_pnl, cost_breakdown):
        self.equity_curve  = equity_curve
        self.trades        = trades_df
        self.daily_pnl     = daily_pnl
        self.cost_breakdown = cost_breakdown

        self._compute_metrics()

    def _compute_metrics(self):
        returns = self.equity_curve.pct_change().dropna()

        self.total_return   = (self.equity_curve.iloc[-1] / self.equity_curve.iloc[0] - 1) * 100
        self.annualized_ret = (1 + self.total_return / 100) ** (252 / len(returns)) - 1
        self.volatility     = returns.std() * np.sqrt(252)
        self.sharpe         = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0

        # max drawdown
        roll_max    = self.equity_curve.cummax()
        drawdown    = (self.equity_curve - roll_max) / roll_max
        self.max_dd = drawdown.min() * 100

        # Calmar
        self.calmar = (self.annualized_ret * 100) / abs(self.max_dd) if self.max_dd != 0 else np.inf

        n_trades = len(self.trades[self.trades['type'] == 'open']) if len(self.trades) > 0 else 0
        self.n_trades = n_trades

    def print_tearsheet(self):
        print("\n" + "=" * 60)
        print("  BACKTEST PERFORMANCE TEAR SHEET")
        print("=" * 60)
        print(f"  Total Return         : {self.total_return:.2f}%")
        print(f"  Annualized Return    : {self.annualized_ret * 100:.2f}%")
        print(f"  Annualized Vol       : {self.volatility * 100:.2f}%")
        print(f"  Sharpe Ratio         : {self.sharpe:.3f}")
        print(f"  Max Drawdown         : {self.max_dd:.2f}%")
        print(f"  Calmar Ratio         : {self.calmar:.3f}")
        print(f"  Number of Trades     : {self.n_trades}")
        print("-" * 60)
        print(f"  Cost Breakdown:")
        for k, v in self.cost_breakdown.items():
            print(f"    {k:<25}: ${v:,.2f}")
        print("=" * 60 + "\n")


class MicrostructureBacktester:
    """
    Main backtesting engine.
    
    Steps:
     1. Load prices and (optionally) volume for ADV
     2. Rolling Johansen to get time-varying beta
     3. Compute spread z-score → signals
     4. For each bar: compute position changes, apply cost model, update equity
     5. Check for cointegration breakdown every REVALIDATION_FREQ_DAYS
    """

    def __init__(self, prices_df, volume_df=None, capital=INITIAL_CAPITAL):
        self.prices   = prices_df.copy()
        self.volume   = volume_df
        self.capital  = capital
        self.tickers  = list(prices_df.columns)
        self.k        = len(self.tickers)

        # will be populated during run
        self.equity_curve  = None
        self.spread_series = None
        self.zscore_series = None
        self.signals_series = None
        self.hedge_betas   = {}   # date -> beta vector

    def _estimate_daily_vol(self, prices_window):
        """Daily volatility estimate for each asset (used in impact model)."""
        log_ret = np.log(prices_window / prices_window.shift(1)).dropna()
        return log_ret.std()

    def _estimate_adv(self, date):
        """
        Estimate Average Daily Volume in dollar terms.
        If volume data is available, use it. Otherwise use price * assumed ADV fraction.
        """
        if self.volume is not None and date in self.volume.index:
            vol_window = self.volume[self.volume.index <= date].tail(20)
            price_window = self.prices[self.prices.index <= date].tail(20)
            adv = (price_window * vol_window).mean()
        else:
            # rough proxy: assume 0.5% of market cap trades daily
            # market cap ≈ price * some_shares, we just use price * scale
            price = self.prices.loc[date] if date in self.prices.index else self.prices.iloc[-1]
            adv = price * 500_000   # very rough, ~500k shares * price
        return adv

    def _taker_fee(self, trade_notional):
        """Flat taker fee on traded dollar notional."""
        return trade_notional * (TAKER_FEE_BPS / 10_000)

    def _borrow_cost(self, short_positions_dollars, n_days=1):
        """Daily accrual of borrow cost on short positions."""
        daily_rate = ANNUAL_BORROW_RATE / 252
        return abs(short_positions_dollars) * daily_rate * n_days

    def _market_impact(self, trade_dollars, asset_vols, adv):
        """
        Square-root market impact model:
            impact = k * sigma * sqrt(|trade| / ADV) * |trade|

        This is the Almgren-Chriss model simplified to one factor.
        The cost grows super-linearly with trade size relative to ADV.
        """
        total_impact = 0.0
        for i, t in enumerate(self.tickers):
            if adv[t] <= 0:
                continue
            q     = abs(trade_dollars[t])
            sigma = asset_vols[t] if t in asset_vols else 0.01
            impact = MARKET_IMPACT_K * sigma * np.sqrt(q / adv[t]) * q
            total_impact += impact
        return total_impact

    def run(self):
        """
        Main backtest loop.
        
        Returns:
            BacktestResult object
        """
        print(f"\n[backtest] Starting backtest on {len(self.prices)} days × {self.k} assets")
        print(f"[backtest] Warmup period: {LOOKBACK_WINDOW} days")

        T         = len(self.prices)
        dates     = self.prices.index
        equity    = self.capital
        cash      = self.capital
        positions = {t: 0.0 for t in self.tickers}   # dollar positions

        equity_history    = []
        trade_log         = []
        daily_pnl_history = []

        total_fees     = 0.0
        total_borrow   = 0.0
        total_impact   = 0.0

        current_beta    = None
        current_signal  = 0
        last_revalidate = 0
        coint_broken    = False

        # ----- phase 1: warmup — collect enough data to run Johansen -----
        for t_idx in range(LOOKBACK_WINDOW, T):
            date = dates[t_idx]

            # ---- re-run Johansen every REVALIDATION_FREQ_DAYS ----
            days_since_reval = t_idx - last_revalidate
            should_revalidate = (days_since_reval >= REVALIDATION_FREQ_DAYS) or (current_beta is None)

            if should_revalidate:
                window_prices = self.prices.iloc[t_idx - LOOKBACK_WINDOW : t_idx]
                try:
                    joh = run_johansen(window_prices, det_order=JOHANSEN_DET_ORDER, k_ar_diff=VECM_LAG_ORDER)
                    rank = joh.cointegration_rank()

                    if rank < MIN_COINTEGRATION_RANK:
                        if not coint_broken:
                            print(f"[backtest] {date.date()}: Cointegration BROKE (rank={rank}) — liquidating")
                            coint_broken = True
                            current_signal = 0
                            # liquidate all positions
                            prices_now = self.prices.iloc[t_idx]
                            for t in self.tickers:
                                if positions[t] != 0:
                                    trade_notional = abs(positions[t])
                                    fee = self._taker_fee(trade_notional)
                                    total_fees += fee
                                    positions[t] = 0.0
                    else:
                        if coint_broken:
                            print(f"[backtest] {date.date()}: Cointegration restored (rank={rank})")
                        coint_broken = False
                        current_beta = joh.get_hedge_ratios(r=1)[:, 0]
                        self.hedge_betas[date] = current_beta.copy()

                    last_revalidate = t_idx

                except Exception as e:
                    logger.warning(f"[backtest] Johansen failed at {date}: {e}")

            if current_beta is None or coint_broken:
                equity_history.append(equity)
                daily_pnl_history.append(0.0)
                continue

            # ---- compute z-score on a recent window ----
            lookback_prices = self.prices.iloc[max(0, t_idx - LOOKBACK_WINDOW) : t_idx + 1]
            spread = compute_spread_from_beta(lookback_prices, current_beta)
            zscore = compute_zscore_from_spread(spread, window=LOOKBACK_WINDOW)

            current_z = zscore.iloc[-1]
            if np.isnan(current_z):
                equity_history.append(equity)
                daily_pnl_history.append(0.0)
                continue

            # ---- signal update ----
            prev_signal = current_signal
            if current_signal == 0:
                if current_z > ENTRY_Z_SCORE:
                    current_signal = -1
                elif current_z < -ENTRY_Z_SCORE:
                    current_signal = 1
            elif current_signal == 1:
                if current_z > -EXIT_Z_SCORE:
                    current_signal = 0
            elif current_signal == -1:
                if current_z < EXIT_Z_SCORE:
                    current_signal = 0

            prices_now = self.prices.iloc[t_idx]

            # ---- compute target positions ----
            target_positions = self._compute_target_positions(
                current_signal, current_beta, prices_now, equity
            )

            # ---- compute trades (changes in position) ----
            trade_dollars = {}
            for t in self.tickers:
                trade_dollars[t] = target_positions.get(t, 0.0) - positions.get(t, 0.0)

            gross_traded = sum(abs(v) for v in trade_dollars.values())

            if gross_traded > 1.0:   # ignore dust trades
                # estimate vol and ADV for cost model
                vol_window = self.prices.iloc[max(0, t_idx - 20) : t_idx]
                asset_vols = self._estimate_daily_vol(vol_window).to_dict()
                adv        = self._estimate_adv(date).to_dict()

                fee    = self._taker_fee(gross_traded)
                impact = self._market_impact(trade_dollars, asset_vols, adv)

                total_fees   += fee
                total_impact += impact

                trade_type = 'open' if prev_signal == 0 else 'close' if current_signal == 0 else 'roll'
                trade_log.append({
                    'date'         : date,
                    'signal'       : current_signal,
                    'type'         : trade_type,
                    'gross_notional': gross_traded,
                    'fee'          : fee,
                    'impact'       : impact,
                    'z_score'      : round(current_z, 4)
                })

            # ---- update positions ----
            for t in self.tickers:
                positions[t] = target_positions.get(t, 0.0)

            # ---- daily PnL from price changes ----
            if t_idx > 0:
                prev_prices = self.prices.iloc[t_idx - 1]
                pnl = sum(
                    positions[t] / prev_prices[t] * (prices_now[t] - prev_prices[t])
                    for t in self.tickers
                    if prev_prices[t] > 0
                )
            else:
                pnl = 0.0

            # ---- borrow cost on shorts ----
            short_exposure = sum(abs(v) for v in positions.values() if v < 0)
            borrow = self._borrow_cost(short_exposure)
            total_borrow += borrow

            daily_pnl = pnl - borrow
            equity   += daily_pnl

            equity_history.append(equity)
            daily_pnl_history.append(daily_pnl)

        # ---- assemble results ----
        result_dates = dates[LOOKBACK_WINDOW:]
        equity_series = pd.Series(equity_history, index=result_dates, name='equity')
        pnl_series    = pd.Series(daily_pnl_history, index=result_dates, name='daily_pnl')
        trades_df     = pd.DataFrame(trade_log) if trade_log else pd.DataFrame()

        cost_breakdown = {
            'Taker fees'      : round(total_fees, 2),
            'Short borrow'    : round(total_borrow, 2),
            'Market impact'   : round(total_impact, 2),
            'Total costs'     : round(total_fees + total_borrow + total_impact, 2),
        }

        result = BacktestResult(equity_series, trades_df, pnl_series, cost_breakdown)
        return result

    def _compute_target_positions(self, signal, beta_vec, prices_now, equity):
        """
        Dollar target positions for each asset based on signal and beta.

        We allocate a fixed fraction of equity to the trade.
        Beta weights determine how much goes to each asset.
        """
        if signal == 0:
            return {t: 0.0 for t in self.tickers}

        # notional: use 80% of equity split equally long/short
        trade_notional = equity * 0.4   # 40% long, 40% short

        pos_beta = np.maximum(beta_vec, 0)
        neg_beta = np.minimum(beta_vec, 0)

        pos_sum = pos_beta.sum()
        neg_sum = abs(neg_beta.sum())

        if pos_sum == 0 or neg_sum == 0:
            return {t: 0.0 for t in self.tickers}

        pos_beta_n = pos_beta / pos_sum
        neg_beta_n = neg_beta / neg_sum

        target = {}
        for i, t in enumerate(self.tickers):
            if signal == 1:   # long spread
                target[t] = pos_beta_n[i] * trade_notional - neg_beta_n[i] * trade_notional
            else:             # short spread
                target[t] = neg_beta_n[i] * trade_notional - pos_beta_n[i] * trade_notional

        return target


def capacity_analysis(backtest_result, prices_df, beta_vec, aum_range=None):
    """
    Estimate how alpha decays as AUM grows.
    
    At higher AUM, market impact grows faster than alpha (sqrt model)
    because sqrt(Q/ADV) increases and Q increases simultaneously.
    So impact ∝ Q^1.5 while PnL ∝ Q → eventually impact > PnL.

    Returns a DataFrame of (AUM, net_sharpe) pairs.
    """
    if aum_range is None:
        aum_range = [1e6, 5e6, 10e6, 25e6, 50e6, 100e6, 250e6]

    base_sharpe   = backtest_result.sharpe
    base_capital  = INITIAL_CAPITAL
    base_impact   = backtest_result.cost_breakdown['Market impact']
    base_pnl      = (backtest_result.equity_curve.iloc[-1] - base_capital)

    rows = []
    for aum in aum_range:
        scale = aum / base_capital
        # PnL scales linearly with AUM (alpha is fixed %)
        scaled_pnl    = base_pnl * scale
        # impact scales as Q^1.5 / ADV^0.5, so roughly scale^1.5
        scaled_impact = base_impact * (scale ** 1.5)
        net_pnl       = scaled_pnl - scaled_impact * scale   # double-count intentional to show decay

        net_return = net_pnl / aum
        # very rough estimate: assume same vol structure
        est_sharpe = base_sharpe * (net_pnl / max(scaled_pnl, 1))
        rows.append({
            'AUM ($M)'     : aum / 1e6,
            'Net PnL ($K)' : round(net_pnl / 1e3, 1),
            'Est. Sharpe'  : round(max(est_sharpe, 0), 3),
            'Impact/PnL %' : round(scaled_impact * scale / max(abs(scaled_pnl), 1) * 100, 1)
        })

    df = pd.DataFrame(rows)
    return df
