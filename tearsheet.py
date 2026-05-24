# tearsheet.py
# Generate visual performance tearsheet + architecture diagram
#
# Run after run_backtest.py (needs equity_curve.csv to exist)
# Or run standalone — it will fetch data and run backtest first.
#
# Usage:  python tearsheet.py

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   # non-interactive backend for saving to file
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

from config import ASSETS, START_DATE, END_DATE, INITIAL_CAPITAL


# ---- colour scheme ----
DARK_BG   = '#0d1117'
ACCENT    = '#58a6ff'
GREEN     = '#3fb950'
RED       = '#f85149'
TEXT      = '#e6edf3'
MUTED     = '#8b949e'


def load_or_run_backtest():
    """Load pre-computed results or run backtest fresh."""
    if os.path.exists('equity_curve.csv'):
        equity = pd.read_csv('equity_curve.csv', index_col=0, parse_dates=True).squeeze()
        trades = pd.read_csv('trades.csv', parse_dates=['date']) if os.path.exists('trades.csv') else pd.DataFrame()
        print("[tearsheet] Loaded pre-computed equity curve")
        return equity, trades
    else:
        print("[tearsheet] No pre-computed results found, running backtest...")
        from run_backtest import main as run_bt
        result = run_bt()
        return result.equity_curve, result.trades


def compute_rolling_metrics(equity, window=63):
    """Rolling Sharpe and rolling drawdown for the equity chart."""
    returns = equity.pct_change().dropna()

    rolling_sharpe = (
        returns.rolling(window).mean() / returns.rolling(window).std()
    ) * np.sqrt(252)

    roll_max  = equity.cummax()
    drawdown  = (equity - roll_max) / roll_max

    return returns, rolling_sharpe, drawdown


def monthly_returns_table(returns):
    """Pivot table of monthly returns (rows=year, cols=month)."""
    monthly = returns.resample('ME').apply(lambda x: (1 + x).prod() - 1)
    monthly.index = monthly.index.to_period('M')
    tbl = monthly.groupby([monthly.index.year, monthly.index.month]).first().unstack()
    tbl.columns = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    return tbl * 100


def plot_tearsheet(equity, trades):
    """Main performance tearsheet — 6 panels."""
    returns, roll_sharpe, drawdown = compute_rolling_metrics(equity)

    fig = plt.figure(figsize=(16, 20), facecolor=DARK_BG)
    gs  = gridspec.GridSpec(4, 2, figure=fig, hspace=0.45, wspace=0.3)

    _style_axes = lambda ax: (
        ax.set_facecolor(DARK_BG),
        ax.tick_params(colors=MUTED, labelsize=9),
        [spine.set_edgecolor('#30363d') for spine in ax.spines.values()]
    )

    # ---- panel 1: equity curve (top full width) ----
    ax1 = fig.add_subplot(gs[0, :])
    _style_axes(ax1)
    ax1.plot(equity.index, equity / 1e6, color=ACCENT, linewidth=1.4, label='Portfolio equity')
    ax1.axhline(INITIAL_CAPITAL / 1e6, color=MUTED, linestyle='--', linewidth=0.8, alpha=0.5, label='Starting capital')
    ax1.set_title('Equity Curve', color=TEXT, fontsize=13, pad=10)
    ax1.set_ylabel('Portfolio Value ($M)', color=MUTED, fontsize=9)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:.1f}M'))
    ax1.legend(fontsize=8, facecolor='#161b22', edgecolor='#30363d', labelcolor=TEXT)

    # shade drawdown periods
    dd_neg = drawdown[drawdown < -0.02]
    if len(dd_neg) > 0:
        ax1.fill_between(equity.index, equity.min() / 1e6, equity / 1e6,
                         where=drawdown < -0.02, alpha=0.15, color=RED, label='DD > 2%')

    # ---- panel 2: drawdown ----
    ax2 = fig.add_subplot(gs[1, :])
    _style_axes(ax2)
    ax2.fill_between(drawdown.index, drawdown * 100, 0, color=RED, alpha=0.65)
    ax2.set_title('Drawdown (%)', color=TEXT, fontsize=13, pad=10)
    ax2.set_ylabel('Drawdown %', color=MUTED, fontsize=9)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}%'))

    # ---- panel 3: rolling sharpe ----
    ax3 = fig.add_subplot(gs[2, 0])
    _style_axes(ax3)
    ax3.plot(roll_sharpe.index, roll_sharpe, color=GREEN, linewidth=1.2)
    ax3.axhline(0, color=MUTED, linewidth=0.8, linestyle='--')
    ax3.set_title('Rolling 63-day Sharpe', color=TEXT, fontsize=11, pad=8)
    ax3.set_ylabel('Sharpe', color=MUTED, fontsize=9)

    # ---- panel 4: return distribution ----
    ax4 = fig.add_subplot(gs[2, 1])
    _style_axes(ax4)
    ret_pct = returns * 100
    ax4.hist(ret_pct, bins=60, color=ACCENT, alpha=0.75, edgecolor='none')
    ax4.axvline(0, color=MUTED, linewidth=1.0, linestyle='--')
    ax4.axvline(ret_pct.mean(), color=GREEN, linewidth=1.2, label=f'Mean={ret_pct.mean():.3f}%')
    ax4.set_title('Daily Return Distribution', color=TEXT, fontsize=11, pad=8)
    ax4.set_xlabel('Daily Return (%)', color=MUTED, fontsize=9)
    ax4.legend(fontsize=8, facecolor='#161b22', edgecolor='#30363d', labelcolor=TEXT)

    # ---- panel 5: monthly returns heatmap ----
    ax5 = fig.add_subplot(gs[3, :])
    _style_axes(ax5)
    try:
        monthly_tbl = monthly_returns_table(returns)
        # clip to available years
        sns.heatmap(
            monthly_tbl,
            ax=ax5,
            cmap='RdYlGn',
            center=0,
            annot=True,
            fmt='.1f',
            linewidths=0.3,
            linecolor='#30363d',
            cbar_kws={'shrink': 0.6},
            annot_kws={'size': 8}
        )
        ax5.set_title('Monthly Returns Heatmap (%)', color=TEXT, fontsize=11, pad=8)
        ax5.set_ylabel('Year', color=MUTED, fontsize=9)
        ax5.tick_params(colors=MUTED, labelsize=8)
    except Exception as e:
        ax5.text(0.5, 0.5, f'Monthly heatmap unavailable\n({e})',
                 ha='center', va='center', color=MUTED, transform=ax5.transAxes)

    # ---- stats box ----
    ann_ret   = (returns.mean() * 252) * 100
    ann_vol   = (returns.std() * np.sqrt(252)) * 100
    sharpe    = ann_ret / ann_vol if ann_vol > 0 else 0
    max_dd    = drawdown.min() * 100
    total_ret = (equity.iloc[-1] / equity.iloc[0] - 1) * 100

    stats_text = (
        f"Total Return: {total_ret:.1f}%\n"
        f"Ann. Return : {ann_ret:.1f}%\n"
        f"Ann. Vol    : {ann_vol:.1f}%\n"
        f"Sharpe      : {sharpe:.3f}\n"
        f"Max DD      : {max_dd:.1f}%"
    )
    fig.text(0.01, 0.98, stats_text, transform=fig.transFigure,
             fontsize=9, color=TEXT, fontfamily='monospace',
             verticalalignment='top',
             bbox=dict(facecolor='#161b22', edgecolor='#30363d', pad=6))

    fig.suptitle(
        'Cross-Asset Stat Arb Engine — Performance Tear Sheet\n'
        f'Assets: {", ".join(ASSETS)}   |   {START_DATE} → {END_DATE}',
        color=TEXT, fontsize=14, y=1.01
    )

    plt.savefig('tearsheet.png', dpi=150, bbox_inches='tight', facecolor=DARK_BG)
    print("[tearsheet] Saved tearsheet.png")
    plt.close()


def plot_architecture_diagram():
    """
    Generate the system architecture diagram.
    Shows data flow: APIs → ZMQ → Math Engine → Risk → Portfolio.
    """
    fig, ax = plt.subplots(figsize=(14, 8), facecolor=DARK_BG)
    ax.set_facecolor(DARK_BG)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')

    def box(ax, x, y, w, h, label, sublabel='', color=ACCENT, fontsize=9):
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle='round,pad=0.1',
            facecolor=color + '22',
            edgecolor=color,
            linewidth=1.5
        )
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2 + (0.15 if sublabel else 0), label,
                ha='center', va='center', color=TEXT, fontsize=fontsize, fontweight='bold')
        if sublabel:
            ax.text(x + w/2, y + h/2 - 0.25, sublabel,
                    ha='center', va='center', color=MUTED, fontsize=7)

    def arrow(ax, x1, y1, x2, y2, label=''):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=MUTED, lw=1.4))
        if label:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mx, my + 0.15, label, ha='center', color=MUTED, fontsize=7)

    # ---- layer 1: data sources ----
    box(ax, 0.3, 5.8, 1.8, 0.9,  'Yahoo Finance',  'daily OHLCV',     color='#f0883e')
    box(ax, 2.3, 5.8, 1.8, 0.9,  'Alpaca / Poly',  'minute ETF bars', color='#f0883e')
    box(ax, 4.3, 5.8, 1.8, 0.9,  'Binance / Bybit', 'HFT crypto ticks',color='#f0883e')

    # ---- layer 2: ZMQ/Redis broker ----
    box(ax, 1.5, 3.9, 5.0, 1.1,  'ZeroMQ PUB/SUB Broker',
        'price_feed channel | non-blocking async', color=ACCENT)

    # ---- layer 3: math engine ----
    box(ax, 0.2, 2.1, 2.3, 1.4,  'Johansen Engine',  'cointegration rank\nβ eigenvectors', color=GREEN)
    box(ax, 2.8, 2.1, 2.3, 1.4,  'VECM Engine',      'α adj. speed\nspread z-score',       color=GREEN)
    box(ax, 5.4, 2.1, 2.3, 1.4,  'Signal Engine',    'entry/exit\nz-score thresholds',     color=GREEN)

    # ---- layer 4: risk ----
    box(ax, 8.2, 2.1, 3.3, 1.4,  'Dynamic Risk Engine',
        'CVXPY hedge re-opt\nN-1 asset fallback', color='#bc8cff')

    # ---- layer 5: execution / backtest ----
    box(ax, 1.5, 0.3, 4.0, 1.1,  'Portfolio Allocator',
        'dollar-neutral sizing | position limits', color='#f0883e')
    box(ax, 6.3, 0.3, 5.3, 1.1,  'Microstructure Backtester',
        'taker fees + borrow + sqrt market impact', color='#f0883e')

    # ---- rolling validation box ----
    box(ax, 8.2, 5.2, 3.3, 1.3,  'Rolling Johansen',
        'rank check every 21 days\nauto-liquidate on rank drop', color=RED)

    # arrows
    for xs in [1.2, 3.2, 5.2]:
        arrow(ax, xs, 5.8, 2.8, 5.0)   # data → broker

    arrow(ax, 4.0, 3.9, 1.3, 3.5)   # broker → johansen
    arrow(ax, 4.0, 3.9, 3.9, 3.5)   # broker → vecm
    arrow(ax, 4.0, 3.9, 6.5, 3.5)   # broker → signals

    arrow(ax, 1.3, 2.1, 3.9, 3.5)   # johansen → vecm
    arrow(ax, 3.9, 2.1, 6.5, 2.1)   # vecm → signals
    arrow(ax, 6.5, 2.1, 8.2, 2.8)   # signals → risk
    arrow(ax, 6.5, 2.1, 5.5, 1.4)   # signals → portfolio
    arrow(ax, 8.2, 2.8, 7.5, 1.4)   # risk → portfolio
    arrow(ax, 7.5, 1.4, 8.5, 1.4)   # portfolio → backtester
    arrow(ax, 9.8, 5.2, 9.8, 3.5)   # rolling val → risk

    ax.set_title(
        'System Architecture — Cross-Asset Stat Arb Engine\n'
        'Praneshwar Kannan Kommiya | B.Tech ME, IIT Roorkee',
        color=TEXT, fontsize=12, pad=10
    )

    plt.tight_layout()
    plt.savefig('architecture.png', dpi=150, bbox_inches='tight', facecolor=DARK_BG)
    print("[tearsheet] Saved architecture.png")
    plt.close()


if __name__ == '__main__':
    equity, trades = load_or_run_backtest()
    plot_tearsheet(equity, trades)
    plot_architecture_diagram()
    print("\nDone. Check tearsheet.png and architecture.png")
