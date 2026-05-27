# Distributed Cross-Asset Statistical Arbitrage Engine

**Praneshwar Kannan Kommiya | B.Tech Mechanical Engineering (3rd Year) | IIT Roorkee**

---

This is a project I built to understand how institutional quant desks extract alpha from *multi-asset* cointegrated relationships rather than just simple pairs. The system runs a full Johansen cointegration test across 6 energy-sector assets, models the mean reversion using a Vector Error Correction Model (VECM), and wraps it all in a distributed pub/sub architecture with a microstructure-aware backtester.

---

## What the project does

At a high level:

1. Downloads daily price data for `XOM, CVX, BP, COP, SLB, XLE` (oil sector — strong cointegration)
2. Runs the **Johansen Test** to find how many independent cointegrating relationships exist
3. Fits a **VECM** to extract hedge ratios (β) and mean-reversion speeds (α)
4. Generates entry/exit signals from the **z-score of the spread**
5. Runs a **microstructure backtester** that accounts for taker fees, short borrow costs, and square-root market impact
6. Has a **dynamic re-hedging engine** (CVXPY) that re-optimizes weights if one asset becomes unavailable (halt, hard-to-borrow, etc.)
7. Streams simulated live data through a **ZeroMQ pub/sub** pipeline

---

## The Econometrics

### Why Johansen over Engle-Granger?

Engle-Granger only handles two-asset pairs and assumes a single cointegrating vector. For a 6-asset basket, we need a matrix approach. The Johansen test works with the Π matrix from the VECM representation:

```
ΔX_t = Π X_{t-1} + Σ Γ_i ΔX_{t-i} + ε_t
```

The rank of Π tells us how many cointegrating relationships exist. If `rank(Π) = r`, we can decompose it as `Π = αβ'`, where:
- **β** (k×r): the cointegrating vectors — these are the hedge ratios. The spread `s_t = β' X_t` is stationary.
- **α** (k×r): adjustment speed matrix — tells you how quickly each asset corrects back to equilibrium when the spread deviates.

The eigenvectors of the Π matrix (via reduced-rank regression) give us β. The test statistic for rank selection is the **Johansen trace statistic**:

```
λ_trace(r) = -T * Σ ln(1 - λ_i)   for i = r+1, ..., k
```

I use the sequential procedure: test H0: rank ≤ 0, then rank ≤ 1, etc., stopping at the first non-rejection.

### VECM and the spread

Once we have β, the spread is just a dot product:

```
spread_t = β₀ * P_XOM_t + β₁ * P_CVX_t + ... + β₅ * P_XLE_t
```

The mean-reversion half-life comes from the largest-magnitude negative α coefficient. Rough estimate:

```
half_life ≈ -log(2) / log(1 + α_min)
```

Typical values for energy stocks: 15–40 trading days. Anything above ~60 days is marginal for a strategy.

### Signal generation

Simple z-score threshold:
- `z > +2.0` → short the spread (expect reversion down)
- `z < -2.0` → long the spread (expect reversion up)
- `|z| < 0.3` → close position (mean-reverted)

The position in each asset is proportional to its β weight, scaled to a fixed notional.

---

## The Cost Model

This is where most papers get lazy and I tried not to. Three cost components:

### 1. Taker fee
Flat 5 bps on gross traded notional. Assumes we're always crossing the spread (conservative).

### 2. Short borrow cost
For all short positions, a daily accrual:
```
daily_borrow = |short_exposure| × (annual_rate / 252)
```
Using 50 bps/year (realistic for large-cap US equities; can be much higher for hard-to-borrow names).

### 3. Market impact (Almgren-Chriss square-root model)
```
impact = k × σ_daily × sqrt(Q / ADV) × Q
```

- `k = 0.1` (empirical constant, around 0.1 for equities per Almgren 2005)
- `Q` = dollar value of the trade
- `ADV` = average daily dollar volume
- The `sqrt(Q/ADV)` term is the key — impact grows faster than linearly with trade size

This model has a crucial implication for capacity: PnL scales ~linearly with AUM, but impact scales as `Q^1.5`. So at some AUM, impact starts eating all your alpha. The `capacity_analysis()` function in `backtest/engine.py` estimates where this crossover happens.

---

## Dynamic Hedging (CVXPY)

If one of the 6 assets becomes unavailable mid-trade (halt, circuit breaker, hard-to-borrow notice), we can't just leave the position unhedged. The `risk/hedging.py` module solves:

```
minimize   ||w_new - w_old||²
subject to:
    Σ w_i = 0           (market neutral)
    |w_i| ≤ 0.25        (position limits)
    w_i = 0  ∀ i ∉ available
```

This is a standard quadratic program that CVXPY solves in <10ms. We stay as close as possible to the original hedge ratios while maintaining market neutrality with the remaining assets.

---

## Distributed Architecture

```
  Yahoo Finance / Alpaca / Binance
           │
           ▼
  [ZeroMQ PUB socket: tcp://*:5555]
           │  (ticker-filtered topics)
           ▼
  [ZeroMQ SUB socket: subscriber.py]
           │
           ▼
  PriceBuffer (accumulates per timestamp)
           │  (emits complete bars)
           ▼
  LiveArbitrageEngine.on_bar()
    ├── RiskMonitor (check halts → CVXPY re-opt)
    ├── compute spread z-score
    └── update signal + log trades
```

The key design choice: the pub/sub is **topic-filtered** (each ticker is a separate ZMQ topic). This means the subscriber only processes topics it subscribed to, and the math engine only wakes up when a complete bar arrives. No polling, no blocking.

For backtesting, the `InMemoryBroker` bypasses ZMQ entirely and calls the same `on_bar()` callback directly. So the signal logic is shared between live and backtest modes — no separate implementation.

---

## Cointegration Breakdown Protocol

Cointegrating relationships don't last forever. The Johansen test is re-run every 21 trading days (roughly monthly) on a rolling 252-day window. If the rank drops to 0 (no cointegration), the engine:

1. Stops generating new signals
2. Closes all open positions at market
3. Waits for the next validation cycle

This prevents the strategy from trading a spread that has permanently broken down (e.g., a merger, acquisition, or structural market change).

---

## Project Structure

```
.
├── config.py               # all parameters in one place
├── data/
│   └── fetcher.py          # yfinance download + cleaning
├── engine/
│   ├── johansen.py         # Johansen test wrapper + rolling test
│   ├── vecm.py             # VECM fit + spread/z-score computation
│   └── signals.py          # z-score → entry/exit signals
├── broker/
│   ├── publisher.py        # ZMQ PUB: replay or live price feed
│   └── subscriber.py       # ZMQ SUB: bar assembly + callback
├── risk/
│   └── hedging.py          # CVXPY re-optimization for N-1 assets
├── backtest/
│   └── engine.py           # full backtester with cost model
├── main.py                 # async orchestrator (ZMQ mode)
├── run_backtest.py         # historical backtest runner
├── tearsheet.py            # performance plots + architecture diagram
└── requirements.txt
```

---

## Setup & Running

**Python 3.9+ required**

```bash
# clone and install
git clone https://github.com/palaemon-codes/cross-asset-stat-arb.git
cd cross-asset-stat-arb
pip install -r requirements.txt

# run the backtest (takes 3-5 mins for rolling Johansen)
python run_backtest.py

# generate tearsheet and architecture diagram
python tearsheet.py

# run the live replay via ZMQ (needs the backtest to run first for warmup)
python main.py --mode replay
```

**Note on Redis**: The codebase references Redis in config but ZeroMQ is the default broker. If you have Redis running locally (`redis-server`), you can swap the broker layer. ZMQ works without any external server — it's just sockets.

---

## Performance (rough numbers from my local run)

| Metric | Value |
|---|---|
| Total Return (2019–2024) | ~34% |
| Annualized Return | ~6.0% |
| Sharpe Ratio | ~1.1–1.4 |
| Max Drawdown | ~8% |
| Avg Hold Period | 12–20 days |

The Sharpe looks decent but the *capacity* is the real limitation. At ~$25M AUM the market impact starts meaningfully degrading net alpha. This is a typical issue for mean-reversion strategies — the alpha itself is small (spread reverts slowly), so position sizes can't be too large.

---

## What I'd improve

- **Tick-level data**: Daily bars are too coarse. The half-life of reversion is 15–30 days, which means entry/exit timing is imprecise. With minute bars from Alpaca the z-score updates faster.
- **Multiple cointegrating vectors**: I'm only using the first eigenvector. If rank r=2, there's a second independent spread to trade — more alpha.
- **Regime detection**: The rolling Johansen catches cointegration breakdowns but doesn't distinguish between temporary noise and structural breaks. A Hidden Markov Model on the spread might help.
- **Real borrow data**: My borrow cost assumption (50 bps/yr) is probably too low for some names. Actual borrow rates from prime brokers can be 5–10% for popular short targets.
- **Live execution**: The `LiveTickPublisher` class in `broker/publisher.py` is a stub. Hooking up Alpaca's WebSocket API is the next step.

---

## References

- Johansen (1988), *Statistical Analysis of Cointegration Vectors*
- Engle & Granger (1987), *Co-integration and Error Correction*
- Almgren & Chriss (2001), *Optimal Execution of Portfolio Transactions*
- Avellaneda & Lee (2010), *Statistical Arbitrage in the US Equities Market*
- statsmodels VECM docs and source

---

*This is a personal learning project, not financial advice. Don't trade this live without understanding the risks.*
