# config.py
# central config - keeping all magic numbers here so its easy to change stuff
# learned the hard way not to hardcode these everywhere

# ---- assets ----
# using oil/energy sector because they're heavily cointegrated
# XLE is the energy ETF which acts as the "basket" for the others
ASSETS = ['XOM', 'CVX', 'BP', 'COP', 'SLB', 'XLE']

# ---- data range ----
START_DATE = '2019-01-01'
END_DATE   = '2024-12-31'

# ---- VECM / Johansen params ----
LOOKBACK_WINDOW   = 252       # ~1 trading year for rolling estimation
VECM_LAG_ORDER    = 2         # lags in VECM (tested 1-5, 2 minimised AIC)
JOHANSEN_DET_ORDER = -1       # -1=no constant, 0=restricted constant, 1=trend
                               # energy prices probably trend so -1 is conservative

# ---- signal thresholds ----
ENTRY_Z_SCORE = 2.0
EXIT_Z_SCORE  = 0.3           # exit close to mean rather than exactly at 0

# ---- transaction costs ----
TAKER_FEE_BPS       = 5       # 5 basis points (0.05%), assume taker always
ANNUAL_BORROW_RATE  = 0.005   # 50 bps/yr on short positions
MARKET_IMPACT_K     = 0.1     # k in: impact = k * sigma * sqrt(Q / ADV)

# assumed average daily volume (fraction of daily market cap that trades)
ADV_PARTICIPATION   = 0.005   # we assume 0.5% of market cap per day

# ---- ZeroMQ ----
ZMQ_PUB_PORT     = 5555
ZMQ_SUB_ADDRESS  = 'tcp://localhost:5555'
ZMQ_PUB_ADDRESS  = 'tcp://*:5555'

# ---- Redis (optional, fallback to ZMQ if unavailable) ----
REDIS_HOST    = 'localhost'
REDIS_PORT    = 6379
REDIS_CHANNEL = 'price_feed'

# ---- rolling cointegration validation ----
REVALIDATION_FREQ_DAYS = 21   # re-run Johansen every ~month
MIN_COINTEGRATION_RANK = 1    # liquidate if rank drops below this

# ---- backtester ----
INITIAL_CAPITAL   = 1_000_000   # 1M USD starting notional
MAX_POSITION_PCT  = 0.25        # max 25% of capital in any single leg
