"""Strategy rules for the Gold vs Dow RSI backtest.

Each rule receives `row` (today's bar) and `prev` (yesterday's bar). Both have:
    gold_open, gold_high, gold_low, gold_close, gold_rsi
    dow_open,  dow_high,  dow_low,  dow_close,  dow_rsi

Signals are evaluated on the daily close and filled at the NEXT day's open.
Return True to act. Set a rule to None to disable it.
"""

RSI_LENGTH = 14

# Which market the trades are taken in: "gold" or "dow".
TRADE_INSTRUMENT = "gold"

# ---- Rules: to be filled in with the user's conditions -----------------------
ENTRY_LONG = None   # e.g. lambda row, prev: prev.gold_rsi < 30 <= row.gold_rsi
EXIT_LONG = None    # e.g. lambda row, prev: row.gold_rsi > 70
ENTRY_SHORT = None
EXIT_SHORT = None

# Optional protective exits, as % of entry price (None = off). Checked intraday
# against high/low; if both hit on the same bar the stop is assumed first.
STOP_LOSS_PCT = None
TAKE_PROFIT_PCT = None
MAX_HOLD_DAYS = None

YEARS = 10
