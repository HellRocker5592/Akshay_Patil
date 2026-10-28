"""Gold vs Dow Jones RSI backtest, run as two separate setups.

  futures: Micro Gold futures (MGC) + Dow Jones futures (YM)
  spot:    Spot gold (XAU/USD)      + Dow Jones Industrial Average index

Data is read from backtest/data/<name>.csv (Date,Open,High,Low,Close) when
present, otherwise downloaded from Yahoo Finance.

Usage:  python backtest/run_backtest.py [futures|spot|all]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import strategy as S

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
OUT_DIR = HERE / "results"

# Dollar value of a 1.00 price move per contract. Spot setups reuse the futures
# multipliers so the $ figures are directly comparable across both setups.
SETUPS = {
    "futures": {
        "gold": {"csv": "MGC", "tickers": ["MGC=F", "GC=F"], "label": "Micro Gold futures (MGC)", "usd_per_point": 10.0},
        "dow": {"csv": "YM", "tickers": ["YM=F"], "label": "Dow futures (YM)", "usd_per_point": 5.0},
        "commission": 2.50,  # per side, per contract
        "slippage_ticks": 1,
        "tick": {"gold": 0.10, "dow": 1.0},
    },
    "spot": {
        "gold": {"csv": "XAUUSD", "tickers": ["XAUUSD=X"], "label": "Spot gold (XAU/USD)", "usd_per_point": 10.0},
        "dow": {"csv": "DJI", "tickers": ["^DJI"], "label": "Dow Jones index (^DJI)", "usd_per_point": 5.0},
        "commission": 0.0,
        "slippage_ticks": 0,
        "tick": {"gold": 0.10, "dow": 1.0},
    },
}


def load_prices(spec, start):
    csv = DATA_DIR / f"{spec['csv']}.csv"
    if csv.exists():
        df = pd.read_csv(csv, parse_dates=["Date"], index_col="Date")
        df.columns = [c.capitalize() for c in df.columns]
        return df[["Open", "High", "Low", "Close"]].sort_index().loc[start:].dropna()
    import yfinance as yf
    for ticker in spec["tickers"]:
        df = yf.download(ticker, start=start, interval="1d", auto_adjust=False, progress=False)
        if df is not None and not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            print(f"  {spec['label']}: downloaded {ticker}, {len(df)} bars")
            return df[["Open", "High", "Low", "Close"]].dropna()
    raise RuntimeError(f"No data for {spec['label']}: add {csv} or allow Yahoo Finance access")


def rsi(close, length):
    """Wilder's RSI, as plotted by TradingView and most charting platforms."""
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    return 100 - 100 / (1 + gain / loss)


def build_frame(setup, start):
    parts = []
    for name in ("gold", "dow"):
        px = load_prices(setup[name], start)
        px.columns = [f"{name}_{c.lower()}" for c in px.columns]
        px[f"{name}_rsi"] = rsi(px[f"{name}_close"], S.RSI_LENGTH)
        parts.append(px)
    # Only days where both markets traded, so every rule sees both RSIs.
    return pd.concat(parts, axis=1, join="inner").dropna()


def backtest(df, setup):
    inst = S.TRADE_INSTRUMENT
    usd_pt = setup[inst]["usd_per_point"]
    slip = setup["slippage_ticks"] * setup["tick"][inst]
    o, h, l = df[f"{inst}_open"].values, df[f"{inst}_high"].values, df[f"{inst}_low"].values
    rows = list(df.itertuples())
    trades, pos, pending = [], None, None

    def close_trade(i, price, reason):
        side = pos["side"]
        exit_px = price - slip * side
        points = (exit_px - pos["entry"]) * side
        trades.append({
            "side": "long" if side == 1 else "short",
            "entry_date": pos["date"].date(), "entry_price": round(pos["entry"], 2),
            "exit_date": rows[i].Index.date(), "exit_price": round(exit_px, 2),
            "days_held": (rows[i].Index - pos["date"]).days,
            "points": round(points, 2),
            "pct": round(points / pos["entry"] * 100, 3),
            "usd": round(points * usd_pt - 2 * setup["commission"], 2),
            "exit_reason": reason,
        })

    for i in range(1, len(rows)):
        # 1) Fill yesterday's signal at today's open.
        if pending == "exit" and pos:
            close_trade(i, o[i], "signal")
            pos = None
        elif pending in (1, -1) and pos is None:
            pos = {"side": pending, "entry": o[i] + slip * pending, "date": rows[i].Index, "bar": i}
        pending = None

        # 2) Intraday stop / target.
        if pos:
            side, entry = pos["side"], pos["entry"]
            stop = entry * (1 - side * S.STOP_LOSS_PCT / 100) if S.STOP_LOSS_PCT else None
            target = entry * (1 + side * S.TAKE_PROFIT_PCT / 100) if S.TAKE_PROFIT_PCT else None
            hit_stop = stop is not None and (l[i] <= stop if side == 1 else h[i] >= stop)
            hit_tgt = target is not None and (h[i] >= target if side == 1 else l[i] <= target)
            if hit_stop:
                # Gap through the stop fills at the open, not the stop price.
                fill = min(o[i], stop) if side == 1 else max(o[i], stop)
                close_trade(i, fill, "stop_loss"); pos = None
            elif hit_tgt:
                fill = max(o[i], target) if side == 1 else min(o[i], target)
                close_trade(i, fill, "take_profit"); pos = None

        # 3) Close-of-day signals, executed next open.
        row, prev = rows[i], rows[i - 1]
        if pos:
            if S.MAX_HOLD_DAYS and i - pos["bar"] + 1 >= S.MAX_HOLD_DAYS:
                pending = "exit"
            elif pos["side"] == 1 and S.EXIT_LONG and S.EXIT_LONG(row, prev):
                pending = "exit"
            elif pos["side"] == -1 and S.EXIT_SHORT and S.EXIT_SHORT(row, prev):
                pending = "exit"
        elif S.ENTRY_LONG and S.ENTRY_LONG(row, prev):
            pending = 1
        elif S.ENTRY_SHORT and S.ENTRY_SHORT(row, prev):
            pending = -1

    if pos:  # mark open trade to the last close
        close_trade(len(rows) - 1, df[f"{inst}_close"].iloc[-1], "end_of_data")
    return pd.DataFrame(trades)


def summarize(trades):
    if trades.empty:
        return {"trades": 0}
    wins, losses = trades[trades.usd > 0], trades[trades.usd <= 0]
    equity = trades.usd.cumsum()
    gross_loss = -losses.usd.sum()
    return {
        "trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate_pct": round(len(wins) / len(trades) * 100, 2),
        "avg_win_pct": round(wins.pct.mean(), 3) if len(wins) else 0,
        "avg_win_usd": round(wins.usd.mean(), 2) if len(wins) else 0,
        "avg_loss_pct": round(losses.pct.mean(), 3) if len(losses) else 0,
        "avg_loss_usd": round(losses.usd.mean(), 2) if len(losses) else 0,
        "largest_win_usd": round(trades.usd.max(), 2),
        "largest_loss_usd": round(trades.usd.min(), 2),
        "net_pnl_usd": round(trades.usd.sum(), 2),
        "profit_factor": round(wins.usd.sum() / gross_loss, 2) if gross_loss else np.inf,
        "max_drawdown_usd": round((equity - equity.cummax().clip(lower=0)).min(), 2),
        "avg_days_held": round(trades.days_held.mean(), 1),
    }


def plot(df, trades, setup, name):
    fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(15, 9), sharex=True, gridspec_kw={"height_ratios": [3, 1.3]})
    ax1.plot(df.index, df.gold_close, color="#C9A227", lw=1.1, label=setup["gold"]["label"])
    ax1.set_ylabel("Gold ($/oz)", color="#C9A227")
    ax2 = ax1.twinx()
    ax2.plot(df.index, df.dow_close, color="#1F5FA8", lw=1.1, label=setup["dow"]["label"])
    ax2.set_ylabel("Dow Jones (points)", color="#1F5FA8")
    if not trades.empty:
        ax = ax1 if S.TRADE_INSTRUMENT == "gold" else ax2
        ent = pd.to_datetime(trades.entry_date)
        ext = pd.to_datetime(trades.exit_date)
        ax.scatter(ent, trades.entry_price, marker="^", color="green", s=28, zorder=5, label="Entry")
        ax.scatter(ext, trades.exit_price, marker="v", color="red", s=28, zorder=5, label="Exit")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left")
    ax1.set_title(f"Gold vs Dow Jones with RSI({S.RSI_LENGTH}) — {name}")
    ax1.grid(alpha=0.3)

    ax3.plot(df.index, df.gold_rsi, color="#C9A227", lw=0.9, label="Gold RSI")
    ax3.plot(df.index, df.dow_rsi, color="#1F5FA8", lw=0.9, label="Dow RSI")
    for lvl in (30, 70):
        ax3.axhline(lvl, color="grey", ls="--", lw=0.8)
    ax3.set_ylim(0, 100)
    ax3.set_ylabel("RSI")
    ax3.legend(loc="upper left")
    ax3.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{name}_chart.png", dpi=130)
    plt.close(fig)


def run(name):
    setup = SETUPS[name]
    cutoff = pd.Timestamp.today().normalize() - pd.DateOffset(years=S.YEARS)
    print(f"\n=== {name.upper()} ===")
    # Load extra history before the cutoff so RSI is warmed up on day one.
    df = build_frame(setup, str((cutoff - pd.DateOffset(days=90)).date()))
    df = df[df.index >= cutoff]
    trades = backtest(df, setup)
    stats = summarize(trades)
    stats = {"period": f"{df.index[0].date()} to {df.index[-1].date()}", "traded": setup[S.TRADE_INSTRUMENT]["label"], **stats}
    OUT_DIR.mkdir(exist_ok=True)
    trades.to_csv(OUT_DIR / f"{name}_trades.csv", index=False)
    pd.Series(stats).to_csv(OUT_DIR / f"{name}_summary.csv", header=["value"])
    plot(df, trades, setup, name)
    for k, v in stats.items():
        print(f"  {k:<18} {v}")
    return stats


if __name__ == "__main__":
    if not any([S.ENTRY_LONG, S.ENTRY_SHORT]):
        sys.exit("No entry rule set: fill in ENTRY_LONG / ENTRY_SHORT in backtest/strategy.py")
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    for n in (SETUPS if which == "all" else [which]):
        run(n)
