"""
Question 1: Do pullbacks in TSMC (TSM) predict later drawdowns in NVIDIA (NVDA)?

Reads the combined dataset, keeps only trading days, calculates
daily returns, builds the engineered features, and prints the key results.
"""
import numpy as np
import pandas as pd
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data" / "processed"


def leftover(stock, market):
    """The part of a stock's daily return that the market doesn't explain."""
    slope, intercept = np.polyfit(market, stock, 1)
    return stock - (slope * market + intercept)


def forward_return(prices, h):
    """% change from each day's close to the close h trading days later."""
    return (prices.shift(-h) / prices - 1) * 100


# Load data and keep only days the market was open
df = pd.read_csv(DATA / "nvidia_gfn_combined.csv", parse_dates=["date"]).sort_values("date")
td = df[df.is_trading_day == 1].copy().reset_index(drop=True)
print(f"Kept {len(td)} trading days out of {len(df)} calendar days")

# Daily returns on trading days only
for t in ["nvda", "tsm", "smh_etf", "qqq"]:
    td[f"{t}_ret"] = td[f"{t}_close"].pct_change() * 100

# Does TSM move before NVDA?
print("\nCorrelation of NVDA with TSM shifted by k trading days (positive = TSM first):")
for k in range(-5, 6):
    r = td["nvda_ret"].corr(td["tsm_ret"].shift(k))
    print(f"  {k:+d}: {r:.3f}")

# Is it TSMC, or shared market movement?
rets = td[["nvda_ret", "tsm_ret", "smh_etf_ret", "qqq_ret"]].dropna()
rets.columns = ["NVDA", "TSM", "SMH", "QQQ"]
print("\nNVDA vs TSM correlation:")
print(f"  raw: {rets['NVDA'].corr(rets['TSM']):.2f}")
print(f"  after removing QQQ: {leftover(rets['NVDA'], rets['QQQ']).corr(leftover(rets['TSM'], rets['QQQ'])):.2f}")
print(f"  after removing SMH: {leftover(rets['NVDA'], rets['SMH']).corr(leftover(rets['TSM'], rets['SMH'])):.2f}")

# What does NVDA do after a big TSM drop?
drop_days = td.index[td["tsm_ret"] <= -3]
print(f"\nDays TSM fell 3% or more: {len(drop_days)}")
for h in [1, 5, 10]:
    nvda_fwd = forward_return(td["nvda_close"], h)
    smh_fwd = forward_return(td["smh_etf_close"], h)
    print(f"  {h:2d} days later:  NVDA {nvda_fwd[drop_days].mean():+.2f}% (normal {nvda_fwd.mean():+.2f}%)"
          f"   SMH {smh_fwd[drop_days].mean():+.2f}% (normal {smh_fwd.mean():+.2f}%)")

# Engineered features
td["nvda_tsm_corr_60d"] = td["nvda_ret"].rolling(60).corr(td["tsm_ret"])
td["nvda_drawdown"] = (td["nvda_close"] / td["nvda_close"].cummax() - 1) * 100
td["tsm_drawdown"] = (td["tsm_close"] / td["tsm_close"].cummax() - 1) * 100
td["tsm_big_drop"] = (td["tsm_ret"] <= -3).astype(int)

# Preview the engineered features
print("\nEngineered features (last 5 trading days):")
print(td[["date", "nvda_tsm_corr_60d", "nvda_drawdown", "tsm_drawdown", "tsm_big_drop"]].tail())