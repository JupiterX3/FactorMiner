import pandas as pd
import numpy as np

def calculate(data, period=12, window=96, **kwargs):
    if 'open_interest' not in data.columns:
        return pd.Series(index=data.index, dtype=float)
    chg = data['open_interest'].pct_change(period)
    mu = chg.rolling(window).mean()
    sd = chg.rolling(window).std()
    return (chg - mu) / sd
