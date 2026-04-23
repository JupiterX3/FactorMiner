import pandas as pd
import numpy as np

def calculate(data, window=48, **kwargs):
    if 'taker_buy_base' not in data.columns:
        return pd.Series(index=data.index, dtype=float)
    vol = data['volume'].replace(0, np.nan)
    imbalance = data['taker_buy_base'] / vol - 0.5
    mu = imbalance.rolling(window).mean()
    sd = imbalance.rolling(window).std()
    return (imbalance - mu) / sd
