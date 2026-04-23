import pandas as pd
import numpy as np

def calculate(data, window=1, **kwargs):
    if 'taker_buy_base' not in data.columns:
        return pd.Series(index=data.index, dtype=float)
    vol = data['volume'].replace(0, np.nan)
    ratio = data['taker_buy_base'] / vol
    if window and window > 1:
        ratio = ratio.rolling(window).mean()
    return ratio
