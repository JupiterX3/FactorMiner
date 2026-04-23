import pandas as pd
import numpy as np

def calculate(data, window=20, **kwargs):
    if 'open_interest' not in data.columns:
        return pd.Series(index=data.index, dtype=float)
    vol = data['volume'].rolling(window).mean().replace(0, np.nan)
    return data['open_interest'] / vol
