import pandas as pd
import numpy as np

def calculate(data, window=168, **kwargs):
    if 'funding_rate' not in data.columns:
        return pd.Series(index=data.index, dtype=float)
    sign = np.sign(data['funding_rate']).fillna(0)
    return sign.rolling(window).mean()
