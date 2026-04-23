import pandas as pd
import numpy as np

def calculate(data, window=240, **kwargs):
    if 'funding_rate' not in data.columns:
        return pd.Series(index=data.index, dtype=float)
    x = data['funding_rate']
    return (x - x.rolling(window).mean()) / x.rolling(window).std()
