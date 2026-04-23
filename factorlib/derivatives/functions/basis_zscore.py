import pandas as pd
import numpy as np

def calculate(data, window=96, **kwargs):
    if 'basis' not in data.columns:
        return pd.Series(index=data.index, dtype=float)
    x = data['basis']
    return (x - x.rolling(window).mean()) / x.rolling(window).std()
