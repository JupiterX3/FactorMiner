import pandas as pd
import numpy as np

def calculate(data, window=48, **kwargs):
    if 'basis' not in data.columns:
        return pd.Series(index=data.index, dtype=float)
    return data['basis'] - data['basis'].rolling(window).mean()
