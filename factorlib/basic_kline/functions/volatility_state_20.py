import pandas as pd
import numpy as np

def calculate(data, **kwargs):
    returns = data['close'].pct_change()
    vol = returns.rolling(window=20).std()
    vol_mean = vol.mean()
    return (vol > vol_mean).astype(int)
