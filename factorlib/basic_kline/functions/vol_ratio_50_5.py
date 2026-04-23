import pandas as pd
import numpy as np

def calculate(data, **kwargs):
    returns = data['close'].pct_change()
    vol_5 = returns.rolling(window=5).std()
    vol_50 = returns.rolling(window=50).std()
    return vol_50 / vol_5
