import pandas as pd
import numpy as np

def calculate(data, **kwargs):
    high = data['high']
    low = data['low']
    close = data['close']
    period = 50

    up_move = high.diff()
    down_move = -low.diff()
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=high.index)

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.rolling(window=period).mean()
    minus_di = 100 * minus_dm.rolling(window=period).mean() / atr
    return minus_di.fillna(0.0)
