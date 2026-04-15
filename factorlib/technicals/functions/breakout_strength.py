def calculate(data, lookback=20, **kwargs):
    import pandas as pd
    import numpy as np
    close = data['close']
    recent_high = close.rolling(window=lookback).max()
    recent_low = close.rolling(window=lookback).min()
    strength = (close - recent_low) / (recent_high - recent_low).replace(0, np.nan)
    return strength.fillna(0.0)

