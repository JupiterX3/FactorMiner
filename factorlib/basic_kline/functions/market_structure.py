def calculate(data, swing_lookback=5, **kwargs):
    import pandas as pd
    high = data['high']
    low = data['low']
    
    swing_high = high[(high.shift(1) < high) & (high.shift(2) < high.shift(1))].rolling(swing_lookback).max()
    
    swing_low = low[(low.shift(1) > low) & (low.shift(2) > low.shift(1))].rolling(swing_lookback).min()
    
    dir_series = (swing_high.ffill() > swing_high.ffill().shift(1)).astype(int) \
                 - (swing_low.ffill() < swing_low.ffill().shift(1)).astype(int)
    
    return dir_series.fillna(0.0)
