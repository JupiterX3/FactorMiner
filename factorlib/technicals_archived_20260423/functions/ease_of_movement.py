def calculate(data, volume_period=14, **kwargs):
    import pandas as pd
    import numpy as np
    distance_moved = ((data['high'] + data['low'])/2 - (data['high'].shift(1) + data['low'].shift(1))/2)
    high_low = data['high'] - data['low']
    box_ratio = data['volume'] / high_low.replace(0, np.nan)
    emv = distance_moved / box_ratio.replace(0, np.nan)
    emv = pd.to_numeric(emv, errors='coerce').fillna(0.0)
    return emv.rolling(window=volume_period).mean().fillna(0.0)

