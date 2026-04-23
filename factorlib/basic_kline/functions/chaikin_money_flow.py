def calculate(data, period=20, **kwargs):
    import pandas as pd
    import numpy as np
    high_low = data['high'] - data['low']
    mfm = ((data['close'] - data['low']) - (data['high'] - data['close'])) / high_low.replace(0, np.nan)
    mfv = mfm * data['volume']
    vol_sum = data['volume'].rolling(window=period).sum().replace(0, np.nan)
    cmf = mfv.rolling(window=period).sum() / vol_sum
    return cmf.fillna(0.0)

