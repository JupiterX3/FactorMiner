def calculate(data, **kwargs):
    import pandas as pd
    import numpy as np
    high = data['high']
    low = data['low']
    close = data['close']
    volume = data['volume']
    mf_multiplier = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)
    mf_volume = mf_multiplier * volume
    ad = mf_volume.cumsum()
    return ad.fillna(0.0)

