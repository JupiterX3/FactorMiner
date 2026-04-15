def calculate(data, fast=3, slow=10, **kwargs):
    import pandas as pd
    import numpy as np
    mfm = ((data['close'] - data['low']) - (data['high'] - data['close'])) / (data['high'] - data['low']).replace(0, np.nan)
    mfv = mfm * data['volume']
    adl = mfv.cumsum()
    cho = adl.ewm(span=int(fast)).mean() - adl.ewm(span=int(slow)).mean()
    return cho.fillna(0.0)


