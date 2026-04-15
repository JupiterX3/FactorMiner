def calculate(data, period=12, method='pct', **kwargs):
    import pandas as pd
    import numpy as np
    close = data['close']
    if method == 'log':
        ratio = close / close.shift(period)
        return np.log(ratio.where(ratio > 0, np.nan)).fillna(0.0)
    else:
        return close.pct_change(periods=period).fillna(0.0)

