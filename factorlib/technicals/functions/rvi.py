def calculate(data, period=10, **kwargs):
    import pandas as pd
    import numpy as np
    close = data['close']
    std = close.rolling(window=period).std()
    rvi = (close - close.rolling(window=period).mean()) / std.replace(0, np.nan)
    return rvi.fillna(0.0)

