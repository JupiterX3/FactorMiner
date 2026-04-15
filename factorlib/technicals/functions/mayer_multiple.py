def calculate(data, period=200, **kwargs):
    import pandas as pd
    import numpy as np
    mm = data['close'] / data['close'].rolling(window=period).mean().replace(0, np.nan)
    return mm.fillna(0.0)

