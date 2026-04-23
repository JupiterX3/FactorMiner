def calculate(data, period=20, **kwargs):
    import pandas as pd
    import numpy as np
    
    ret = data['close'].pct_change()
    vol = ret.shift(1).rolling(window=period).std()
    var = ret / vol.replace(0, np.nan)
    return var.fillna(0.0)

