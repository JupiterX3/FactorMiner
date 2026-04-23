def calculate(data, fast=5, slow=20, **kwargs):
    import pandas as pd
    import numpy as np
    vol = data['volume']
    slow_ma = vol.rolling(slow).mean().replace(0, np.nan)
    result = vol.rolling(fast).mean() / slow_ma
    return result.fillna(0.0)

