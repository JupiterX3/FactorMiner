def calculate(data, r=25, s=13, **kwargs):
    import pandas as pd
    import numpy as np
    close = data['close']
    m = close.diff()
    abs_m = m.abs()
    ema1 = m.ewm(span=r).mean()
    ema2 = ema1.ewm(span=s).mean()
    ema1_abs = abs_m.ewm(span=r).mean()
    ema2_abs = ema1_abs.ewm(span=s).mean()
    tsi = 100 * ema2 / ema2_abs.replace(0, np.nan)
    return tsi.fillna(0.0)

