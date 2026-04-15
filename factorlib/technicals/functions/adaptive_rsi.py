def calculate(data, period=14, method='ema', **kwargs):
    import pandas as pd
    import numpy as np
    close = data['close']
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    if method == 'wilders':
        gain_avg = gain.ewm(alpha=1/period, adjust=False).mean()
        loss_avg = loss.ewm(alpha=1/period, adjust=False).mean()
    else:
        gain_avg = gain.ewm(span=period, adjust=False).mean()
        loss_avg = loss.ewm(span=period, adjust=False).mean()
    rs = gain_avg / loss_avg.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(0.0)

