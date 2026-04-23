import pandas as pd
import numpy as np
from _alpha_ops import *  # noqa: F401,F403

def calculate(data, **kwargs):
    close = data['close']
    open_ = data['open']
    high = data['high']
    low = data['low']
    volume = data['volume']
    ret = returns(close)
    vwap = vwap_proxy(data)
    return pd.Series(np.where(((ref(close, 20) - ref(close, 10)) / 10 - (ref(close, 10) - close) / 10) > 0.25, -1.0, np.where(((ref(close, 20) - ref(close, 10)) / 10 - (ref(close, 10) - close) / 10) < 0, 1.0, -(close - ref(close, 1)))), index=close.index)
