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
    return pd.Series(np.where(ts_min(delta(close, 1), 4) > 0, delta(close, 1), np.where(ts_max(delta(close, 1), 4) < 0, delta(close, 1), -delta(close, 1))), index=close.index)
