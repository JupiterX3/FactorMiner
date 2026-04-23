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
    return pd.Series(np.where(ts_sum(close, 8) / 8 + ts_std(close, 8) < ts_sum(close, 2) / 2, -1.0, np.where(ts_sum(close, 2) / 2 < ts_sum(close, 8) / 8 - ts_std(close, 8), 1.0, np.where(volume / (adv(volume, 20) + EPS) >= 1.0, 1.0, -1.0))), index=close.index)
