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
    return pd.Series(np.where(adv(volume, 20) < volume, -ts_rank(delta(close, 7).abs(), 60) * sign(delta(close, 7)), -1.0), index=close.index)
