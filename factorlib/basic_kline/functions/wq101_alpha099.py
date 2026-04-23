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
    return pd.Series(np.where(ts_corr(ts_sum((high + low) / 2, 20), ts_sum(adv(volume, 60), 20), 9) < ts_corr(low, volume, 6), -1.0, 1.0), index=close.index)
