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
    return sign(delta(volume, 1)) * (-1 * delta(close, 1))
