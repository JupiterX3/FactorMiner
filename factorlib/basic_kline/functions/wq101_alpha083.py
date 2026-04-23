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
    return (ref((high - low) / (ts_sum(close, 5) / 5 + EPS), 2) * volume) / (((high - low) / (ts_sum(close, 5) / 5 + EPS)) / (vwap - close + EPS) + EPS)
