import pandas as pd
import numpy as np
from _alpha_ops import *  # noqa: F401,F403

def calculate(data, **kwargs):
    ohlc = (data['open'] + data['high'] + data['low'] + data['close']) / 4.0
    return ref(ohlc, 1) / (data['close'] + EPS)
