import pandas as pd
import numpy as np

def calculate(data, **kwargs):
    close = data['close']
    volume = data['volume']
    return close.pct_change().rolling(window=100).corr(volume.pct_change())
