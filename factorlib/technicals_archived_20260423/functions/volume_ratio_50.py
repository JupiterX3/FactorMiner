import pandas as pd
import numpy as np

def calculate(data, **kwargs):
    volume = data['volume']
    vol_ma = volume.rolling(window=50).mean()
    return volume / vol_ma
