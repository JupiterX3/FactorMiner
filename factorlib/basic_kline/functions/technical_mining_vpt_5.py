import pandas as pd
import numpy as np

def calculate(data, **kwargs):
    close = data['close']
    volume = data['volume']
    vpt = ((close.pct_change()) * volume).cumsum()
    return vpt
