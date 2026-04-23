import pandas as pd
import numpy as np

def calculate(data, **kwargs):
    volume = data['volume']
    return volume.rolling(window=50).apply(lambda x: x.autocorr())
