import pandas as pd
import numpy as np

def calculate(data, **kwargs):
    volume = data['volume']
    return volume.rolling(window=20).rank(pct=True)
