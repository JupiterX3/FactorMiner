import pandas as pd
import numpy as np

def calculate(data, **kwargs):
    high = data['high']
    return high.rolling(window=20).max()
