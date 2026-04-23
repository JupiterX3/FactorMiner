import pandas as pd
import numpy as np

def calculate(data, **kwargs):
    low = data['low']
    return low.rolling(window=5).min()
