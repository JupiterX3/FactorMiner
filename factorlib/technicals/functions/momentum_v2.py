import pandas as pd
import numpy as np

def calculate(data, period=10, **kwargs):
    close = data['close']
    return close - close.shift(period)
