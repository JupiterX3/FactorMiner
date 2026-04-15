import pandas as pd
import numpy as np

def calculate(data, **kwargs):
    close = data['close']
    return close / close.shift(20) - 1
