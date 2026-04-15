import pandas as pd
import numpy as np

def calculate(data, period=50, **kwargs):
    return data['close'].rolling(window=period).quantile(0.9)
