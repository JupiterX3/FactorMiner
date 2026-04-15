import pandas as pd
import numpy as np

def calculate(data, period=20, std_dev=2, **kwargs):
    sma = data['close'].rolling(window=period).mean()
    return sma
