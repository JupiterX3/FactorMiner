import pandas as pd
import numpy as np

def calculate(data, **kwargs):
    close = data['close']
    return (close.rolling(window=100).std() / close.rolling(window=100).mean())
