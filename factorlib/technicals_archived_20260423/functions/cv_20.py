import pandas as pd
import numpy as np

def calculate(data, **kwargs):
    close = data['close']
    return (close.rolling(window=20).std() / close.rolling(window=20).mean())
