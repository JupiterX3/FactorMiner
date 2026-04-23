import pandas as pd
import numpy as np

def calculate(data, **kwargs):
    close = data['close']
    return (data['high'] - data['low']) / close
