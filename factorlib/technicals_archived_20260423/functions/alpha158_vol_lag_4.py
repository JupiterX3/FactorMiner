import pandas as pd
import numpy as np
from _alpha_ops import *  # noqa: F401,F403

def calculate(data, **kwargs):
    return ref(data['volume'], 4) / (data['volume'] + EPS)
