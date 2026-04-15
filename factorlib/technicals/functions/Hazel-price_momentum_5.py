import pandas as pd
import numpy as np

def calculate(data: pd.DataFrame, window: int = 5, **kwargs) -> pd.Series:
    # ç¸å¯¹ 5 æ ¹K çä»·æ ¼å¨éã?
    if data is None or len(data) == 0 or "close" not in data.columns:
        return pd.Series(dtype=float)
    close = pd.to_numeric(data["close"], errors="coerce")
    base = close.shift(window)
    return (close - base) / base
