import pandas as pd
import numpy as np

def calculate(data: pd.DataFrame, **kwargs) -> pd.Series:
    # è¿å æä½ä»· åãè¥åç¼ºå¤±åè¿å NaN åºåã?
    if data is None or len(data) == 0:
        return pd.Series(dtype=float)
    if "low" not in data.columns:
        return pd.Series(np.nan, index=data.index)
    series = pd.to_numeric(data["low"], errors="coerce")
    return series
