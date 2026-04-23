def calculate(data, window=50, **kwargs):
    import pandas as pd
    # ç®åï¼ä¸æ²¿=é«ç¹æ»å¨çº¿æ§åå½æçï¼ ä¸æ²¿=ä½ç¹æ»å¨çº¿æ§åå½æçï¼æ¶æ(åå·ä¸ç»å¯¹å¼åå°?è§ä¸ºä¸è§å½¢åæ?
    high = data['high']
    low = data['low']
    def slope(s):
        import numpy as np
        x = np.arange(len(s))
        if len(s) < 2:
            return 0.0
        A = np.vstack([x, np.ones(len(x))]).T
        m, _ = np.linalg.lstsq(A, s.values, rcond=None)[0]
        return m
    up_slope = high.rolling(window=window).apply(slope, raw=False)
    down_slope = low.rolling(window=window).apply(slope, raw=False)
    score = (up_slope.abs() + down_slope.abs())
    return (-score).fillna(0.0)

