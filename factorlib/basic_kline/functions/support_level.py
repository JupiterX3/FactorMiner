def calculate(data, period=20, **kwargs):
    import pandas as pd
    # è¿periodåçå±é¨æä½ä½ä¸ºæ¯æä¼°è®?
    return data['low'].rolling(window=period).min().fillna(0.0)

