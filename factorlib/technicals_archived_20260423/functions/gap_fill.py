def calculate(data, lookback=20, **kwargs):
    import pandas as pd
    # ç®åçï¼è¥å½åæ¶çä»·åå°åä¸æ ¹æ¶çä»·ä¸å½æ ¹å¼çä»·ä¹é´ï¼ç¼ºå£åºåï¼ï¼åè§ä¸ºåè¡¥
    prev_close = data['close'].shift(1)
    open_ = data['open']
    close = data['close']
    filled = ((close - prev_close) * (open_ - prev_close) <= 0).astype(float)
    # å¯éï¼è¦æ±å?lookback ååºç°è¿è·³ç©ºï¼æ­¤å¤ç®åç´æ¥è¾åºï¼
    return filled.fillna(0.0)

