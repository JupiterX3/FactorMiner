
def calculate(data, fast_period=12, slow_period=26, signal_period=9):
    """
    璁＄畻MACD鎸囨爣
    
    鍙傛暟:
        - fast_period: 蹇嚎鍛ㄦ湡
        - slow_period: 鎱㈢嚎鍛ㄦ湡
        - signal_period: 淇″彿绾垮懆鏈?
    """
    # 1. 璁＄畻蹇嚎鍜屾參绾?
    fast_ema = data['close'].ewm(span=fast_period).mean()
    slow_ema = data['close'].ewm(span=slow_period).mean()
    
    # 2. 璁＄畻MACD绾?
    macd_line = fast_ema - slow_ema
    
    # 3. 璁＄畻淇″彿绾?
    signal_line = macd_line.ewm(span=signal_period).mean()
    
    # 4. 璁＄畻MACD鏌辩姸鍥?
    histogram = macd_line - signal_line
    
    return macd_line  # 杩斿洖MACD绾?
