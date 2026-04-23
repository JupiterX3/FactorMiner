def calculate(data, shift=26, **kwargs):
    # 馃毃 淇鏈潵鍑芥暟闂锛?
    # 鍘熸潵鐨勯敊璇細浣跨敤 shift(-int(shift)) 鑾峰彇鏈潵鏁版嵁
    # 淇鍚庯細浣跨敤鍘嗗彶鏁版嵁锛屾ā鎷烠hikou Span鐨勫欢杩熸樉绀烘晥鏋?
    
    # Chikou Span: 鏀剁洏浠峰悜鍓嶇Щ浣嶏紙鍘嗗彶鏁版嵁锛?
    # 娉ㄦ剰锛氳繖鏄负浜嗘ā鎷烠hikou Span鐨勬樉绀烘晥鏋滐紝瀹為檯浣跨敤鏃堕渶瑕佸欢杩焥hift涓懆鏈?
    return data['close'].shift(int(shift)).ffill().fillna(0.0)

