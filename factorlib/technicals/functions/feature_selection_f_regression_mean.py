import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

FACTOR_NAME = "feature_selection_f_regression_mean"

def calculate(data: pd.DataFrame, **kwargs) -> pd.Series:
    try:
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if data is None or len(data) == 0:
            return pd.Series(dtype=float)
        
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            print(f"Missing columns: {missing_cols}")
            return pd.Series(index=data.index, dtype=float)
        
        model_paths = [
            Path(__file__).parent.parent / "models" / f"{FACTOR_NAME}.pkl",
            Path(__file__).parent.parent.parent / "minactors" / "models" / f"{FACTOR_NAME}.pkl",
            Path(__file__).parent.parent.parent / "technicals" / "models" / f"{FACTOR_NAME}.pkl",
            Path.cwd() / "factorlib" / "minactors" / "models" / f"{FACTOR_NAME}.pkl",
        ]
        
        artifact_file = None
        for path in model_paths:
            if path.exists():
                artifact_file = path
                break
        
        if artifact_file is None:
            print(f"Model file not found: {FACTOR_NAME}.pkl")
            return pd.Series(index=data.index, dtype=float)
        
        with open(artifact_file, 'rb') as f:
            artifact = pickle.load(f)
        
        model = artifact.get("model")
        feature_columns = artifact.get("feature_columns", [])
        scaler = artifact.get("scaler")
        
        if model is None:
            print("Model file corrupted: cannot load model")
            return pd.Series(index=data.index, dtype=float)
        
        features = _build_features(data)
        
        missing = [c for c in feature_columns if c not in features.columns]
        if missing:
            print(f"Missing feature columns: {missing}")
            for c in missing:
                features[c] = np.nan
        
        X = features[feature_columns]
        
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.ffill().bfill()
        
        if scaler is not None:
            X_scaled = scaler.transform(X)
        else:
            X_scaled = X.values
        
        y_pred = model.predict(X_scaled)
        return pd.Series(y_pred, index=data.index)
        
    except Exception as e:
        print(f"Error calculating {FACTOR_NAME}: {e}")
        import traceback
        traceback.print_exc()
    return pd.Series(index=data.index, dtype=float)

def _build_features(data: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(index=data.index)
    
    features['price_momentum_1'] = data['close'].pct_change(1)
    features['price_momentum_5'] = data['close'].pct_change(5)
    features['price_momentum_10'] = data['close'].pct_change(10)
    
    features['volume_ratio'] = data['volume'] / data['volume'].rolling(20).mean()
    features['volume_momentum'] = data['volume'].pct_change(5)
    
    features['volatility_10'] = data['close'].rolling(10).std() / data['close'].rolling(10).mean()
    features['volatility_20'] = data['close'].rolling(20).std() / data['close'].rolling(20).mean()
    
    features['trend_5'] = (data['close'] - data['close'].shift(5)) / data['close'].shift(5)
    features['trend_10'] = (data['close'] - data['close'].shift(10)) / data['close'].shift(10)
    
    features['price_position_20'] = (data['close'] - data['low'].rolling(20).min()) / (data['high'].rolling(20).max() - data['low'].rolling(20).min())
    
    features['ma_5'] = data['close'] / data['close'].rolling(5).mean() - 1
    features['ma_10'] = data['close'] / data['close'].rolling(10).mean() - 1
    features['ma_20'] = data['close'] / data['close'].rolling(20).mean() - 1
    
    return features
