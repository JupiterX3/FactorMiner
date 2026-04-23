#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子优化器模块
提供因子参数优化、因子组合优化和自动因子选择功能
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.covariance import LedoitWolf
import warnings
warnings.filterwarnings('ignore')

class FactorOptimizer:
    """因子优化器主类"""
    
    def __init__(self, data=None, returns=None):
        self.data = data
        self.returns = returns
        self.scaler = StandardScaler()
        self.best_params = {}
        self.optimization_history = []
        
    def set_data(self, data, returns):
        """设置数据和收益率"""
        self.data = data
        self.returns = returns
        
    def optimize_factor_parameters(self, factor_func, param_grid, metric='ic'):
        """
        优化因子参数
        
        Args:
            factor_func: 因子计算函数
            param_grid: 参数网格
            metric: 优化指标 ('ic', 'ir', 'effectiveness_score')
            
        Returns:
            best_params: 最佳参数
            best_score: 最佳得分
        """
        print(f"开始优化因子参数，参数网格大小: {len(param_grid)}")
        
        best_score = -np.inf
        best_params = None
        
        for i, params in enumerate(param_grid):
            try:
                factor = factor_func(self.data, **params)
                score = self._calculate_metric(factor, self.returns, metric)
                
                self.optimization_history.append({
                    'params': params,
                    'score': score,
                    'iteration': i
                })
                
                if score > best_score:
                    best_score = score
                    best_params = params
                    
                if (i + 1) % 10 == 0:
                    print(f"进度: {i+1}/{len(param_grid)}, 当前最佳得分: {best_score:.4f}")
                    
            except Exception as e:
                print(f"参数 {params} 计算失败: {e}")
                continue
        
        self.best_params = best_params
        print(f"参数优化完成，最佳得分: {best_score:.4f}")
        print(f"最佳参数: {best_params}")
        
        return best_params, best_score
    
    def optimize_factor_combination(self, factors_df, max_factors=10, method='greedy'):
        """
        优化因子组合
        
        Args:
            factors_df: 因子DataFrame
            max_factors: 最大因子数量
            method: 优化方法 ('greedy', 'genetic', 'lasso')
            
        Returns:
            best_combination: 最佳因子组合
            best_score: 最佳得分
        """
        print(f"开始优化因子组合，总因子数: {len(factors_df.columns)}")
        
        if method == 'greedy':
            return self._greedy_factor_selection(factors_df, max_factors)
        elif method == 'genetic':
            return self._genetic_factor_selection(factors_df, max_factors)
        elif method == 'lasso':
            return self._lasso_factor_selection(factors_df, max_factors)
        else:
            raise ValueError(f"不支持的优化方法: {method}")
    
    def _greedy_factor_selection(self, factors_df, max_factors):
        """贪婪因子选择"""
        print("使用贪婪算法选择因子...")
        
        available_factors = list(factors_df.columns)
        selected_factors = []
        best_score = -np.inf
        
        for i in range(min(max_factors, len(available_factors))):
            current_best_factor = None
            current_best_score = -np.inf
            
            for factor in available_factors:
                test_factors = selected_factors + [factor]
                test_df = factors_df[test_factors]
                score = self._calculate_combination_score(test_df, self.returns)
                
                if score > current_best_score:
                    current_best_score = score
                    current_best_factor = factor
            
            if current_best_factor:
                selected_factors.append(current_best_factor)
                available_factors.remove(current_best_factor)
                best_score = current_best_score
                
                print(f"第 {i+1} 轮: 选择因子 {current_best_factor}, 得分: {best_score:.4f}")
        
        return selected_factors, best_score
    
    def _genetic_factor_selection(self, factors_df, max_factors, population_size=50, generations=100):
        """遗传算法因子选择"""
        print("使用遗传算法选择因子...")
        
        n_factors = len(factors_df.columns)
        factor_names = list(factors_df.columns)
        
        population = []
        for _ in range(population_size):
            selected = np.random.choice([0, 1], size=n_factors, p=[0.7, 0.3])
            population.append(selected)
        
        best_individual = None
        best_score = -np.inf
        
        for generation in range(generations):
            fitness_scores = []
            for individual in population:
                if individual.sum() > 0 and individual.sum() <= max_factors:
                    selected_factors = [factor_names[i] for i in range(n_factors) if individual[i]]
                    score = self._calculate_combination_score(factors_df[selected_factors], self.returns)
                    fitness_scores.append(score)
                else:
                    fitness_scores.append(-np.inf)
            
            best_idx = np.argmax(fitness_scores)
            if fitness_scores[best_idx] > best_score:
                best_score = fitness_scores[best_idx]
                best_individual = population[best_idx].copy()
            
            selected_indices = self._tournament_selection(fitness_scores, population_size)
            new_population = [population[i] for i in selected_indices]
            
            for i in range(0, population_size, 2):
                if i + 1 < population_size:
                    if np.random.random() < 0.8:
                        crossover_point = np.random.randint(1, n_factors)
                        new_population[i], new_population[i+1] = self._crossover(
                            new_population[i], new_population[i+1], crossover_point
                        )
                    
                    if np.random.random() < 0.1:
                        new_population[i] = self._mutate(new_population[i])
                    if np.random.random() < 0.1:
                        new_population[i+1] = self._mutate(new_population[i+1])
            
            population = new_population
            
            if generation % 20 == 0:
                print(f"第 {generation} 代: 最佳得分: {best_score:.4f}")
        
        selected_factors = [factor_names[i] for i in range(n_factors) if best_individual[i]]
        return selected_factors, best_score
    
    def _lasso_factor_selection(self, factors_df, max_factors):
        """Lasso回归因子选择"""
        print("使用Lasso回归选择因子...")
        
        aligned = self._align_factors_returns(factors_df, self.returns)
        if aligned is None:
            return [], -np.inf
        X_df, y_s = aligned
        X = X_df.values
        y = y_s.values
        
        X_scaled = self.scaler.fit_transform(X)
        
        lasso = Lasso(alpha=0.01, max_iter=1000)
        lasso.fit(X_scaled, y)
        
        coefficients = lasso.coef_
        selected_indices = np.where(np.abs(coefficients) > 1e-6)[0]
        
        factor_importance = [(i, abs(coefficients[i])) for i in selected_indices]
        factor_importance.sort(key=lambda x: x[1], reverse=True)
        
        selected_factors = [factors_df.columns[i] for i, _ in factor_importance[:max_factors]]
        
        score = self._calculate_combination_score(factors_df[selected_factors], self.returns)
        
        return selected_factors, score
    
    def _tournament_selection(self, fitness_scores, population_size, tournament_size=3):
        """锦标赛选择"""
        selected_indices = []
        for _ in range(population_size):
            tournament_indices = np.random.choice(len(fitness_scores), tournament_size)
            tournament_scores = [fitness_scores[i] for i in tournament_indices]
            winner_idx = tournament_indices[np.argmax(tournament_scores)]
            selected_indices.append(winner_idx)
        return selected_indices
    
    def _crossover(self, parent1, parent2, crossover_point):
        """交叉操作"""
        child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
        child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
        return child1, child2
    
    def _mutate(self, individual, mutation_rate=0.1):
        """变异操作"""
        for i in range(len(individual)):
            if np.random.random() < mutation_rate:
                individual[i] = 1 - individual[i]
        return individual
    
    def _align_factors_returns(self, factors_df, returns, min_rows=10):
        """
        对齐因子与收益率数据，使用 thresh 保留部分有效行
        
        Args:
            factors_df: 因子DataFrame
            returns: 收益率Series
            min_rows: 最少有效行数
            
        Returns:
            (aligned_factors_df, aligned_returns_series) 或 None
        """
        if factors_df.empty or returns is None:
            return None
        
        thresh = max(1, len(factors_df.columns) // 2)
        factors_valid = factors_df.dropna(thresh=thresh)
        
        common_index = factors_valid.index.intersection(returns.dropna().index)
        if len(common_index) < min_rows:
            return None
        
        return factors_valid.loc[common_index], returns.loc[common_index]
    
    def _calculate_metric(self, factor, returns, metric):
        """计算单个指标"""
        factor_aligned = factor.dropna()
        common_index = factor_aligned.index.intersection(returns.dropna().index)
        if len(common_index) < 10:
            return -np.inf
        factor_aligned = factor_aligned.loc[common_index]
        returns_aligned = returns.loc[common_index]
        
        if metric == 'ic':
            return factor_aligned.corr(returns_aligned)
        elif metric == 'ir':
            ic = factor_aligned.corr(returns_aligned)
            ic_std = factor_aligned.rolling(20).corr(returns_aligned).std()
            return ic / ic_std if ic_std > 0 else 0
        elif metric == 'effectiveness_score':
            ic = abs(factor_aligned.corr(returns_aligned))
            win_rate = self._calculate_win_rate(factor_aligned, returns_aligned)
            return (ic + win_rate) / 2
        else:
            raise ValueError(f"不支持的指标: {metric}")
    
    def _calculate_combination_score(self, factors_df, returns):
        """计算因子组合得分"""
        if factors_df.empty:
            return -np.inf
        
        aligned = self._align_factors_returns(factors_df, returns)
        if aligned is None:
            return -np.inf
        factors_aligned, returns_aligned = aligned
        
        combined_factor = factors_aligned.mean(axis=1)
        ic = combined_factor.corr(returns_aligned)
        win_rate = self._calculate_win_rate(combined_factor, returns_aligned)
        score = (abs(ic) + win_rate) / 2
        
        return score
    
    def _calculate_win_rate(self, factor, returns):
        """
        计算胜率（排除因子值或收益率为0的样本）
        """
        factor_direction = np.sign(factor)
        returns_direction = np.sign(returns)
        
        valid_mask = (factor_direction != 0) & (returns_direction != 0)
        if valid_mask.sum() == 0:
            return 0.5
        
        correct_predictions = (factor_direction[valid_mask] == returns_direction[valid_mask]).mean()
        return correct_predictions
    
    def create_ensemble_factor(self, factors_df, method='equal_weight', **kwargs):
        """
        创建集成因子
        
        Args:
            factors_df: 因子DataFrame
            method: 集成方法 ('equal_weight', 'ic_weight', 'ml_weight', 'max_icir_weight')
            **kwargs: 不同方法的扩展参数
            
        Returns:
            ensemble_factor: 集成因子
        """
        print(f"创建集成因子，方法: {method}")
        
        if method == 'equal_weight':
            return factors_df.mean(axis=1)
        elif method == 'ic_weight':
            return self._create_ic_weighted_factor(factors_df)
        elif method == 'ml_weight':
            return self._create_ml_weighted_factor(factors_df)
        elif method == 'max_icir_weight':
            return self._create_max_icir_weighted_factor(factors_df, **kwargs)
        else:
            raise ValueError(f"不支持的集成方法: {method}")
    
    def _create_ic_weighted_factor(self, factors_df):
        """创建IC加权因子"""
        ic_scores = {}
        for col in factors_df.columns:
            ic = self._calculate_metric(factors_df[col], self.returns, 'ic')
            ic_scores[col] = abs(ic)
        
        total_ic = sum(ic_scores.values())
        if total_ic == 0:
            return factors_df.mean(axis=1)
        
        weights = {col: ic / total_ic for col, ic in ic_scores.items()}
        
        weighted_factor = pd.Series(0, index=factors_df.index)
        for col, weight in weights.items():
            weighted_factor += weight * factors_df[col]
        
        return weighted_factor

    def _project_l1(self, weights, enforce_non_negative=False):
        """L1 归一化投影，避免权重规模失控。"""
        w = np.asarray(weights, dtype=float)
        w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
        if enforce_non_negative:
            w = np.clip(w, 0.0, None)
        l1 = np.abs(w).sum()
        if l1 <= 0:
            n = len(w)
            if n == 0:
                return w
            return np.ones(n, dtype=float) / float(n)
        return w / l1

    def _build_ic_matrix(self, factors_df, window=60, min_periods=20):
        """
        构建滚动 IC 矩阵 [T, K]：
        每一行表示一个时间窗口内各因子的 IC。
        """
        aligned = self._align_factors_returns(factors_df, self.returns, min_rows=max(window, min_periods))
        if aligned is None:
            return None, None
        factors_aligned, returns_aligned = aligned
        k = len(factors_aligned.columns)
        if k == 0:
            return None, None

        window = max(int(window), 5)
        min_periods = max(int(min_periods), 5)
        ic_rows = []
        idx_values = []

        for end_i in range(min_periods, len(factors_aligned) + 1):
            start_i = max(0, end_i - window)
            xw = factors_aligned.iloc[start_i:end_i]
            yw = returns_aligned.iloc[start_i:end_i]
            if len(xw) < min_periods:
                continue
            row = []
            for col in factors_aligned.columns:
                corr = xw[col].corr(yw)
                row.append(corr if np.isfinite(corr) else np.nan)
            if np.isfinite(np.asarray(row, dtype=float)).sum() == 0:
                continue
            ic_rows.append(row)
            idx_values.append(factors_aligned.index[end_i - 1])

        if not ic_rows:
            return None, None

        ic_matrix = pd.DataFrame(ic_rows, columns=factors_aligned.columns, index=idx_values, dtype=float)
        # 允许少量缺失，按列中位数补齐以便协方差估计稳定
        med = ic_matrix.median(axis=0, skipna=True)
        ic_matrix = ic_matrix.fillna(med).fillna(0.0)
        return ic_matrix, factors_aligned

    def _create_max_icir_weighted_factor(
        self,
        factors_df,
        ic_window=60,
        min_periods=20,
        use_ledoit_wolf=True,
        enforce_non_negative=False,
        ridge=1e-6,
        return_weights=False
    ):
        """
        最大化 ICIR 的因子组合：
        ICIR(w) = (mu^T w) / sqrt(w^T Sigma w), 约束 ||w||_1 = 1。

        当 return_weights=True 时，额外返回每个因子对应的权重字典 {factor_id: weight}，
        未参与求解的因子（例如 IC 矩阵里被剔除的列）权重为 0。
        """
        ic_matrix, factors_aligned = self._build_ic_matrix(
            factors_df,
            window=ic_window,
            min_periods=min_periods
        )
        if ic_matrix is None or factors_aligned is None or ic_matrix.empty:
            if return_weights:
                n = max(len(factors_df.columns), 1)
                fallback_w = {c: 1.0 / n for c in factors_df.columns}
                return factors_df.mean(axis=1), fallback_w
            return factors_df.mean(axis=1)

        mu = ic_matrix.mean(axis=0).to_numpy(dtype=float)
        x = ic_matrix.to_numpy(dtype=float)
        k = x.shape[1]

        cov = None
        if use_ledoit_wolf and len(x) >= 2:
            try:
                cov = LedoitWolf().fit(x).covariance_
            except Exception:
                cov = None
        if cov is None:
            cov = np.cov(x, rowvar=False)
        cov = np.asarray(cov, dtype=float)
        if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
            if return_weights:
                n = max(len(factors_df.columns), 1)
                fallback_w = {c: 1.0 / n for c in factors_df.columns}
                return factors_df.mean(axis=1), fallback_w
            return factors_df.mean(axis=1)
        cov = cov + np.eye(k, dtype=float) * float(ridge)

        try:
            w = np.linalg.solve(cov, mu)
        except np.linalg.LinAlgError:
            w, *_ = np.linalg.lstsq(cov, mu, rcond=None)
        w = self._project_l1(w, enforce_non_negative=enforce_non_negative)

        active_cols = list(ic_matrix.columns)
        weighted_factor = pd.Series(0.0, index=factors_df.index, dtype=float)
        for i, col in enumerate(active_cols):
            if col in factors_df.columns:
                weighted_factor = weighted_factor.add(factors_df[col] * w[i], fill_value=0.0)

        if return_weights:
            weights_map = {col: 0.0 for col in factors_df.columns}
            for i, col in enumerate(active_cols):
                weights_map[col] = float(w[i])
            return weighted_factor, weights_map
        return weighted_factor
    
    def _create_ml_weighted_factor(self, factors_df):
        """创建ML加权因子（全量训练，仅用于离线分析，不适用于回测）"""
        aligned = self._align_factors_returns(factors_df, self.returns)
        if aligned is None:
            return factors_df.mean(axis=1)
        X_df, y_s = aligned
        
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_df.values, y_s.values)
        
        feature_importance = rf.feature_importances_
        
        weighted_factor = pd.Series(0, index=factors_df.index)
        for i, col in enumerate(factors_df.columns):
            weighted_factor += feature_importance[i] * factors_df[col]
        
        return weighted_factor

    def _create_ml_weighted_factor_ts_cv(self, factors_df, n_splits=5):
        """
        使用时序交叉验证创建ML加权因子（旧版本：全局权重，仍有前瞻偏差）
        
        警告：此方法会聚合所有折的特征重要性作为全局权重，
        早期样本仍可能受后期信息影响。建议使用 _create_ml_weighted_factor_walk_forward。
        
        Args:
            factors_df: 因子DataFrame
            n_splits: 时序交叉验证折数
            
        Returns:
            ensemble_factor: 集成因子Series
        """
        aligned = self._align_factors_returns(factors_df, self.returns)
        if aligned is None:
            return factors_df.mean(axis=1)
        X_df, y_s = aligned

        X = X_df.values
        y = y_s.values
        index = X_df.index

        if len(X) < 20:
            return factors_df.mean(axis=1)

        tscv = TimeSeriesSplit(n_splits=min(n_splits, max(2, len(X) // 10)))

        feature_weights = np.zeros(len(factors_df.columns))
        fold_count = 0

        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)

            try:
                rf = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=5)
                rf.fit(X_train_scaled, y_train)
                feature_weights += rf.feature_importances_
                fold_count += 1
            except Exception:
                continue

        if fold_count == 0 or feature_weights.sum() <= 0:
            return factors_df.mean(axis=1)

        feature_weights = feature_weights / feature_weights.sum()

        weighted_factor = pd.Series(0, index=factors_df.index)
        for i, col in enumerate(factors_df.columns):
            col_idx = list(X_df.columns).index(col) if col in X_df.columns else -1
            if col_idx >= 0:
                weighted_factor += feature_weights[col_idx] * factors_df[col]

        return weighted_factor

    def _create_ml_weighted_factor_walk_forward(self, factors_df, min_train_size=50, refit_freq=10):
        """
        真正的逐时点 walk-forward 动态权重，完全避免前瞻偏差
        
        对每个时间点 t：
        1. 只用 t 之前的数据训练模型
        2. 用该模型预测 t 时刻的因子权重
        3. 这样每个时间点的权重都是独立的，不会受未来信息影响
        
        为提高效率，采用滚动重训练策略：
        - min_train_size: 最小训练样本数
        - refit_freq: 每隔多少个时间点重新训练一次模型
        
        Args:
            factors_df: 因子DataFrame
            min_train_size: 最小训练样本数，默认50
            refit_freq: 模型重训练频率，默认每10期重训练一次
            
        Returns:
            ensemble_factor: 集成因子Series（每个时间点使用该点之前训练的权重）
        """
        aligned = self._align_factors_returns(factors_df, self.returns)
        if aligned is None:
            return factors_df.mean(axis=1)
        X_df, y_s = aligned

        X = X_df.values
        y = y_s.values
        n_samples = len(X)
        n_features = X.shape[1]
        index = X_df.index

        if n_samples < min_train_size + 10:
            return factors_df.mean(axis=1)

        feature_weights_timeline = np.zeros((n_samples, n_features))
        current_weights = np.ones(n_features) / n_features
        last_train_idx = -refit_freq

        for t in range(n_samples):
            if t < min_train_size:
                feature_weights_timeline[t] = current_weights
                continue

            if t - last_train_idx >= refit_freq:
                train_end = t
                train_start = 0
                X_train = X[train_start:train_end]
                y_train = y[train_start:train_end]

                valid_mask = ~(np.isnan(X_train).any(axis=1) | np.isnan(y_train))
                X_train_clean = X_train[valid_mask]
                y_train_clean = y_train[valid_mask]

                if len(X_train_clean) >= min_train_size // 2:
                    try:
                        scaler = StandardScaler()
                        X_train_scaled = scaler.fit_transform(X_train_clean)

                        rf = RandomForestRegressor(
                            n_estimators=30,
                            random_state=42,
                            max_depth=4,
                            min_samples_leaf=5,
                            n_jobs=-1
                        )
                        rf.fit(X_train_scaled, y_train_clean)
                        current_weights = rf.feature_importances_
                        if current_weights.sum() > 0:
                            current_weights = current_weights / current_weights.sum()
                        else:
                            current_weights = np.ones(n_features) / n_features
                        last_train_idx = t
                    except Exception:
                        pass

            feature_weights_timeline[t] = current_weights

        weighted_factor = pd.Series(0.0, index=factors_df.index)
        factor_cols = list(X_df.columns)

        for t in range(n_samples):
            for i, col in enumerate(factor_cols):
                if col in factors_df.columns:
                    val = factors_df[col].iloc[t]
                    if pd.notna(val):
                        weighted_factor.iloc[t] += feature_weights_timeline[t, i] * val

        return weighted_factor
    
    def optimize_hyperparameters(self, factors_df, model_type='ensemble'):
        """
        优化超参数
        
        Args:
            factors_df: 因子DataFrame
            model_type: 模型类型 ('ensemble', 'linear', 'ml')
            
        Returns:
            best_params: 最佳参数
            best_score: 最佳得分
        """
        print(f"优化超参数，模型类型: {model_type}")
        
        if model_type == 'ensemble':
            return self._optimize_ensemble_params(factors_df)
        elif model_type == 'linear':
            return self._optimize_linear_params(factors_df)
        elif model_type == 'ml':
            return self._optimize_ml_params(factors_df)
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
