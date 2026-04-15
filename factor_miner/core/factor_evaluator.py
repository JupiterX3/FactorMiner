#!/usr/bin/env python3
"""
数据科学统计模块
包含因子评估的常用统计指标：IC值、IR比率、正比例等
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from concurrent.futures import ThreadPoolExecutor
from scipy import stats
from sklearn.metrics import mutual_info_score
import re
import warnings
warnings.filterwarnings('ignore')


class FactorStatistics:
    """
    因子统计评估类
    提供各种因子评估的统计指标
    """
    
    def __init__(self):
        """初始化统计评估器"""
        pass
    
    def calculate_ic(self, factor: pd.Series, returns: pd.Series, 
                    method: str = 'pearson') -> float:
        """
        计算信息系数(Information Coefficient, IC)
        
        Args:
            factor: 因子值序列
            returns: 收益率序列
            method: 相关系数计算方法 ('pearson', 'spearman', 'kendall')
            
        Returns:
            float: IC值
        """
        # 对齐数据
        factor = factor.dropna()
        returns = returns.loc[factor.index]
        
        if len(factor) < 10:
            return np.nan
        
        # 计算相关系数
        if method == 'pearson':
            correlation = factor.corr(returns)
        elif method == 'spearman':
            correlation = factor.corr(returns, method='spearman')
        elif method == 'kendall':
            correlation = factor.corr(returns, method='kendall')
        else:
            raise ValueError(f"不支持的相关系数方法: {method}")
            
        return correlation
    
    def calculate_rank_ic(self, factor: pd.Series, returns: pd.Series) -> float:
        """
        计算排序IC (Rank IC)
        
        Args:
            factor: 因子值序列
            returns: 收益率序列
            
        Returns:
            float: Rank IC值
        """
        return self.calculate_ic(factor, returns, method='spearman')
    
    def calculate_mutual_information(self, factor: pd.Series, returns: pd.Series, 
                                   bins: int = 10) -> float:
        """
        计算互信息(Mutual Information)
        
        Args:
            factor: 因子值序列
            returns: 收益率序列
            bins: 分箱数量
            
        Returns:
            float: 互信息值
        """
        # 对齐数据
        factor = factor.dropna()
        returns = returns.loc[factor.index]
        
        if len(factor) < 10:
            return np.nan
        
        # 处理NaN值
        factor_clean = factor.dropna()
        returns_clean = returns.loc[factor_clean.index].dropna()
        
        if len(factor_clean) < 10 or len(returns_clean) < 10:
            return np.nan
        
        # 确保两个序列长度一致
        common_index = factor_clean.index.intersection(returns_clean.index)
        if len(common_index) < 10:
            return np.nan
            
        factor_clean = factor_clean.loc[common_index]
        returns_clean = returns_clean.loc[common_index]
        
        try:
            # 分箱处理
            factor_binned = pd.cut(factor_clean, bins=bins, labels=False, duplicates='drop')
            returns_binned = pd.cut(returns_clean, bins=bins, labels=False, duplicates='drop')
            
            # 处理分箱后的NaN值
            valid_mask = ~(factor_binned.isna() | returns_binned.isna())
            if valid_mask.sum() < 10:
                return np.nan
                
            factor_binned = factor_binned[valid_mask]
            returns_binned = returns_binned[valid_mask]
            
            # 计算互信息
            mi = mutual_info_score(factor_binned, returns_binned)
            return mi
        except Exception:
            return np.nan
    
    def calculate_ic_ir(self, ic_series: pd.Series) -> float:
        """
        计算信息比率(Information Ratio, IR)
        
        Args:
            ic_series: IC值序列
            
        Returns:
            float: IR值
        """
        if len(ic_series) == 0:
            return np.nan
        
        ic_mean = ic_series.mean()
        ic_std = ic_series.std()
        
        if ic_std == 0:
            return np.nan
            
        return ic_mean / ic_std
    
    def calculate_ic_positive_ratio(self, ic_series: pd.Series) -> float:
        """
        计算IC正比例
        
        Args:
            ic_series: IC值序列
            
        Returns:
            float: 正IC比例
        """
        if len(ic_series) == 0:
            return np.nan
            
        return (ic_series > 0).mean()
    
    def calculate_ic_stability(self, ic_series: pd.Series, window: int = 20) -> float:
        """
        计算IC稳定性
        
        Args:
            ic_series: IC值序列
            window: 滚动窗口大小
            
        Returns:
            float: IC稳定性指标
        """
        if len(ic_series) < window:
            return np.nan
        
        # 计算滚动IC的标准差
        rolling_ic_std = ic_series.rolling(window=window).std()
        ic_stability = 1 / (1 + rolling_ic_std.mean())
        
        return ic_stability
    
    def calculate_factor_returns(self, factor: pd.Series, returns: pd.Series, 
                               n_groups: int = 5) -> Dict[str, float]:
        """
        计算因子收益率统计
        
        Args:
            factor: 因子值序列
            returns: 收益率序列
            n_groups: 分组数量
            
        Returns:
            Dict: 包含各种收益率统计指标
        """
        # 对齐数据
        factor = factor.dropna()
        returns = returns.loc[factor.index]
        
        if len(factor) < n_groups * 10:
            return {}
        
        # 分组
        try:
            factor_quantiles = pd.qcut(factor, n_groups, labels=False, duplicates='drop')
        except ValueError:
            # 如果唯一值太少，使用cut
            factor_quantiles = pd.cut(factor, n_groups, labels=False, duplicates='drop')
        
        # 计算各组收益率
        group_returns = returns.groupby(factor_quantiles).mean()

        # 计算统计指标
        long_short_return = group_returns.iloc[-1] - group_returns.iloc[0]  # 多空收益
        long_return = group_returns.iloc[-1]  # 多头收益
        short_return = group_returns.iloc[0]  # 空头收益

        # 为与现有输出兼容保留该字段（原始收益波动）
        returns_std = returns.std()

        # 使用分组组合收益序列估计“组合口径”Sharpe，避免分子/分母口径不一致
        # 说明：单序列分位回测信息有限，此处给出近似口径，优于直接除以原始收益std
        factor_rank = factor_quantiles.astype(float)
        long_mask = factor_rank == factor_rank.max()
        short_mask = factor_rank == factor_rank.min()
        ls_series = returns.where(long_mask, 0.0) - returns.where(short_mask, 0.0)
        ls_std = ls_series.std()
        if ls_std is None or (not np.isfinite(ls_std)) or ls_std <= 0:
            sharpe_ratio = np.nan
        else:
            sharpe_ratio = ls_series.mean() / ls_std
        
        return {
            'long_short_return': long_short_return,
            'long_return': long_return,
            'short_return': short_return,
            'returns_std': returns_std,
            'sharpe_ratio': sharpe_ratio,
            'group_returns': group_returns.to_dict()
        }
    
    def calculate_factor_turnover(self, factor: pd.Series, window: int = 20) -> float:
        """
        计算因子换手率
        
        Args:
            factor: 因子值序列
            window: 滚动窗口大小
            
        Returns:
            float: 换手率
        """
        if len(factor) < window + 1:
            return np.nan
        
        # 计算因子变化
        factor_change = factor.diff().abs()
        
        # 计算滚动平均换手率
        turnover = factor_change.rolling(window=window).mean()
        
        return turnover.mean()
    
    def calculate_factor_decay(self, factor: pd.Series, returns: pd.Series, 
                             max_lag: int = 10) -> Dict[str, float]:
        """
        计算因子衰减特征
        
        分析因子值对不同期限未来收益的预测能力衰减情况。
        lag=0: 因子值预测当期收益
        lag=N: 因子值预测N期后的收益
        
        Args:
            factor: 因子值序列
            returns: 收益率序列
            max_lag: 最大滞后期数
            
        Returns:
            Dict: 包含各期IC值和衰减率
        """
        ic_decay = {}
        
        for lag in range(max_lag + 1):
            if lag == 0:
                ic = self.calculate_ic(factor, returns)
            else:
                lagged_returns = returns.shift(-lag)
                ic = self.calculate_ic(factor, lagged_returns)
            
            ic_decay[f'ic_lag_{lag}'] = ic
        
        ic_values = [v for v in ic_decay.values() if np.isfinite(v)]
        if len(ic_values) > 1:
            decay_rate = (ic_values[0] - ic_values[-1]) / len(ic_values)
        else:
            decay_rate = np.nan
        
        ic_decay['decay_rate'] = decay_rate
        
        return ic_decay
    
    def calculate_factor_correlation(self, factors: pd.DataFrame) -> pd.DataFrame:
        """
        计算因子间相关性矩阵
        
        Args:
            factors: 因子DataFrame
            
        Returns:
            pd.DataFrame: 相关性矩阵
        """
        return factors.corr()
    
    def calculate_factor_autocorrelation(self, factor: pd.Series, 
                                       max_lag: int = 10) -> Dict[str, float]:
        """
        计算因子自相关性
        
        Args:
            factor: 因子值序列
            max_lag: 最大滞后期数
            
        Returns:
            Dict: 包含各期自相关系数
        """
        autocorr = {}
        
        for lag in range(1, max_lag + 1):
            if len(factor) > lag:
                corr = factor.autocorr(lag=lag)
                autocorr[f'autocorr_lag_{lag}'] = corr
        
        return autocorr
    
    def calculate_factor_skewness_kurtosis(self, factor: pd.Series) -> Dict[str, float]:
        """
        计算因子偏度和峰度
        
        Args:
            factor: 因子值序列
            
        Returns:
            Dict: 包含偏度和峰度
        """
        factor_clean = factor.dropna()
        
        if len(factor_clean) < 10:
            return {'skewness': np.nan, 'kurtosis': np.nan}
        
        skewness = factor_clean.skew()
        kurtosis = factor_clean.kurtosis()
        
        return {
            'skewness': skewness,
            'kurtosis': kurtosis
        }
    
    def calculate_factor_win_rate(self, factor: pd.Series, returns: pd.Series, 
                                threshold: float = 0.5) -> float:
        """
        计算因子胜率（修复：使用历史因子预测当期收益）
        
        Args:
            factor: 因子值序列
            returns: 收益率序列
            threshold: 胜率阈值
            
        Returns:
            float: 胜率
        """
        # 对齐数据
        factor = factor.dropna()
        returns = returns.loc[factor.index]
        
        if len(factor) < 10:
            return np.nan
        
        # 因子与收益已在上游完成时点对齐，这里直接计算方向一致率
        factor_sign = np.sign(factor)
        returns_sign = np.sign(returns)
        
        # 计算胜率（历史因子预测当期收益的正确性）
        win_rate = (factor_sign == returns_sign).mean()
        
        return win_rate
    
    def comprehensive_factor_analysis(self, factor: pd.Series, returns: pd.Series,
                                    factor_name: str = "factor") -> Dict[str, Union[float, Dict]]:
        """
        综合因子分析
        
        Args:
            factor: 因子值序列
            returns: 收益率序列
            factor_name: 因子名称
            
        Returns:
            Dict: 包含所有统计指标的综合分析结果
        """
        aligned_factor = factor.shift(1)
        analysis = {
            'factor_name': factor_name,
            'data_length': len(aligned_factor.dropna()),
            'missing_ratio': aligned_factor.isna().mean()
        }
        
        # 基础IC指标
        analysis['ic_pearson'] = self.calculate_ic(aligned_factor, returns, 'pearson')
        analysis['ic_spearman'] = self.calculate_ic(aligned_factor, returns, 'spearman')
        analysis['ic_kendall'] = self.calculate_ic(aligned_factor, returns, 'kendall')
        analysis['mutual_information'] = self.calculate_mutual_information(aligned_factor, returns)
        
        # 因子收益率分析
        factor_returns = self.calculate_factor_returns(aligned_factor, returns)
        analysis.update(factor_returns)
        
        # 因子特征
        skewness_kurtosis = self.calculate_factor_skewness_kurtosis(aligned_factor)
        analysis.update(skewness_kurtosis)
        
        # 因子换手率
        analysis['turnover'] = self.calculate_factor_turnover(aligned_factor)
        
        # 因子胜率
        analysis['win_rate'] = self.calculate_factor_win_rate(aligned_factor, returns)
        
        # 因子自相关性
        autocorr = self.calculate_factor_autocorrelation(aligned_factor)
        analysis.update(autocorr)
        
        # 因子衰减特征（使用原始factor，因为需要分析对不同期限收益的预测能力）
        # 注意：这里不shift，因为是分析因子对不同滞后期收益的预测能力
        factor_decay = self.calculate_factor_decay(factor, returns, max_lag=5)
        analysis['decay_rate'] = factor_decay.get('decay_rate')
        for lag in range(1, 6):
            analysis[f'ic_lag_{lag}'] = factor_decay.get(f'ic_lag_{lag}')
        
        return analysis
    
    def batch_factor_analysis(self, factors: pd.DataFrame, returns: pd.Series,
                            progress_bar: bool = True) -> pd.DataFrame:
        """
        批量因子分析
        
        Args:
            factors: 因子DataFrame
            returns: 收益率序列
            progress_bar: 是否显示进度条
            
        Returns:
            pd.DataFrame: 包含所有因子分析结果的DataFrame
        """
        results = []
        
        if progress_bar:
            from tqdm import tqdm
            factor_iterator = tqdm(factors.columns, desc="分析因子")
        else:
            factor_iterator = factors.columns
        
        for factor_name in factor_iterator:
            factor = factors[factor_name]
            analysis = self.comprehensive_factor_analysis(factor, returns, factor_name)
            results.append(analysis)
        
        return pd.DataFrame(results)
    
    def calculate_rolling_ic(self, factor: pd.Series, returns: pd.Series,
                           window: int = 60) -> pd.Series:
        """
        计算滚动IC
        
        Args:
            factor: 因子值序列
            returns: 收益率序列
            window: 滚动窗口大小
            
        Returns:
            pd.Series: 滚动IC序列
        """
        rolling_ic = pd.Series(index=factor.index, dtype=float)
        
        for i in range(window, len(factor)):
            factor_window = factor.iloc[i-window:i]
            returns_window = returns.iloc[i-window:i]
            ic = self.calculate_ic(factor_window, returns_window)
            rolling_ic.iloc[i] = ic
        
        return rolling_ic
    
    def calculate_ic_ir_series(self, rolling_ic: pd.Series, window: int = 20) -> pd.Series:
        """
        计算滚动IR序列
        
        Args:
            rolling_ic: 滚动IC序列
            window: 滚动窗口大小
            
        Returns:
            pd.Series: 滚动IR序列
        """
        rolling_ir = pd.Series(index=rolling_ic.index, dtype=float)
        
        for i in range(window, len(rolling_ic)):
            ic_window = rolling_ic.iloc[i-window:i]
            ir = self.calculate_ic_ir(ic_window)
            rolling_ir.iloc[i] = ir
        
        return rolling_ir
    
    def get_factor_ranking(self, factors: pd.DataFrame, returns: pd.Series,
                          metric: str = 'ic_pearson', top_n: int = 10) -> pd.DataFrame:
        """
        获取因子排名
        
        Args:
            factors: 因子DataFrame
            returns: 收益率序列
            metric: 排序指标
            top_n: 返回前N个因子
            
        Returns:
            pd.DataFrame: 因子排名结果
        """
        # 批量分析
        analysis_df = self.batch_factor_analysis(factors, returns)
        
        # 按指定指标排序
        if metric in analysis_df.columns:
            analysis_df = analysis_df.sort_values(metric, key=abs, ascending=False)
        
        return analysis_df.head(top_n)
    
    def calculate_factor_effectiveness_score(self, factor: pd.Series, returns: pd.Series) -> float:
        """
        计算因子有效性综合评分
        
        Args:
            factor: 因子值序列
            returns: 收益率序列
            
        Returns:
            float: 综合评分 (0-1)
        """
        # 计算各项指标
        aligned_factor = factor.shift(1)
        ic = abs(self.calculate_ic(aligned_factor, returns))
        mi = self.calculate_mutual_information(aligned_factor, returns)
        win_rate = self.calculate_factor_win_rate(aligned_factor, returns)
        turnover = self.calculate_factor_turnover(aligned_factor)
        
        # 标准化指标
        ic_score = min(ic * 10, 1.0)  # IC转换为0-1
        mi_score = min(mi * 5, 1.0)   # 互信息转换为0-1
        win_rate_score = win_rate     # 胜率已经是0-1
        turnover_score = max(0, 1 - turnover)  # 换手率越低越好
        
        # 综合评分 (加权平均)
        effectiveness_score = (
            ic_score * 0.4 +
            mi_score * 0.3 +
            win_rate_score * 0.2 +
            turnover_score * 0.1
        )
        
        return effectiveness_score


class FactorEvaluator:
    """
    因子评估器
    提供高级的因子评估功能
    """
    
    def __init__(self):
        """初始化因子评估器"""
        self.stats = FactorStatistics()
    
    def evaluate_single_factor(self, factor: pd.Series, returns: pd.Series,
                             factor_name: str = "factor") -> Dict:
        """
        评估单个因子
        
        Args:
            factor: 因子值序列
            returns: 收益率序列
            factor_name: 因子名称
            
        Returns:
            Dict: 评估结果
        """
        return self.stats.comprehensive_factor_analysis(factor, returns, factor_name)
    
    def evaluate_multiple_factors(self, factors: pd.DataFrame, returns: pd.Series,
                                metrics: List[str] = None) -> pd.DataFrame:
        """
        评估多个因子
        
        Args:
            factors: 因子DataFrame
            returns: 收益率序列
            metrics: 评估指标列表
            
        Returns:
            pd.DataFrame: 评估结果
        """
        if metrics is None:
            metrics = ['ic_pearson', 'ic_spearman', 'mutual_information', 
                      'long_short_return', 'sharpe_ratio', 'win_rate']
        
        # 批量分析
        analysis_df = self.stats.batch_factor_analysis(factors, returns)
        
        # 选择指定指标
        if metrics:
            available_metrics = [m for m in metrics if m in analysis_df.columns]
            analysis_df = analysis_df[['factor_name'] + available_metrics]
        
        return analysis_df
    
    def get_best_factors(self, factors: pd.DataFrame, returns: pd.Series,
                        metric: str = 'ic_pearson', top_n: int = 10) -> pd.DataFrame:
        """
        获取最佳因子
        
        Args:
            factors: 因子DataFrame
            returns: 收益率序列
            metric: 排序指标
            top_n: 返回前N个因子
            
        Returns:
            pd.DataFrame: 最佳因子列表
        """
        return self.stats.get_factor_ranking(factors, returns, metric, top_n)
    
    def calculate_factor_effectiveness_scores(self, factors: pd.DataFrame, 
                                            returns: pd.Series) -> pd.Series:
        """
        计算所有因子的有效性评分
        
        Args:
            factors: 因子DataFrame
            returns: 收益率序列
            
        Returns:
            pd.Series: 因子有效性评分
        """
        scores = {}
        
        for factor_name in factors.columns:
            factor = factors[factor_name]
            score = self.stats.calculate_factor_effectiveness_score(factor, returns)
            scores[factor_name] = score
        
        return pd.Series(scores)
    
    def generate_factor_report(self, factors: pd.DataFrame, returns: pd.Series,
                             output_file: str = None) -> str:
        """
        生成因子评估报告
        
        Args:
            factors: 因子DataFrame
            returns: 收益率序列
            output_file: 输出文件路径
            
        Returns:
            str: 报告内容
        """
        # 获取最佳因子
        best_factors = self.get_best_factors(factors, returns, 'ic_pearson', 20)
        
        # 计算有效性评分
        effectiveness_scores = self.calculate_factor_effectiveness_scores(factors, returns)
        
        # 生成报告
        report = []
        report.append("=" * 80)
        report.append("因子评估报告")
        report.append("=" * 80)
        report.append(f"评估时间: {pd.Timestamp.now()}")
        report.append(f"因子总数: {len(factors.columns)}")
        report.append(f"数据长度: {len(returns)}")
        report.append("")
        
        # 整体统计
        ic_values = []
        for factor_name in factors.columns:
            factor = factors[factor_name]
            ic = self.stats.calculate_ic(factor.shift(1), returns)
            if not np.isnan(ic):
                ic_values.append(ic)
        
        if ic_values:
            ic_series = pd.Series(ic_values)
            report.append("整体IC统计:")
            report.append(f"  IC均值: {ic_series.mean():.4f}")
            report.append(f"  IC标准差: {ic_series.std():.4f}")
            report.append(f"  IC IR: {self.stats.calculate_ic_ir(ic_series):.4f}")
            report.append(f"  IC正比例: {self.stats.calculate_ic_positive_ratio(ic_series):.4f}")
            report.append("")
        
        # 最佳因子
        report.append("最佳因子排名 (按IC绝对值):")
        report.append("-" * 60)
        for i, (_, row) in enumerate(best_factors.iterrows()):
            factor_name = row['factor_name']
            ic = row.get('ic_pearson', 0)
            effectiveness = effectiveness_scores.get(factor_name, 0)
            report.append(f"{i+1:2d}. {factor_name:25s} | IC: {ic:6.4f} | 有效性: {effectiveness:.3f}")
        
        report.append("")
        report.append("=" * 80)
        
        report_content = "\n".join(report)
        
        # 保存报告
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report_content)
        
        return report_content


# 便捷函数
def calculate_ic(factor: pd.Series, returns: pd.Series, method: str = 'pearson') -> float:
    """便捷函数：计算IC"""
    stats = FactorStatistics()
    return stats.calculate_ic(factor, returns, method)


def calculate_ic_ir(ic_series: pd.Series) -> float:
    """便捷函数：计算IR"""
    stats = FactorStatistics()
    return stats.calculate_ic_ir(ic_series)


def calculate_factor_effectiveness_score(factor: pd.Series, returns: pd.Series) -> float:
    """便捷函数：计算因子有效性评分"""
    stats = FactorStatistics()
    return stats.calculate_factor_effectiveness_score(factor, returns)


def evaluate_factors(factors: pd.DataFrame, returns: pd.Series) -> pd.DataFrame:
    """便捷函数：评估因子"""
    evaluator = FactorEvaluator()
    return evaluator.evaluate_multiple_factors(factors, returns)


class CrossSectionalEvaluator:
    """
    截面因子评估器
    用于多币种截面因子评估
    
    注意：不建议使用事件因子进行截面评估
    事件因子特征：仅0/1取值，unique值=2
    截面评估缺陷：
    1. 取值区分度低：只有两个值，无法有效分组
    2. 数据稀疏：大部分时间点因子值为0，有效信号稀少
    3. 分组效果差：难以形成有意义的多空组合
    """
    
    def __init__(
        self,
        n_groups: int = 5,
        normalize_method: str = 'rank_centered',
        predict_step: int = 1,
        sample_step: int = 1,
        base_timeframe: str = '1h',
        factor_timeframe: Optional[str] = None,
        factor_bar_mode: str = 'completed',
        max_lookback: int = 200,
        min_coverage: float = 0.3,
        min_valid_count: int = 30,
        min_group_size: int = 5,
        treat_zero_as_invalid: bool = True,
        enable_data_cleaning: bool = False,
        remove_zero_volume: bool = True,
        liquidity_filter_ratio: float = 0.5,
        enable_outlier_treatment: bool = False,
        outlier_method: str = 'mad',
        outlier_group_minutes: int = 30,
        outlier_mad_n: float = 5.0,
        outlier_winsor_lower: float = 0.01,
        outlier_winsor_upper: float = 0.99,
        compute_fsc: bool = False,
        compute_ic_decay_curve: bool = False,
        ic_decay_max_lag: int = 5,
        transaction_cost: float = 0.001,
    ):
        """
        初始化截面评估器
        
        Args:
            n_groups: 分组数量，默认5组
            normalize_method: 因子标准化方法
                - 'rank': rank标准化，将因子值转换为排名并归一化到[0,1]
                - 'rank_centered': rank标准化并中心化到[-1,1]
                - 'none': 不进行标准化
            predict_step: 预测时间步（N），收益定义为 t -> t+N
            sample_step: 采样时间步（M），每隔 M 个时间点参与评估
            base_timeframe: 收益评估时间框架（执行/收益周期）
            factor_timeframe: 因子计算时间框架（为空时与 base_timeframe 相同）
            factor_bar_mode: 因子K线口径
                - 'completed': 仅使用已完成的高周期K线
                - 'intrabar': 使用当前高周期桶内的盘中快照（近似）
                - 'intrabar_strict': 严格按交易所边界做盘中重放（高精度，较慢）
            max_lookback: intrabar_strict 模式回放时保留的最大历史bar数量
            min_coverage: 每个截面的最小覆盖率阈值（有效样本/截面样本）
            min_valid_count: 每个截面的最小有效样本数阈值
            min_group_size: 分组评估时每组最小样本数
            treat_zero_as_invalid: 是否将 0 值视为无效信号（稀疏因子推荐开启）
        """
        self.n_groups = n_groups
        self.normalize_method = normalize_method
        try:
            self.predict_step = max(1, int(predict_step))
        except Exception:
            self.predict_step = 1
        try:
            self.sample_step = max(1, int(sample_step))
        except Exception:
            self.sample_step = 1
        self.base_timeframe = str(base_timeframe or '1h').lower()
        self.factor_timeframe = str(factor_timeframe or self.base_timeframe).lower()
        self.factor_bar_mode = str(factor_bar_mode or 'completed').lower()
        if self.factor_bar_mode not in ('completed', 'intrabar', 'intrabar_strict'):
            self.factor_bar_mode = 'completed'
        try:
            self.max_lookback = max(1, int(max_lookback))
        except Exception:
            self.max_lookback = 200
        try:
            self.min_coverage = float(min_coverage)
        except Exception:
            self.min_coverage = 0.3
        self.min_coverage = min(max(self.min_coverage, 0.0), 1.0)
        try:
            self.min_valid_count = max(1, int(min_valid_count))
        except Exception:
            self.min_valid_count = 30
        try:
            self.min_group_size = max(1, int(min_group_size))
        except Exception:
            self.min_group_size = 5
        self.treat_zero_as_invalid = bool(treat_zero_as_invalid)
        self.enable_data_cleaning = bool(enable_data_cleaning)
        self.remove_zero_volume = bool(remove_zero_volume)
        try:
            self.liquidity_filter_ratio = float(liquidity_filter_ratio)
        except Exception:
            self.liquidity_filter_ratio = 0.5
        self.liquidity_filter_ratio = min(max(self.liquidity_filter_ratio, 0.0), 1.0)
        self.enable_outlier_treatment = bool(enable_outlier_treatment)
        self.outlier_method = str(outlier_method or 'mad').lower()
        if self.outlier_method not in ('mad', 'winsor'):
            self.outlier_method = 'mad'
        try:
            self.outlier_group_minutes = max(1, int(outlier_group_minutes))
        except Exception:
            self.outlier_group_minutes = 30
        try:
            self.outlier_mad_n = float(outlier_mad_n)
        except Exception:
            self.outlier_mad_n = 5.0
        try:
            self.outlier_winsor_lower = float(outlier_winsor_lower)
            self.outlier_winsor_upper = float(outlier_winsor_upper)
        except Exception:
            self.outlier_winsor_lower, self.outlier_winsor_upper = 0.01, 0.99
        if self.outlier_winsor_lower >= self.outlier_winsor_upper:
            self.outlier_winsor_lower, self.outlier_winsor_upper = 0.01, 0.99
        self.compute_fsc = bool(compute_fsc)
        self.compute_ic_decay_curve = bool(compute_ic_decay_curve)
        try:
            self.ic_decay_max_lag = max(1, int(ic_decay_max_lag))
        except Exception:
            self.ic_decay_max_lag = 5
        try:
            self.transaction_cost = float(transaction_cost)
        except (TypeError, ValueError):
            self.transaction_cost = 0.001
        self.stats = FactorStatistics()

    def _normalize_factor_cross_sectional(self, factor_vals: np.ndarray) -> np.ndarray:
        """
        截面因子标准化
        
        Args:
            factor_vals: 因子值数组
            
        Returns:
            标准化后的因子值数组
        """
        if self.normalize_method == 'none':
            return factor_vals
        
        valid_mask = ~np.isnan(factor_vals)
        if valid_mask.sum() < 2:
            return factor_vals
        
        result = factor_vals.copy()
        valid_vals = factor_vals[valid_mask]
        n = len(valid_vals)
        
        ranks = pd.Series(valid_vals).rank(method='average').values
        
        if self.normalize_method == 'rank':
            result[valid_mask] = (ranks - 1) / (n - 1)
        elif self.normalize_method == 'rank_centered':
            result[valid_mask] = 2 * (ranks - 1) / (n - 1) - 1
        
        return result

    def _to_series(self, factor_values) -> Optional[pd.Series]:
        """将引擎输出统一为 Series。"""
        if factor_values is None:
            return None
        if hasattr(factor_values, 'columns'):
            try:
                factor_values = factor_values.iloc[:, 0]
            except Exception:
                factor_values = factor_values.squeeze()
        if not isinstance(factor_values, pd.Series):
            try:
                factor_values = pd.Series(factor_values)
            except Exception:
                return None
        return factor_values

    def _get_valid_signal_mask(self, factor_series: pd.Series, returns_series: pd.Series) -> pd.Series:
        """构造有效信号掩码：先过滤 NaN/inf，再按需过滤 0 值。"""
        valid = factor_series.notna() & returns_series.notna()
        valid = valid & np.isfinite(factor_series.to_numpy(dtype=float, copy=False))
        valid = valid & np.isfinite(returns_series.to_numpy(dtype=float, copy=False))
        if self.treat_zero_as_invalid:
            valid = valid & (factor_series != 0)
        return valid

    def _timeframe_to_pandas_freq(self, timeframe: str) -> str:
        """将常见交易周期字符串转为 pandas 频率。"""
        tf = str(timeframe or '').strip().lower()
        m = re.fullmatch(r'(\d+)\s*([mhdw])', tf)
        if not m:
            return tf
        n = int(m.group(1))
        unit = m.group(2)
        if unit == 'm':
            return f'{n}min'
        if unit == 'h':
            return f'{n}h'
        if unit == 'd':
            return f'{n}D'
        if unit == 'w':
            return f'{n}W'
        return tf

    def _timeframe_to_timedelta(self, timeframe: str) -> Optional[pd.Timedelta]:
        """将常见交易周期字符串转为 Timedelta。"""
        tf = str(timeframe or '').strip().lower()
        m = re.fullmatch(r'(\d+)\s*([mhdw])', tf)
        if not m:
            return None
        n = int(m.group(1))
        unit = m.group(2)
        if unit == 'm':
            return pd.Timedelta(minutes=n)
        if unit == 'h':
            return pd.Timedelta(hours=n)
        if unit == 'd':
            return pd.Timedelta(days=n)
        if unit == 'w':
            return pd.Timedelta(weeks=n)
        return None

    def _resample_ohlcv_completed(self, market_data: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        生成“已完成bar”口径的高周期K线。
        说明：使用右闭右标注 (a, b]，并丢弃最后未完成桶。
        """
        freq = self._timeframe_to_pandas_freq(timeframe)
        agg = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
        }
        bars = market_data[['open', 'high', 'low', 'close', 'volume']].resample(
            freq, label='right', closed='right'
        ).agg(agg)
        bars = bars.dropna(subset=['open', 'high', 'low', 'close'])
        if bars.empty:
            return bars

        # 仅保留“在样本时刻已经完成”的bar（例如 01:15 时只允许用到 01:00 的 1h bar）
        try:
            max_completed_ts = market_data.index.max().floor(freq)
            bars = bars[bars.index <= max_completed_ts]
        except Exception:
            pass
        return bars

    def _build_intrabar_snapshots(self, market_data: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        构造高周期桶内盘中快照（近似 intrabar 口径）：
        每个基础周期时点都生成该高周期桶从起点到当前的 OHLCV。
        """
        freq = self._timeframe_to_pandas_freq(timeframe)
        snaps = market_data[['open', 'high', 'low', 'close', 'volume']].copy()
        bucket_start = snaps.index.floor(freq)
        g = snaps.groupby(bucket_start)
        snaps['open'] = g['open'].transform('first')
        snaps['high'] = g['high'].cummax()
        snaps['low'] = g['low'].cummin()
        snaps['volume'] = g['volume'].cumsum()
        snaps = snaps.dropna(subset=['open', 'high', 'low', 'close'])
        return snaps

    def _clean_market_data(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """按高频实践做可选清洗：前向填充 + 剔除零成交量。"""
        md = market_data.copy().sort_index()
        if not self.enable_data_cleaning:
            return md
        ohlcv_cols = [c for c in ['open', 'high', 'low', 'close', 'volume'] if c in md.columns]
        if ohlcv_cols:
            md[ohlcv_cols] = md[ohlcv_cols].ffill()
        if self.remove_zero_volume and 'volume' in md.columns:
            md = md[md['volume'] > 0]
        return md

    def _apply_liquidity_filter(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        按日均成交额过滤低流动性标的：
        liquidity_filter_ratio=0.5 表示过滤后 50%，仅保留前 50%。
        """
        if (not self.enable_data_cleaning) or self.liquidity_filter_ratio <= 0:
            return data_dict
        liquidity = []
        for symbol, md in data_dict.items():
            if md is None or md.empty or ('close' not in md.columns) or ('volume' not in md.columns):
                continue
            amount = (md['close'] * md['volume']).replace([np.inf, -np.inf], np.nan).dropna()
            if len(amount) == 0:
                continue
            liquidity.append((symbol, float(amount.mean())))
        if len(liquidity) < 2:
            return data_dict
        keep_ratio = 1.0 - self.liquidity_filter_ratio
        keep_n = max(1, int(np.ceil(len(liquidity) * keep_ratio)))
        sorted_liq = sorted(liquidity, key=lambda x: x[1], reverse=True)
        keep_symbols = {sym for sym, _ in sorted_liq[:keep_n]}
        return {sym: md for sym, md in data_dict.items() if sym in keep_symbols}

    def _apply_outlier_treatment(self, factor_series: pd.Series) -> pd.Series:
        """对因子值做可选异常值处理（MAD 或 Winsor），按固定分钟分组独立处理。"""
        if (not self.enable_outlier_treatment) or factor_series is None or factor_series.empty:
            return factor_series
        fs = factor_series.copy()
        try:
            group_key = fs.index.floor(f'{self.outlier_group_minutes}min')
        except Exception:
            return fs

        def _clip_mad(x: pd.Series) -> pd.Series:
            x = x.astype(float)
            med = x.median()
            mad = (x - med).abs().median()
            if not np.isfinite(mad) or mad <= 0:
                return x
            lower = med - self.outlier_mad_n * mad
            upper = med + self.outlier_mad_n * mad
            return x.clip(lower, upper)

        def _clip_winsor(x: pd.Series) -> pd.Series:
            x = x.astype(float)
            lower = x.quantile(self.outlier_winsor_lower)
            upper = x.quantile(self.outlier_winsor_upper)
            if not np.isfinite(lower) or not np.isfinite(upper):
                return x
            return x.clip(lower, upper)

        if self.outlier_method == 'winsor':
            return fs.groupby(group_key, group_keys=False).apply(_clip_winsor)
        return fs.groupby(group_key, group_keys=False).apply(_clip_mad)

    def _build_intrabar_strict_factor_series(
        self,
        market_data: pd.DataFrame,
        factor_id: str,
        engine
    ) -> Optional[pd.Series]:
        """
        严格交易所锚定 + intrabar 重放：
        - 高周期桶按交易所标准边界（自然整点/整分）划分；
        - 每个基础时刻 t 构造：历史已完成高周期bar + 当前桶内未完成bar(快照)；
        - 对该“重放切片”重新计算因子，仅取最后一个值作为 t 的高周期因子。
        """
        freq = self._timeframe_to_pandas_freq(self.factor_timeframe)
        tf_delta = self._timeframe_to_timedelta(self.factor_timeframe)
        if tf_delta is None:
            return None

        base = market_data[['open', 'high', 'low', 'close', 'volume']].copy()
        if base.empty:
            return None

        completed_bars = self._resample_ohlcv_completed(base, self.factor_timeframe)
        bucket_start = base.index.floor(freq)
        g = base.groupby(bucket_start)

        partial = pd.DataFrame(index=base.index)
        partial['open'] = g['open'].transform('first')
        partial['high'] = g['high'].cummax()
        partial['low'] = g['low'].cummin()
        partial['close'] = base['close']
        partial['volume'] = g['volume'].cumsum()
        partial['bucket_start'] = bucket_start

        completed_ohlcv = completed_bars[['open', 'high', 'low', 'close', 'volume']]
        max_lookback = max(1, int(self.max_lookback))
        bucket_groups = list(partial.groupby('bucket_start', sort=True))

        def process_bucket(bucket_item):
            bucket_start_ts, group = bucket_item
            bucket_end = bucket_start_ts + tf_delta
            hist = completed_ohlcv[completed_ohlcv.index < bucket_end].iloc[-max_lookback:]
            bucket_results = []
            for ts, row in group.iterrows():
                current_bar = pd.DataFrame(
                    {
                        'open': [row['open']],
                        'high': [row['high']],
                        'low': [row['low']],
                        'close': [row['close']],
                        'volume': [row['volume']],
                    },
                    index=pd.DatetimeIndex([bucket_end]),
                )
                replay_input = pd.concat([hist, current_bar])
                replay_input = replay_input[~replay_input.index.duplicated(keep='last')].sort_index()
                factor_replay = self._to_series(engine.compute_single_factor(factor_id, replay_input))
                value = np.nan if (factor_replay is None or factor_replay.empty) else factor_replay.iloc[-1]
                bucket_results.append((ts, value))
            return bucket_results

        max_workers = min(4, len(bucket_groups)) if bucket_groups else 1
        results_map = {}
        if max_workers <= 1:
            for item in bucket_groups:
                for ts, value in process_bucket(item):
                    results_map[ts] = value
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for bucket_results in executor.map(process_bucket, bucket_groups):
                    for ts, value in bucket_results:
                        results_map[ts] = value

        values = [results_map.get(ts, np.nan) for ts in base.index]
        return pd.Series(values, index=base.index, dtype=float)

    def _resolve_signal_shift(self, factor_id: str, engine, default_trade_lag: int = 1) -> int:
        """
        解析截面评估中“信号相对收益”的延迟（shift）：
        - 默认：信号延迟 1 根K线（与历史1根K线对齐收益）
        - 若因子定义在 metadata/parameters/computation_data 中声明 lookahead，则额外滞后该 lookahead
        """
        shift_n = default_trade_lag

        try:
            if not hasattr(engine, "storage") or engine.storage is None:
                return int(shift_n)

            factor_def = engine.storage.load_factor_definition(factor_id)
            if not factor_def:
                return int(shift_n)

            # 允许在 metadata/parameters/computation_data 中声明（向后兼容）
            meta = (factor_def.metadata or {})
            params = (factor_def.parameters or {})
            comp_data = (factor_def.computation_data or {})

            # 覆盖默认 trade_lag
            for k in ("lag", "signal_lag", "trade_lag", "shift"):
                if k in meta:
                    shift_n = meta.get(k)
                    break
                if k in params:
                    shift_n = params.get(k)
                    break
                if k in comp_data:
                    shift_n = comp_data.get(k)
                    break

            lookahead = 0
            for k in ("lookahead", "future_bars"):
                if k in meta:
                    lookahead = meta.get(k, 0) or 0
                    break
                if k in params:
                    lookahead = params.get(k, 0) or 0
                    break
                if k in comp_data:
                    lookahead = comp_data.get(k, 0) or 0
                    break

            # shift_n 表示：信号延后了多少根K线用于对齐 returns
            shift_n = int(shift_n) + int(lookahead)
        except Exception:
            # 解析失败则回退默认
            return int(default_trade_lag)

        return max(0, int(shift_n))
    
    def prepare_cross_sectional_data(
        self, 
        data_dict: Dict[str, pd.DataFrame],
        factor_id: str,
        engine
    ) -> pd.DataFrame:
        """
        准备截面评估数据
        
        Args:
            data_dict: {symbol: market_data} 字典
            factor_id: 因子ID
            engine: 因子引擎
            
        Returns:
            DataFrame with columns: [date, symbol, factor_value, returns]
        """
        all_dfs = []
        trade_shift = self._resolve_signal_shift(factor_id, engine, default_trade_lag=1)
        cleaned_dict = {}
        for symbol, market_data in data_dict.items():
            if market_data is None or market_data.empty:
                continue
            cleaned = self._clean_market_data(market_data)
            if cleaned is not None and (not cleaned.empty):
                cleaned_dict[symbol] = cleaned
        filtered_dict = self._apply_liquidity_filter(cleaned_dict)
        
        for symbol, market_data in filtered_dict.items():
            if market_data is None or market_data.empty:
                continue
            
            try:
                market_data = market_data.copy().sort_index()
                # 未来收益：预测 t -> t+predict_step 的收益率
                market_data['future_returns'] = (
                    market_data['close'].pct_change(periods=self.predict_step).shift(-self.predict_step)
                )

                base_index = market_data.index
                use_same_tf = self.factor_timeframe == self.base_timeframe

                if use_same_tf:
                    factor_input = market_data
                    factor_on_base = self._to_series(engine.compute_single_factor(factor_id, factor_input))
                elif self.factor_bar_mode == 'intrabar_strict':
                    factor_on_base = self._build_intrabar_strict_factor_series(
                        market_data, factor_id, engine
                    )
                elif self.factor_bar_mode == 'intrabar':
                    factor_input = self._build_intrabar_snapshots(market_data, self.factor_timeframe)
                    factor_raw = self._to_series(engine.compute_single_factor(factor_id, factor_input))
                    factor_on_base = factor_raw.reindex(base_index) if factor_raw is not None else None
                else:
                    factor_input = self._resample_ohlcv_completed(market_data, self.factor_timeframe)
                    if factor_input is None or factor_input.empty:
                        continue
                    factor_raw = self._to_series(engine.compute_single_factor(factor_id, factor_input))
                    factor_on_base = factor_raw.reindex(base_index, method='ffill') if factor_raw is not None else None

                if factor_on_base is None:
                    continue

                # 统一在基础周期索引上对齐，再施加交易时滞
                common_idx = factor_on_base.index.intersection(market_data.index)
                if len(common_idx) == 0:
                    continue
                fv = factor_on_base.loc[common_idx].shift(trade_shift)
                fv = self._apply_outlier_treatment(fv)
                rt = market_data.loc[common_idx, 'future_returns']
                base_mask = rt.notna()
                if base_mask.sum() == 0:
                    continue
                eval_idx = common_idx[base_mask.values]
                sym_df = pd.DataFrame({
                    'date': eval_idx,
                    'symbol': symbol,
                    'factor_value': fv.loc[eval_idx].values,
                    'returns': rt.loc[eval_idx].values
                })
                all_dfs.append(sym_df)
            except Exception as e:
                print(f"处理 {symbol} 时出错: {e}")
                continue
        
        if not all_dfs:
            return pd.DataFrame()
        
        df = pd.concat(all_dfs, ignore_index=True)
        df['date'] = pd.to_datetime(df['date'])
        if self.sample_step > 1 and not df.empty:
            sampled_dates = pd.DatetimeIndex(sorted(df['date'].drop_duplicates())).to_series().iloc[::self.sample_step]
            df = df[df['date'].isin(sampled_dates.values)]
        return df
    
    def calculate_cross_sectional_ic(
        self, 
        cs_data: pd.DataFrame
    ) -> Dict[str, float]:
        """
        计算截面IC
        
        每个时间点计算因子值与收益率的截面相关性，然后取均值
        
        Args:
            cs_data: 截面数据 DataFrame
            
        Returns:
            Dict: 包含IC均值、IC标准差、ICIR等
        """
        ic_series = []
        date_counts = []
        coverage_series = []
        
        for date, group in cs_data.groupby('date'):
            if len(group) < 10:
                continue
            
            total_count = len(group)
            valid_mask = self._get_valid_signal_mask(group['factor_value'], group['returns'])
            valid_count = int(valid_mask.sum())
            coverage = (valid_count / total_count) if total_count > 0 else 0.0
            if coverage < self.min_coverage or valid_count < self.min_valid_count:
                continue
            group_valid = group.loc[valid_mask]
            factor_vals = group_valid['factor_value'].to_numpy(dtype=float)
            return_vals = group_valid['returns'].to_numpy(dtype=float)
            if len(factor_vals) < 10:
                continue

            factor_vals = self._normalize_factor_cross_sectional(factor_vals)

            fv_std = np.nanstd(factor_vals)
            rt_std = np.nanstd(return_vals)
            if (not np.isfinite(fv_std)) or (not np.isfinite(rt_std)) or fv_std <= 0 or rt_std <= 0:
                continue
            
            try:
                ic_pearson = np.corrcoef(factor_vals, return_vals)[0, 1]
                ic_spearman = stats.spearmanr(factor_vals, return_vals)[0]

                if not np.isfinite(ic_pearson):
                    ic_pearson = np.nan
                if not np.isfinite(ic_spearman):
                    ic_spearman = np.nan

                # 同一天两者都不可计算则跳过
                if not np.isfinite(ic_pearson) and not np.isfinite(ic_spearman):
                    continue
                
                ic_series.append({
                    'date': date,
                    'ic_pearson': ic_pearson,
                    'ic_spearman': ic_spearman,
                    'n_symbols': len(factor_vals),
                    'coverage': coverage,
                    'valid_count': valid_count,
                })
                date_counts.append(len(factor_vals))
                coverage_series.append(coverage)
            except Exception:
                continue
        
        if not ic_series:
            return {'ic_mean': np.nan, 'ic_std': np.nan, 'icir': np.nan}
        
        ic_df = pd.DataFrame(ic_series)
        
        ic_mean = ic_df['ic_pearson'].mean()
        ic_std = ic_df['ic_pearson'].std()
        # std 可能为 NaN（例如仅 1 个时间点），此时 IR 应保持为 NaN，
        # 否则会被错误显示为 0
        if ic_std is None or (not np.isfinite(ic_std)) or ic_std <= 0:
            icir = np.nan
        else:
            icir = ic_mean / ic_std
        
        rank_ic_mean = ic_df['ic_spearman'].mean()
        rank_ic_std = ic_df['ic_spearman'].std()
        if rank_ic_std is None or (not np.isfinite(rank_ic_std)) or rank_ic_std <= 0:
            rank_icir = np.nan
        else:
            rank_icir = rank_ic_mean / rank_ic_std
        
        # 仅统计可计算的 IC(非 NaN) 天数，避免 NaN 被当作 False(从而被计入 0)
        ic_pearson_vals = ic_df['ic_pearson'].to_numpy(dtype=float)
        finite_mask = np.isfinite(ic_pearson_vals)
        if finite_mask.sum() == 0:
            ic_positive_ratio = np.nan
        else:
            ic_positive_ratio = (ic_pearson_vals[finite_mask] > 0).mean()
        
        return {
            'ic_mean': ic_mean,
            'ic_std': ic_std,
            'icir': icir,
            'rank_ic_mean': rank_ic_mean,
            'rank_ic_std': rank_ic_std,
            'rank_icir': rank_icir,
            'ic_positive_ratio': ic_positive_ratio,
            'n_periods': len(ic_df),
            'avg_symbols_per_period': np.mean(date_counts),
            'avg_coverage': float(np.mean(coverage_series)) if coverage_series else np.nan,
        }
    
    def calculate_cross_sectional_returns(
        self, 
        cs_data: pd.DataFrame,
        timeframe: Optional[str] = None,
        transaction_cost: float = 0.001
    ) -> Dict[str, float]:
        """
        计算截面多空收益
        
        每个时间点按因子值分组，计算多空收益
        
        Args:
            cs_data: 截面数据 DataFrame
            timeframe: 时间框架
            transaction_cost: 单边交易成本费率（默认0.001），用于计算扣成本后指标
            
        Returns:
            Dict: 包含多空收益、胜率、夏普比率等
        """
        period_returns = []
        coverage_series = []
        monotonicity_values = []
        turnover_values = []
        fsc_values = []
        prev_long_symbols = None
        prev_short_symbols = None
        prev_exposure = None
        
        for date, group in cs_data.groupby('date'):
            if len(group) < self.n_groups:
                continue
            
            total_count = len(group)
            valid_mask = self._get_valid_signal_mask(group['factor_value'], group['returns'])
            valid_count = int(valid_mask.sum())
            coverage = (valid_count / total_count) if total_count > 0 else 0.0
            if coverage < self.min_coverage or valid_count < self.min_valid_count:
                continue
            if valid_count < self.n_groups * self.min_group_size:
                continue
            group = group.loc[valid_mask].copy()
            if len(group) < self.n_groups:
                continue
            
            try:
                normalized_factor = self._normalize_factor_cross_sectional(group['factor_value'].values)
                group['factor_normalized'] = normalized_factor
                group['factor_rank'] = pd.Series(
                    normalized_factor, index=group.index
                ).rank(ascending=True, method='first')
                group['group'] = pd.qcut(
                    group['factor_rank'], 
                    self.n_groups, 
                    labels=False, 
                    duplicates='drop'
                )
                
                group_returns = group.groupby('group')['returns'].mean()
                
                if len(group_returns) >= 2:
                    long_return = group_returns.iloc[-1]
                    short_return = group_returns.iloc[0]
                    long_short = long_return - short_return
                    long_group_id = group_returns.index[-1]
                    short_group_id = group_returns.index[0]
                    long_symbols = set(group.loc[group['group'] == long_group_id, 'symbol'].tolist())
                    short_symbols = set(group.loc[group['group'] == short_group_id, 'symbol'].tolist())

                    if prev_long_symbols is not None and prev_short_symbols is not None:
                        changed = len(long_symbols - prev_long_symbols) + len(short_symbols - prev_short_symbols)
                        base_size = max(len(long_symbols) + len(short_symbols), 1)
                        turnover_values.append(changed / base_size)
                    prev_long_symbols = long_symbols
                    prev_short_symbols = short_symbols

                    if len(group_returns) >= 3:
                        try:
                            mono = stats.spearmanr(
                                np.asarray(group_returns.index, dtype=float),
                                group_returns.values
                            )[0]
                            if np.isfinite(mono):
                                monotonicity_values.append(float(mono))
                        except Exception:
                            pass

                    if self.compute_fsc:
                        exposure = group.set_index('symbol')['factor_normalized'].astype(float)
                        if prev_exposure is not None:
                            common_symbols = prev_exposure.index.intersection(exposure.index)
                            if len(common_symbols) >= max(self.min_group_size, 5):
                                rho = prev_exposure.loc[common_symbols].corr(
                                    exposure.loc[common_symbols],
                                    method='spearman'
                                )
                                if np.isfinite(rho):
                                    fsc_values.append(float(rho))
                        prev_exposure = exposure
                    
                    period_returns.append({
                        'date': date,
                        'long_return': long_return,
                        'short_return': short_return,
                        'long_short_return': long_short,
                        'n_groups': len(group_returns),
                        'coverage': coverage,
                        'valid_count': valid_count,
                    })
                    coverage_series.append(coverage)
            except Exception as e:
                continue
        
        if not period_returns:
            return {
                'long_short_return': np.nan,
                'long_return': np.nan,
                'short_return': np.nan,
                'sharpe_ratio': np.nan,
                'win_rate': np.nan,
                'total_return': np.nan,
                'max_drawdown': np.nan,
            }
        
        returns_df = pd.DataFrame(period_returns)
        
        avg_long_short = returns_df['long_short_return'].mean()
        avg_long = returns_df['long_return'].mean()
        avg_short = returns_df['short_return'].mean()
        
        returns_std = returns_df['long_short_return'].std()
        # std 可能为 NaN（例如仅 1 个时间点），保持 NaN 以避免被误当成 0
        if returns_std is None or (not np.isfinite(returns_std)) or returns_std <= 0:
            sharpe_ratio = np.nan
        else:
            sharpe_ratio = avg_long_short / returns_std
        
        win_rate = (returns_df['long_short_return'] > 0).mean()
        
        periods_per_year = 365
        tf = str(timeframe or '').lower()
        if tf.endswith('h'):
            try:
                hours = float(tf[:-1] or 1)
                periods_per_year = (24 / hours) * 365
            except Exception:
                periods_per_year = 24 * 365
        elif tf.endswith('m'):
            try:
                minutes = float(tf[:-1] or 1)
                periods_per_year = (24 * 60 / minutes) * 365
            except Exception:
                periods_per_year = 24 * 60 * 365
        elif tf.endswith('d'):
            periods_per_year = 365
        elif tf.endswith('w'):
            periods_per_year = 52
        annualized_return = avg_long_short * periods_per_year
        annualized_std = returns_std * np.sqrt(periods_per_year)
        if annualized_std is None or (not np.isfinite(annualized_std)) or annualized_std <= 0:
            annualized_sharpe = np.nan
        else:
            annualized_sharpe = annualized_return / annualized_std

        cumulative = (1.0 + returns_df['long_short_return']).cumprod()
        total_return = float(cumulative.iloc[-1] - 1.0) if len(cumulative) > 0 else np.nan
        running_max = cumulative.cummax()
        drawdown = cumulative / running_max - 1.0
        max_drawdown = float(drawdown.min()) if len(drawdown) > 0 else np.nan

        turnover_mean = float(np.mean(turnover_values)) if turnover_values else np.nan
        monotonicity_mean = float(np.mean(monotonicity_values)) if monotonicity_values else np.nan
        monotonicity_abs_mean = float(np.mean(np.abs(monotonicity_values))) if monotonicity_values else np.nan
        monotonicity_positive_ratio = (
            float(np.mean(np.asarray(monotonicity_values) > 0)) if monotonicity_values else np.nan
        )
        fsc_mean = float(np.mean(fsc_values)) if fsc_values else np.nan
        
        result = {
            'long_short_return': avg_long_short,
            'long_return': avg_long,
            'short_return': avg_short,
            'sharpe_ratio': sharpe_ratio,
            'annualized_sharpe': annualized_sharpe,
            'win_rate': win_rate,
            'n_periods': len(returns_df),
            'returns_std': returns_std,
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'turnover': turnover_mean,
            'monotonicity_mean': monotonicity_mean,
            'monotonicity_abs_mean': monotonicity_abs_mean,
            'monotonicity_positive_ratio': monotonicity_positive_ratio,
            'fsc': fsc_mean,
            'avg_coverage': float(np.mean(coverage_series)) if coverage_series else np.nan,
        }

        if turnover_mean is not None and np.isfinite(turnover_mean) and turnover_mean > 0:
            turnover_cost_per_period = float(turnover_mean) * transaction_cost
        elif np.isfinite(avg_long_short):
            turnover_cost_per_period = 2.0 * transaction_cost
        else:
            turnover_cost_per_period = np.nan

        if np.isfinite(avg_long_short) and np.isfinite(turnover_cost_per_period):
            avg_ls_after_cost = float(avg_long_short) - turnover_cost_per_period
            result['long_short_return_after_cost'] = avg_ls_after_cost
            result['turnover_cost_per_period'] = turnover_cost_per_period

            if returns_std is not None and np.isfinite(returns_std) and returns_std > 0:
                result['sharpe_ratio_after_cost'] = avg_ls_after_cost / float(returns_std)
            else:
                result['sharpe_ratio_after_cost'] = np.nan

            ann_ret_after = avg_ls_after_cost * periods_per_year
            if annualized_std is not None and np.isfinite(annualized_std) and annualized_std > 0:
                result['annualized_sharpe_after_cost'] = ann_ret_after / float(annualized_std)
            else:
                result['annualized_sharpe_after_cost'] = np.nan

            net_returns = returns_df['long_short_return'] - turnover_cost_per_period
            cum_net = (1.0 + net_returns).cumprod()
            result['total_return_after_cost'] = float(cum_net.iloc[-1] - 1.0) if len(cum_net) > 0 else np.nan
            running_max_net = cum_net.cummax()
            dd_net = cum_net / running_max_net - 1.0
            result['max_drawdown_after_cost'] = float(dd_net.min()) if len(dd_net) > 0 else np.nan
            result['win_rate_after_cost'] = float((net_returns > 0).mean())
        else:
            result['long_short_return_after_cost'] = np.nan
            result['turnover_cost_per_period'] = np.nan
            result['sharpe_ratio_after_cost'] = np.nan
            result['annualized_sharpe_after_cost'] = np.nan
            result['total_return_after_cost'] = np.nan
            result['max_drawdown_after_cost'] = np.nan
            result['win_rate_after_cost'] = np.nan

        return result

    def calculate_cross_sectional_ic_decay(
        self,
        cs_data: pd.DataFrame,
        max_lag: Optional[int] = None
    ) -> Dict[str, Union[float, List[Dict[str, float]]]]:
        """
        计算截面 IC 衰减曲线（可选复杂指标）：
        对每个 lag，按 symbol 将 returns 向后平移，再做按日截面 Rank IC。
        """
        if max_lag is None:
            max_lag = self.ic_decay_max_lag
        max_lag = max(1, int(max_lag))
        curve = []

        base = cs_data[['date', 'symbol', 'factor_value', 'returns']].copy()
        base = base.sort_values(['symbol', 'date'])

        for lag in range(max_lag + 1):
            tmp = base.copy()
            if lag > 0:
                tmp['returns_lag'] = tmp.groupby('symbol')['returns'].shift(-lag)
            else:
                tmp['returns_lag'] = tmp['returns']
            tmp = tmp.dropna(subset=['factor_value', 'returns_lag'])

            lag_ics = []
            for _, group in tmp.groupby('date'):
                if len(group) < max(self.min_valid_count, 10):
                    continue
                valid = self._get_valid_signal_mask(group['factor_value'], group['returns_lag'])
                gv = group.loc[valid]
                if len(gv) < max(self.min_valid_count, 10):
                    continue
                fv = self._normalize_factor_cross_sectional(gv['factor_value'].to_numpy(dtype=float))
                rv = gv['returns_lag'].to_numpy(dtype=float)
                std_f = np.nanstd(fv)
                std_r = np.nanstd(rv)
                if (not np.isfinite(std_f)) or (not np.isfinite(std_r)) or std_f <= 0 or std_r <= 0:
                    continue
                ric = stats.spearmanr(fv, rv)[0]
                if np.isfinite(ric):
                    lag_ics.append(float(ric))

            rank_ic_mean = float(np.mean(lag_ics)) if lag_ics else np.nan
            curve.append({
                'lag': lag,
                'rank_ic_mean': rank_ic_mean,
                'n_periods': len(lag_ics),
            })

        finite_vals = [x['rank_ic_mean'] for x in curve if np.isfinite(x.get('rank_ic_mean', np.nan))]
        if len(finite_vals) >= 2:
            decay_rate = (finite_vals[0] - finite_vals[-1]) / max(len(finite_vals) - 1, 1)
        else:
            decay_rate = np.nan
        return {
            'curve': curve,
            'decay_rate': decay_rate,
        }
    
    def evaluate_cross_sectional(
        self, 
        data_dict: Dict[str, pd.DataFrame],
        factor_id: str,
        engine,
        timeframe: Optional[str] = None
    ) -> Dict:
        """
        综合截面评估
        
        Args:
            data_dict: {symbol: market_data} 字典
            factor_id: 因子ID
            engine: 因子引擎
            
        Returns:
            Dict: 综合评估结果
        """
        cs_data = self.prepare_cross_sectional_data(data_dict, factor_id, engine)
        
        if cs_data.empty:
            return {
                'success': False,
                'message': '无法准备截面数据',
                'factor_id': factor_id
            }
        
        ic_results = self.calculate_cross_sectional_ic(cs_data)
        return_results = self.calculate_cross_sectional_returns(cs_data, timeframe=timeframe, transaction_cost=self.transaction_cost)
        ic_decay_results = None
        if self.compute_ic_decay_curve:
            try:
                ic_decay_results = self.calculate_cross_sectional_ic_decay(
                    cs_data,
                    max_lag=self.ic_decay_max_lag
                )
            except Exception:
                ic_decay_results = None
        
        unique_symbols = cs_data['symbol'].nunique()
        unique_dates = cs_data['date'].nunique()
        
        n_periods_ic = ic_results.get('n_periods', 0) or 0
        n_periods_returns = return_results.get('n_periods', 0) or 0
        ic_skip_rate = (1 - n_periods_ic / unique_dates) if unique_dates > 0 else 0.0
        returns_skip_rate = (1 - n_periods_returns / unique_dates) if unique_dates > 0 else 0.0
        avg_coverage_ic = ic_results.get('avg_coverage', np.nan)
        avg_coverage_returns = return_results.get('avg_coverage', np.nan)
        
        return {
            'success': True,
            'factor_id': factor_id,
            'n_symbols': unique_symbols,
            'n_periods': unique_dates,
            'n_periods_total': unique_dates,
            'n_periods_ic': n_periods_ic,
            'n_periods_returns': n_periods_returns,
            'ic': ic_results,
            'returns': return_results,
            'ic_decay': ic_decay_results,
            'coverage': {
                'avg_coverage_ic': avg_coverage_ic,
                'avg_coverage_returns': avg_coverage_returns,
                'ic_skip_rate': ic_skip_rate,
                'returns_skip_rate': returns_skip_rate,
                'n_periods_total': unique_dates,
                'n_periods_ic': n_periods_ic,
                'n_periods_returns': n_periods_returns,
            },
            'summary': {
                'ic_mean': ic_results.get('ic_mean'),
                'icir': ic_results.get('icir'),
                'rank_ic_mean': ic_results.get('rank_ic_mean'),
                'long_short_return': return_results.get('long_short_return'),
                'long_short_return_after_cost': return_results.get('long_short_return_after_cost'),
                'sharpe_ratio': return_results.get('sharpe_ratio'),
                'sharpe_ratio_after_cost': return_results.get('sharpe_ratio_after_cost'),
                'win_rate': return_results.get('win_rate'),
                'win_rate_after_cost': return_results.get('win_rate_after_cost'),
                'max_drawdown': return_results.get('max_drawdown'),
                'max_drawdown_after_cost': return_results.get('max_drawdown_after_cost'),
                'total_return': return_results.get('total_return'),
                'total_return_after_cost': return_results.get('total_return_after_cost'),
                'turnover': return_results.get('turnover'),
                'turnover_cost_per_period': return_results.get('turnover_cost_per_period'),
                'fsc': return_results.get('fsc'),
                'monotonicity_abs_mean': return_results.get('monotonicity_abs_mean'),
                'avg_coverage': avg_coverage_returns,
                'skip_rate': returns_skip_rate,
            },
            'settings': {
                'predict_step': self.predict_step,
                'sample_step': self.sample_step,
                'base_timeframe': self.base_timeframe,
                'factor_timeframe': self.factor_timeframe,
                'factor_bar_mode': self.factor_bar_mode,
                'max_lookback': self.max_lookback,
                'min_coverage': self.min_coverage,
                'min_valid_count': self.min_valid_count,
                'min_group_size': self.min_group_size,
                'treat_zero_as_invalid': self.treat_zero_as_invalid,
                'enable_data_cleaning': self.enable_data_cleaning,
                'remove_zero_volume': self.remove_zero_volume,
                'liquidity_filter_ratio': self.liquidity_filter_ratio,
                'enable_outlier_treatment': self.enable_outlier_treatment,
                'outlier_method': self.outlier_method,
                'outlier_group_minutes': self.outlier_group_minutes,
                'outlier_mad_n': self.outlier_mad_n,
                'outlier_winsor_lower': self.outlier_winsor_lower,
                'outlier_winsor_upper': self.outlier_winsor_upper,
                'compute_fsc': self.compute_fsc,
                'compute_ic_decay_curve': self.compute_ic_decay_curve,
                'ic_decay_max_lag': self.ic_decay_max_lag,
            },
        }


def evaluate_cross_sectional(
    data_dict: Dict[str, pd.DataFrame],
    factor_id: str,
    engine,
    n_groups: int = 5,
    timeframe: Optional[str] = None,
    normalize_method: str = 'rank',
    predict_step: int = 1,
    sample_step: int = 1,
    factor_timeframe: Optional[str] = None,
    factor_bar_mode: str = 'completed',
    max_lookback: int = 200,
    min_coverage: float = 0.3,
    min_valid_count: int = 30,
    min_group_size: int = 5,
    treat_zero_as_invalid: bool = True,
    enable_data_cleaning: bool = False,
    remove_zero_volume: bool = True,
    liquidity_filter_ratio: float = 0.5,
    enable_outlier_treatment: bool = False,
    outlier_method: str = 'mad',
    outlier_group_minutes: int = 30,
    outlier_mad_n: float = 5.0,
    outlier_winsor_lower: float = 0.01,
    outlier_winsor_upper: float = 0.99,
    compute_fsc: bool = False,
    compute_ic_decay_curve: bool = False,
    ic_decay_max_lag: int = 5,
) -> Dict:
    """
    便捷函数：截面因子评估
    
    Args:
        data_dict: {symbol: market_data} 字典
        factor_id: 因子ID
        engine: 因子引擎
        n_groups: 分组数量
        timeframe: 时间框架
        normalize_method: 因子标准化方法 ('rank', 'rank_centered', 'none')
        predict_step: 预测时间步（N），收益定义为 t -> t+N
        sample_step: 采样时间步（M），每隔 M 个时间点参与评估
        factor_timeframe: 因子计算时间框架（为空则与 timeframe 相同）
        factor_bar_mode: 因子K线口径 ('completed' | 'intrabar' | 'intrabar_strict')
        max_lookback: intrabar_strict 模式回放时保留的最大历史bar数量
        min_coverage: 每个截面的最小覆盖率阈值（有效样本/截面样本）
        min_valid_count: 每个截面的最小有效样本数阈值
        min_group_size: 分组评估时每组最小样本数
        treat_zero_as_invalid: 是否将 0 值视为无效信号（稀疏因子推荐开启）
        
    Returns:
        Dict: 评估结果
    """
    evaluator = CrossSectionalEvaluator(
        n_groups=n_groups,
        normalize_method=normalize_method,
        predict_step=predict_step,
        sample_step=sample_step,
        base_timeframe=(timeframe or '1h'),
        factor_timeframe=factor_timeframe,
        factor_bar_mode=factor_bar_mode,
        max_lookback=max_lookback,
        min_coverage=min_coverage,
        min_valid_count=min_valid_count,
        min_group_size=min_group_size,
        treat_zero_as_invalid=treat_zero_as_invalid,
        enable_data_cleaning=enable_data_cleaning,
        remove_zero_volume=remove_zero_volume,
        liquidity_filter_ratio=liquidity_filter_ratio,
        enable_outlier_treatment=enable_outlier_treatment,
        outlier_method=outlier_method,
        outlier_group_minutes=outlier_group_minutes,
        outlier_mad_n=outlier_mad_n,
        outlier_winsor_lower=outlier_winsor_lower,
        outlier_winsor_upper=outlier_winsor_upper,
        compute_fsc=compute_fsc,
        compute_ic_decay_curve=compute_ic_decay_curve,
        ic_decay_max_lag=ic_decay_max_lag,
    )
    return evaluator.evaluate_cross_sectional(data_dict, factor_id, engine, timeframe=timeframe)
