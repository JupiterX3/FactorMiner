"""
遗传编程截面因子挖掘器
借鉴AlphaGPT的算子组合思想，使用遗传编程(GP)在多币种截面空间中搜索最优因子表达式

核心流程：
1. 加载多币种数据，预计算基础特征
2. 随机生成初始种群（表达式树）
3. 截面评估每个个体的适应度（IC/IR）
4. 选择/交叉/变异进化
5. 返回Top-K因子

与AlphaGPT的区别：
- AlphaGPT用RL(Transformer)生成公式，本模块用GP
- AlphaGPT用回测PnL评估，本模块用截面IC/IR评估
- AlphaGPT面向Meme币，本模块面向通用加密货币
"""

import pandas as pd
import numpy as np
import copy
import random
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class NodeType(Enum):
    FEATURE = "feature"
    OPERATOR = "operator"
    CONSTANT = "constant"


@dataclass
class OpSpec:
    name: str
    arity: int
    func: Callable
    is_cross_sectional: bool = False
    is_time_series: bool = False


FEATURES_CONFIG = [
    {"name": "returns", "desc": "收益率"},
    {"name": "log_ret", "desc": "对数收益率"},
    {"name": "volatility", "desc": "波动率"},
    {"name": "volume_ratio", "desc": "成交量比率"},
    {"name": "price_position", "desc": "价格位置"},
    {"name": "momentum", "desc": "动量"},
    {"name": "high_low_range", "desc": "振幅"},
    {"name": "close_open_diff", "desc": "收盘-开盘差"},
    {"name": "volume_price_corr", "desc": "量价相关"},
    {"name": "ma_deviation", "desc": "均线偏离"},
]

CONSTANTS_POOL = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 60.0, -1.0, 0.01, 0.1]


def _ts_delay(x: pd.Series, d: int = 1) -> pd.Series:
    return x.shift(d)


def _ts_mean(x: pd.Series, w: int = 20) -> pd.Series:
    return x.rolling(window=w, min_periods=1).mean()


def _ts_std(x: pd.Series, w: int = 20) -> pd.Series:
    return x.rolling(window=w, min_periods=1).std()


def _ts_max(x: pd.Series, w: int = 20) -> pd.Series:
    return x.rolling(window=w, min_periods=1).max()


def _ts_min(x: pd.Series, w: int = 20) -> pd.Series:
    return x.rolling(window=w, min_periods=1).min()


def _ts_rank(x: pd.Series, w: int = 20) -> pd.Series:
    return x.rolling(window=w, min_periods=1).rank(pct=True)


def _ts_zscore(x: pd.Series, w: int = 20) -> pd.Series:
    m = x.rolling(window=w, min_periods=1).mean()
    s = x.rolling(window=w, min_periods=1).std()
    return (x - m) / (s + 1e-8)


def _ts_decay(x: pd.Series, w: int = 5) -> pd.Series:
    weights = np.array([0.9 ** i for i in range(w)])[::-1]
    weights = weights / weights.sum()
    return x.rolling(window=w, min_periods=1).apply(lambda v: np.dot(v, weights[:len(v)]), raw=True)


def _ts_momentum(x: pd.Series, w: int = 10) -> pd.Series:
    return x / x.shift(w) - 1


def _ts_skewness(x: pd.Series, w: int = 20) -> pd.Series:
    return x.rolling(window=w, min_periods=max(3, w // 2)).skew()


def _ts_kurtosis(x: pd.Series, w: int = 20) -> pd.Series:
    return x.rolling(window=w, min_periods=max(4, w // 2)).kurt()


def _ts_corr(x: pd.Series, y: pd.Series, w: int = 20) -> pd.Series:
    return x.rolling(window=w, min_periods=2).corr(y)


def _ts_cov(x: pd.Series, y: pd.Series, w: int = 20) -> pd.Series:
    return x.rolling(window=w, min_periods=2).cov(y)


def _cs_rank(x: pd.Series) -> pd.Series:
    return x.rank(pct=True)


def _cs_zscore(x: pd.Series) -> pd.Series:
    m = x.mean()
    s = x.std()
    if s < 1e-8:
        return x * 0
    return (x - m) / s


def _cs_mad_norm(x: pd.Series) -> pd.Series:
    med = x.median()
    mad = (x - med).abs().median()
    if mad < 1e-8:
        return x * 0
    return ((x - med) / mad).clip(-5, 5)


def _safe_add(x, y):
    return x + y


def _safe_sub(x, y):
    return x - y


def _safe_mul(x, y):
    return x * y


def _safe_div(x, y):
    return x / (y + 1e-8)


def _safe_abs(x):
    return x.abs()


def _safe_neg(x):
    return -x


def _safe_sqrt(x):
    return np.sqrt(x.abs())


def _safe_log(x):
    return np.log(x.abs() + 1e-8)


def _safe_sign(x):
    return np.sign(x)


def _safe_max(x, y):
    return np.maximum(x, y)


def _safe_min(x, y):
    return np.minimum(x, y)


def _safe_gate(x, y, z):
    return np.where(x > 0, y, z)


OPS_REGISTRY: Dict[str, OpSpec] = {
    "add": OpSpec("add", 2, _safe_add),
    "sub": OpSpec("sub", 2, _safe_sub),
    "mul": OpSpec("mul", 2, _safe_mul),
    "div": OpSpec("div", 2, _safe_div),
    "max": OpSpec("max", 2, _safe_max),
    "min": OpSpec("min", 2, _safe_min),
    "abs": OpSpec("abs", 1, _safe_abs),
    "neg": OpSpec("neg", 1, _safe_neg),
    "sqrt": OpSpec("sqrt", 1, _safe_sqrt),
    "log": OpSpec("log", 1, _safe_log),
    "sign": OpSpec("sign", 1, _safe_sign),
    "delay": OpSpec("delay", 1, _ts_delay, is_time_series=True),
    "ts_mean": OpSpec("ts_mean", 1, _ts_mean, is_time_series=True),
    "ts_std": OpSpec("ts_std", 1, _ts_std, is_time_series=True),
    "ts_max": OpSpec("ts_max", 1, _ts_max, is_time_series=True),
    "ts_min": OpSpec("ts_min", 1, _ts_min, is_time_series=True),
    "ts_rank": OpSpec("ts_rank", 1, _ts_rank, is_time_series=True),
    "ts_zscore": OpSpec("ts_zscore", 1, _ts_zscore, is_time_series=True),
    "ts_decay": OpSpec("ts_decay", 1, _ts_decay, is_time_series=True),
    "ts_momentum": OpSpec("ts_momentum", 1, _ts_momentum, is_time_series=True),
    "ts_skewness": OpSpec("ts_skewness", 1, _ts_skewness, is_time_series=True),
    "ts_kurtosis": OpSpec("ts_kurtosis", 1, _ts_kurtosis, is_time_series=True),
    "ts_corr": OpSpec("ts_corr", 2, _ts_corr, is_time_series=True),
    "ts_cov": OpSpec("ts_cov", 2, _ts_cov, is_time_series=True),
    "cs_rank": OpSpec("cs_rank", 1, _cs_rank, is_cross_sectional=True),
    "cs_zscore": OpSpec("cs_zscore", 1, _cs_zscore, is_cross_sectional=True),
    "cs_mad_norm": OpSpec("cs_mad_norm", 1, _cs_mad_norm, is_cross_sectional=True),
    "gate": OpSpec("gate", 3, _safe_gate),
}

ARITH_OPS = ["add", "sub", "mul", "div", "max", "min"]
UNARY_OPS = ["abs", "neg", "sqrt", "log", "sign"]
TS_OPS = ["delay", "ts_mean", "ts_std", "ts_max", "ts_min", "ts_rank",
          "ts_zscore", "ts_decay", "ts_momentum", "ts_skewness", "ts_kurtosis"]
CS_OPS = ["cs_rank", "cs_zscore", "cs_mad_norm"]
BINARY_TS_OPS = ["ts_corr", "ts_cov"]
TERNARY_OPS = ["gate"]


class ExpressionNode:
    """表达式树节点"""

    def __init__(self, node_type: NodeType, value: Any = None,
                 op_name: str = None, children: List['ExpressionNode'] = None,
                 window: int = None):
        self.node_type = node_type
        self.value = value
        self.op_name = op_name
        self.children = children or []
        self.window = window

    def __deepcopy__(self, memo):
        new_node = ExpressionNode(
            node_type=self.node_type,
            value=self.value,
            op_name=self.op_name,
            children=[copy.deepcopy(c, memo) for c in self.children],
            window=self.window
        )
        return new_node

    def depth(self) -> int:
        if not self.children:
            return 1
        return 1 + max(c.depth() for c in self.children)

    def size(self) -> int:
        return 1 + sum(c.size() for c in self.children)

    def to_string(self) -> str:
        if self.node_type == NodeType.FEATURE:
            return str(self.value)
        elif self.node_type == NodeType.CONSTANT:
            return str(self.value)
        elif self.node_type == NodeType.OPERATOR:
            op = OPS_REGISTRY.get(self.op_name)
            if not op:
                return "?"

            win_str = f"(w={self.window})" if self.window else ""

            if op.arity == 1:
                child_str = self.children[0].to_string() if self.children else "?"
                return f"{self.op_name}{win_str}({child_str})"
            elif op.arity == 2:
                left = self.children[0].to_string() if len(self.children) > 0 else "?"
                right = self.children[1].to_string() if len(self.children) > 1 else "?"
                if self.op_name in ARITH_OPS:
                    return f"({left} {self.op_name} {right})"
                return f"{self.op_name}{win_str}({left}, {right})"
            elif op.arity == 3:
                a = self.children[0].to_string() if len(self.children) > 0 else "?"
                b = self.children[1].to_string() if len(self.children) > 1 else "?"
                c = self.children[2].to_string() if len(self.children) > 2 else "?"
                return f"gate({a}, {b}, {c})"
            return self.op_name
        return "?"

    def __repr__(self):
        return self.to_string()


class ExpressionTree:
    """表达式树（GP个体）"""

    def __init__(self, root: ExpressionNode = None, fitness: float = None):
        self.root = root
        self.fitness = fitness
        self.ic_mean = None
        self.ic_std = None
        self.icir = None
        self.rank_ic_mean = None
        self.rank_icir = None
        self.long_short_return = None
        self.eval_detail = None

    def __deepcopy__(self, memo):
        new_tree = ExpressionTree(
            root=copy.deepcopy(self.root, memo),
            fitness=self.fitness
        )
        new_tree.ic_mean = self.ic_mean
        new_tree.ic_std = self.ic_std
        new_tree.icir = self.icir
        new_tree.rank_ic_mean = self.rank_ic_mean
        new_tree.rank_icir = self.rank_icir
        new_tree.long_short_return = self.long_short_return
        new_tree.eval_detail = self.eval_detail
        return new_tree

    def depth(self) -> int:
        return self.root.depth() if self.root else 0

    def size(self) -> int:
        return self.root.size() if self.root else 0

    def to_string(self) -> str:
        return self.root.to_string() if self.root else "empty"

    def __repr__(self):
        fit_str = f", fitness={self.fitness:.4f}" if self.fitness is not None else ""
        return f"Expr({self.to_string()}{fit_str})"


class FeatureEngine:
    """基础特征计算引擎"""

    @staticmethod
    def compute_features(data: pd.DataFrame) -> Dict[str, pd.Series]:
        features = {}
        c = data['close']
        o = data['open']
        h = data['high']
        l = data['low']
        v = data['volume']

        features['returns'] = c.pct_change()
        features['log_ret'] = np.log(c / (c.shift(1) + 1e-8))
        features['volatility'] = features['returns'].rolling(20, min_periods=1).std()
        features['volume_ratio'] = v / (v.rolling(20, min_periods=1).mean() + 1e-8)
        features['price_position'] = (c - l.rolling(20, min_periods=1).min()) / \
            (h.rolling(20, min_periods=1).max() - l.rolling(20, min_periods=1).min() + 1e-8)
        features['momentum'] = c / (c.shift(10) + 1e-8) - 1
        features['high_low_range'] = (h - l) / (c + 1e-8)
        features['close_open_diff'] = (c - o) / (o + 1e-8)
        features['volume_price_corr'] = features['returns'].rolling(20, min_periods=2).corr(v.pct_change())
        features['ma_deviation'] = (c - c.rolling(20, min_periods=1).mean()) / (c.rolling(20, min_periods=1).std() + 1e-8)

        for key in features:
            features[key] = features[key].replace([np.inf, -np.inf], np.nan).fillna(0)

        return features


class GPMiner:
    """
    遗传编程截面因子挖掘器

    借鉴AlphaGPT的"基础特征+算子组合"范式，使用GP在多币种截面空间中搜索因子表达式
    """

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.population_size = self.config.get('population_size', 200)
        self.max_generations = self.config.get('max_generations', 30)
        self.max_depth = self.config.get('max_depth', 5)
        self.crossover_rate = self.config.get('crossover_rate', 0.7)
        self.mutation_rate = self.config.get('mutation_rate', 0.2)
        self.elitism_rate = self.config.get('elitism_rate', 0.1)
        self.tournament_size = self.config.get('tournament_size', 5)
        self.min_ic_threshold = self.config.get('min_ic', 0.02)
        self.min_ir_threshold = self.config.get('min_ir', 0.1)
        self.min_coverage_threshold = float(self.config.get('min_coverage', 0.2))
        self.max_factor_count = self.config.get('max_factors', 15)
        self.max_correlation = self.config.get('max_correlation', 0.7)
        self.stagnation_limit = self.config.get('stagnation_limit', 5)
        self.eval_workers = max(1, int(self.config.get('eval_workers', 1)))
        self.feature_names = [f['name'] for f in FEATURES_CONFIG]
        self.population: List[ExpressionTree] = []
        self.best_individuals: List[ExpressionTree] = []
        self.generation_stats: List[Dict] = []
        self._progress_callback = None
        self._stop_requested = False
        self._eval_cache: Dict[str, Dict] = {}

    def set_progress_callback(self, callback: Callable):
        self._progress_callback = callback

    def request_stop(self):
        self._stop_requested = True

    def _report_progress(self, progress_pct: int, message: str, detail: Dict = None):
        if self._progress_callback:
            progress = max(0, min(int(progress_pct), 100))
            self._progress_callback(progress, message, detail or {})

    def _random_window(self) -> int:
        return random.choice([5, 10, 20, 30, 60])

    def _random_constant(self) -> float:
        return random.choice(CONSTANTS_POOL)

    def _random_feature(self) -> str:
        return random.choice(self.feature_names)

    def _random_operator(self, allow_cs: bool = True, allow_ts: bool = True) -> str:
        pool = list(ARITH_OPS) + list(UNARY_OPS)
        if allow_ts:
            pool += list(TS_OPS) + list(BINARY_TS_OPS)
        if allow_cs:
            pool += list(CS_OPS)
        pool += list(TERNARY_OPS)
        return random.choice(pool)

    def _create_random_node(self, max_depth: int, allow_cs: bool = True,
                            allow_ts: bool = True) -> ExpressionNode:
        if max_depth <= 1 or random.random() < 0.3:
            if random.random() < 0.8:
                return ExpressionNode(NodeType.FEATURE, value=self._random_feature())
            else:
                return ExpressionNode(NodeType.CONSTANT, value=self._random_constant())

        op_name = self._random_operator(allow_cs=allow_cs, allow_ts=allow_ts)
        op = OPS_REGISTRY[op_name]
        children = []
        for _ in range(op.arity):
            child = self._create_random_node(
                max_depth - 1,
                allow_cs=allow_cs and op_name not in CS_OPS,
                allow_ts=allow_ts
            )
            children.append(child)

        window = None
        if op.is_time_series:
            window = self._random_window()

        return ExpressionNode(
            node_type=NodeType.OPERATOR,
            op_name=op_name,
            children=children,
            window=window
        )

    def _create_random_individual(self) -> ExpressionTree:
        depth = random.randint(2, self.max_depth)
        root = self._create_random_node(depth)
        return ExpressionTree(root=root)

    def _initialize_population(self):
        self.population = [self._create_random_individual() for _ in range(self.population_size)]

    def _evaluate_node(self, node: ExpressionNode,
                       features: Dict[str, pd.Series],
                       node_cache: Dict[int, pd.Series] = None) -> pd.Series:
        if node_cache is None:
            node_cache = {}
        node_id = id(node)
        cached = node_cache.get(node_id)
        if cached is not None:
            return cached

        if node.node_type == NodeType.FEATURE:
            result = features.get(node.value, pd.Series(0, dtype=float))
            node_cache[node_id] = result
            return result
        elif node.node_type == NodeType.CONSTANT:
            idx = next(iter(features.values())).index
            result = pd.Series(float(node.value), index=idx, dtype=float)
            node_cache[node_id] = result
            return result
        elif node.node_type == NodeType.OPERATOR:
            op = OPS_REGISTRY.get(node.op_name)
            if not op:
                idx = next(iter(features.values())).index
                result = pd.Series(0, dtype=float, index=idx)
                node_cache[node_id] = result
                return result

            child_values = [self._evaluate_node(c, features, node_cache) for c in node.children]

            while len(child_values) < op.arity:
                idx = next(iter(features.values())).index
                child_values.append(pd.Series(0, dtype=float, index=idx))

            try:
                if op.is_time_series and node.window:
                    if op.arity == 1:
                        result = op.func(child_values[0], node.window)
                    elif op.arity == 2:
                        result = op.func(child_values[0], child_values[1], node.window)
                    else:
                        result = op.func(*child_values)
                else:
                    result = op.func(*child_values)

                if isinstance(result, (int, float)):
                    idx = next(iter(features.values())).index
                    result = pd.Series(float(result), index=idx, dtype=float)

                result = result.replace([np.inf, -np.inf], np.nan).fillna(0)
                node_cache[node_id] = result
                return result
            except Exception:
                idx = next(iter(features.values())).index
                result = pd.Series(0, dtype=float, index=idx)
                node_cache[node_id] = result
                return result

        idx = next(iter(features.values())).index
        result = pd.Series(0, dtype=float, index=idx)
        node_cache[node_id] = result
        return result

    def _evaluate_individual_on_symbol(self, individual: ExpressionTree,
                                       features: Dict[str, pd.Series]) -> pd.Series:
        node_cache: Dict[int, pd.Series] = {}
        return self._evaluate_node(individual.root, features, node_cache)

    def _evaluate_cross_sectional(self, individual: ExpressionTree,
                                  all_features: Dict[str, Dict[str, pd.Series]],
                                  all_returns: Dict[str, pd.Series]) -> Dict:
        factor_values = {}
        for symbol, features in all_features.items():
            try:
                fv = self._evaluate_individual_on_symbol(individual, features)
                if fv is not None and len(fv) > 0:
                    factor_values[symbol] = fv
            except Exception:
                continue

        if len(factor_values) < 3:
            return {
                'ic_mean': 0.0, 'ic_std': 1.0, 'icir': 0.0,
                'rank_ic_mean': 0.0, 'rank_icir': 0.0,
                'long_short_return': 0.0, 'n_symbols': len(factor_values),
                'n_periods': 0, 'total_periods': 0, 'coverage_rate': 0.0, 'fitness': -999.0
            }

        valid_symbols = sorted(set(factor_values.keys()) & set(all_returns.keys()))
        if len(valid_symbols) < 3:
            return {
                'ic_mean': 0.0, 'ic_std': 1.0, 'icir': 0.0,
                'rank_ic_mean': 0.0, 'rank_icir': 0.0,
                'long_short_return': 0.0, 'n_symbols': len(factor_values),
                'n_periods': 0, 'total_periods': 0, 'coverage_rate': 0.0, 'fitness': -999.0
            }

        factor_df = pd.concat([factor_values[s] for s in valid_symbols], axis=1, keys=valid_symbols)
        ret_df = pd.concat([all_returns[s] for s in valid_symbols], axis=1, keys=valid_symbols)

        # 将截面计算对齐到同一时间网格，避免逐行构造DataFrame再groupby的高开销
        aligned_idx = factor_df.index.intersection(ret_df.index)
        if len(aligned_idx) < 50:
            return {
                'ic_mean': 0.0, 'ic_std': 1.0, 'icir': 0.0,
                'rank_ic_mean': 0.0, 'rank_icir': 0.0,
                'long_short_return': 0.0, 'n_symbols': len(factor_values),
                'n_periods': 0, 'total_periods': 0, 'coverage_rate': 0.0, 'fitness': -999.0
            }

        factor_mat = factor_df.loc[aligned_idx].to_numpy(dtype=float)
        ret_mat = ret_df.loc[aligned_idx].to_numpy(dtype=float)

        ic_list = []
        rank_ic_list = []
        ls_returns = []

        for t in range(factor_mat.shape[0]):
            fv_row = factor_mat[t]
            rv_row = ret_mat[t]
            valid_mask = np.isfinite(fv_row) & np.isfinite(rv_row)
            if valid_mask.sum() < 3:
                continue
            fv = fv_row[valid_mask]
            rv = rv_row[valid_mask]

            fv_std = np.std(fv)
            rv_std = np.std(rv)
            if fv_std < 1e-10 or rv_std < 1e-10:
                continue

            try:
                ic = np.corrcoef(fv, rv)[0, 1]
                fv_rank = np.argsort(np.argsort(fv)).astype(float) + 1.0
                rv_rank = np.argsort(np.argsort(rv)).astype(float) + 1.0
                rank_ic = np.corrcoef(fv_rank, rv_rank)[0, 1]

                if np.isfinite(ic):
                    ic_list.append(ic)
                if np.isfinite(rank_ic):
                    rank_ic_list.append(rank_ic)

                ranked = fv_rank / max(len(fv_rank), 1)
                long_mask = ranked > 0.8
                short_mask = ranked < 0.2
                if long_mask.sum() > 0 and short_mask.sum() > 0:
                    ls_ret = rv[long_mask].mean() - rv[short_mask].mean()
                    ls_returns.append(ls_ret)
            except Exception:
                continue

        if not ic_list:
            return {
                'ic_mean': 0.0, 'ic_std': 1.0, 'icir': 0.0,
                'rank_ic_mean': 0.0, 'rank_icir': 0.0,
                'long_short_return': 0.0, 'n_symbols': len(factor_values),
                'n_periods': 0, 'total_periods': len(aligned_idx), 'coverage_rate': 0.0, 'fitness': -999.0
            }

        ic_mean = np.mean(ic_list)
        ic_std = np.std(ic_list) if len(ic_list) > 1 else 1.0
        icir = ic_mean / ic_std if ic_std > 1e-8 else 0.0

        rank_ic_mean = np.mean(rank_ic_list) if rank_ic_list else 0.0
        rank_ic_std = np.std(rank_ic_list) if len(rank_ic_list) > 1 else 1.0
        rank_icir = rank_ic_mean / rank_ic_std if rank_ic_std > 1e-8 else 0.0

        ls_return = np.mean(ls_returns) if ls_returns else 0.0

        # 方向无关的适应度：正向/反向因子都可通过绝对预测力参与竞争
        fitness = abs(icir) + 0.3 * abs(rank_icir)

        return {
            'ic_mean': float(ic_mean),
            'ic_std': float(ic_std),
            'icir': float(icir),
            'rank_ic_mean': float(rank_ic_mean),
            'rank_icir': float(rank_icir),
            'long_short_return': float(ls_return),
            'n_symbols': len(factor_values),
            'n_periods': len(ic_list),
            'total_periods': len(aligned_idx),
            'coverage_rate': float(len(ic_list) / max(len(aligned_idx), 1)),
            'fitness': float(fitness)
        }

    def _tournament_select(self) -> ExpressionTree:
        candidates = random.sample(self.population, min(self.tournament_size, len(self.population)))
        best = max(candidates, key=lambda x: x.fitness if x.fitness is not None else -999)
        return best

    def _get_all_nodes(self, node: ExpressionNode) -> List[Tuple[ExpressionNode, ExpressionNode, int]]:
        result = []
        for i, child in enumerate(node.children):
            result.append((node, child, i))
            result.extend(self._get_all_nodes(child))
        return result

    def _crossover(self, parent1: ExpressionTree, parent2: ExpressionTree) -> Tuple[ExpressionTree, ExpressionTree]:
        child1 = copy.deepcopy(parent1)
        child2 = copy.deepcopy(parent2)

        nodes1 = self._get_all_nodes(child1.root) if child1.root.children else []
        nodes2 = self._get_all_nodes(child2.root) if child2.root.children else []

        if not nodes1 or not nodes2:
            return child1, child2

        p1, n1, idx1 = random.choice(nodes1)
        p2, n2, idx2 = random.choice(nodes2)

        new_depth1 = n2.depth() + (child1.root.depth() - n1.depth())
        new_depth2 = n1.depth() + (child2.root.depth() - n2.depth())

        if new_depth1 <= self.max_depth + 2 and new_depth2 <= self.max_depth + 2:
            p1.children[idx1] = copy.deepcopy(n2)
            p2.children[idx2] = copy.deepcopy(n1)

        return child1, child2

    def _mutate(self, individual: ExpressionTree) -> ExpressionTree:
        mutant = copy.deepcopy(individual)

        if not mutant.root.children:
            mutant.root = self._create_random_node(random.randint(2, self.max_depth))
            return mutant

        all_nodes = self._get_all_nodes(mutant.root)
        if not all_nodes:
            return mutant

        parent, node, idx = random.choice(all_nodes)

        mutation_type = random.random()

        if mutation_type < 0.4:
            parent.children[idx] = self._create_random_node(random.randint(1, 3))
        elif mutation_type < 0.7:
            if node.node_type == NodeType.OPERATOR and node.op_name in TS_OPS:
                node.window = self._random_window()
            elif node.node_type == NodeType.FEATURE:
                node.value = self._random_feature()
            elif node.node_type == NodeType.CONSTANT:
                node.value = self._random_constant()
        else:
            if node.node_type == NodeType.OPERATOR:
                old_arity = OPS_REGISTRY[node.op_name].arity
                same_arity_ops = [op for op, spec in OPS_REGISTRY.items()
                                  if spec.arity == old_arity and op != node.op_name]
                if same_arity_ops:
                    node.op_name = random.choice(same_arity_ops)

        return mutant

    def _select_diverse_best(self, candidates: List[ExpressionTree],
                             all_features: Dict[str, Dict[str, pd.Series]],
                             max_count: int) -> List[ExpressionTree]:
        if not candidates:
            return []

        factor_series_cache: Dict[str, Dict[str, pd.Series]] = {}

        def _get_factor_values(individual: ExpressionTree) -> Dict[str, pd.Series]:
            expr_key = individual.to_string()
            cached_values = factor_series_cache.get(expr_key)
            if cached_values is not None:
                return cached_values

            factor_values = {}
            for symbol, features in all_features.items():
                try:
                    fv = self._evaluate_individual_on_symbol(individual, features)
                    factor_values[symbol] = fv
                except Exception:
                    continue
            factor_series_cache[expr_key] = factor_values
            return factor_values

        selected = [candidates[0]]
        candidate_factors = {}
        first_fv = _get_factor_values(candidates[0])
        candidate_factors[candidates[0].to_string()] = first_fv

        for cand in candidates[1:]:
            if len(selected) >= max_count:
                break

            cand_fv = _get_factor_values(cand)

            is_diverse = True
            for existing_fv in candidate_factors.values():
                corr_sum = 0.0
                corr_count = 0
                for symbol in set(cand_fv.keys()) & set(existing_fv.keys()):
                    c1 = cand_fv.get(symbol)
                    c2 = existing_fv.get(symbol)
                    if c1 is not None and c2 is not None and len(c1) > 10:
                        common_idx = c1.index.intersection(c2.index)
                        if len(common_idx) > 10:
                            v1 = c1.loc[common_idx].to_numpy(dtype=float)
                            v2 = c2.loc[common_idx].to_numpy(dtype=float)
                            valid = np.isfinite(v1) & np.isfinite(v2)
                            if valid.sum() < 3:
                                continue
                            x = v1[valid]
                            y = v2[valid]
                            x = x - x.mean()
                            y = y - y.mean()
                            x_std = x.std()
                            y_std = y.std()
                            if x_std < 1e-12 or y_std < 1e-12:
                                continue
                            corr = float((x * y).mean() / (x_std * y_std + 1e-12))
                            if np.isfinite(corr):
                                corr_sum += abs(corr)
                                corr_count += 1
                                if (corr_sum / corr_count) > self.max_correlation:
                                    is_diverse = False
                                    break
                if not is_diverse:
                    is_diverse = False
                    break

            if is_diverse:
                selected.append(cand)
                candidate_factors[cand.to_string()] = cand_fv

        return selected

    def mine(self, data_dict: Dict[str, pd.DataFrame],
             progress_callback: Callable = None) -> Dict:
        """
        执行GP截面因子挖掘

        Args:
            data_dict: {symbol: market_data} 多币种数据字典
            progress_callback: 进度回调函数 callback(progress_pct, message, detail)

        Returns:
            Dict: 挖掘结果
        """
        self._stop_requested = False
        self._eval_cache = {}
        self.set_progress_callback(progress_callback)

        self._report_progress(0, "正在计算基础特征...")

        all_features = {}
        all_returns = {}
        for symbol, data in data_dict.items():
            if data is None or data.empty:
                continue
            features = FeatureEngine.compute_features(data)
            all_features[symbol] = features
            all_returns[symbol] = data['close'].pct_change().shift(-1).fillna(0)

        if len(all_features) < 3:
            return {
                'success': False,
                'error': f'有效币种数量不足（{len(all_features)}个），截面挖掘至少需要3个币种',
                'factors': []
            }

        self._report_progress(2, f"已加载 {len(all_features)} 个币种的特征数据，开始初始化种群...")

        self._initialize_population()

        self._report_progress(5, f"种群初始化完成（{self.population_size}个个体），开始进化...")

        best_fitness_ever = -999.0
        stagnation_count = 0
        stopped_early = False
        completed_generations = 0

        for gen in range(self.max_generations):
            if self._stop_requested:
                stop_pct = int((gen + 1) / self.max_generations * 85)
                self._report_progress(stop_pct, "用户请求停止挖掘")
                stopped_early = True
                completed_generations = gen
                break

            gen_start = time.time()

            expr_to_inds: Dict[str, List[ExpressionTree]] = {}
            expr_to_repr: Dict[str, ExpressionTree] = {}
            for ind in self.population:
                expr_key = ind.to_string()
                expr_to_inds.setdefault(expr_key, []).append(ind)
                if expr_key not in expr_to_repr:
                    expr_to_repr[expr_key] = ind

            missing_exprs = [expr for expr in expr_to_inds.keys() if expr not in self._eval_cache]
            if missing_exprs:
                if self.eval_workers > 1 and len(missing_exprs) > 1:
                    with ThreadPoolExecutor(max_workers=min(self.eval_workers, len(missing_exprs))) as executor:
                        future_map = {
                            executor.submit(
                                self._evaluate_cross_sectional,
                                expr_to_repr[expr],
                                all_features,
                                all_returns
                            ): expr for expr in missing_exprs
                        }
                        for future in as_completed(future_map):
                            expr_key = future_map[future]
                            try:
                                self._eval_cache[expr_key] = future.result()
                            except Exception:
                                self._eval_cache[expr_key] = {
                                    'ic_mean': 0.0, 'ic_std': 1.0, 'icir': 0.0,
                                    'rank_ic_mean': 0.0, 'rank_icir': 0.0,
                                    'long_short_return': 0.0, 'n_symbols': 0,
                                    'n_periods': 0, 'total_periods': 0, 'coverage_rate': 0.0, 'fitness': -999.0
                                }
                else:
                    for expr_key in missing_exprs:
                        self._eval_cache[expr_key] = self._evaluate_cross_sectional(
                            expr_to_repr[expr_key], all_features, all_returns
                        )

            for expr_key, inds in expr_to_inds.items():
                eval_result = self._eval_cache.get(expr_key)
                if eval_result is None:
                    eval_result = {
                        'ic_mean': 0.0, 'ic_std': 1.0, 'icir': 0.0,
                        'rank_ic_mean': 0.0, 'rank_icir': 0.0,
                        'long_short_return': 0.0, 'n_symbols': 0,
                        'n_periods': 0, 'total_periods': 0, 'coverage_rate': 0.0, 'fitness': -999.0
                    }
                    self._eval_cache[expr_key] = eval_result
                for ind in inds:
                    ind.fitness = eval_result['fitness']
                    ind.ic_mean = eval_result['ic_mean']
                    ind.ic_std = eval_result['ic_std']
                    ind.icir = eval_result['icir']
                    ind.rank_ic_mean = eval_result['rank_ic_mean']
                    ind.rank_icir = eval_result['rank_icir']
                    ind.long_short_return = eval_result['long_short_return']
                    ind.eval_detail = eval_result

            self.population.sort(key=lambda x: x.fitness if x.fitness is not None else -999, reverse=True)

            gen_best = self.population[0]
            gen_best_fitness = gen_best.fitness if gen_best.fitness is not None else -999

            gen_stats = {
                'generation': gen + 1,
                'best_fitness': gen_best_fitness,
                'best_ic': gen_best.ic_mean,
                'best_icir': gen_best.icir,
                'avg_fitness': np.mean([ind.fitness for ind in self.population
                                        if ind.fitness is not None]),
                'best_expression': gen_best.to_string(),
                'time': time.time() - gen_start
            }
            self.generation_stats.append(gen_stats)

            if gen_best_fitness > best_fitness_ever:
                best_fitness_ever = gen_best_fitness
                stagnation_count = 0
            else:
                stagnation_count += 1

            if stagnation_count >= self.stagnation_limit:
                stale_pct = int((gen + 1) / self.max_generations * 85)
                self._report_progress(
                    stale_pct,
                    f"第{gen+1}代: 连续{self.stagnation_limit}代无改善，提前终止",
                    gen_stats
                )
                break

            elite_count = max(1, int(self.elitism_rate * self.population_size))
            new_population = [copy.deepcopy(ind) for ind in self.population[:elite_count]]

            while len(new_population) < self.population_size:
                if random.random() < self.crossover_rate:
                    p1 = self._tournament_select()
                    p2 = self._tournament_select()
                    c1, c2 = self._crossover(p1, p2)
                    new_population.append(c1)
                    if len(new_population) < self.population_size:
                        new_population.append(c2)
                elif random.random() < self.crossover_rate + self.mutation_rate:
                    p = self._tournament_select()
                    m = self._mutate(p)
                    new_population.append(m)
                else:
                    new_population.append(self._create_random_individual())

            self.population = new_population[:self.population_size]

            progress_pct = int((gen + 1) / self.max_generations * 85)
            self._report_progress(
                progress_pct,
                f"第{gen+1}/{self.max_generations}代: "
                f"最佳IC={gen_best.ic_mean:.4f}, ICIR={gen_best.icir:.4f}, "
                f"表达式={gen_best.to_string()[:60]}",
                gen_stats
            )

        self._report_progress(92, "进化完成，正在筛选多样化因子...")

        valid_individuals = [
            ind for ind in self.population
            if ind.fitness is not None
            and abs(ind.ic_mean) >= self.min_ic_threshold
            and abs(ind.icir) >= self.min_ir_threshold
            and (ind.eval_detail or {}).get('coverage_rate', 0.0) >= self.min_coverage_threshold
        ]
        valid_individuals.sort(key=lambda x: x.fitness if x.fitness is not None else -999, reverse=True)

        self.best_individuals = self._select_diverse_best(
            valid_individuals, all_features, self.max_factor_count
        )

        self._report_progress(97, "正在生成因子序列...")

        factors_result = []
        for i, ind in enumerate(self.best_individuals):
            factor_data = {}
            for symbol, features in all_features.items():
                try:
                    fv = self._evaluate_individual_on_symbol(ind, features)
                    factor_data[symbol] = fv
                except Exception:
                    continue

            factors_result.append({
                'factor_id': f"gp_cs_{i+1}",
                'name': f"GP截面因子_{i+1}",
                'expression': ind.to_string(),
                'ic_mean': ind.ic_mean,
                'ic_std': ind.ic_std,
                'icir': ind.icir,
                'rank_ic_mean': ind.rank_ic_mean,
                'rank_icir': ind.rank_icir,
                'long_short_return': ind.long_short_return,
                'n_symbols': ind.eval_detail.get('n_symbols', 0) if ind.eval_detail else 0,
                'n_periods': ind.eval_detail.get('n_periods', 0) if ind.eval_detail else 0,
                'total_periods': ind.eval_detail.get('total_periods', 0) if ind.eval_detail else 0,
                'coverage_rate': ind.eval_detail.get('coverage_rate', 0.0) if ind.eval_detail else 0.0,
                'fitness': ind.fitness,
                'direction': 'negative' if (ind.ic_mean is not None and ind.ic_mean < 0) else 'positive',
                'factor_data': factor_data,
                'depth': ind.depth(),
                'size': ind.size(),
            })

        self._report_progress(100, f"挖掘完成！共发现 {len(factors_result)} 个有效因子")

        return {
            'success': True,
            'factors': factors_result,
            'generation_stats': self.generation_stats,
            'total_evaluated': len(self.population) * self.max_generations,
            'n_symbols': len(all_features),
            'stopped': stopped_early,
            'actual_generations': completed_generations if stopped_early else self.max_generations,
            'config': {
                'population_size': self.population_size,
                'max_generations': self.max_generations,
                'max_depth': self.max_depth,
                'crossover_rate': self.crossover_rate,
                'mutation_rate': self.mutation_rate,
            }
        }


def run_cross_sectional_mining(
    data_dict: Dict[str, pd.DataFrame],
    config: Dict = None,
    progress_callback: Callable = None
) -> Dict:
    """
    便捷函数：运行截面因子挖掘

    Args:
        data_dict: {symbol: market_data} 多币种数据字典
        config: 挖掘配置
        progress_callback: 进度回调

    Returns:
        Dict: 挖掘结果
    """
    miner = GPMiner(config)
    return miner.mine(data_dict, progress_callback)
