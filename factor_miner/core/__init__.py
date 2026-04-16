"""
核心模块
包含因子挖掘的核心功能
"""

from .data_loader import DataLoader
from .factor_builder import FactorBuilder
from .factor_evaluator import FactorEvaluator
from .factor_optimizer import FactorOptimizer
from .gp_miner import GPMiner, run_cross_sectional_mining

TORCH_AVAILABLE = False
try:
    from .rl_miner import RLMiner, run_rl_cross_sectional_mining
    TORCH_AVAILABLE = True
except ImportError:
    RLMiner = None
    run_rl_cross_sectional_mining = None

__all__ = [
    'DataLoader',
    'FactorBuilder',
    'FactorEvaluator',
    'FactorOptimizer',
    'GPMiner',
    'run_cross_sectional_mining',
    'RLMiner',
    'run_rl_cross_sectional_mining',
    'TORCH_AVAILABLE',
] 