"""
RL截面因子挖掘器
借鉴AlphaGPT的REINFORCE+Transformer架构，在多币种截面空间中搜索最优因子表达式

核心流程：
1. 加载多币种数据，预计算基础特征 → GPU Tensor
2. Transformer自回归生成token序列（公式）
3. StackVM执行公式 → 因子值
4. 截面回测评估 → reward信号
5. REINFORCE策略梯度更新
6. LoRD正则化

与AlphaGPT的对应关系：
- AlphaGPT: Meme币DEX回测 → 本模块: 主流币CEX截面回测
- AlphaGPT: 12算子 → 本模块: 27算子（增加时序窗口+截面算子）
- AlphaGPT: PostgreSQL数据 → 本模块: Binance API数据
- AlphaGPT: liquidity/fdv特征 → 本模块: volume/mcap特征
- LoRD正则化: 直接复用
"""

import logging
import warnings
from typing import Dict, List, Callable

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

TORCH_AVAILABLE = False
try:
    import math
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.distributions import Categorical
    import pandas as pd
    import numpy as np
    TORCH_AVAILABLE = True
except ImportError:
    logger.warning("PyTorch未安装，RL挖掘器不可用。请安装: pip install torch")


FEATURE_NAMES = [
    'RET', 'VOL', 'V_CHG', 'PV', 'TREND', 'HL_RANGE',
    'CLOSE_POS', 'MA_DEV', 'VOLATILITY', 'MOMENTUM',
]
INPUT_DIM = len(FEATURE_NAMES)


if TORCH_AVAILABLE:

    # ==================== 算子定义 ====================

    def _ts_delay(x, d=1):
        if d == 0:
            return x
        pad = torch.zeros((x.shape[0], d), device=x.device, dtype=x.dtype)
        return torch.cat([pad, x[:, :-d]], dim=1)

    def _ts_mean(x, w=5):
        pad = torch.zeros((x.shape[0], w - 1), device=x.device, dtype=x.dtype)
        x_pad = torch.cat([pad, x], dim=1)
        return x_pad.unfold(1, w, 1).mean(dim=-1)

    def _ts_std(x, w=5):
        pad = torch.zeros((x.shape[0], w - 1), device=x.device, dtype=x.dtype)
        x_pad = torch.cat([pad, x], dim=1)
        return x_pad.unfold(1, w, 1).std(dim=-1)

    def _ts_max(x, w=5):
        pad = torch.full((x.shape[0], w - 1), float('-inf'), device=x.device, dtype=x.dtype)
        x_pad = torch.cat([pad, x], dim=1)
        return x_pad.unfold(1, w, 1).max(dim=-1)[0]

    def _ts_min(x, w=5):
        pad = torch.full((x.shape[0], w - 1), float('inf'), device=x.device, dtype=x.dtype)
        x_pad = torch.cat([pad, x], dim=1)
        return x_pad.unfold(1, w, 1).min(dim=-1)[0]

    def _ts_rank(x, w=5):
        pad = torch.zeros((x.shape[0], w - 1), device=x.device, dtype=x.dtype)
        x_pad = torch.cat([pad, x], dim=1)
        windows = x_pad.unfold(1, w, 1)
        last = windows[..., -1].unsqueeze(-1)
        # 返回当前值在滚动窗口内的分位秩，形状保持 [N, T]
        return (windows <= last).float().mean(dim=-1)

    def _ts_zscore(x, w=5):
        m = _ts_mean(x, w)
        s = _ts_std(x, w)
        return (x - m) / (s + 1e-6)

    def _ts_decay(x):
        return x + 0.8 * _ts_delay(x, 1) + 0.6 * _ts_delay(x, 2)

    def _ts_momentum(x, w=5):
        delayed = _ts_delay(x, w)
        return x / (delayed + 1e-6) - 1

    def _op_gate(condition, x, y):
        mask = (condition > 0).float()
        return mask * x + (1.0 - mask) * y

    def _op_jump(x):
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True) + 1e-6
        z = (x - mean) / std
        return torch.relu(z - 3.0)

    def _cs_rank(x):
        return x.argsort(dim=0).argsort(dim=0).float() / (x.shape[0] - 1 + 1e-6)

    def _cs_zscore(x):
        m = x.mean(dim=0, keepdim=True)
        s = x.std(dim=0, keepdim=True) + 1e-6
        return (x - m) / s

    def _cs_mad_norm(x):
        med = x.median(dim=0, keepdim=True)[0]
        mad = (x - med).abs().median(dim=0, keepdim=True)[0] + 1e-6
        return torch.clamp((x - med) / mad, -5.0, 5.0)

    def _safe_div(x, y):
        return x / (y + 1e-6)

    def _safe_sqrt(x):
        return torch.sqrt(torch.abs(x) + 1e-8)

    def _safe_log(x):
        return torch.log(torch.abs(x) + 1e-8)

    OPS_CONFIG = [
        ('ADD',       lambda x, y: x + y,                  2),
        ('SUB',       lambda x, y: x - y,                  2),
        ('MUL',       lambda x, y: x * y,                  2),
        ('DIV',       lambda x, y: _safe_div(x, y),        2),
        ('MAX',       lambda x, y: torch.max(x, y),        2),
        ('MIN',       lambda x, y: torch.min(x, y),        2),
        ('NEG',       lambda x: -x,                         1),
        ('ABS',       lambda x: torch.abs(x),               1),
        ('SIGN',      lambda x: torch.sign(x),              1),
        ('SQRT',      lambda x: _safe_sqrt(x),              1),
        ('LOG',       lambda x: _safe_log(x),               1),
        ('GATE',      lambda c, x, y: _op_gate(c, x, y),   3),
        ('JUMP',      lambda x: _op_jump(x),                1),
        ('DECAY',     lambda x: _ts_decay(x),               1),
        ('DELAY1',    lambda x: _ts_delay(x, 1),            1),
        ('DELAY3',    lambda x: _ts_delay(x, 3),            1),
        ('MA5',       lambda x: _ts_mean(x, 5),             1),
        ('MA10',      lambda x: _ts_mean(x, 10),            1),
        ('MA20',      lambda x: _ts_mean(x, 20),            1),
        ('STD5',      lambda x: _ts_std(x, 5),              1),
        ('STD10',     lambda x: _ts_std(x, 10),             1),
        ('RANK5',     lambda x: _ts_rank(x, 5),             1),
        ('MOM5',      lambda x: _ts_momentum(x, 5),         1),
        ('MOM10',     lambda x: _ts_momentum(x, 10),        1),
        ('CS_RANK',   lambda x: _cs_rank(x),                1),
        ('CS_ZSCORE', lambda x: _cs_zscore(x),              1),
        ('CS_MAD',    lambda x: _cs_mad_norm(x),            1),
    ]

    # ==================== StackVM ====================

    class StackVM:
        def __init__(self):
            self.feat_offset = INPUT_DIM
            self.op_map = {i + self.feat_offset: cfg[1] for i, cfg in enumerate(OPS_CONFIG)}
            self.arity_map = {i + self.feat_offset: cfg[2] for i, cfg in enumerate(OPS_CONFIG)}
            self.op_names = {i + self.feat_offset: cfg[0] for i, cfg in enumerate(OPS_CONFIG)}
            self.feat_names = FEATURE_NAMES

        def execute(self, formula_tokens, feat_tensor):
            stack = []
            try:
                for token in formula_tokens:
                    token = int(token)
                    if token < self.feat_offset:
                        stack.append(feat_tensor[:, token, :])
                    elif token in self.op_map:
                        arity = self.arity_map[token]
                        if len(stack) < arity:
                            return None
                        args = []
                        for _ in range(arity):
                            args.append(stack.pop())
                        args.reverse()
                        res = self.op_map[token](*args)
                        if torch.isnan(res).any() or torch.isinf(res).any():
                            res = torch.nan_to_num(res, nan=0.0, posinf=1.0, neginf=-1.0)
                        stack.append(res)
                    else:
                        return None
                if len(stack) == 1:
                    return stack[0]
                return None
            except Exception:
                return None

        def decode_formula(self, formula_tokens):
            parts = []
            for t in formula_tokens:
                t = int(t)
                if t < self.feat_offset:
                    parts.append(FEATURE_NAMES[t])
                elif t in self.op_names:
                    parts.append(self.op_names[t])
                else:
                    parts.append(f'?{t}')
            return ' '.join(parts)

    # ==================== 特征工程 ====================

    class RLFeatureEngineer:
        @staticmethod
        def compute_features(raw_dict):
            c = raw_dict['close']
            o = raw_dict['open']
            h = raw_dict['high']
            l = raw_dict['low']
            v = raw_dict['volume']

            ret = torch.log(c / (torch.roll(c, 1, dims=1) + 1e-9))
            ret[:, 0] = 0.0

            log_vol = torch.log1p(v)
            vol_chg = (v - torch.roll(v, 1, dims=1)) / (torch.roll(v, 1, dims=1) + 1.0)
            vol_chg[:, 0] = 0.0

            pv = ret * log_vol

            pad5 = torch.zeros((c.shape[0], 4), device=c.device, dtype=c.dtype)
            c_pad = torch.cat([pad5, c], dim=1)
            ma5 = c_pad.unfold(1, 5, 1).mean(dim=-1)
            trend = (c - ma5) / (ma5 + 1e-6)

            hl_range = (h - l) / (c + 1e-6)
            close_pos = (c - l) / (h - l + 1e-6)
            ma_dev = (c - ma5) / (ma5 + 1e-6)

            ret_sq = ret ** 2
            pad10 = torch.zeros((ret_sq.shape[0], 9), device=c.device, dtype=c.dtype)
            ret_sq_pad = torch.cat([pad10, ret_sq], dim=1)
            volatility = torch.sqrt(ret_sq_pad.unfold(1, 10, 1).mean(dim=-1) + 1e-9)

            pad5r = torch.zeros((ret.shape[0], 4), device=c.device, dtype=c.dtype)
            ret_pad = torch.cat([pad5r, ret], dim=1)
            momentum = ret_pad.unfold(1, 5, 1).sum(dim=-1)

            def robust_norm(t):
                median = torch.nanmedian(t, dim=1, keepdim=True)[0]
                mad = torch.nanmedian(torch.abs(t - median), dim=1, keepdim=True)[0] + 1e-6
                norm = (t - median) / mad
                return torch.clamp(norm, -5.0, 5.0)

            features = torch.stack([
                robust_norm(ret),
                robust_norm(log_vol),
                robust_norm(vol_chg),
                robust_norm(pv),
                robust_norm(trend),
                robust_norm(hl_range),
                close_pos,
                robust_norm(ma_dev),
                robust_norm(volatility),
                robust_norm(momentum),
            ], dim=1)

            return features

    # ==================== 截面回测评估器 ====================

    class CrossSectionalBacktest:
        def __init__(self, trade_size=10000.0, min_volume=100000.0, base_fee=0.001,
                     long_quantile=0.8, short_quantile=0.2):
            self.trade_size = trade_size
            self.min_volume = min_volume
            self.base_fee = base_fee
            self.long_quantile = long_quantile
            self.short_quantile = short_quantile

        def evaluate(self, factors, raw_data, target_ret):
            volume = raw_data['volume']
            usd_volume = volume * raw_data['close']
            is_safe = (usd_volume > self.min_volume).float()

            # 改为截面分位开仓：对每个时间点，在"safe"币种内部按因子值排名，
            # 前 (1-long_quantile) 分位开多，后 short_quantile 分位开空。
            # 这样对因子分布中心不敏感，避免原先 sigmoid(factor)>0.65 阈值
            # 让大量正常因子 activity 恒为 0 → score=-10 的问题。
            large_neg = torch.finfo(factors.dtype).min
            factor_masked = torch.where(is_safe > 0, factors, torch.full_like(factors, large_neg))
            rank_ord = factor_masked.argsort(dim=0).argsort(dim=0).float()

            n_symbols_f = float(factors.shape[0])
            n_safe = is_safe.sum(dim=0, keepdim=True)
            denom = (n_safe - 1.0).clamp(min=1.0)
            relative_rank = (rank_ord - (n_symbols_f - n_safe)) / denom
            relative_rank = torch.clamp(relative_rank, 0.0, 1.0)

            long_pos = (relative_rank >= self.long_quantile).float() * is_safe
            short_pos = (relative_rank <= self.short_quantile).float() * is_safe
            position = long_pos - short_pos  # 取值 {-1, 0, +1}

            impact = self.trade_size / (volume * raw_data['close'] + 1e-9)
            impact = torch.clamp(impact, 0.0, 0.02)
            total_cost = self.base_fee + impact

            prev_pos = torch.roll(position, 1, dims=1)
            prev_pos[:, 0] = 0
            turnover = torch.abs(position - prev_pos)
            tx_cost = turnover * total_cost

            gross_pnl = position * target_ret
            net_pnl = gross_pnl - tx_cost

            cum_ret = net_pnl.sum(dim=1)

            big_drawdowns = (net_pnl < -0.05).float().sum(dim=1)
            score = cum_ret - (big_drawdowns * 0.5)

            activity = torch.abs(position).sum(dim=1)
            score = torch.where(activity < 5, torch.tensor(-10.0, device=score.device), score)

            final_fitness = torch.median(score)
            return final_fitness, cum_ret.mean().item()

    # ==================== Transformer策略网络 ====================

    class RMSNorm(nn.Module):
        def __init__(self, d_model, eps=1e-6):
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(d_model))

        def forward(self, x):
            rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
            return (x / rms) * self.weight

    class QKNorm(nn.Module):
        def __init__(self, d_head, eps=1e-6):
            super().__init__()
            self.eps = eps
            self.scale = nn.Parameter(torch.ones(1, 1, 1, d_head) * (d_head ** -0.5))

        def forward(self, q, k):
            q_norm = F.normalize(q, p=2, dim=-1)
            k_norm = F.normalize(k, p=2, dim=-1)
            return q_norm * self.scale, k_norm * self.scale

    class SwiGLU(nn.Module):
        def __init__(self, d_in, d_ff):
            super().__init__()
            self.w = nn.Linear(d_in, d_ff * 2)
            self.fc = nn.Linear(d_ff, d_in)

        def forward(self, x):
            x_glu = self.w(x)
            x, gate = x_glu.chunk(2, dim=-1)
            x = x * F.silu(gate)
            return self.fc(x)

    class LoopedTransformerLayer(nn.Module):
        def __init__(self, d_model, nhead, dim_feedforward, num_loops=3, dropout=0.1):
            super().__init__()
            self.num_loops = num_loops
            self.qk_norm = QKNorm(d_model // nhead)
            self.attention = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout)
            self.norm1 = RMSNorm(d_model)
            self.norm2 = RMSNorm(d_model)
            self.ffn = SwiGLU(d_model, dim_feedforward)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x, mask=None, is_causal=False):
            for _ in range(self.num_loops):
                x_norm = self.norm1(x)
                attn_out, _ = self.attention(x_norm, x_norm, x_norm, attn_mask=mask, is_causal=is_causal)
                x = x + self.dropout(attn_out)
                x_norm = self.norm2(x)
                ffn_out = self.ffn(x_norm)
                x = x + self.dropout(ffn_out)
            return x

    class LoopedTransformer(nn.Module):
        def __init__(self, d_model, nhead, num_layers, dim_feedforward, num_loops=3, dropout=0.1):
            super().__init__()
            self.layers = nn.ModuleList([
                LoopedTransformerLayer(d_model, nhead, dim_feedforward, num_loops, dropout)
                for _ in range(num_layers)
            ])

        def forward(self, x, mask=None, is_causal=False):
            for layer in self.layers:
                x = layer(x, mask=mask, is_causal=is_causal)
            return x

    class MTPHead(nn.Module):
        def __init__(self, d_model, vocab_size, num_tasks=3):
            super().__init__()
            self.num_tasks = num_tasks
            self.task_heads = nn.ModuleList([
                nn.Linear(d_model, vocab_size) for _ in range(num_tasks)
            ])
            self.task_weights = nn.Parameter(torch.ones(num_tasks) / num_tasks)
            self.task_router = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Linear(d_model // 2, num_tasks)
            )

        def forward(self, x):
            task_logits = self.task_router(x)
            task_probs = F.softmax(task_logits, dim=-1)
            task_outputs = [head(x) for head in self.task_heads]
            task_outputs = torch.stack(task_outputs, dim=1)
            weighted = (task_probs.unsqueeze(-1) * task_outputs).sum(dim=1)
            return weighted, task_probs

    class AlphaPolicy(nn.Module):
        def __init__(self, vocab_size, d_model=64, nhead=4, num_layers=2,
                     dim_feedforward=128, num_loops=3, max_formula_len=16, dropout=0.1):
            super().__init__()
            self.d_model = d_model
            self.vocab_size = vocab_size
            self.max_formula_len = max_formula_len

            self.token_emb = nn.Embedding(vocab_size, d_model)
            self.pos_emb = nn.Parameter(torch.zeros(1, max_formula_len + 1, d_model))

            self.blocks = LoopedTransformer(
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
                dim_feedforward=dim_feedforward,
                num_loops=num_loops,
                dropout=dropout,
            )

            self.ln_f = RMSNorm(d_model)
            self.mtp_head = MTPHead(d_model, vocab_size, num_tasks=3)
            self.head_critic = nn.Linear(d_model, 1)

        def forward(self, idx):
            B, T = idx.size()
            x = self.token_emb(idx) + self.pos_emb[:, :T, :]

            mask = nn.Transformer.generate_square_subsequent_mask(T).to(idx.device)
            x = self.blocks(x, mask=mask, is_causal=True)
            x = self.ln_f(x)

            last_emb = x[:, -1, :]
            logits, task_probs = self.mtp_head(last_emb)
            value = self.head_critic(last_emb)

            return logits, value, task_probs

    # ==================== LoRD正则化 ====================

    class NewtonSchulzLowRankDecay:
        def __init__(self, named_parameters, decay_rate=1e-3, num_iterations=5, target_keywords=None):
            self.decay_rate = decay_rate
            self.num_iterations = num_iterations
            self.target_keywords = target_keywords or ["q_proj", "k_proj", "attention", "qk_norm"]
            self.params_to_decay = []

            for name, param in named_parameters:
                if not param.requires_grad or param.ndim != 2:
                    continue
                if not any(k in name for k in self.target_keywords):
                    continue
                self.params_to_decay.append((name, param))

        @torch.no_grad()
        def step(self):
            for name, W in self.params_to_decay:
                orig_dtype = W.dtype
                X = W.float()
                r, c = X.shape

                transposed = False
                if r > c:
                    X = X.T
                    transposed = True

                norm = X.norm() + 1e-8
                X = X / norm

                Y = X
                I = torch.eye(X.shape[-1], device=X.device, dtype=X.dtype)

                for _ in range(self.num_iterations):
                    A = Y.T @ Y
                    Y = 0.5 * Y @ (3.0 * I - A)

                if transposed:
                    Y = Y.T

                W.sub_(self.decay_rate * Y.to(orig_dtype))

    # ==================== RL训练引擎 ====================

    class RLMiner:
        """
        RL截面因子挖掘器

        借鉴AlphaGPT的REINFORCE+LoopedTransformer架构，
        在多币种截面空间中搜索最优因子表达式
        """

        def __init__(self, config: Dict = None):
            if not TORCH_AVAILABLE:
                raise ImportError("RL挖掘器需要PyTorch。请安装: pip install torch")

            self.config = config or {}
            self.device = self._resolve_device(self.config.get('device', 'auto'))

            self.batch_size = int(self.config.get('batch_size', 4096 if self.device.type == 'cuda' else 256))
            self.train_steps = int(self.config.get('train_steps', 500))
            self.max_formula_len = int(self.config.get('max_formula_len', 16))
            self.lr = float(self.config.get('lr', 1e-3))
            self.lord_decay_rate = float(self.config.get('lord_decay_rate', 1e-3))
            self.use_lord = bool(self.config.get('use_lord', True))
            self.entropy_coef = float(self.config.get('entropy_coef', 0.01))
            self.value_coef = float(self.config.get('value_coef', 0.5))
            self.max_factors = int(self.config.get('max_factors', 15))
            self.min_reward_threshold = float(self.config.get('min_reward', 0.0))
            self.min_coverage_threshold = float(self.config.get('min_coverage', 0.2))
            self.max_correlation = float(self.config.get('max_correlation', 0.7))
            self.eval_interval = int(self.config.get('eval_interval', 50))
            self.top_k_keep = int(self.config.get('top_k_keep', 3))
            self.collect_temperature = float(self.config.get('collect_temperature', 1.0))
            self.collect_top_k = int(self.config.get('collect_top_k', 0))
            # 截面分位开仓阈值：默认做多 top20%、做空 bottom20%
            self.long_quantile = float(self.config.get('long_quantile', 0.8))
            self.short_quantile = float(self.config.get('short_quantile', 0.2))
            # 策略坍缩检测与自动重启：连续 collapse_window 步
            # rewards.std() 都低于 collapse_std_threshold 时，
            # 对 head_actor 注入噪声强制重新探索
            self.collapse_window = int(self.config.get('collapse_window', 20))
            self.collapse_std_threshold = float(self.config.get('collapse_std_threshold', 1e-3))
            self.collapse_bias_noise = float(self.config.get('collapse_bias_noise', 0.5))
            self.collapse_weight_noise = float(self.config.get('collapse_weight_noise', 0.05))
            self.collapse_weight_scale = float(self.config.get('collapse_weight_scale', 0.9))

            self.vm = StackVM()
            self.bt = CrossSectionalBacktest(
                trade_size=float(self.config.get('trade_size', 10000.0)),
                min_volume=float(self.config.get('min_volume', 100000.0)),
                base_fee=float(self.config.get('base_fee', 0.001)),
                long_quantile=self.long_quantile,
                short_quantile=self.short_quantile,
            )

            self.vocab_size = INPUT_DIM + len(OPS_CONFIG)
            self.arity_tensor = torch.zeros(self.vocab_size, dtype=torch.long, device=self.device)
            for token, arity in self.vm.arity_map.items():
                self.arity_tensor[token] = arity
            self.max_arity = max(cfg[2] for cfg in OPS_CONFIG)
            self.model = AlphaPolicy(
                vocab_size=self.vocab_size,
                d_model=int(self.config.get('d_model', 64)),
                nhead=int(self.config.get('nhead', 4)),
                num_layers=int(self.config.get('num_layers', 2)),
                dim_feedforward=int(self.config.get('dim_feedforward', 128)),
                num_loops=int(self.config.get('num_loops', 3)),
                max_formula_len=self.max_formula_len,
                dropout=float(self.config.get('dropout', 0.1)),
            ).to(self.device)

            self.opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr)

            self.lord_opt = None
            if self.use_lord:
                # 对齐 AlphaGPT 原版：只衰减注意力相关权重，
                # 不要对 head_actor（输出投影到 vocab 的线性层）施加低秩衰减，
                # 否则会持续把 actor 的 logits 投影拉向 rank-1，导致策略完全坍缩
                self.lord_opt = NewtonSchulzLowRankDecay(
                    self.model.named_parameters(),
                    decay_rate=self.lord_decay_rate,
                    target_keywords=["q_proj", "k_proj", "attention", "qk_norm"],
                )

            self.best_score = -float('inf')
            self.best_formula = None
            self.best_formulas: List[Dict] = []
            self.training_history: List[Dict] = []
            self._progress_callback = None
            self._stop_requested = False
            self._reward_std_history: List[float] = []

        def _build_postfix_action_mask(self, stack_slots, step_idx):
            B = stack_slots.shape[0]
            mask = torch.full((B, self.vocab_size), float('-inf'), device=self.device)
            remaining_after = self.max_formula_len - step_idx - 1

            active = stack_slots > 0
            empty = ~active

            delta = torch.ones((B, self.vocab_size), device=self.device, dtype=torch.long)
            op_tokens = self.arity_tensor > 0
            if op_tokens.any():
                delta[:, op_tokens] = 1 - self.arity_tensor[op_tokens].unsqueeze(0)
            next_slots = stack_slots.unsqueeze(1) + delta

            max_inc = remaining_after
            max_dec = remaining_after * max(self.max_arity - 1, 0)
            min_possible = next_slots - max_dec
            max_possible = next_slots + max_inc
            feasible = (next_slots >= 0) & (min_possible <= 1) & (max_possible >= 1)

            # 栈为空时只能先放特征，保证可执行表达式前缀合法
            if empty.any():
                feat_ok = feasible[empty, :self.vm.feat_offset]
                mask[empty, :self.vm.feat_offset] = torch.where(feat_ok, torch.zeros_like(mask[empty, :self.vm.feat_offset]), mask[empty, :self.vm.feat_offset])

            # 对于非空样本：特征恒可选（+1槽位）
            if active.any():
                feat_ok = feasible[active, :self.vm.feat_offset]
                mask[active, :self.vm.feat_offset] = torch.where(feat_ok, torch.zeros_like(mask[active, :self.vm.feat_offset]), mask[active, :self.vm.feat_offset])

                # 仅当当前槽位足够时，允许选择对应arity的算子
                op_slots = stack_slots.unsqueeze(1).expand(B, self.vocab_size)
                arity_expand = self.arity_tensor.unsqueeze(0).expand(B, self.vocab_size)
                op_allowed = active.unsqueeze(1) & (arity_expand > 0) & (op_slots >= arity_expand) & feasible
                mask[op_allowed] = 0.0

            # 最后一步必须收敛为单一结果栈槽
            if remaining_after == 0:
                final_ok = next_slots == 1
                mask = torch.where(final_ok, mask, torch.full_like(mask, float('-inf')))
                still_invalid = ~torch.isfinite(mask).any(dim=1)
                if still_invalid.any():
                    # 优先开放能让 next_slots==1 的算子（arity == stack_slots），
                    # 避免原先直接开放 feature 导致必产出非法公式（VM 一律返回 None）
                    needed_arity = stack_slots.unsqueeze(1)
                    arity_match = (self.arity_tensor.unsqueeze(0) == needed_arity) & (self.arity_tensor.unsqueeze(0) > 0)
                    fallback_slot = still_invalid.unsqueeze(1) & arity_match
                    mask = torch.where(fallback_slot, torch.zeros_like(mask), mask)
                    # 栈深超过 max_arity 的极端情况，退让到 feature：
                    # VM 会拒绝这条公式返回 -5，仍然是合理的学习信号
                    still_invalid = ~torch.isfinite(mask).any(dim=1)
                    if still_invalid.any():
                        mask[still_invalid, :self.vm.feat_offset] = 0.0

            return mask

        def _resolve_device(self, requested_device):
            """
            解析用户请求的设备，并在CUDA不可用时安全回退到CPU。
            支持: auto/cuda/cpu 或 torch.device 可识别字符串。
            """
            request = str(requested_device or 'auto').strip().lower()
            cuda_available = bool(torch.cuda.is_available())

            if request in ('auto', ''):
                return torch.device('cuda' if cuda_available else 'cpu')

            if request.startswith('cuda'):
                if cuda_available:
                    return torch.device(request)
                logger.warning(
                    "请求使用CUDA设备(%s)，但当前PyTorch未启用CUDA，已自动回退到CPU。",
                    requested_device
                )
                return torch.device('cpu')

            try:
                return torch.device(request)
            except Exception:
                logger.warning("无法识别设备配置: %s，已自动回退到CPU。", requested_device)
                return torch.device('cpu')

        def mine(self, data_dict: Dict, progress_callback: Callable = None) -> Dict:
            self._progress_callback = progress_callback
            self._stop_requested = False

            self._report_progress(0, "正在准备GPU数据...")

            raw_data, feat_tensor, target_ret = self._prepare_data(data_dict)
            n_symbols, n_features, n_periods = feat_tensor.shape

            self._report_progress(5, f"数据就绪: {n_symbols}币种 × {n_periods}期 × {n_features}特征, 设备={self.device}")

            self._report_progress(10, f"开始RL训练: BS={self.batch_size}, Steps={self.train_steps}")

            self.model.train()

            for step in range(self.train_steps):
                if self._stop_requested:
                    self._report_progress(-1, "用户停止训练")
                    break

                inp = torch.zeros((self.batch_size, 1), dtype=torch.long, device=self.device)
                stack_slots = torch.zeros(self.batch_size, dtype=torch.long, device=self.device)

                log_probs = []
                values = []
                tokens_list = []
                entropies = []

                for t in range(self.max_formula_len):
                    logits, value, _ = self.model(inp)
                    logits = torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0)
                    logits = torch.clamp(logits, -20.0, 20.0)
                    action_mask = self._build_postfix_action_mask(stack_slots, t)
                    masked_logits = logits + action_mask
                    if not torch.isfinite(masked_logits).any(dim=1).all():
                        fallback = ~torch.isfinite(masked_logits).any(dim=1)
                        masked_logits[fallback, :self.vm.feat_offset] = logits[fallback, :self.vm.feat_offset]
                    dist = Categorical(logits=masked_logits)
                    action = dist.sample()

                    log_probs.append(dist.log_prob(action))
                    values.append(value.squeeze(-1))
                    tokens_list.append(action)
                    # 收集每步的真实 Categorical 熵 -Σ p·log p，
                    # 代替原先只用采样 logprob 的 `-p·log p` 近似
                    entropies.append(dist.entropy())
                    inp = torch.cat([inp, action.unsqueeze(1)], dim=1)
                    is_op = action >= self.vm.feat_offset
                    delta = torch.ones_like(stack_slots)
                    if is_op.any():
                        delta[is_op] = 1 - self.arity_tensor[action[is_op]]
                    stack_slots = torch.clamp(stack_slots + delta, min=0)

                seqs = torch.stack(tokens_list, dim=1)

                rewards = torch.zeros(self.batch_size, device=self.device)
                reward_cache = {}
                valid_count = 0

                for i in range(self.batch_size):
                    formula = seqs[i].tolist()
                    formula_key = tuple(formula)
                    if formula_key in reward_cache:
                        rewards[i] = reward_cache[formula_key]
                        continue
                    res = self.vm.execute(formula, feat_tensor)

                    if res is None:
                        rewards[i] = -5.0
                        reward_cache[formula_key] = rewards[i]
                        continue

                    if res.std() < 1e-4:
                        rewards[i] = -2.0
                        reward_cache[formula_key] = rewards[i]
                        continue

                    score, ret_val = self.bt.evaluate(res, raw_data, target_ret)

                    ic_bonus = 0.0
                    try:
                        ic_vals = []
                        n_p = res.shape[1] if res.dim() == 2 else 0
                        for t_idx in range(min(n_p, res.shape[1])):
                            fv_t = res[:, t_idx]
                            tr_t = target_ret[:, t_idx]
                            valid_m = ~(torch.isnan(fv_t) | torch.isnan(tr_t) | torch.isinf(fv_t) | torch.isinf(tr_t))
                            if valid_m.sum().item() >= 3:
                                fv_v = fv_t[valid_m]
                                tr_v = tr_t[valid_m]
                                fc = fv_v - fv_v.mean()
                                tc = tr_v - tr_v.mean()
                                fs = fc.std()
                                ts = tc.std()
                                if fs > 1e-8 and ts > 1e-8:
                                    ic_val = ((fc * tc).mean() / (fs * ts)).item()
                                    if not math.isnan(ic_val):
                                        ic_vals.append(abs(ic_val))
                        if ic_vals:
                            ic_bonus = float(np.median(ic_vals)) * 2.0
                    except Exception:
                        pass

                    final_score = score + ic_bonus
                    rewards[i] = final_score
                    reward_cache[formula_key] = rewards[i]
                    valid_count += 1

                    if final_score.item() > self.best_score:
                        self.best_score = final_score.item()
                        self.best_formula = formula
                        self._report_progress(
                            -1,
                            f"[!] 新最优: Score={final_score:.2f}, Ret={ret_val:.2%}, "
                            f"ICBonus={ic_bonus:.3f}, 公式={self.vm.decode_formula(formula)}"
                        )

                rewards = torch.nan_to_num(rewards, nan=-5.0, posinf=10.0, neginf=-10.0)

                values_tensor = torch.stack(values, dim=0)

                adv = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
                adv = adv.unsqueeze(0).expand_as(values_tensor).detach()

                policy_loss = 0
                for t in range(len(log_probs)):
                    policy_loss += -log_probs[t] * adv[t]
                policy_loss = policy_loss.mean()

                value_loss = 0
                for t in range(len(values_tensor)):
                    value_loss += F.mse_loss(values_tensor[t], rewards.detach())
                value_loss = value_loss.mean()

                if entropies:
                    entropy = torch.stack(entropies, dim=0).mean()
                    entropy = torch.nan_to_num(entropy, nan=0.0, posinf=0.0, neginf=0.0)
                else:
                    entropy = torch.zeros((), device=self.device)

                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                if not torch.isfinite(loss):
                    self._report_progress(
                        -1,
                        f"[!] Step {step+1}: 检测到非有限loss，已跳过更新以保持训练稳定"
                    )
                    continue

                self.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.opt.step()
                with torch.no_grad():
                    for p in self.model.parameters():
                        if p is not None:
                            p.copy_(torch.nan_to_num(p, nan=0.0, posinf=1.0, neginf=-1.0))

                if self.use_lord and self.lord_opt:
                    self.lord_opt.step()

                avg_reward = rewards.mean().item()
                max_reward = rewards.max().item()
                valid_ratio = valid_count / max(self.batch_size, 1)

                self.training_history.append({
                    'step': step,
                    'avg_reward': avg_reward,
                    'max_reward': max_reward,
                    'valid_ratio': valid_ratio,
                    'best_score': self.best_score,
                    'policy_loss': policy_loss.item(),
                    'value_loss': value_loss.item(),
                    'entropy': entropy.item(),
                })

                # 策略坍缩检测：连续 collapse_window 步 rewards.std() 都低于阈值，
                # 认定为策略卡死在退化公式上，直接对 head_actor 注入噪声重启探索，
                # 比依赖熵正则自然恢复更可靠
                reward_std_val = float(rewards.std().item())
                self._reward_std_history.append(reward_std_val)
                if len(self._reward_std_history) > self.collapse_window:
                    self._reward_std_history.pop(0)
                if (len(self._reward_std_history) >= self.collapse_window and
                        all(s < self.collapse_std_threshold for s in self._reward_std_history)):
                    with torch.no_grad():
                        mtp_head = getattr(self.model, 'mtp_head', None)
                        if mtp_head is not None:
                            for head in mtp_head.task_heads:
                                if head.bias is not None:
                                    head.bias.add_(torch.randn_like(head.bias) * self.collapse_bias_noise)
                                head.weight.mul_(self.collapse_weight_scale)
                                head.weight.add_(torch.randn_like(head.weight) * self.collapse_weight_noise)
                            if hasattr(mtp_head, 'task_router'):
                                for p in mtp_head.task_router.parameters():
                                    p.add_(torch.randn_like(p) * self.collapse_weight_noise)
                    self._reward_std_history.clear()
                    self._report_progress(
                        -1,
                        f"[!] Step {step+1}: 连续{self.collapse_window}步 rewards.std()<{self.collapse_std_threshold}，"
                        f"策略疑似坍缩，已扰动 mtp_head 强制重启探索"
                    )

                if step % self.eval_interval == 0 or step == self.train_steps - 1:
                    self._collect_top_formulas(feat_tensor, raw_data, target_ret)

                progress_pct = 10 + int((step + 1) / self.train_steps * 80)
                self._report_progress(
                    progress_pct,
                    f"Step {step+1}/{self.train_steps}: "
                    f"AvgRew={avg_reward:.3f}, MaxRew={max_reward:.3f}, Valid={valid_ratio:.1%}, "
                    f"Best={self.best_score:.3f}"
                )

            self._report_progress(92, "训练完成，正在筛选多样化因子...")

            self._collect_top_formulas(feat_tensor, raw_data, target_ret)

            diverse_formulas = self._diversify_formulas(feat_tensor, raw_data, target_ret)

            factors_result = []
            for f_info in diverse_formulas:
                formula_tokens = f_info['formula']
                factor_values = self.vm.execute(formula_tokens, feat_tensor)

                ic_metrics = self._compute_cross_sectional_ic(factor_values, target_ret, n_symbols)
                if ic_metrics['coverage_rate'] < self.min_coverage_threshold:
                    continue

                factor_idx = len(factors_result) + 1
                factors_result.append({
                    'factor_id': f'rl_cs_{factor_idx}',
                    'name': f'RL_CrossSectional_{factor_idx}',
                    'expression': self.vm.decode_formula(formula_tokens),
                    'score': f_info['score'],
                    'avg_return': f_info['avg_return'],
                    'ic_mean': ic_metrics['ic_mean'],
                    'icir': ic_metrics['icir'],
                    'rank_ic_mean': ic_metrics['rank_ic_mean'],
                    'rank_icir': ic_metrics['rank_icir'],
                    'long_short_return': ic_metrics['long_short_return'],
                    'n_symbols': n_symbols,
                    'n_periods': ic_metrics['n_periods'],
                    'total_periods': ic_metrics['total_periods'],
                    'coverage_rate': ic_metrics['coverage_rate'],
                })
                if len(factors_result) >= self.max_factors:
                    break

            self._report_progress(100, f"挖掘完成！共发现 {len(factors_result)} 个有效因子")

            return {
                'success': True,
                'factors': factors_result,
                'training_history': self.training_history,
                'best_score': self.best_score,
                'best_formula': self.vm.decode_formula(self.best_formula) if self.best_formula else None,
                'total_evaluated': self.batch_size * self.train_steps,
                'n_symbols': n_symbols,
                'n_periods': n_periods,
                'device': str(self.device),
                'config': {
                    'mode': 'cross_sectional_rl',
                    'batch_size': self.batch_size,
                    'train_steps': self.train_steps,
                    'max_formula_len': self.max_formula_len,
                    'lr': self.lr,
                    'use_lord': self.use_lord,
                    'd_model': self.model.d_model,
                    'vocab_size': self.vocab_size,
                }
            }

        def stop(self):
            self._stop_requested = True

        def _compute_cross_sectional_ic(self, factor_values, target_ret, n_symbols):
            if factor_values is None:
                return {
                    'ic_mean': 0.0, 'icir': 0.0,
                    'rank_ic_mean': 0.0, 'rank_icir': 0.0,
                    'long_short_return': 0.0, 'n_periods': 0, 'total_periods': 0, 'coverage_rate': 0.0,
                }

            ic_list = []
            rank_ic_list = []
            ls_returns = []

            n_periods = factor_values.shape[1] if factor_values.dim() == 2 else 0

            for t in range(n_periods):
                fv_t = factor_values[:, t]
                tr_t = target_ret[:, t]

                valid_mask = ~(torch.isnan(fv_t) | torch.isnan(tr_t) | torch.isinf(fv_t) | torch.isinf(tr_t))
                valid_count = valid_mask.sum().item()

                if valid_count < 3:
                    continue

                fv_valid = fv_t[valid_mask]
                tr_valid = tr_t[valid_mask]

                fv_centered = fv_valid - fv_valid.mean()
                tr_centered = tr_valid - tr_valid.mean()
                fv_std = fv_centered.std()
                tr_std = tr_centered.std()

                if fv_std < 1e-8 or tr_std < 1e-8:
                    continue

                ic = (fv_centered * tr_centered).mean() / (fv_std * tr_std + 1e-8)
                if math.isnan(ic.item()):
                    continue
                ic_list.append(ic.item())

                fv_rank = fv_valid.argsort().argsort().float()
                tr_rank = tr_valid.argsort().argsort().float()
                fv_rank_centered = fv_rank - fv_rank.mean()
                tr_rank_centered = tr_rank - tr_rank.mean()
                rank_ic = (fv_rank_centered * tr_rank_centered).mean() / (fv_rank_centered.std() * tr_rank_centered.std() + 1e-8)
                if not math.isnan(rank_ic.item()):
                    rank_ic_list.append(rank_ic.item())

                median_val = fv_valid.median()
                long_mask = fv_valid > median_val
                short_mask = fv_valid < median_val
                if long_mask.sum() > 0 and short_mask.sum() > 0:
                    ls_ret = tr_valid[long_mask].mean().item() - tr_valid[short_mask].mean().item()
                    ls_returns.append(ls_ret)

            if not ic_list:
                return {
                    'ic_mean': 0.0, 'icir': 0.0,
                    'rank_ic_mean': 0.0, 'rank_icir': 0.0,
                    'long_short_return': 0.0, 'n_periods': 0, 'total_periods': n_periods, 'coverage_rate': 0.0,
                }

            ic_mean = float(np.mean(ic_list))
            ic_std = float(np.std(ic_list)) if len(ic_list) > 1 else 1.0
            icir = ic_mean / ic_std if ic_std > 1e-8 else 0.0

            rank_ic_mean = float(np.mean(rank_ic_list)) if rank_ic_list else 0.0
            rank_ic_std = float(np.std(rank_ic_list)) if len(rank_ic_list) > 1 else 1.0
            rank_icir = rank_ic_mean / rank_ic_std if rank_ic_std > 1e-8 else 0.0

            long_short_return = float(np.mean(ls_returns)) if ls_returns else 0.0

            return {
                'ic_mean': ic_mean,
                'icir': icir,
                'rank_ic_mean': rank_ic_mean,
                'rank_icir': rank_icir,
                'long_short_return': long_short_return,
                'n_periods': len(ic_list),
                'total_periods': n_periods,
                'coverage_rate': float(len(ic_list) / max(n_periods, 1)),
            }

        def _prepare_data(self, data_dict: Dict):
            symbols = sorted(data_dict.keys())
            n_symbols = len(symbols)

            all_dates = sorted(set().union(*[set(df.index) for df in data_dict.values()]))
            aligned_index = pd.Index(all_dates)
            min_coverage = float(self.config.get('min_date_coverage', 0.8))

            # 用DataFrame重索引一次性对齐时序，避免逐时间点赋值导致的三重循环开销
            aligned_frames = []
            for sym in symbols:
                df = data_dict[sym]
                frame = df.reindex(aligned_index).copy()
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    if col not in frame.columns:
                        frame[col] = np.nan
                frame = frame[['open', 'high', 'low', 'close', 'volume']]
                aligned_frames.append(frame)

            if aligned_frames:
                close_valid = np.stack([np.isfinite(f['close'].to_numpy(dtype=np.float32)) for f in aligned_frames], axis=0)
                coverage = close_valid.mean(axis=0)
                keep_mask = coverage >= min_coverage
                if keep_mask.any():
                    aligned_index = aligned_index[keep_mask]
                    aligned_frames = [f.loc[aligned_index] for f in aligned_frames]

            for i in range(len(aligned_frames)):
                aligned_frames[i] = aligned_frames[i].ffill().bfill().fillna(0.0)

            raw_tensors = {}
            for col in ['open', 'high', 'low', 'close', 'volume']:
                arr = np.stack([f[col].to_numpy(dtype=np.float32) for f in aligned_frames], axis=0)
                raw_tensors[col] = torch.tensor(arr, device=self.device)

            for col in ['open', 'high', 'low', 'close']:
                raw_tensors[col] = torch.nan_to_num(raw_tensors[col], nan=1.0, posinf=1.0, neginf=1.0).clamp(min=1e-6)
            raw_tensors['volume'] = torch.nan_to_num(raw_tensors['volume'], nan=1.0, posinf=1e12, neginf=1.0)
            raw_tensors['volume'] = raw_tensors['volume'].clamp(min=1.0)

            feat_tensor = RLFeatureEngineer.compute_features(raw_tensors)
            feat_tensor = torch.nan_to_num(feat_tensor, nan=0.0, posinf=5.0, neginf=-5.0)
            feat_tensor = torch.clamp(feat_tensor, -5.0, 5.0)

            c = raw_tensors['close']
            t1 = torch.roll(c, -1, dims=1)
            t2 = torch.roll(c, -2, dims=1)
            target_ret = torch.log(t2 / (t1 + 1e-9))
            target_ret[:, -2:] = 0.0
            target_ret = torch.nan_to_num(target_ret, nan=0.0, posinf=1.0, neginf=-1.0)
            target_ret = torch.clamp(target_ret, -1.0, 1.0)

            return raw_tensors, feat_tensor, target_ret

        def _collect_top_formulas(self, feat_tensor, raw_data, target_ret):
            self.model.eval()
            with torch.no_grad():
                n_sample = min(self.batch_size * 2, 2048)
                inp = torch.zeros((n_sample, 1), dtype=torch.long, device=self.device)
                stack_slots = torch.zeros(n_sample, dtype=torch.long, device=self.device)
                tokens_list = []

                for step_idx in range(self.max_formula_len):
                    logits, _, _ = self.model(inp)
                    logits = torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0)
                    logits = torch.clamp(logits, -20.0, 20.0)
                    action_mask = self._build_postfix_action_mask(stack_slots, step_idx)
                    logits = logits + action_mask
                    if not torch.isfinite(logits).any(dim=1).all():
                        fallback = ~torch.isfinite(logits).any(dim=1)
                        logits[fallback, :self.vm.feat_offset] = 0.0
                    temp = max(self.collect_temperature, 1e-6)
                    logits = logits / temp
                    if self.collect_top_k > 0 and self.collect_top_k < logits.shape[-1]:
                        topk_vals, topk_idx = torch.topk(logits, self.collect_top_k, dim=-1)
                        filtered_logits = torch.full_like(logits, float('-inf'))
                        filtered_logits.scatter_(1, topk_idx, topk_vals)
                        logits = filtered_logits
                    dist = Categorical(logits=logits)
                    action = dist.sample()
                    tokens_list.append(action)
                    inp = torch.cat([inp, action.unsqueeze(1)], dim=1)
                    is_op = action >= self.vm.feat_offset
                    delta = torch.ones_like(stack_slots)
                    if is_op.any():
                        delta[is_op] = 1 - self.arity_tensor[action[is_op]]
                    stack_slots = torch.clamp(stack_slots + delta, min=0)

                seqs = torch.stack(tokens_list, dim=1)
                unique_formulas = {}
                for formula in seqs.tolist():
                    unique_formulas.setdefault(tuple(formula), formula)

                existing_formula_keys = {tuple(f.get('formula', [])) for f in self.best_formulas}
                eval_cache = {}

                for formula_key, formula in unique_formulas.items():
                    if formula_key in existing_formula_keys:
                        continue

                    cached = eval_cache.get(formula_key)
                    if cached is not None:
                        score_val, ret_val = cached
                    else:
                        res = self.vm.execute(formula, feat_tensor)
                        if res is None or res.std() < 1e-6:
                            continue
                        score, ret_val = self.bt.evaluate(res, raw_data, target_ret)
                        score_val = score.item()
                        eval_cache[formula_key] = (score_val, ret_val)

                    if score_val > self.min_reward_threshold:
                        self.best_formulas.append({
                            'formula': formula,
                            'score': score_val,
                            'avg_return': ret_val,
                            'expression': self.vm.decode_formula(formula),
                        })
                        existing_formula_keys.add(formula_key)

            self.best_formulas.sort(key=lambda x: x['score'], reverse=True)
            self.best_formulas = self.best_formulas[:self.top_k_keep * 5]
            self.model.train()

        def _diversify_formulas(self, feat_tensor, raw_data, target_ret):
            if not self.best_formulas:
                return []

            formula_eval_cache = {}
            scored = []
            for f_info in self.best_formulas:
                formula = f_info.get('formula', [])
                formula_key = tuple(formula)
                cached = formula_eval_cache.get(formula_key)
                if cached is None:
                    res = self.vm.execute(formula, feat_tensor)
                    if res is None:
                        continue
                    score, ret_val = self.bt.evaluate(res, raw_data, target_ret)
                    flat = res.flatten()
                    valid = torch.isfinite(flat)
                    if valid.sum() < 3:
                        continue
                    flat_valid = flat[valid]
                    centered = flat_valid - flat_valid.mean()
                    norm = centered.norm()
                    if norm < 1e-8:
                        continue
                    normed_vec = centered / norm
                    cached = (score.item(), ret_val, normed_vec)
                    formula_eval_cache[formula_key] = cached

                score_val, ret_val, normed_vec = cached
                scored.append({
                    **f_info,
                    'score': score_val,
                    'avg_return': ret_val,
                    '_normed_vector': normed_vec,
                })

            scored.sort(key=lambda x: x['score'], reverse=True)

            selected = []
            selected_vectors = []
            for f_info in scored:
                if len(selected) >= self.max_factors:
                    break

                if not selected:
                    selected.append(f_info)
                    selected_vectors.append(f_info.get('_normed_vector'))
                    continue

                vec_new = f_info.get('_normed_vector')
                if vec_new is None:
                    continue

                too_similar = False
                for vec_exist in selected_vectors:
                    if vec_exist is None:
                        continue
                    min_len = min(vec_new.shape[0], vec_exist.shape[0])
                    if min_len < 3:
                        continue
                    corr = torch.dot(vec_new[:min_len], vec_exist[:min_len]).item()
                    if math.isnan(corr):
                        corr = 0.0
                    if abs(corr) > self.max_correlation:
                        too_similar = True
                        break

                if not too_similar:
                    selected.append(f_info)
                    selected_vectors.append(vec_new)

            for f in selected:
                f.pop('_normed_vector', None)

            return selected

        def _report_progress(self, pct, message, detail=None):
            if self._progress_callback:
                self._progress_callback(pct, message, detail)


    def run_rl_cross_sectional_mining(
        data_dict: Dict,
        config: Dict = None,
        progress_callback: Callable = None
    ) -> Dict:
        miner = RLMiner(config)
        return miner.mine(data_dict, progress_callback)
