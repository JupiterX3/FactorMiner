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
        def __init__(self, trade_size=10000.0, min_volume=1000000.0, base_fee=0.001):
            self.trade_size = trade_size
            self.min_volume = min_volume
            self.base_fee = base_fee

        def evaluate(self, factors, raw_data, target_ret):
            volume = raw_data['volume']
            signal = torch.sigmoid(factors)

            is_safe = (volume > self.min_volume).float()
            position = (signal > 0.65).float() * is_safe

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

            big_drawdowns = (net_pnl < -0.03).float().sum(dim=1)
            score = cum_ret - (big_drawdowns * 1.5)

            activity = position.sum(dim=1)
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
            self.head_actor = nn.Linear(d_model, vocab_size)
            self.head_critic = nn.Linear(d_model, 1)

        def forward(self, idx):
            B, T = idx.size()
            x = self.token_emb(idx) + self.pos_emb[:, :T, :]

            mask = nn.Transformer.generate_square_subsequent_mask(T).to(idx.device)
            x = self.blocks(x, mask=mask, is_causal=True)
            x = self.ln_f(x)

            last_emb = x[:, -1, :]
            logits = self.head_actor(last_emb)
            value = self.head_critic(last_emb)

            return logits, value

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
            self.device = torch.device(
                self.config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
            )

            self.batch_size = int(self.config.get('batch_size', 512 if self.device.type == 'cuda' else 64))
            self.train_steps = int(self.config.get('train_steps', 500))
            self.max_formula_len = int(self.config.get('max_formula_len', 16))
            self.lr = float(self.config.get('lr', 1e-3))
            self.lord_decay_rate = float(self.config.get('lord_decay_rate', 1e-3))
            self.use_lord = bool(self.config.get('use_lord', True))
            self.entropy_coef = float(self.config.get('entropy_coef', 0.01))
            self.value_coef = float(self.config.get('value_coef', 0.5))
            self.max_factors = int(self.config.get('max_factors', 15))
            self.min_reward_threshold = float(self.config.get('min_reward', 0.0))
            self.max_correlation = float(self.config.get('max_correlation', 0.7))
            self.eval_interval = int(self.config.get('eval_interval', 50))
            self.top_k_keep = int(self.config.get('top_k_keep', 3))

            self.vm = StackVM()
            self.bt = CrossSectionalBacktest(
                trade_size=float(self.config.get('trade_size', 10000.0)),
                min_volume=float(self.config.get('min_volume', 1000000.0)),
                base_fee=float(self.config.get('base_fee', 0.001)),
            )

            self.vocab_size = INPUT_DIM + len(OPS_CONFIG)
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
                self.lord_opt = NewtonSchulzLowRankDecay(
                    self.model.named_parameters(),
                    decay_rate=self.lord_decay_rate,
                    target_keywords=["q_proj", "k_proj", "attention", "qk_norm", "head_actor"],
                )

            self.best_score = -float('inf')
            self.best_formula = None
            self.best_formulas: List[Dict] = []
            self.training_history: List[Dict] = []
            self._progress_callback = None
            self._stop_requested = False

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

                log_probs = []
                values = []
                tokens_list = []

                for t in range(self.max_formula_len):
                    logits, value = self.model(inp)
                    dist = Categorical(logits=logits)
                    action = dist.sample()

                    log_probs.append(dist.log_prob(action))
                    values.append(value.squeeze(-1))
                    tokens_list.append(action)
                    inp = torch.cat([inp, action.unsqueeze(1)], dim=1)

                seqs = torch.stack(tokens_list, dim=1)

                rewards = torch.zeros(self.batch_size, device=self.device)

                for i in range(self.batch_size):
                    formula = seqs[i].tolist()
                    res = self.vm.execute(formula, feat_tensor)

                    if res is None:
                        rewards[i] = -5.0
                        continue

                    if res.std() < 1e-4:
                        rewards[i] = -2.0
                        continue

                    score, ret_val = self.bt.evaluate(res, raw_data, target_ret)
                    rewards[i] = score

                    if score.item() > self.best_score:
                        self.best_score = score.item()
                        self.best_formula = formula
                        self._report_progress(
                            -1,
                            f"[!] 新最优: Score={score:.2f}, Ret={ret_val:.2%}, "
                            f"公式={self.vm.decode_formula(formula)}"
                        )

                adv = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

                policy_loss = 0
                for t in range(len(log_probs)):
                    policy_loss += -log_probs[t] * adv.detach()
                policy_loss = policy_loss.mean()

                value_loss = 0
                values_tensor = torch.stack(values, dim=0)
                for t in range(len(values_tensor)):
                    value_loss += F.mse_loss(values_tensor[t], rewards.detach())
                value_loss = value_loss.mean()

                entropy = 0
                for t in range(len(log_probs)):
                    entropy += -(log_probs[t].exp() * log_probs[t]).mean()

                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                self.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.opt.step()

                if self.use_lord and self.lord_opt:
                    self.lord_opt.step()

                avg_reward = rewards.mean().item()
                max_reward = rewards.max().item()

                self.training_history.append({
                    'step': step,
                    'avg_reward': avg_reward,
                    'max_reward': max_reward,
                    'best_score': self.best_score,
                    'policy_loss': policy_loss.item(),
                    'value_loss': value_loss.item(),
                    'entropy': entropy.item(),
                })

                if step % self.eval_interval == 0 or step == self.train_steps - 1:
                    self._collect_top_formulas(feat_tensor, raw_data, target_ret)

                progress_pct = 10 + int((step + 1) / self.train_steps * 80)
                self._report_progress(
                    progress_pct,
                    f"Step {step+1}/{self.train_steps}: "
                    f"AvgRew={avg_reward:.3f}, MaxRew={max_reward:.3f}, "
                    f"Best={self.best_score:.3f}"
                )

            self._report_progress(92, "训练完成，正在筛选多样化因子...")

            self._collect_top_formulas(feat_tensor, raw_data, target_ret)

            diverse_formulas = self._diversify_formulas(feat_tensor, raw_data, target_ret)

            factors_result = []
            for i, f_info in enumerate(diverse_formulas[:self.max_factors]):
                formula_tokens = f_info['formula']
                factor_values = self.vm.execute(formula_tokens, feat_tensor)

                factor_data = {}
                if factor_values is not None:
                    symbols_list = list(data_dict.keys())
                    for si, sym in enumerate(symbols_list):
                        dates = data_dict[sym].index
                        n_dates = min(len(dates), factor_values.shape[1])
                        factor_data[sym] = {
                            'dates': [str(d) for d in dates[:n_dates]],
                            'values': factor_values[si, :n_dates].cpu().tolist(),
                        }

                factors_result.append({
                    'factor_id': f'rl_cs_{i+1}',
                    'name': f'RL_CrossSectional_{i+1}',
                    'expression': self.vm.decode_formula(formula_tokens),
                    'score': f_info['score'],
                    'avg_return': f_info['avg_return'],
                    'formula_tokens': formula_tokens,
                    'factor_data': factor_data,
                })

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

        def _prepare_data(self, data_dict: Dict):
            symbols = sorted(data_dict.keys())
            n_symbols = len(symbols)

            all_dates = sorted(set().union(*[set(df.index) for df in data_dict.values()]))
            n_periods = len(all_dates)
            date_idx = {d: i for i, d in enumerate(all_dates)}

            raw_tensors = {}
            for col in ['open', 'high', 'low', 'close', 'volume']:
                arr = np.zeros((n_symbols, n_periods), dtype=np.float32)
                for si, sym in enumerate(symbols):
                    df = data_dict[sym]
                    if col in df.columns:
                        for d, v in df[col].items():
                            if d in date_idx:
                                arr[si, date_idx[d]] = v
                raw_tensors[col] = torch.tensor(arr, device=self.device)

            raw_tensors['volume'] = raw_tensors['volume'].clamp(min=1.0)

            feat_tensor = RLFeatureEngineer.compute_features(raw_tensors)

            c = raw_tensors['close']
            t1 = torch.roll(c, -1, dims=1)
            t2 = torch.roll(c, -2, dims=1)
            target_ret = torch.log(t2 / (t1 + 1e-9))
            target_ret[:, -2:] = 0.0

            return raw_tensors, feat_tensor, target_ret

        def _collect_top_formulas(self, feat_tensor, raw_data, target_ret):
            self.model.eval()
            with torch.no_grad():
                n_sample = min(self.batch_size * 2, 2048)
                inp = torch.zeros((n_sample, 1), dtype=torch.long, device=self.device)
                tokens_list = []

                for _ in range(self.max_formula_len):
                    logits, _ = self.model(inp)
                    action = torch.argmax(logits, dim=-1)
                    tokens_list.append(action)
                    inp = torch.cat([inp, action.unsqueeze(1)], dim=1)

                seqs = torch.stack(tokens_list, dim=1)

                for i in range(n_sample):
                    formula = seqs[i].tolist()
                    res = self.vm.execute(formula, feat_tensor)
                    if res is None or res.std() < 1e-4:
                        continue

                    score, ret_val = self.bt.evaluate(res, raw_data, target_ret)

                    if score.item() > self.min_reward_threshold:
                        exists = any(f['formula'] == formula for f in self.best_formulas)
                        if not exists:
                            self.best_formulas.append({
                                'formula': formula,
                                'score': score.item(),
                                'avg_return': ret_val,
                                'expression': self.vm.decode_formula(formula),
                            })

            self.best_formulas.sort(key=lambda x: x['score'], reverse=True)
            self.best_formulas = self.best_formulas[:self.top_k_keep * 5]
            self.model.train()

        def _diversify_formulas(self, feat_tensor, raw_data, target_ret):
            if not self.best_formulas:
                return []

            scored = []
            for f_info in self.best_formulas:
                res = self.vm.execute(f_info['formula'], feat_tensor)
                if res is None:
                    continue
                score, ret_val = self.bt.evaluate(res, raw_data, target_ret)
                f_info['score'] = score.item()
                f_info['avg_return'] = ret_val
                f_info['factor_values'] = res
                scored.append(f_info)

            scored.sort(key=lambda x: x['score'], reverse=True)

            selected = []
            for f_info in scored:
                if len(selected) >= self.max_factors:
                    break

                if not selected:
                    selected.append(f_info)
                    continue

                res_new = f_info.get('factor_values')
                if res_new is None:
                    continue

                too_similar = False
                for existing in selected:
                    res_exist = existing.get('factor_values')
                    if res_exist is None:
                        continue
                    corr = torch.corrcoef(torch.stack([
                        res_new.flatten(), res_exist.flatten()
                    ]))[0, 1].item()
                    if math.isnan(corr):
                        corr = 0.0
                    if abs(corr) > self.max_correlation:
                        too_similar = True
                        break

                if not too_similar:
                    selected.append(f_info)

            for f in selected:
                f.pop('factor_values', None)

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
