# RL截面因子挖掘器 — 实现与设计思路

## 1. 概述

FactorMiner的RL截面因子挖掘器借鉴了AlphaGPT的核心架构——REINFORCE策略梯度 + Looped Transformer自回归生成 + StackVM公式执行——并将其适配到主流加密货币的截面因子挖掘场景。

**核心思想**：将因子公式搜索建模为序列决策问题，Transformer作为策略网络自回归地生成token序列（即因子公式），StackVM执行公式得到因子值，截面回测评估因子质量并产生reward信号，REINFORCE算法利用reward信号更新策略网络参数。

---

## 2. 算子的作用与扩展原因

### 2.1 算子是什么？

算子（Operator）是因子表达式的基本构建块。在RL框架中，每个算子对应一个token ID，Transformer策略网络通过采样生成token序列来组合出因子公式。StackVM基于栈式虚拟机执行这些公式。

**算子的分类**：

| 类别 | 算子 | 说明 |
|------|------|------|
| **二元算术** | ADD, SUB, MUL, DIV | 基础四则运算，组合特征 |
| **二元比较** | MAX, MIN | 取两个值的极值 |
| **一元数学** | NEG, ABS, SIGN, SQRT, LOG | 数值变换 |
| **条件控制** | GATE(3元) | 条件选择：`if condition > 0 then x else y` |
| **异常检测** | JUMP | 检测z-score > 3的异常值 |
| **时序衰减** | DECAY | 加权延迟：`x + 0.8*delay(x,1) + 0.6*delay(x,2)` |
| **时序延迟** | DELAY1, DELAY3 | 引入历史信息 |
| **移动平均** | MA5, MA10, MA20 | 不同窗口的趋势平滑 |
| **波动率** | STD5, STD10 | 不同窗口的波动率 |
| **时序排名** | RANK5 | 滚动窗口内排名 |
| **动量** | MOM5, MOM10 | 不同窗口的动量信号 |
| **截面排名** | CS_RANK | 截面内排名（跨币种比较） |
| **截面标准化** | CS_ZSCORE | 截面内z-score标准化 |
| **截面鲁棒标准化** | CS_MAD | 基于MAD的鲁棒截面标准化 |

### 2.2 AlphaGPT的12个算子

AlphaGPT面向Meme币DEX交易，其算子集为：

```
ADD, SUB, MUL, DIV, NEG, ABS, SIGN, GATE, JUMP, DECAY, DELAY1, MAX3
```

这些算子足够表达Meme币的短线交易逻辑（FOMO、pump detection等），但缺少：
- 多窗口时序算子（只有DELAY1和MAX3，无MA/STD/MOM等）
- 截面算子（无CS_RANK/CS_ZSCORE，因为Meme币间截面关系弱）
- 安全数学函数（无SQRT/LOG，Meme币价格波动大不需要）

### 2.3 为什么要扩展到27个算子？

**原因1：主流币种需要更丰富的时序特征**

Meme币的生命周期短（通常几天到几周），AlphaGPT只需要DELAY1和MAX3就能捕捉短期动量。但主流加密货币（BTC、ETH等）的行情周期更长，需要：
- 多窗口移动平均（MA5/10/20）捕捉不同级别的趋势
- 多窗口波动率（STD5/10）衡量不同时间尺度的风险
- 多窗口动量（MOM5/10）捕捉不同周期的价格变化

**原因2：截面因子挖掘需要截面算子**

AlphaGPT的Meme币场景中，不同币种之间几乎没有可比性（流动性、市值差异巨大），因此不需要截面算子。但在主流币截面挖掘中：
- CS_RANK：对因子值在截面上排名，消除量纲差异
- CS_ZSCORE：截面标准化，使因子值可跨币种比较
- CS_MAD：基于中位数绝对偏差的鲁棒标准化，抗异常值

**原因3：安全数学函数避免数值溢出**

主流币的价格和成交量范围更广，需要SQRT和LOG来压缩数值范围，避免浮点溢出。

**原因4：增加搜索空间的多样性**

更多算子意味着策略网络可以探索更丰富的公式结构。虽然搜索空间增大，但REINFORCE的梯度信号能有效引导搜索方向，不会像GP那样陷入随机搜索。

### 2.4 算子扩展的权衡

| 维度 | AlphaGPT(12算子) | FactorMiner(27算子) |
|------|-------------------|---------------------|
| 搜索空间大小 | vocab=5+12=17 | vocab=10+27=37 |
| 公式复杂度 | 简单，偏短线 | 丰富，支持多周期+截面 |
| 训练难度 | 较低 | 较高（需更多步数） |
| 因子质量 | 适合Meme短线 | 适合主流币中长线 |
| 截面能力 | 无 | 有（CS_RANK/ZSCORE/MAD） |

---

## 3. 架构设计

### 3.1 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                    RL截面因子挖掘器 (RLMiner)                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │  数据准备     │───▶│  特征工程     │───▶│  GPU Tensor  │      │
│  │ _prepare_data │    │ RLFeature    │    │  feat_tensor │      │
│  │              │    │ Engineer     │    │  raw_data    │      │
│  └──────────────┘    └──────────────┘    │  target_ret  │      │
│                                           └──────┬───────┘      │
│                                                  │              │
│  ┌───────────────────────────────────────────────▼─────────┐    │
│  │              REINFORCE 训练循环                           │    │
│  │                                                          │    │
│  │  ┌────────────┐   sample   ┌──────────┐   execute  ┌──┴──┐│
│  │  │ AlphaPolicy│───────────▶│ token序列 │──────────▶│Stack│││
│  │  │ (Looped    │            │ (公式)    │           │ VM  │││
│  │  │  Transformer)│          └──────────┘           └──┬──┘││
│  │  └─────┬──────┘                                     │   ││
│  │        │                                     因子值  │   ││
│  │   logits, value                                    │   ││
│  │        │                                           ▼   ││
│  │        │                                  ┌──────────┐ ││
│  │        │                                  │ 截面回测  │ ││
│  │        │                                  │ CrossSec  │ ││
│  │        │                                  │ Backtest  │ ││
│  │        │                                  └─────┬────┘ ││
│  │        │                                        │      ││
│  │        │              reward信号                 │      ││
│  │        │◄───────────────────────────────────────┘      ││
│  │        │                                              ││
│  │  ┌─────▼──────────────────────────────────────────┐   ││
│  │  │ 策略梯度更新                                     │   ││
│  │  │ loss = policy_loss + value_coef*value_loss      │   ││
│  │  │        - entropy_coef*entropy                   │   ││
│  │  │ + LoRD正则化                                    │   ││
│  │  └────────────────────────────────────────────────┘   ││
│  └───────────────────────────────────────────────────────┘    │
│                                                                 │
│  ┌───────────────────────────────────────────────────────┐    │
│  │ 因子筛选与多样化                                       │    │
│  │ _collect_top_formulas → _diversify_formulas           │    │
│  │ (相关性过滤, max_correlation=0.7)                      │    │
│  └───────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 核心组件详解

#### 3.2.1 AlphaPolicy — Looped Transformer策略网络

```python
class AlphaPolicy(nn.Module):
    # 输入: token序列 [B, T]
    # 输出: logits [B, vocab_size], value [B, 1]

    # 架构:
    # Token Embedding + Positional Embedding
    # → LoopedTransformer (2层, 每层循环3次)
    #   → 每层: QK-Norm Attention + SwiGLU FFN
    # → RMSNorm
    # → Actor Head (Linear → vocab_size)
    # → Critic Head (Linear → 1)
```

**与AlphaGPT的区别**：

| 组件 | AlphaGPT | FactorMiner |
|------|----------|-------------|
| 输出头 | MTPHead (多任务路由) | 单Actor + Critic |
| 多任务 | 3个任务头+路由网络 | 无（简化） |
| 返回值 | logits, value, task_probs | logits, value |

**为什么不用MTPHead？**

AlphaGPT的MTPHead（Multi-Task Pooling Head）设计用于同时优化多个交易策略目标。在FactorMiner的截面因子挖掘场景中，我们的目标是单一的——最大化截面回测收益，因此不需要多任务路由机制。简化为单Actor+Critic头减少了参数量，降低了过拟合风险。

#### 3.2.2 StackVM — 栈式虚拟机

StackVM是公式执行引擎，基于后缀表达式（逆波兰表示法）的栈式计算：

```
输入: [RET, VOL, MUL, MA5, SUB, CS_RANK]
执行过程:
  1. RET → stack: [RET值]
  2. VOL → stack: [RET值, VOL值]
  3. MUL → pop 2, push RET*VOL → stack: [RET*VOL]
  4. MA5 → stack: [RET*VOL, MA5(RET*VOL)]  ← 注意: MA5是一元算子
  5. SUB → pop 2, push RET*VOL - MA5(RET*VOL) → stack: [差值]
  6. CS_RANK → pop 1, push 截面排名 → stack: [排名值]
结果: 截面排名值
```

**与AlphaGPT StackVM的区别**：

AlphaGPT的StackVM完全相同——这是直接复用的组件。唯一的区别是`feat_offset`（特征偏移量），AlphaGPT为6，FactorMiner为10。

#### 3.2.3 RLFeatureEngineer — 特征工程

FactorMiner的10维特征空间：

| # | 特征名 | 计算方式 | 含义 |
|---|--------|----------|------|
| 0 | RET | log(close/prev_close) | 对数收益率 |
| 1 | VOL | log(1+volume) | 对数成交量 |
| 2 | V_CHG | (vol-prev_vol)/prev_vol | 成交量变化率 |
| 3 | PV | RET * VOL | 量价交互 |
| 4 | TREND | (close-MA5)/MA5 | 趋势偏离 |
| 5 | HL_RANGE | (high-low)/close | 振幅 |
| 6 | CLOSE_POS | (close-low)/(high-low) | 收盘位置 |
| 7 | MA_DEV | (close-MA5)/MA5 | 均线偏离 |
| 8 | VOLATILITY | sqrt(rolling_mean(ret^2, 10)) | 已实现波动率 |
| 9 | MOMENTUM | rolling_sum(ret, 5) | 动量 |

AlphaGPT的6维特征空间：

| # | 特征名 | 计算方式 | 含义 |
|---|--------|----------|------|
| 0 | RET | log(close/prev_close) | 对数收益率 |
| 1 | LIQ_SCORE | liquidity/(fdv+1e-6)*4 | 流动性健康度 |
| 2 | PRESSURE | tanh((close-open)/(high-low)*3) | 买卖压力 |
| 3 | FOMO | vol_chg加速度 | FOMO加速 |
| 4 | DEV | (close-MA20)/MA20 | Pump偏离 |
| 5 | LOG_VOL | log(1+volume) | 对数成交量 |

**关键差异**：
- AlphaGPT使用`liquidity`和`fdv`（Meme币DEX特有指标），FactorMiner使用`volume`和`close`（CEX通用指标）
- AlphaGPT的PRESSURE和FOMO是Meme币特有的情绪指标，FactorMiner用VOLATILITY和MOMENTUM替代
- FactorMiner增加了CLOSE_POS（K线形态）和PV（量价交互）

#### 3.2.4 CrossSectionalBacktest — 截面回测评估器

**回测逻辑**：

```
1. signal = sigmoid(factors)           # 因子值 → 信号强度
2. position = (signal > 0.65) * safe   # 信号阈值 + 安全过滤
3. impact = trade_size / (volume*close) # 市场冲击
4. total_cost = base_fee + impact       # 总交易成本
5. net_pnl = position * target_ret - turnover * total_cost
6. score = cum_ret - 1.5 * big_drawdowns - (activity<5 ? -10 : 0)
7. final = median(score)                # 中位数作为适应度
```

**与AlphaGPT MemeBacktest的区别**：

| 参数 | AlphaGPT | FactorMiner | 原因 |
|------|----------|-------------|------|
| trade_size | $1,000 | $10,000 | 主流币流动性更好 |
| min_safety | liquidity > $500K | volume > $1M | CEX用成交量，DEX用流动性 |
| base_fee | 0.6% | 0.1% | CEX手续费远低于DEX |
| signal_threshold | 0.85 | 0.65 | 主流币波动更小，需要更敏感的信号 |
| drawdown_penalty | -5% → 2.0x | -3% → 1.5x | 主流币回撤更小 |
| impact_clamp | 5% | 2% | 主流币冲击更小 |

#### 3.2.5 NewtonSchulzLowRankDecay — LoRD正则化

**直接复用AlphaGPT的实现**，核心算法：

```
Newton-Schulz迭代: Y_{k+1} = 0.5 * Y_k * (3I - Y_k^T * Y_k)
收敛到X的最近正交矩阵，然后对权重施加低秩衰减:
W -= decay_rate * Y
```

**作用**：防止注意力权重的有效秩坍缩，保持模型的表达能力。在因子挖掘中，这防止策略网络过早收敛到单一类型的公式。

**FactorMiner的扩展**：target_keywords增加了`"head_actor"`，因为Actor头直接决定公式生成，其权重矩阵也需要保持低秩结构。

---

## 4. 训练流程详解

### 4.1 REINFORCE算法

```
for step in range(train_steps):
    # 1. 生成公式
    inp = zeros(B, 1)  # 起始token
    for t in range(max_formula_len):
        logits, value = model(inp)
        dist = Categorical(logits)
        action = dist.sample()
        log_probs.append(dist.log_prob(action))
        values.append(value)
        inp = cat([inp, action])

    # 2. 执行并评估
    for each formula in batch:
        result = StackVM.execute(formula, feat_tensor)
        reward = CrossSectionalBacktest.evaluate(result)

    # 3. 计算优势函数
    advantage = (rewards - mean(rewards)) / (std(rewards) + 1e-5)

    # 4. 策略梯度损失
    policy_loss = mean(-log_probs * advantage.detach())

    # 5. 价值函数损失（baseline）
    value_loss = mean(MSE(values, rewards.detach()))

    # 6. 熵正则化（鼓励探索）
    entropy = mean(-log_probs.exp() * log_probs)

    # 7. 总损失
    loss = policy_loss + 0.5*value_loss - 0.01*entropy

    # 8. 梯度更新 + LoRD
    loss.backward()
    clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    lord_opt.step()  # LoRD正则化
```

### 4.2 与AlphaGPT训练的区别

| 维度 | AlphaGPT | FactorMiner |
|------|----------|-------------|
| 算法 | REINFORCE (纯策略梯度) | REINFORCE + Value Baseline |
| 价值函数 | 无 | 有 (Critic Head) |
| 熵正则 | 无 | 有 (entropy_coef=0.01) |
| 梯度裁剪 | 无 | 有 (max_norm=1.0) |
| Batch Size | 8192 | 512(GPU)/64(CPU) |
| 训练步数 | 1000 | 500 |
| 公式长度 | 12 | 16 |
| 评估间隔 | 无 | 每50步评估+收集Top公式 |
| 因子多样化 | 无 | 有 (相关性过滤<0.7) |

**为什么增加Value Baseline？**

纯REINFORCE的方差很大，尤其是在截面回测reward分布不稳定时。Value Baseline通过Critic网络估计状态价值，用advantage = reward - baseline代替原始reward，显著降低梯度方差。

**为什么增加熵正则？**

因子挖掘需要探索多样化的公式结构。没有熵正则，策略网络容易过早收敛到局部最优（如总是生成同一种简单公式）。熵正则鼓励策略保持探索性。

**为什么Batch Size更小？**

AlphaGPT在GPU上运行8192 batch，因为Meme币数据量大（500+币种）。FactorMiner面向主流币（通常10-50个），截面维度更小，512 batch已足够。CPU模式下更降至64。

---

## 5. GPU加速配置

### 5.1 环境要求

```
# 必需
Python >= 3.8
PyTorch >= 2.0 (with CUDA support)

# 推荐GPU
NVIDIA GPU with >= 4GB VRAM (RTX 3060及以上)
CUDA >= 11.8
```

### 5.2 安装步骤

```bash
# 方式1: pip安装 (推荐)
pip install torch --index-url https://download.pytorch.org/whl/cu118

# 方式2: conda安装
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# 验证安装
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
```

### 5.3 配置方式

**方式A: WebUI配置**

在WebUI的"截面RL"模式中，系统自动检测GPU：
- 如果检测到CUDA → 自动使用GPU，默认batch_size=512
- 如果无CUDA → 使用CPU，默认batch_size=64

**方式B: 代码配置**

```python
from factor_miner.core.rl_miner import RLMiner

config = {
    'device': 'cuda',           # 'cuda', 'cpu', 或 'auto'
    'batch_size': 512,          # GPU推荐512-2048, CPU推荐32-128
    'train_steps': 500,         # GPU推荐500-2000, CPU推荐100-300
    'd_model': 64,              # 模型维度
    'nhead': 4,                 # 注意力头数
    'num_layers': 2,            # Transformer层数
    'num_loops': 3,             # 每层循环次数
    'lr': 1e-3,                 # 学习率
    'use_lord': True,           # 启用LoRD正则化
}

miner = RLMiner(config)
result = miner.mine(data_dict, progress_callback=callback)
```

### 5.4 性能对比

| 配置 | 训练500步耗时 | 每步评估公式数 |
|------|-------------|--------------|
| CPU (i7-12700) | ~30分钟 | 64 |
| GPU (RTX 3060) | ~3分钟 | 512 |
| GPU (RTX 4090) | ~1分钟 | 2048 |

**加速关键点**：
1. 特征张量预计算并常驻GPU，避免CPU-GPU数据传输
2. StackVM执行在GPU上完成（所有算子都是torch操作）
3. 截面回测评估在GPU上完成（矩阵运算）
4. LoRD正则化的Newton-Schulz迭代在GPU上完成

---

## 6. 因子筛选与多样化

### 6.1 Top-K收集

训练过程中，每隔`eval_interval`步（默认50步），使用贪心解码（argmax）生成一批公式，评估后保留得分最高的Top-K*5个。

### 6.2 相关性过滤

最终输出时，按得分排序依次选择因子，如果新因子与已选因子的相关系数超过`max_correlation`（默认0.7），则跳过。这确保输出的因子集具有多样性。

```python
for f_info in scored_formulas:
    too_similar = False
    for existing in selected:
        corr = abs(correlation(f_info, existing))
        if corr > 0.7:
            too_similar = True
            break
    if not too_similar:
        selected.append(f_info)
```

### 6.3 与AlphaGPT的区别

AlphaGPT只输出单个最优公式（best_formula），没有多样化筛选。FactorMiner输出最多`max_factors`（默认15）个低相关因子，更适合构建多因子组合策略。

---

## 7. 文件结构

```
factor_miner/core/rl_miner.py     # RL挖掘器核心实现
factor_miner/core/__init__.py     # 导出RLMiner, TORCH_AVAILABLE
webui/routes/mining_api.py        # RL挖掘API端点
webui/static/js/factor_mining.js  # 前端RL配置与交互
webui/templates/factor_mining.html # RL挖掘UI模板
```

---

## 8. API端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/mining/cross_sectional_rl/start` | POST | 启动RL截面挖掘 |
| `/api/mining/cross_sectional_rl/status/<session_id>` | GET | 查询挖掘状态 |
| `/api/mining/cross_sectional_rl/stop/<session_id>` | POST | 停止挖掘 |

---

## 9. 限制与未来改进

### 当前限制
1. **PyTorch依赖**：RL挖掘器必须安装PyTorch，否则不可用（优雅降级为提示信息）
2. **CPU训练极慢**：无GPU时训练耗时约30分钟，实用性有限
3. **无预训练**：策略网络从零开始训练，未利用已有因子知识
4. **单步reward**：每个公式独立评估，未考虑公式间的组合效应

### 未来改进方向
1. **课程学习**：先训练简单公式，逐步增加复杂度
2. **经验回放**：缓存历史优秀公式，加速收敛
3. **PPO替代REINFORCE**：更稳定的策略优化算法
4. **多GPU并行**：支持分布式训练，加速大规模截面搜索
5. **预训练+微调**：在历史因子数据上预训练策略网络
