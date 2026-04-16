# 截面因子挖掘完整流程 — 与AlphaGPT对比

## 1. 截面因子挖掘概述

截面因子挖掘（Cross-Sectional Factor Mining）是指在多个资产（如多种加密货币）的同一时间截面上，寻找能够预测未来收益的统计特征（因子）。与时间序列因子不同，截面因子利用的是**资产间的相对排序关系**，而非单个资产的时间序列模式。

**核心问题**：给定N个资产在T个时间点的数据，找到一个函数f，使得f在每个时间截面上对N个资产的排序与未来收益的排序高度相关。

---

## 2. FactorMiner截面因子挖掘完整流程

FactorMiner提供两种截面因子挖掘方法：**遗传编程（GP）** 和 **强化学习（RL）**。

### 2.1 整体流程图

```
┌─────────────────────────────────────────────────────────────────────┐
│                     截面因子挖掘完整流程                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Step 1: 数据获取                                                   │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ 用户选择交易对(≥3) + 时间框架 + 日期范围 + 数据源              │   │
│  │ → Binance API / Yahoo Finance → DataFrame字典               │   │
│  │   {symbol: DataFrame(open,high,low,close,volume)}           │   │
│  └────────────────────────────┬────────────────────────────────┘   │
│                               │                                     │
│  Step 2: 数据对齐与预处理       │                                     │
│  ┌────────────────────────────▼────────────────────────────────┐   │
│  │ • 对齐所有交易对的时间索引（取交集）                           │   │
│  │ • 前向填充缺失值                                             │   │
│  │ • 对齐为 (N_symbols, N_periods) 矩阵                        │   │
│  └────────────────────────────┬────────────────────────────────┘   │
│                               │                                     │
│  Step 3: 特征工程              │                                     │
│  ┌────────────────────────────▼────────────────────────────────┐   │
│  │ GP路径: pandas计算10维特征 (returns, volatility, etc.)       │   │
│  │ RL路径: torch计算10维特征 → GPU Tensor (N, 10, T)           │   │
│  │                                                              │   │
│  │ 特征列表: RET, VOL, V_CHG, PV, TREND, HL_RANGE,            │   │
│  │          CLOSE_POS, MA_DEV, VOLATILITY, MOMENTUM            │   │
│  └────────────────────────────┬────────────────────────────────┘   │
│                               │                                     │
│              ┌────────────────┴────────────────┐                    │
│              │                                 │                    │
│    Step 4a: GP搜索              Step 4b: RL搜索                     │
│    ┌──────────────────┐        ┌──────────────────────┐            │
│    │ 初始化种群        │        │ 初始化Transformer     │            │
│    │ (随机表达式树)    │        │ 策略网络              │            │
│    │                   │        │                       │            │
│    │ for gen in gens:  │        │ for step in steps:    │            │
│    │   评估适应度      │        │   采样公式序列        │            │
│    │   (IC/IR)        │        │   StackVM执行         │            │
│    │   锦标赛选择      │        │   截面回测评估        │            │
│    │   交叉+变异      │        │   REINFORCE更新       │            │
│    │   精英保留       │        │   LoRD正则化          │            │
│    │                   │        │                       │            │
│    │ 输出: Top-K因子   │        │ 输出: Top-K因子       │            │
│    └────────┬─────────┘        └──────────┬───────────┘            │
│             │                              │                        │
│             └──────────────┬───────────────┘                        │
│                            │                                        │
│  Step 5: 因子筛选与多样化   │                                        │
│  ┌─────────────────────────▼────────────────────────────────────┐  │
│  │ • 按得分排序                                                 │  │
│  │ • 相关性过滤 (|corr| > 0.7 则剔除)                           │  │
│  │ • 最多保留 max_factors 个因子                                 │  │
│  └────────────────────────────┬─────────────────────────────────┘  │
│                               │                                     │
│  Step 6: 结果输出             │                                     │
│  ┌────────────────────────────▼─────────────────────────────────┐  │
│  │ • 因子表达式 (人类可读)                                      │  │
│  │ • 因子值时序 (每个币种的因子值序列)                            │  │
│  │ • 评估指标 (score, avg_return, IC, IR)                       │  │
│  │ • 训练历史 (RL模式)                                          │  │
│  │ • 保存到 factorlib (可选)                                    │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 Step 1: 数据获取

**输入**：用户在WebUI中选择参数
- 交易对列表（至少3个，如BTCUSDT, ETHUSDT, SOLUSDT...）
- 时间框架（1d, 4h, 1h等）
- 起止日期
- 数据源（Binance期货 / Yahoo Finance）

**输出**：`Dict[str, pd.DataFrame]`，每个DataFrame包含open/high/low/close/volume列

**数据源差异**：

| 数据源 | 适用场景 | 特点 |
|--------|----------|------|
| Binance期货 | 主流加密货币 | 高频数据，流动性好，有永续合约 |
| Yahoo Finance | 股票/ETF | 日频数据，适合传统金融 |

### 2.3 Step 2: 数据对齐与预处理

```
原始数据: 每个币种的DataFrame，时间索引可能不完全对齐

对齐逻辑:
1. 取所有币种时间索引的交集
2. 前向填充缺失值
3. 转为 (N_symbols, N_periods) 矩阵

GP路径: 保持pandas DataFrame格式
RL路径: 转为torch.Tensor并移至GPU
```

### 2.4 Step 3: 特征工程

**10维特征空间**（GP和RL共用相同特征定义，实现方式不同）：

| # | 特征 | 公式 | 金融含义 |
|---|------|------|----------|
| 0 | RET | log(C_t / C_{t-1}) | 对数收益率 |
| 1 | VOL | log(1 + Volume_t) | 对数成交量（压缩量级） |
| 2 | V_CHG | (V_t - V_{t-1}) / V_{t-1} | 成交量变化率 |
| 3 | PV | RET * VOL | 量价交互因子 |
| 4 | TREND | (C_t - MA5_t) / MA5_t | 趋势偏离度 |
| 5 | HL_RANGE | (H_t - L_t) / C_t | 日内振幅 |
| 6 | CLOSE_POS | (C_t - L_t) / (H_t - L_t) | K线收盘位置 |
| 7 | MA_DEV | (C_t - MA5_t) / MA5_t | 均线偏离 |
| 8 | VOLATILITY | sqrt(mean(ret^2, 10)) | 已实现波动率 |
| 9 | MOMENTUM | sum(ret, 5) | 5日动量 |

**所有特征都经过鲁棒标准化**（基于MAD，抗异常值），并clip到[-5, 5]。

### 2.5 Step 4a: GP搜索流程

**遗传编程（Genetic Programming）** 将因子公式表示为表达式树，通过模拟自然选择来搜索最优公式。

```
初始化:
  population = [随机生成表达式树 for _ in range(pop_size)]

表达式树示例:
        MUL
       /   \
    RET    CS_RANK
            |
          MA5
            |
          VOL

对应公式: RET * CS_RANK(MA5(VOL))

进化循环 (for gen in range(max_generations)):
  1. 适应度评估:
     - 对每个个体，计算因子值
     - 截面IC (Information Coefficient): corr(factor, forward_return)
     - 截面IR (Information Ratio): mean(IC) / std(IC)
     - 适应度 = IR * |mean(IC)|

  2. 锦标赛选择:
     - 随机选3个个体，取最优者
     - 重复pop_size次

  3. 交叉:
     - 随机选两个父代的子树交换
     - 概率: crossover_rate (默认0.7)

  4. 变异:
     - 随机替换子树 / 改变节点类型
     - 概率: mutation_rate (默认0.2)

  5. 精英保留:
     - 保留top-k最优个体不变

输出: Top-K因子（按IR排序）
```

**GP的算子集**（与RL不同，GP使用pandas实现）：

| 类别 | 算子 |
|------|------|
| 算术 | add, sub, mul, div |
| 数学 | abs, sign, sqrt, log |
| 时序 | ts_delay, ts_mean, ts_std, ts_max, ts_min, ts_rank, ts_zscore |
| 截面 | cs_rank, cs_zscore, cs_mad_norm |
| 常数 | 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 60.0, -1.0, 0.01, 0.1 |

### 2.6 Step 4b: RL搜索流程

详见 [rl_implementation_design.md](rl_implementation_design.md)，核心步骤：

```
1. Transformer自回归生成token序列（公式）
2. StackVM执行公式 → 因子值
3. 截面回测评估 → reward信号
4. REINFORCE + Value Baseline 策略梯度更新
5. LoRD正则化
6. 周期性收集Top公式 + 相关性过滤
```

### 2.7 Step 5: 因子筛选与多样化

两种方法共享相同的筛选逻辑：

```
1. 按得分排序（GP用IR，RL用回测Score）
2. 贪心选择: 依次选择得分最高的因子
3. 相关性过滤: 如果新因子与已选因子的|corr| > 0.7，跳过
4. 最多保留 max_factors 个因子
```

### 2.8 Step 6: 结果输出

每个因子包含：
- `factor_id`: 唯一标识符
- `name`: 因子名称
- `expression`: 人类可读的公式表达式
- `score`: 评估得分
- `avg_return`: 平均收益
- `factor_data`: 每个币种的因子值时序数据

---

## 3. AlphaGPT完整流程

### 3.1 AlphaGPT架构

```
┌─────────────────────────────────────────────────────────────────┐
│                    AlphaGPT 完整流程                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Step 1: 数据获取                                               │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │ PostgreSQL数据库 → SQL查询 → 500个Meme币的OHLCV+Liquidity │ │
│  │ 数据源: Birdeye/DexScreener API → PostgreSQL              │ │
│  └──────────────────────────┬────────────────────────────────┘ │
│                             │                                   │
│  Step 2: 特征工程           │                                   │
│  ┌──────────────────────────▼────────────────────────────────┐ │
│  │ FeatureEngineer.compute_features()                        │ │
│  │ 6维特征: RET, LIQ_SCORE, PRESSURE, FOMO, DEV, LOG_VOL    │ │
│  │ → GPU Tensor (500, 6, T)                                  │ │
│  └──────────────────────────┬────────────────────────────────┘ │
│                             │                                   │
│  Step 3: RL训练             │                                   │
│  ┌──────────────────────────▼────────────────────────────────┐ │
│  │ AlphaGPT模型 (Looped Transformer + MTPHead)               │ │
│  │ REINFORCE策略梯度 (纯策略梯度, 无Value Baseline)           │ │
│  │ LoRD正则化                                                │ │
│  │ BS=8192, Steps=1000, FormulaLen=12                        │ │
│  └──────────────────────────┬────────────────────────────────┘ │
│                             │                                   │
│  Step 4: 回测评估           │                                   │
│  ┌──────────────────────────▼────────────────────────────────┐ │
│  │ MemeBacktest:                                             │ │
│  │ • 信号阈值: sigmoid > 0.85                                │ │
│  │ • 安全过滤: liquidity > $500K                             │ │
│  │ • 手续费: 0.6% (DEX swap + gas + Jito tip)               │ │
│  │ • 冲击成本: trade_size / liquidity (clamp 5%)             │ │
│  │ • 适应度: median(cum_ret - 2.0*big_drawdowns)            │ │
│  └──────────────────────────┬────────────────────────────────┘ │
│                             │                                   │
│  Step 5: 输出               │                                   │
│  ┌──────────────────────────▼────────────────────────────────┐ │
│  │ 单个最优公式 → best_meme_strategy.json                     │ │
│  │ 训练历史 → training_history.json                           │ │
│  └───────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 AlphaGPT核心组件

| 组件 | 实现 | 说明 |
|------|------|------|
| 策略网络 | AlphaGPT (LoopedTransformer + MTPHead) | 多任务路由输出头 |
| 公式执行 | StackVM | 栈式虚拟机 |
| 回测评估 | MemeBacktest | DEX Meme币回测 |
| 正则化 | NewtonSchulzLowRankDecay | LoRD低秩衰减 |
| 数据加载 | CryptoDataLoader | PostgreSQL → GPU Tensor |
| 特征工程 | FeatureEngineer | 6维Meme币特征 |
| 算子集 | OPS_CONFIG | 12个算子 |

---

## 4. 详细对比：FactorMiner vs AlphaGPT

### 4.1 目标场景对比

| 维度 | AlphaGPT | FactorMiner |
|------|----------|-------------|
| **目标市场** | Solana Meme币 (DEX) | 主流加密货币 (CEX) |
| **交易对数量** | 500+ | 3-50 |
| **交易对类型** | 新创建的Meme代币 | BTC/ETH/SOL等主流币 |
| **数据频率** | 分钟级 | 日频/4小时 |
| **数据来源** | PostgreSQL (Birdeye/DexScreener) | Binance API / Yahoo Finance |
| **交易场所** | DEX (Jupiter) | CEX (Binance) |
| **策略目标** | 短线Pump交易 | 截面因子投资 |
| **因子用途** | 单币种交易信号 | 多币种截面排序 |

**为什么不同？**

AlphaGPT面向Meme币的短线交易，这些币种生命周期短、波动极大、流动性差，策略核心是捕捉pump信号。FactorMiner面向主流币的截面因子投资，这些币种相对稳定、流动性好，策略核心是找到跨币种的相对价值信号。

### 4.2 数据层对比

| 维度 | AlphaGPT | FactorMiner | 不同原因 |
|------|----------|-------------|----------|
| 数据存储 | PostgreSQL | API实时获取 | Meme币需要历史数据存储，主流币数据可实时获取 |
| 特殊字段 | liquidity, fdv | 无 | DEX需要链上流动性数据，CEX不需要 |
| 数据量 | 500+币种 | 3-50币种 | Meme币数量远多于主流币 |
| 数据对齐 | pivot by address | pivot by symbol | 相同逻辑，命名不同 |
| 目标收益 | log(open_{t+2}/open_{t+1}) | log(close_{t+2}/close_{t+1}) | DEX用open（流动性差），CEX用close |

### 4.3 特征工程对比

| # | AlphaGPT特征 | FactorMiner特征 | 不同原因 |
|---|-------------|----------------|----------|
| 0 | RET (收益率) | RET (收益率) | 通用特征 |
| 1 | LIQ_SCORE (流动性健康度) | VOL (对数成交量) | DEX用liquidity/fdv，CEX用volume |
| 2 | PRESSURE (买卖压力) | V_CHG (量变化率) | Meme币关注买卖压力，主流币关注量变化 |
| 3 | FOMO (FOMO加速) | PV (量价交互) | Meme币有FOMO效应，主流币用量价关系 |
| 4 | DEV (Pump偏离) | TREND (趋势偏离) | Meme币关注pump，主流币关注趋势 |
| 5 | LOG_VOL (对数量) | HL_RANGE (振幅) | Meme币5维足够，主流币需要更多 |
| 6 | - | CLOSE_POS (K线位置) | 主流币K线形态更有意义 |
| 7 | - | MA_DEV (均线偏离) | 主流币趋势跟踪重要 |
| 8 | - | VOLATILITY (波动率) | 主流币风险管理需要 |
| 9 | - | MOMENTUM (动量) | 主流币动量效应显著 |

**核心差异**：AlphaGPT的特征围绕Meme币的"pump-dump"模式设计（LIQ_SCORE, PRESSURE, FOMO, DEV），FactorMiner的特征围绕主流币的"趋势-动量-波动"模式设计。

### 4.4 算子集对比

| 类别 | AlphaGPT (12个) | FactorMiner RL (27个) | 扩展原因 |
|------|-----------------|----------------------|----------|
| 算术 | ADD, SUB, MUL, DIV | ADD, SUB, MUL, DIV | 相同 |
| 比较 | - | MAX, MIN | 主流币需要极值比较 |
| 数学 | NEG, ABS, SIGN | NEG, ABS, SIGN, SQRT, LOG | 主流币数值范围大，需要压缩 |
| 条件 | GATE | GATE | 相同 |
| 异常 | JUMP | JUMP | 相同 |
| 衰减 | DECAY | DECAY | 相同 |
| 延迟 | DELAY1 | DELAY1, DELAY3 | 主流币需要更长回看 |
| 极值 | MAX3 | - | MAX3是Meme币特有的3日极值 |
| 均线 | - | MA5, MA10, MA20 | 主流币需要多窗口趋势 |
| 波动 | - | STD5, STD10 | 主流币需要波动率度量 |
| 排名 | - | RANK5 | 主流币时序排名有意义 |
| 动量 | - | MOM5, MOM10 | 主流币动量效应显著 |
| 截面 | - | CS_RANK, CS_ZSCORE, CS_MAD | 截面因子挖掘的核心算子 |

**为什么AlphaGPT不需要截面算子？**

AlphaGPT的Meme币场景中，不同币种之间缺乏可比性——一个新创建的Meme币和一个已经涨了10倍的Meme币在截面上没有可比的排名关系。而主流币（BTC/ETH/SOL等）在截面上高度可比，截面排名和标准化是核心操作。

### 4.5 模型架构对比

| 组件 | AlphaGPT | FactorMiner | 不同原因 |
|------|----------|-------------|----------|
| Transformer层数 | 2 | 2 | 相同 |
| 注意力头数 | 4 | 4 | 相同 |
| d_model | 64 | 64 | 相同 |
| dim_ff | 128 | 128 | 相同 |
| 循环次数 | 3 | 3 | 相同 |
| 输出头 | MTPHead (3任务路由) | Actor+Critic | AlphaGPT多任务，FactorMiner单任务+baseline |
| QK-Norm | 有 | 有 | 相同 |
| SwiGLU | 有 | 有 | 相同 |
| RMSNorm | 有 | 有 | 相同 |
| 位置编码 | 可学习 | 可学习 | 相同 |
| 因果掩码 | 有 | 有 | 相同 |

**MTPHead vs Actor+Critic**：

AlphaGPT的MTPHead设计用于同时优化多个交易策略（如不同持仓时间、不同风险偏好），通过路由网络动态选择任务头。FactorMiner使用更简单的Actor+Critic结构，因为截面因子挖掘的目标单一——最大化截面回测收益。Critic网络提供Value Baseline，降低REINFORCE的方差。

### 4.6 训练算法对比

| 维度 | AlphaGPT | FactorMiner | 不同原因 |
|------|----------|-------------|----------|
| 算法 | REINFORCE | REINFORCE + Value Baseline | FactorMiner需要更稳定的梯度 |
| 价值函数 | 无 | Critic网络 | 降低策略梯度方差 |
| 熵正则 | 无 | 有 (coef=0.01) | 鼓励探索多样化公式 |
| 梯度裁剪 | 无 | 有 (norm=1.0) | 防止梯度爆炸 |
| Batch Size | 8192 | 512/64 | Meme币500+种，主流币10-50种 |
| 训练步数 | 1000 | 500 | 主流币搜索空间更聚焦 |
| 公式长度 | 12 | 16 | 主流币需要更复杂的因子表达式 |
| LoRD | 有 | 有 | 直接复用 |
| LoRD目标 | q_proj, k_proj, attention, qk_norm | + head_actor | Actor头也需要低秩约束 |
| 评估间隔 | 无 | 每50步 | 定期收集优秀公式 |
| 进度监控 | tqdm | 回调函数 | WebUI需要实时进度 |

**为什么FactorMiner增加Value Baseline？**

AlphaGPT的8192 batch已经足够大，batch内的reward均值是较好的baseline。FactorMiner的batch更小（512），batch内reward波动更大，需要显式的Critic网络来估计baseline。

**为什么FactorMiner增加熵正则？**

较小的batch和更复杂的算子集（27 vs 12）增加了策略过早收敛的风险。熵正则确保策略在训练过程中保持足够的探索性。

### 4.7 回测评估对比

| 维度 | AlphaGPT MemeBacktest | FactorMiner CrossSectionalBacktest | 不同原因 |
|------|----------------------|-----------------------------------|----------|
| 交易金额 | $1,000 | $10,000 | 主流币流动性更好 |
| 安全过滤 | liquidity > $500K | volume > $1M | DEX用流动性，CEX用成交量 |
| 手续费 | 0.6% | 0.1% | DEX手续费远高于CEX |
| 冲击模型 | trade_size/liquidity (5%上限) | trade_size/(volume*close) (2%上限) | DEX冲击大，CEX冲击小 |
| 信号阈值 | sigmoid > 0.85 | sigmoid > 0.65 | Meme币需要更高置信度 |
| 回撤惩罚 | -5% → 2.0x | -3% → 1.5x | Meme币回撤更大 |
| 最低活跃度 | 5笔 | 5笔 | 相同 |
| 适应度 | median(score) | median(score) | 相同 |

**为什么信号阈值不同？**

Meme币的噪音信号多（假突破、pump-and-dump），需要更高的信号阈值（0.85）来过滤噪音。主流币的价格信号更可靠，0.65的阈值即可。

### 4.8 输出结果对比

| 维度 | AlphaGPT | FactorMiner |
|------|----------|-------------|
| 输出数量 | 1个最优公式 | 最多15个低相关因子 |
| 多样化 | 无 | 相关性过滤 (|corr|<0.7) |
| 输出格式 | JSON文件 | WebUI展示 + API返回 |
| 因子值 | 不输出 | 输出每个币种的因子值时序 |
| 训练历史 | JSON文件 | 实时展示训练曲线 |
| 可视化 | 无 | WebUI图表 |

**为什么FactorMiner输出多个因子？**

AlphaGPT的目标是找到单个最优交易策略，直接用于自动交易。FactorMiner的目标是构建因子库，多个低相关因子可以组合成多因子策略，提高Sharpe比率和策略鲁棒性。

---

## 5. GP vs RL：FactorMiner内部两种方法对比

| 维度 | GP (遗传编程) | RL (强化学习) |
|------|--------------|--------------|
| **搜索方式** | 进化算法（选择/交叉/变异） | 策略梯度（梯度下降） |
| **搜索引导** | 无梯度，靠适应度选择 | 有梯度信号引导 |
| **公式表示** | 表达式树 | Token序列 |
| **评估指标** | 截面IC/IR | 截面回测PnL |
| **算子实现** | pandas (CPU) | torch (GPU) |
| **GPU需求** | 不需要 | 推荐 |
| **依赖** | 无额外依赖 | 需要PyTorch |
| **搜索效率** | 中等（随机探索为主） | 较高（梯度引导） |
| **公式复杂度** | 受树深度限制 | 受序列长度限制 |
| **多样性** | 依赖种群多样性 | 依赖熵正则 |
| **适用场景** | 快速原型/无GPU环境 | GPU环境/需要高质量因子 |
| **训练时间** | 2-5分钟 | 1-30分钟（取决于设备） |
| **可解释性** | 高（表达式树直观） | 中（token序列需解码） |

**选择建议**：
- 无GPU或快速探索 → GP
- 有GPU且需要高质量因子 → RL
- 两者结合 → 先GP快速筛选，再RL精细搜索

---

## 6. 完整截面因子挖掘最佳实践

### 6.1 数据准备

```
1. 选择10-30个流动性好的主流币
2. 使用日频数据（1d timeframe）
3. 至少1年的历史数据
4. 确保数据无大量缺失值
```

### 6.2 GP挖掘配置

```python
config = {
    'population_size': 200,
    'max_generations': 100,
    'crossover_rate': 0.7,
    'mutation_rate': 0.2,
    'max_depth': 5,
    'tournament_size': 3,
}
```

### 6.3 RL挖掘配置

```python
config = {
    'device': 'cuda',           # GPU加速
    'batch_size': 512,          # GPU推荐512+
    'train_steps': 500,         # 足够收敛
    'max_formula_len': 16,      # 允许较复杂的公式
    'lr': 1e-3,                 # 学习率
    'use_lord': True,           # 启用LoRD
    'entropy_coef': 0.01,       # 熵正则
    'max_factors': 15,          # 最多输出15个因子
    'max_correlation': 0.7,     # 相关性阈值
}
```

### 6.4 因子组合

```
1. 从GP和RL结果中分别选出Top因子
2. 合并去重（相关性过滤）
3. 使用ICIR加权构建多因子组合
4. 样本外验证
```

---

## 7. 技术实现细节

### 7.1 并发与异步

- GP和RL挖掘都在后台线程中运行
- WebUI通过轮询API获取进度
- 支持中途停止（`_stop_requested`标志）

### 7.2 内存管理

- RL模式下，特征张量常驻GPU内存
- GP模式下，数据以pandas DataFrame形式保存在内存
- 因子值在输出时转为Python list（JSON序列化）

### 7.3 错误处理

- StackVM执行失败 → 返回None → reward=-5.0
- 因子值全零/常数 → reward=-2.0
- 活跃度不足 → reward=-10.0
- NaN/Inf → 自动替换为0.0/±1.0
- PyTorch未安装 → 优雅降级，提示安装

---

## 8. 总结

FactorMiner的截面因子挖掘系统在AlphaGPT的基础上进行了以下关键改造：

1. **场景适配**：从Meme币DEX交易 → 主流币CEX截面投资
2. **特征扩展**：从6维Meme特征 → 10维主流币特征
3. **算子扩展**：从12个基础算子 → 27个算子（含截面算子）
4. **算法增强**：从纯REINFORCE → REINFORCE + Value Baseline + 熵正则
5. **输出多样化**：从单公式输出 → 多因子筛选 + 相关性过滤
6. **双引擎**：GP + RL两种搜索方法，适应不同场景
7. **WebUI集成**：完整的配置、监控、可视化界面

这些改造使得FactorMiner能够有效地在主流加密货币的截面空间中挖掘高质量、多样化的因子，为量化投资提供更可靠的信号来源。
