在币安（Binance）平台，除了基础的 OHLCV（开高低收成交量）外，还有多类**免费**且极具深度的历史数据。这些数据非常适合进行截面因子（Cross-sectional Factors）挖掘，特别是合约市场（Futures）的特有指标。

以下是满足你“实时获取、支持历史、覆盖面广”要求的核心数据类别：

### 1. 衍生品市场指标（Futures Metrics）
这是做截面因子最核心的池子。虽然 REST API 通常只提供最近 30 天的数据，但币安的**公开数据服务 (data.binance.vision)** 提供了完整的历史归档。

| 数据类型 | 描述 | 获取方式 (实时) | 获取方式 (历史/S3) |
| :--- | :--- | :--- | :--- |
| **持仓量 (Open Interest)** | 全网未平仓合约总量。 | `/fapi/v1/openInterest` | `data/futures/um/daily/metrics` |
| **多空持仓人数比** | 大户/全网的多空人数对比 (Long/Short Ratio)。 | `/futures/data/globalLongShortAccountRatio` | `data/futures/um/daily/metrics` |
| **大户净持仓比** | 统计持仓量前 20% 账户的净头寸方向。 | `/futures/data/topLongShortPositionRatio` | `data/futures/um/daily/metrics` |
| **主动买入卖出量** | Taker Buy/Sell Volume。用于计算主动成交压力。 | `/futures/data/takerbuy_sell_vol` | `data/futures/um/daily/metrics` |
| **基差 (Basis)** | 永续价格与现货价格/指数价格的偏离。 | `/fapi/v1/premiumIndexKlines` | `data/futures/um/daily/metrics` |

* **因子建议**：计算 OI 变化率（$\Delta OI / Vol$）或者大户/散户情绪背离因子。

---

### 2. 资金费率相关数据 (Funding Data)
资金费率是截面策略中最常用的因子之一（如 Funding Arbitrage 或 Carry Factor）。

* **资金费率历史 (Funding Rate History)**
    * **实时**：`GET /fapi/v1/fundingRate`
    * **历史**：可以获取自上线以来的所有 8 小时（或 1 小时）资金费。
    * **特点**：覆盖所有 U 本位和币本位永续合约。
* **预测资金费率**：
    * **实时**：`GET /fapi/v1/premiumIndex`
    * **用途**：实时监控即时的资金费变化，比已结算的历史数据更有前瞻性。

---

### 3. 强平快照 (Liquidation Snapshots)
强平数据在截面策略中常用于捕捉“流动性枯竭”或“情绪反转”因子。

* **数据内容**：每一笔被强平订单的价格、数量、方向和时间戳。
* **获取方式**：
    * **实时**：WebSocket 订阅 `!forceOrder@arr`。
    * **历史**：`data.binance.vision` 中的 `liquidationSnapshot` 路径，按天汇总。
* **因子建议**：统计过去 1 小时内各币种强平金额占成交量的比值（强平强度）。

---

### 4. 深度与订单簿统计 (Order Book Aggregation)
虽然获取全量 L2 历史深度很重，但你可以直接利用**免费的 AggTrade（聚合交易）**数据自行加工：

* **AggTrade 历史**：
    * **下载地址**：`data.binance.vision/data/futures/um/daily/aggTrades/`
    * **包含字段**：成交价格、数量、成交时间、**是否为买方做市（判断 Taker 方向）**。
* **加工因子**：
    * **波动率因子**：实现波动率 (Realized Volatility)。
    * **偏度/峰度因子**：基于 Tick 级别的收益率分布。
    * **成交集中度**：单笔成交额的 Gini 系数或大单占比。

---

### 总结与获取策略

如果你要构建**截面回测框架**，建议的路径是：

1.  **离线下载**：去 [data.binance.vision](https://data.binance.vision/) 爬取 `metrics` 和 `fundingRate` 的历史 ZIP 包。这些数据通常以 5 分钟为粒度（Metrics）或按事件触发（Funding）。
2.  **数据加工**：将 5min 的 Metrics 数据（持仓、多空比）通过 `resample('1h').last()` 处理成每小时截面。
3.  **实时增量**：使用 REST API 维护最近 30 天的数据更新。
4.  **一致性检查**：注意币安的 `pair` (如 BTCUSD) 和 `symbol` (如 BTCUSDT) 在不同接口中的区别，确保截面对象的一致性。

**注意点**：`data.binance.vision` 上的 `metrics` 文件包含了你需要的绝大部分非 OHLCV 指标，它是币安官方专门为量化研究者准备的数据集。