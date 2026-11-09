# Sequoia A 股 SQLite 数据库

该目录把三份 Parquet 快照整理为一个可增量更新的 SQLite 数据库：

- `daily_k`：全 A 股原始未复权日 K；主键为 `(thscode, trade_date)`。
- `adjustment_events`：分红、送股、配股事件；同一股票同一除权日允许有多条不同事件。
- `source_imports`：源文件、SHA-256、覆盖日期和导入状态。
- `v_data_coverage`：各数据集的行数、股票数和日期范围。

10 年日 K 先导入，最近 10 日快照随后以 UPSERT 覆盖重叠日期。重复执行时，SHA-256
相同且已成功导入的文件会自动跳过；放入更新日期的新 Parquet 后再次执行即可增量同步。
毫秒时间戳按 `Asia/Shanghai`（含历史夏令时规则）转换为交易日。

策略执行还会使用以下本地数据库：

- `fundamentals.sqlite3`：A 类基本面历史缓存。
- `selection_history.sqlite3`：每日买入、持有、卖出事件、当前持仓以及在线策略权重。

## 建库

```powershell
python -m pip install -r Sequoia/requirements.txt
python Sequoia/build_sqlite.py
```

默认输出为 `Sequoia/a_share_market.sqlite3`。也可指定其他位置：

```powershell
python Sequoia/build_sqlite.py --database D:\data\a_share_market.sqlite3
```

## Python 调用

从仓库根目录运行：

```python
from Sequoia import AShareDataStore

with AShareDataStore() as store:
    bars = store.get_daily_k(
        ["000001.SZ", "600000.SH"],
        start_date="2026-10-01",
        end_date="2026-11-02",
    )
    events = store.get_adjustment_events("000001.SZ")
    bars_df = store.to_dataframe(bars)
```

也可直接使用 SQL：

```python
import sqlite3

connection = sqlite3.connect("Sequoia/a_share_market.sqlite3")
rows = connection.execute(
    """
    SELECT trade_date, open_price, high_price, low_price, close_price, volume
    FROM daily_k
    WHERE thscode = ? AND trade_date BETWEEN ? AND ?
    ORDER BY trade_date
    """,
    ("000001.SZ", "2026-10-01", "2026-11-02"),
).fetchall()
```

价格字段均为原始未复权价。`adjustment_events` 保存公司行为事件，不会擅自假设某种
前复权或后复权口径；需要复权行情的模块可按自己的基准日和口径使用这些事件计算。

## ABCD 策略选股与 Teams 推送

仓库根目录的 `main.py` 会运行日 K 可支持的 B/C 类策略，并使用 D 类低波动、
风险平价和差分进化算法对候选池排序。A 类策略读取单独的基本面缓存；缓存没有
准备好时会明确跳过，不会用行情字段伪造基本面结论。
每个交易日的买入、持有、卖出状态及价格快照保存在
`Sequoia/selection_history.sqlite3`；同一交易日重复运行会复用原记录。
龙虎榜使用 Financial-API 的“全部榜”，与本地行情采用同一交易日；当天新上榜、
连续上榜和离榜分别记录为买入、持有、卖出。接口原始字段保存在
`dragon_tiger_runs`、`dragon_tiger_events` 两张表，同日重复运行复用已有记录。

目前接入的策略：

- A 类：华泰 FFScore、罗伯·瑞克超额现金流。
- B 类：高质量动量、上下影线、振幅隐藏结构、纯真波动率、隔夜与日间动量、
  均线收敛与发散。
- C 类：ICU 均线、鳄鱼线、RSRS、QRS、均线通道突破、底部形态识别。
- D 类：低波动组合、风险平价、DE 组合优化。

所有策略首次运行时权重均为 1。每次平仓后，系统使用扣除买入和卖出手续费后的
实际收益更新入场时命中的策略权重。候选股票按命中策略权重之和排序，默认取
权重最高的 Top 30；Teams 消息在这 30 只股票内部按权重总分从小到大排列。
历史库保留每次交易和调权流水，方便复核。

### 同步 A 类基本面数据

基本面同步器读取 `config.json` 中相同的 Financial-API 配置，把估值以及最近
8 期利润表、资产负债表和现金流量表写入
`Sequoia/fundamentals.sqlite3`：

```powershell
# 建议首次分批同步；重复运行会跳过已有完整缓存的股票
python Sequoia/sync_fundamentals.py --limit-symbols 200

# 指定股票验证
python Sequoia/sync_fundamentals.py --symbols 600519.SH,000001.SZ

# 刷新已有股票
python Sequoia/sync_fundamentals.py --symbols 600519.SH --refresh
```

首次全 A 同步涉及每只股票三张报表的请求，适合分批执行。估值快照按每批最多
100 只请求，报表同步可中断续跑。

### 完整运行顺序

首次使用前安装依赖并复制配置：

```powershell
python -m pip install -r Sequoia/requirements.txt
Copy-Item Sequoia/config.example.json Sequoia/config.json
```

填写 `financial_api.x_api_key`、`teams.webhook_url`、交易参数、在线权重参数和退出
参数。`Sequoia/config.json` 已被 Git 忽略。

运行顺序：

```powershell
# 1. 通过 Financial-API 获取最近三年行情并更新 SQLite
python Sequoia/fetch_three_year_data.py

# 2. 可选：同步 A 类基本面缓存
python Sequoia/sync_fundamentals.py --limit-symbols 200

# 3. 首次从权重 1 开始，并清空旧持仓记录
python main.py --reset-history --no-name-lookup

# 4. 后续每日终端预览
python main.py --no-name-lookup

# 5. 实际发送 Teams
python main.py --send
```

### 通过 API 获取最近三年数据

`fetch_three_year_data.py` 调用 Financial-API 官方全市场 dump：下载约 10 年的
未复权日 K 和全部复权事件，在本地流式过滤指定日期窗口，再 UPSERT 到
`Sequoia/a_share_market.sqlite3`。默认获取截至上海时区当天的最近 3 个日历年：

```powershell
# 先验证 API Key 和下载接口，不下载大文件
python Sequoia/fetch_three_year_data.py --check-only

# 下载、过滤最近三年并更新 SQLite
python Sequoia/fetch_three_year_data.py

# 固定回测截止日，并保留过滤后的 Parquet
python Sequoia/fetch_three_year_data.py `
  --as-of 2026-11-04 `
  --years 3 `
  --output-dir Sequoia\data
```

首次下载的是全市场约 10 年文件，体积较大；脚本不会保存短时有效的预签名 URL。
重复执行会按 `(thscode, trade_date)` 更新日 K，并按复权事件唯一键去重。

Financial-API 当前公开估值接口只返回最新快照，不提供历史逐日 PE/PB/PCF。
脚本不会把当前估值回填到过去日期。A 类基本面策略仍可通过
`sync_fundamentals.py` 获取带报告发布日期的多期财务报表，但三年回测若使用
估值因子，还必须接入真实的历史估值数据源，否则应跳过对应策略。

需要在选股前更新主流程依赖的全部增量数据时：

```powershell
python main.py --backfill --no-name-lookup

# 也可以单独运行更新器
python Sequoia/update_daily_k.py
```

`--backfill` 和 `update_daily_k.py` 会依次执行：

1. 下载最近 10 个交易日的全市场日 K，按 `(thscode, trade_date)` UPSERT。
2. 下载全市场完整复权事件 dump，并原子替换本地复权事件缓存。
3. 每批最多 100 只刷新全市场最新 PE/PB/PS/PCF 估值快照。
4. 按持久化游标轮换刷新一批股票的利润表、资产负债表和现金流量表；优先补齐
   缺少三类年报的股票。

财报 API 只支持逐只股票查询，因此不会在每天一次性请求全市场约 1.7 万个接口。
默认每日刷新 50 只股票，可在配置中调整：

```json
{
  "financial_api": {
    "update_fundamentals": true,
    "incremental_report_batch_size": 50
  }
}
```

设置 `update_fundamentals` 为 `false` 可只更新行情和复权事件；将
`incremental_report_batch_size` 设为 `0` 可刷新估值但跳过财报轮换。

默认按照在线策略权重总分选择 Top 30。
`--max-results N` 仅用于临时调试。

程序优先使用 Financial-API 估值快照解析股票名称，接口不可用时才使用备用名称源。

## 在线策略权重

`main.py` 不再依赖 `rollingScore.py` 或 `strategy_performance.parquet`。全部注册
策略从权重 1 开始，买入时保存当时命中的策略；卖出时计算扣除双边手续费后的
实际收益，并更新这些入场策略：

```text
新权重 = clamp(
    旧权重 × exp(学习率 × tanh(净收益率 / 收益尺度)),
    最低权重,
    最高权重
)
```

盈利交易提高权重，亏损交易降低权重，极端收益通过 `tanh` 限幅。一次股票交易
命中多个策略时，每个入场策略都记录一次相同的交易结果。调权只使用已经完成的
交易，不使用未实现收益，也不会改变当天已经生成的候选排名。

权重和统计保存在 `selection_history.sqlite3`：

- `strategy_weights`：当前权重、交易次数、胜次数、累计收益和平均收益。
- `strategy_weight_updates`：每次平仓对应的旧权重、新权重和净收益率。

主要配置：

| 配置项 | 默认值 | 含义 |
|---|---:|---|
| `strategy_weights.initial_weight` | 1.0 | 所有策略初始权重 |
| `strategy_weights.learning_rate` | 0.25 | 单笔交易调权速度 |
| `strategy_weights.return_scale` | 0.10 | 10% 收益对应约一个标准更新尺度 |
| `strategy_weights.min_weight` | 0.20 | 权重下限 |
| `strategy_weights.max_weight` | 5.0 | 权重上限 |

从头开始记录时执行：

```powershell
python main.py --reset-history --no-name-lookup
```

该命令会先把旧数据库备份为
`selection_history_before_reset_YYYYMMDD_HHMMSS.sqlite3`，再清空旧选股记录、
持仓、冷却期和权重，最后把全部策略重新初始化为 1。

实际选股买入使用 `trading` 配置：每只股票预算 100,000 元，数量按 100 股整数倍
向下取整，并确保股票成交额加 0.025% 买入手续费后不超过预算：

```text
股数 = floor(100000 / (股价 × 1.00025) / 100) × 100
```

预算不足买入 100 股时跳过该股票。持仓会保存实际股数；卖出时按原持仓股数计算
0.1% 手续费和净到账金额。

主要配置：

| 配置项 | 默认值 | 含义 |
|---|---:|---|
| `trading.budget_per_stock` | 100000 | 每只股票含买入手续费的资金上限 |
| `trading.lot_size` | 100 | 买入股数单位 |
| `trading.buy_fee_rate` | 0.00025 | 买入手续费率 0.025% |
| `trading.sell_fee_rate` | 0.001 | 卖出手续费率 0.1% |

### 卖出与风险控制

`exit_rules` 默认启用以下规则：

- 扣除买卖手续费后的收益达到 20% 时止盈。
- 扣费后收益跌至 -10% 时止损。
- 持仓达到 60 个交易日时卖出。
- 当前价较持仓期最高价回撤 12% 时卖出。
- 策略总分连续 3 个交易日下降，或跌出目标名单后连续 3 个交易日未恢复时卖出。
- 全市场等权净值低于 20 日均线且 20 日收益不高于 -5% 时，清空可交易持仓并
  暂停新买入。
- 卖出后冷却 5 个交易日，避免相同信号立即买回。

止盈、止损和收益展示均按卖出手续费后的净到账额相对初始实际投入计算。退出事件
会记录卖出原因、持仓交易日数、净收益率和从持仓期高点的回撤。

同一交易日同时命中多条规则时，卖出原因优先级为：大盘风险控制、止损、回撤止损、
止盈、最大持仓期限、策略分数下降确认。当前实现使用数据库最新日 K 收盘价记录
买卖事件；用于真实下单时，应根据实际执行系统选择下一交易日开盘价或真实成交价。
