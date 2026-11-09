# 日 K 强化学习选股

这是一个简单的 Actor-Critic 选股模型。网络共享主干严格为两层、每层
256 个神经元（`Linear -> ReLU -> Linear -> ReLU`），输入完全来自
`Sequoia/a_share_market.sqlite3` 的 daily K line。

模型使用前一交易日及更早的收益、波动率、均线、价格区间、成交量和成交额
特征，在下一交易日开盘决定是否建仓，因此不会用当日收盘信息回到当日成交。
训练默认读取最近十年、平均成交额最高的 800 只股票。

持仓卖出规则与根目录 `main.py` 保持同一参数口径及判断顺序：

1. 大盘风险关闭；
2. 扣费后亏损达到止损线；
3. 从持仓期高点回撤达到阈值；
4. 扣费后收益达到止盈线；
5. 达到最大持仓交易日；
6. 模型分数连续多日下降。

参数从 `Sequoia/config.json` 的 `trading` 和 `exit_rules` 读取，缺失时采用
`main.py` 的默认值。每个自然月末用组合月收益训练：正收益为正奖励，负收益
乘 `negative_penalty` 后作为处罚（默认 1.5 倍）。

## 安装与训练

```powershell
python -m pip install -r ML/requirements.txt
python -m ML.train --years 10 --epochs 1
```

完整十年训练可能需要较长时间。先验证流程可缩小股票池：

```powershell
python -m ML.train --years 1 --universe-size 50 --daily-candidates 32
```

模型写入 `ML/artifacts/actor_critic.pt`，逐月收益和奖励写入同名
`actor_critic.monthly.csv`。

## 生成最新候选

```powershell
python -m ML.predict --top 30
```

输出 `ML/artifacts/latest_candidates.csv`。这是下一交易日的候选买入名单；
实际交易前仍应进行停牌、涨跌停、ST、退市风险和滑点检查。

这是研究和回测脚手架，不构成投资建议。为控制十年数据的内存占用，默认股票池
按全区间平均成交额筛选，因此存在股票池/存续偏差；正式研究可改为逐日历史成分股。
