"""可由 Sequoia 行情与基本面缓存复现的选股策略。

这里把研究目录中的 notebook 口径收敛成无第三方量化框架依赖的 pandas 实现，
供根目录 ``main.py`` 调用。所有函数只产生因子值或布尔信号，不负责持仓记录。
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


TECHNICAL_POINTS = {
    "B/振幅隐藏结构": 15,
    "B/纯真波动率": 20,
    "B/隔夜与日间动量": 20,
    "B/均线收敛与发散": 15,
    "C/RSRS择时": 20,
    "C/QRS择时": 15,
    "C/均线通道突破": 15,
    "C/底部形态识别": 10,
    "D/低波动组合": 10,
    "D/风险平价": 10,
}

FUNDAMENTAL_POINTS = {
    "A/华泰FFScore": 30,
    "A/罗伯·瑞克超额现金流": 25,
}


def _rolling(
    data: pd.DataFrame,
    column: str,
    window: int,
    operation: str,
    min_periods: int | None = None,
) -> pd.Series:
    grouped = data.groupby("thscode", sort=False)[column]
    values = grouped.rolling(
        window, min_periods=window if min_periods is None else min_periods
    )
    return (
        getattr(values, operation)()
        .reset_index(level=0, drop=True)
        .sort_index()
    )


def add_extended_features(data: pd.DataFrame) -> pd.DataFrame:
    """添加新增 B/C/D 策略共用的日线特征。"""
    frame = data.copy()
    grouped = frame.groupby("thscode", sort=False)

    previous_close = grouped["close_price"].shift(1)
    frame["overnight_return"] = frame["open_price"] / previous_close - 1
    frame["daytime_return"] = frame["close_price"] / frame["open_price"] - 1
    frame["overnight_momentum_20"] = _rolling(
        frame, "overnight_return", 20, "sum"
    )
    frame["daytime_momentum_20"] = _rolling(
        frame, "daytime_return", 20, "sum"
    )

    frame["amplitude"] = frame["high_price"] / frame["low_price"] - 1
    low_20 = _rolling(frame, "close_price", 20, "min")
    high_20 = _rolling(frame, "close_price", 20, "max")
    price_position = (frame["close_price"] - low_20) / (high_20 - low_20)
    frame["_high_price_amplitude"] = frame["amplitude"].where(
        price_position.ge(0.8)
    )
    frame["_low_price_amplitude"] = frame["amplitude"].where(
        price_position.le(0.2)
    )
    frame["ideal_amplitude_20"] = (
        _rolling(
            frame,
            "_high_price_amplitude",
            20,
            "mean",
            min_periods=2,
        )
        - _rolling(
            frame,
            "_low_price_amplitude",
            20,
            "mean",
            min_periods=2,
        )
    )

    for window in (9, 10, 18, 60):
        frame[f"ma_{window}"] = _rolling(
            frame, "close_price", window, "mean"
        )
    ma_matrix = frame[["close_price", "ma_5", "ma_10", "ma_20", "ma_60"]]
    frame["ma_convergence"] = ma_matrix.std(axis=1) / frame["close_price"]
    frame["ma_convergence_previous"] = grouped["ma_convergence"].shift(5)

    cross_up = frame["ma_9"].gt(frame["ma_18"]) & grouped["ma_9"].shift(
        1
    ).le(grouped["ma_18"].shift(1))
    frame["_ma_cross_up"] = cross_up.astype(float)
    frame["recent_ma_cross_up"] = _rolling(
        frame, "_ma_cross_up", 12, "max", min_periods=1
    ).gt(0)
    frame["channel_high_12_previous"] = grouped["high_price"].transform(
        lambda values: values.shift(1).rolling(12, min_periods=12).max()
    )

    # OLS(high = alpha + beta * low) 的向量化解析解。
    n = 18
    mean_high = _rolling(frame, "high_price", n, "mean")
    mean_low = _rolling(frame, "low_price", n, "mean")
    frame["_high_low"] = frame["high_price"] * frame["low_price"]
    frame["_low_square"] = frame["low_price"] ** 2
    mean_high_low = _rolling(frame, "_high_low", n, "mean")
    mean_low_square = _rolling(frame, "_low_square", n, "mean")
    covariance = mean_high_low - mean_high * mean_low
    variance_low = mean_low_square - mean_low**2
    beta = covariance / variance_low.replace(0, np.nan)
    std_high = _rolling(frame, "high_price", n, "std")
    std_low = _rolling(frame, "low_price", n, "std")
    correlation = covariance / (std_high * std_low)
    r_squared = correlation.clip(-1, 1) ** 2
    frame["rsrs_beta"] = beta
    beta_mean = _rolling(frame, "rsrs_beta", 120, "mean")
    beta_std = _rolling(frame, "rsrs_beta", 120, "std")
    beta_z = (beta - beta_mean) / beta_std.replace(0, np.nan)
    frame["qrs_signal"] = beta_z * r_squared
    frame["rsrs_signal"] = frame["qrs_signal"] * beta

    # 低成本的底部形态识别：两个相近低点、颈线恢复且中期均线转强。
    frame["_low_first_half"] = grouped["low_price"].transform(
        lambda values: values.shift(15).rolling(15, min_periods=12).min()
    )
    frame["_low_second_half"] = grouped["low_price"].transform(
        lambda values: values.rolling(15, min_periods=12).min()
    )
    bottom_distance = (
        frame["_low_first_half"] / frame["_low_second_half"] - 1
    ).abs()
    frame["double_bottom"] = (
        bottom_distance.le(0.06)
        & frame["close_price"].gt(frame["ma_20"])
        & frame["ma_20"].gt(grouped["ma_20"].shift(5))
        & frame["ret_20"].gt(0)
    )
    return frame


def pure_volatility_latest(
    features: pd.DataFrame, latest_date: str
) -> pd.Series:
    """计算剔除过去 6 期横截面信息后的纯真波动率残差。

    原研究使用月频特质波动率横截面。本地没有 FF/CH3 风格因子数据，因此以
    20 日收益波动率作为基础波动率，并严格保留“本期截面对过去 6 期截面回归
    取残差”的核心去相关步骤。
    """
    monthly = (
        features.assign(month=features["trade_date"].str.slice(0, 7))
        .sort_values(["thscode", "trade_date"])
        .groupby(["month", "thscode"], sort=False)
        .tail(1)
        .pivot(index="month", columns="thscode", values="vol_20")
        .sort_index()
    )
    if len(monthly) < 7:
        return pd.Series(dtype=float, name="pure_volatility")
    sample = monthly.tail(7)
    y = sample.iloc[-1]
    x = sample.iloc[:-1].T
    valid = y.notna() & x.notna().all(axis=1)
    if valid.sum() < 20:
        return pd.Series(dtype=float, name="pure_volatility")
    design = np.column_stack(
        [np.ones(int(valid.sum())), x.loc[valid].to_numpy(dtype=float)]
    )
    target = y.loc[valid].to_numpy(dtype=float)
    coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
    residual = target - design @ coefficients
    return pd.Series(
        residual,
        index=x.index[valid],
        name="pure_volatility",
    )


def technical_signals(
    features: pd.DataFrame, latest_date: str
) -> tuple[pd.DataFrame, list[str]]:
    """返回新增技术策略信号以及实际运行的策略名。"""
    latest = features.loc[features["trade_date"].eq(latest_date)].copy()
    liquid = latest["avg_turnover_20"].ge(20_000_000)

    pure_vol = pure_volatility_latest(features, latest_date)
    latest["pure_volatility"] = latest["thscode"].map(pure_vol)

    def quantile(column: str, value: float) -> float:
        return float(latest.loc[liquid, column].quantile(value))

    latest["B/振幅隐藏结构"] = (
        liquid
        & latest["ideal_amplitude_20"].le(
            quantile("ideal_amplitude_20", 0.10)
        )
        & latest["ret_20"].gt(0)
    )
    latest["B/纯真波动率"] = (
        liquid
        & latest["pure_volatility"].le(quantile("pure_volatility", 0.10))
        & latest["ret_20"].gt(0)
    )
    latest["B/隔夜与日间动量"] = (
        liquid
        & latest["overnight_momentum_20"].ge(
            quantile("overnight_momentum_20", 0.80)
        )
        & latest["daytime_momentum_20"].gt(0)
    )
    latest["B/均线收敛与发散"] = (
        liquid
        & latest["ma_convergence_previous"].le(
            latest.loc[liquid, "ma_convergence_previous"].quantile(0.20)
        )
        & latest["ma_convergence"].gt(latest["ma_convergence_previous"] * 1.2)
        & latest["close_price"].gt(latest["ma_20"])
    )
    latest["C/RSRS择时"] = liquid & latest["rsrs_signal"].gt(0.70)
    latest["C/QRS择时"] = liquid & latest["qrs_signal"].gt(0.70)
    latest["C/均线通道突破"] = (
        liquid
        & latest["recent_ma_cross_up"]
        & latest["ma_9"].gt(latest["ma_18"])
        & latest["close_price"].gt(latest["channel_high_12_previous"])
    )
    latest["C/底部形态识别"] = liquid & latest["double_bottom"]
    latest["D/低波动组合"] = (
        liquid
        & latest["vol_120"].le(quantile("vol_120", 0.10))
        & latest["ret_20"].gt(0)
    )
    return latest, [
        name for name in TECHNICAL_POINTS if name != "D/风险平价"
    ]


def apply_risk_parity(
    candidates: pd.DataFrame, selected_assets: int = 20
) -> pd.DataFrame:
    """按逆波动风险预算给候选池前若干名增加风险平价策略分。"""
    if candidates.empty:
        candidates["risk_parity_weight"] = pd.Series(dtype=float)
        return candidates
    inverse_vol = 1 / candidates["vol_120"].replace(0, np.nan)
    candidates["risk_parity_weight"] = (
        inverse_vol / inverse_vol.sum()
    ).fillna(0)
    selected = candidates.nlargest(
        min(selected_assets, len(candidates)), "risk_parity_weight"
    ).index
    candidates.loc[selected, "strategies"] = candidates.loc[
        selected, "strategies"
    ].apply(lambda names: [*names, "D/风险平价"])
    return candidates


def _latest_two(
    reports: pd.DataFrame, statement_type: str
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    subset = reports.loc[reports["statement_type"].eq(statement_type)].sort_values(
        "period_end_ms", ascending=False
    )
    if len(subset) < 2:
        return None
    return (
        json.loads(subset.iloc[0]["payload_json"]),
        json.loads(subset.iloc[1]["payload_json"]),
    )


def load_fundamental_signals(
    database: Path,
    market_database: Path,
    latest: pd.DataFrame,
    latest_date: str,
) -> tuple[dict[str, list[str]], str]:
    """从同步缓存计算 FFScore 与罗伯·瑞克策略。

    返回 ``symbol -> strategy names``。缓存不存在时不伪造基本面信号。
    """
    if not database.is_file():
        return {}, "A 类基本面缓存不存在"
    connection = sqlite3.connect(database)
    try:
        valuations = pd.read_sql_query(
            """
            SELECT v.*
            FROM valuation_snapshots v
            JOIN (
                SELECT thscode, MAX(snapshot_at) AS snapshot_at
                FROM valuation_snapshots GROUP BY thscode
            ) x USING (thscode, snapshot_at)
            """,
            connection,
        )
        reports = pd.read_sql_query(
            "SELECT thscode, statement_type, period_end_ms, payload_json "
            "FROM fundamental_reports WHERE period = 'annual'",
            connection,
        )
    except (sqlite3.DatabaseError, pd.errors.DatabaseError):
        return {}, "A 类基本面缓存结构不完整"
    finally:
        connection.close()
    if valuations.empty or reports.empty:
        return {}, "A 类基本面缓存为空"

    one_year_ago = str(pd.Timestamp(latest_date) - pd.Timedelta(days=365))[:10]
    market_connection = sqlite3.connect(market_database)
    try:
        dividends = pd.read_sql_query(
            """
            SELECT thscode, SUM(dividend_per_share) AS dividend_per_share
            FROM adjustment_events
            WHERE ex_date > ? AND ex_date <= ?
            GROUP BY thscode
            """,
            market_connection,
            params=(one_year_ago, latest_date),
        )
    finally:
        market_connection.close()

    values = valuations.merge(
        latest[["thscode", "close_price"]], on="thscode", how="inner"
    ).merge(dividends, on="thscode", how="left")
    values["dividend_per_share"] = values["dividend_per_share"].fillna(0)
    values["dividend_yield"] = (
        values["dividend_per_share"] / values["close_price"]
    )
    report_groups = {
        symbol: group for symbol, group in reports.groupby("thscode")
    }
    rows: list[dict[str, Any]] = []
    for row in values.itertuples(index=False):
        symbol_reports = report_groups.get(row.thscode)
        if symbol_reports is None:
            continue
        income = _latest_two(symbol_reports, "income")
        balance = _latest_two(symbol_reports, "balance")
        cash_flow = _latest_two(symbol_reports, "cash_flow")
        if not income or not balance or not cash_flow:
            continue
        current_income, previous_income = income
        current_balance, previous_balance = balance
        current_cash, _ = cash_flow

        def number(payload: dict[str, Any], key: str) -> float:
            value = payload.get(key)
            return float(value) if value is not None else np.nan

        assets = number(current_balance, "assets_total")
        previous_assets = number(previous_balance, "assets_total")
        equity = number(current_balance, "holder_equity_total")
        previous_equity = number(previous_balance, "holder_equity_total")
        debt = number(current_balance, "total_debt")
        previous_debt = number(previous_balance, "total_debt")
        current_assets = number(current_balance, "total_current_assets")
        previous_current_assets = number(
            previous_balance, "total_current_assets"
        )
        income_value = number(current_income, "operating_income")
        previous_income_value = number(previous_income, "operating_income")
        cost = number(current_income, "operating_costs")
        previous_cost = number(previous_income, "operating_costs")
        profit = number(current_income, "net_profit")
        previous_profit = number(previous_income, "net_profit")
        operating_cash = number(current_cash, "act_cash_flow_net")
        capex = number(current_cash, "pay_fixed_assets_etc_cash")

        roe = profit / equity
        previous_roe = previous_profit / previous_equity
        current_asset_turnover = income_value / current_assets
        previous_asset_turnover = previous_income_value / previous_current_assets
        total_turnover = income_value / assets
        previous_total_turnover = previous_income_value / previous_assets
        leverage = debt / assets
        previous_leverage = previous_debt / previous_assets
        ffscore = sum(
            int(bool(condition))
            for condition in (
                roe > 0,
                roe > previous_roe,
                current_asset_turnover > previous_asset_turnover,
                total_turnover > previous_total_turnover,
                leverage < previous_leverage,
            )
        )
        free_cash_flow = operating_cash - capex
        rows.append(
            {
                "thscode": row.thscode,
                "ffscore": ffscore,
                "pe_ttm": row.pe_ttm,
                "pb_mrq": row.pb_mrq,
                "pcf_ttm": row.pcf_ttm,
                "dividend_yield": row.dividend_yield,
                "debt_ratio": debt / assets,
                "free_cash_flow": free_cash_flow,
            }
        )
    metrics = pd.DataFrame(rows)
    if metrics.empty:
        return {}, "A 类缓存没有足够的连续两年报表"

    positive_pe = metrics.loc[metrics["pe_ttm"].gt(0), "pe_ttm"]
    positive_pcf = metrics.loc[metrics["pcf_ttm"].gt(0), "pcf_ttm"]
    average_pe = positive_pe.mean()
    average_pcf = positive_pcf.mean()
    average_dividend = metrics["dividend_yield"].mean()
    metrics["A/华泰FFScore"] = (
        metrics["ffscore"].ge(4)
        & metrics["pb_mrq"].gt(0)
        & metrics["pb_mrq"].le(metrics["pb_mrq"].quantile(0.30))
    )
    metrics["A/罗伯·瑞克超额现金流"] = (
        metrics["pb_mrq"].between(0, 3)
        & metrics["pe_ttm"].between(0, average_pe)
        & metrics["pcf_ttm"].between(0, average_pcf * 0.8)
        & metrics["dividend_yield"].gt(average_dividend)
        & metrics["debt_ratio"].lt(0.33)
        & metrics["free_cash_flow"].gt(0)
    )
    mapping: dict[str, list[str]] = {}
    for _, row in metrics.iterrows():
        names = [
            name for name in FUNDAMENTAL_POINTS if bool(row[name])
        ]
        if names:
            mapping[str(row["thscode"])] = names
    return mapping, f"A 类分析 {len(metrics)} 只缓存股票"
