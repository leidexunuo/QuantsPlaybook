"""从 Sequoia/daily_k 读取十年日 K 并构造无未来函数特征。"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


RAW_COLUMNS = [
    "thscode",
    "trade_date",
    "open_price",
    "high_price",
    "low_price",
    "close_price",
    "volume",
    "turnover",
]

FEATURE_COLUMNS = [
    "ret_1",
    "ret_5",
    "ret_20",
    "ret_60",
    "ret_120",
    "vol_20",
    "vol_60",
    "close_ma_5",
    "close_ma_20",
    "close_ma_60",
    "range_20",
    "close_location_20",
    "turnover_log",
    "turnover_ratio_20",
    "volume_ratio_20",
    "market_ret_20",
    "market_below_ma_20",
]


@dataclass(frozen=True)
class FeatureScaler:
    mean: dict[str, float]
    std: dict[str, float]

    @classmethod
    def fit(cls, frame: pd.DataFrame) -> "FeatureScaler":
        values = frame[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
        mean = values.mean().fillna(0.0)
        std = values.std().replace(0.0, np.nan).fillna(1.0)
        return cls(mean=mean.to_dict(), std=std.to_dict())

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        for name in FEATURE_COLUMNS:
            result[name] = (
                (result[name] - self.mean[name]) / self.std[name]
            ).clip(-10.0, 10.0)
        result[FEATURE_COLUMNS] = result[FEATURE_COLUMNS].fillna(0.0)
        return result


def subtract_years(value: date, years: int) -> date:
    try:
        return value.replace(year=value.year - years)
    except ValueError:
        return value.replace(year=value.year - years, day=28)


def database_coverage(database: Path) -> tuple[str, str]:
    uri = f"file:{database.resolve().as_posix()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        row = connection.execute(
            "SELECT MIN(trade_date), MAX(trade_date) FROM daily_k"
        ).fetchone()
    if not row or row[0] is None or row[1] is None:
        raise RuntimeError("daily_k 表为空")
    return str(row[0]), str(row[1])


def select_liquid_universe(
    connection: sqlite3.Connection,
    start_date: str,
    end_date: str,
    universe_size: int,
    minimum_rows: int = 120,
) -> list[str]:
    rows = connection.execute(
        """
        SELECT thscode
        FROM daily_k
        WHERE trade_date BETWEEN ? AND ?
          AND close_price > 0
          AND turnover > 0
        GROUP BY thscode
        HAVING COUNT(*) >= ?
        ORDER BY AVG(turnover) DESC
        LIMIT ?
        """,
        (start_date, end_date, minimum_rows, universe_size),
    ).fetchall()
    symbols = [str(row[0]) for row in rows]
    if not symbols:
        raise RuntimeError("指定区间没有满足条件的股票")
    return symbols


def load_daily_bars(
    database: Path,
    years: int = 10,
    universe_size: int = 800,
    end_date: str | None = None,
    warmup_days: int = 240,
) -> tuple[pd.DataFrame, str, str]:
    """读取流动性较好的股票，返回含指标预热区间的行情。"""
    if not database.is_file():
        raise FileNotFoundError(f"行情数据库不存在: {database}")
    _, database_end = database_coverage(database)
    effective_end = date.fromisoformat(end_date or database_end)
    decision_start = subtract_years(effective_end, years)
    query_start = decision_start - timedelta(days=warmup_days)

    uri = f"file:{database.resolve().as_posix()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        symbols = select_liquid_universe(
            connection,
            decision_start.isoformat(),
            effective_end.isoformat(),
            universe_size,
        )
        placeholders = ",".join("?" for _ in symbols)
        bars = pd.read_sql_query(
            f"""
            SELECT {", ".join(RAW_COLUMNS)}
            FROM daily_k
            WHERE trade_date BETWEEN ? AND ?
              AND thscode IN ({placeholders})
            ORDER BY thscode, trade_date
            """,
            connection,
            params=(
                query_start.isoformat(),
                effective_end.isoformat(),
                *symbols,
            ),
        )
    if bars.empty:
        raise RuntimeError("未读取到训练行情")
    return bars, decision_start.isoformat(), effective_end.isoformat()


def _rolling(
    frame: pd.DataFrame,
    column: str,
    window: int,
    operation: str,
    min_periods: int | None = None,
) -> pd.Series:
    grouped = frame.groupby("thscode", sort=False)[column]
    rolling = grouped.rolling(
        window,
        min_periods=window if min_periods is None else min_periods,
    )
    return (
        getattr(rolling, operation)()
        .reset_index(level=0, drop=True)
        .sort_index()
    )


def build_features(
    bars: pd.DataFrame,
    decision_start: str,
    decision_end: str,
) -> pd.DataFrame:
    """特征整体后移一日：当日开盘交易只使用前一日及更早的数据。"""
    data = bars.sort_values(["thscode", "trade_date"]).copy()
    grouped = data.groupby("thscode", sort=False)
    data["_daily_return"] = grouped["close_price"].pct_change(
        fill_method=None
    )
    for window in (1, 5, 20, 60, 120):
        data[f"_ret_{window}"] = grouped["close_price"].pct_change(
            window, fill_method=None
        )
    for window in (20, 60):
        data[f"_vol_{window}"] = _rolling(
            data, "_daily_return", window, "std"
        )
    for window in (5, 20, 60):
        moving_average = _rolling(data, "close_price", window, "mean")
        data[f"_close_ma_{window}"] = data["close_price"] / moving_average - 1

    high_20 = _rolling(data, "high_price", 20, "max")
    low_20 = _rolling(data, "low_price", 20, "min")
    data["_range_20"] = high_20 / low_20.replace(0.0, np.nan) - 1.0
    data["_close_location_20"] = (
        (data["close_price"] - low_20)
        / (high_20 - low_20).replace(0.0, np.nan)
    )
    turnover_mean = _rolling(data, "turnover", 20, "mean")
    volume_mean = _rolling(data, "volume", 20, "mean")
    data["_turnover_log"] = np.log1p(data["turnover"].clip(lower=0.0))
    data["_turnover_ratio_20"] = (
        data["turnover"] / turnover_mean.replace(0.0, np.nan)
    )
    data["_volume_ratio_20"] = (
        data["volume"] / volume_mean.replace(0.0, np.nan)
    )
    data["_avg_turnover_20"] = turnover_mean

    market_return = data.groupby("trade_date")["_daily_return"].mean()
    market_nav = (1.0 + market_return.fillna(0.0)).cumprod()
    market_ma = market_nav.rolling(20, min_periods=20).mean()
    market_ret_20 = market_nav.pct_change(20, fill_method=None)
    # 整体后移一个市场交易日，停牌股票恢复交易后也不会得到陈旧的大盘状态。
    market = pd.DataFrame(
        {
            "market_ret_20": market_ret_20,
            "market_below_ma_20": (market_nav < market_ma).astype(float),
        }
    ).shift(1)
    data = data.merge(
        market,
        left_on="trade_date",
        right_index=True,
        how="left",
        validate="many_to_one",
    )

    source_names = [
        "_ret_1",
        "_ret_5",
        "_ret_20",
        "_ret_60",
        "_ret_120",
        "_vol_20",
        "_vol_60",
        "_close_ma_5",
        "_close_ma_20",
        "_close_ma_60",
        "_range_20",
        "_close_location_20",
        "_turnover_log",
        "_turnover_ratio_20",
        "_volume_ratio_20",
        "_avg_turnover_20",
    ]
    shifted = data.groupby("thscode", sort=False)[source_names].shift(1)
    shifted.columns = [name.removeprefix("_") for name in source_names]
    data[shifted.columns] = shifted
    data["feature_ready"] = data[FEATURE_COLUMNS].notna().all(axis=1)
    data["market_return_20_raw"] = data["market_ret_20"]
    data["market_below_ma_20_raw"] = data["market_below_ma_20"]
    data["market_risk_off"] = (
        data["market_below_ma_20"].gt(0.5)
        & data["market_ret_20"].le(-0.05)
    )
    result = data.loc[
        data["trade_date"].between(decision_start, decision_end)
    ].copy()
    result.replace([np.inf, -np.inf], np.nan, inplace=True)
    return result.reset_index(drop=True)
