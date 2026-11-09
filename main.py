"""使用 QuantsPlaybook 的 A/B/C/D 策略选股并推送到 Teams。

日线策略直接读取 A 股行情 SQLite；A 类 FFScore 与超额现金流策略读取
``Sequoia/fundamentals.sqlite3`` 增量缓存。Financial-API 的 X-api-key 与
Teams Webhook 统一从 ``Sequoia/config.json`` 读取，该文件被 Git 忽略。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib.util
import json
import math
import os
import sqlite3
import sys
import urllib.parse
import urllib.error
import urllib.request
import time
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from Sequoia.strategies import (
    FUNDAMENTAL_POINTS,
    TECHNICAL_POINTS,
    add_extended_features,
    apply_risk_parity,
    load_fundamental_signals,
    technical_signals,
)
from Sequoia.strategy_registry import STRATEGY_NAMES


ROOT = Path(__file__).resolve().parent
DEFAULT_DATABASE = ROOT / "Sequoia" / "a_share_market.sqlite3"
DEFAULT_HISTORY_DATABASE = ROOT / "Sequoia" / "selection_history.sqlite3"
DEFAULT_FUNDAMENTAL_DATABASE = ROOT / "Sequoia" / "fundamentals.sqlite3"
DEFAULT_CONFIG = ROOT / "Sequoia" / "config.json"
DEFAULT_ML_MODEL = ROOT / "ML" / "artifacts" / "actor_critic.pt"
ICU_MODULE = ROOT / "C-择时类" / "ICU均线" / "src" / "icu_ma.py"
DE_MODULE = ROOT / "D-组合优化" / "DE算法下的组合优化" / "py" / "DE_algorithm.py"
TEAMS_WEBHOOK_ENV = "TEAMS_WEBHOOK_URL"
STRATEGY_POINTS: dict[str, float] = {}
DRAGON_TIGER_BOARD_TYPE = "all"


@dataclass(frozen=True)
class TradingSettings:
    budget_per_stock: float = 100_000.0
    lot_size: int = 100
    buy_fee_rate: float = 0.00025
    sell_fee_rate: float = 0.001

    def validate(self) -> None:
        if self.budget_per_stock <= 0:
            raise ValueError("trading.budget_per_stock 必须大于 0")
        if self.lot_size < 100:
            raise ValueError("trading.lot_size 不能小于 100 股")
        if self.buy_fee_rate < 0 or self.sell_fee_rate < 0:
            raise ValueError("trading 买卖手续费率不能为负")


@dataclass(frozen=True)
class ExitSettings:
    take_profit_rate: float = 0.20
    stop_loss_rate: float = 0.10
    max_holding_days: int = 60
    trailing_drawdown_rate: float = 0.12
    score_decline_confirmation_days: int = 3
    market_ma_window: int = 20
    market_return_window: int = 20
    market_return_threshold: float = -0.05
    reentry_cooldown_days: int = 5

    def validate(self) -> None:
        if self.take_profit_rate <= 0:
            raise ValueError("exit_rules.take_profit_rate 必须大于 0")
        if not 0 < self.stop_loss_rate < 1:
            raise ValueError("exit_rules.stop_loss_rate 必须在 (0, 1) 内")
        if self.max_holding_days <= 0:
            raise ValueError("exit_rules.max_holding_days 必须大于 0")
        if not 0 < self.trailing_drawdown_rate < 1:
            raise ValueError(
                "exit_rules.trailing_drawdown_rate 必须在 (0, 1) 内"
            )
        if self.score_decline_confirmation_days <= 0:
            raise ValueError(
                "exit_rules.score_decline_confirmation_days 必须大于 0"
            )
        if self.market_ma_window < 2 or self.market_return_window < 2:
            raise ValueError("大盘风控窗口至少为 2")
        if self.market_return_threshold >= 0:
            raise ValueError("market_return_threshold 必须小于 0")
        if self.reentry_cooldown_days < 0:
            raise ValueError("reentry_cooldown_days 不能为负")


@dataclass(frozen=True)
class StrategyWeightSettings:
    initial_weight: float = 1.0
    learning_rate: float = 0.25
    return_scale: float = 0.10
    min_weight: float = 0.20
    max_weight: float = 5.0

    def validate(self) -> None:
        if self.initial_weight <= 0:
            raise ValueError("strategy_weights.initial_weight 必须大于 0")
        if self.learning_rate <= 0:
            raise ValueError("strategy_weights.learning_rate 必须大于 0")
        if self.return_scale <= 0:
            raise ValueError("strategy_weights.return_scale 必须大于 0")
        if not 0 < self.min_weight <= self.initial_weight:
            raise ValueError(
                "strategy_weights.min_weight 必须大于 0 且不高于初始权重"
            )
        if self.max_weight < self.initial_weight:
            raise ValueError(
                "strategy_weights.max_weight 不能低于初始权重"
            )


def calculate_buy_quantity(
    price: float,
    settings: TradingSettings,
) -> int:
    """按含手续费预算向下取整到整手，预算不足一手时返回 0。"""
    if not math.isfinite(price) or price <= 0:
        return 0
    affordable_shares = settings.budget_per_stock / (
        price * (1.0 + settings.buy_fee_rate)
    )
    return int(affordable_shares // settings.lot_size) * settings.lot_size


def transaction_values(
    price: float,
    quantity: int,
    fee_rate: float,
    side: str,
) -> tuple[float, float, float]:
    """返回交易额、手续费、实际支出（买）或净到账（卖）。"""
    if side not in {"buy", "sell"}:
        raise ValueError("side 必须是 buy 或 sell")
    trade_amount = float(price * quantity)
    fee = float(trade_amount * fee_rate)
    cash_amount = trade_amount + fee if side == "buy" else trade_amount - fee
    return trade_amount, fee, cash_amount


def market_risk_status(
    features: pd.DataFrame,
    settings: ExitSettings,
) -> tuple[bool, str]:
    """使用全市场等权收益判断是否进入风险关闭状态。"""
    market_returns = (
        features.groupby("trade_date", sort=True)["daily_return"]
        .mean()
        .dropna()
    )
    required = max(settings.market_ma_window, settings.market_return_window) + 1
    if len(market_returns) < required:
        return False, "大盘数据不足，未触发风控"
    market_nav = (1.0 + market_returns).cumprod()
    market_ma = market_nav.rolling(settings.market_ma_window).mean()
    market_return = (
        market_nav.iloc[-1] / market_nav.iloc[-1 - settings.market_return_window]
        - 1.0
    )
    below_ma = market_nav.iloc[-1] < market_ma.iloc[-1]
    risk_off = bool(
        below_ma and market_return <= settings.market_return_threshold
    )
    summary = (
        f"等权市场{settings.market_return_window}日收益"
        f"{market_return:.2%}，"
        f"{'低于' if below_ma else '高于'}"
        f"{settings.market_ma_window}日均线"
    )
    return risk_off, summary

PRICE_COLUMNS = [
    "thscode",
    "trade_date",
    "open_price",
    "high_price",
    "low_price",
    "close_price",
    "volume",
    "turnover",
]


def load_module(module_name: str, path: Path):
    """Load an existing strategy module whose directory is not a Python package."""
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载模块: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_history(database: Path, trading_days: int = 260) -> tuple[pd.DataFrame, str]:
    if not database.is_file():
        raise FileNotFoundError(f"SQLite 数据库不存在: {database}")
    connection = sqlite3.connect(database)
    try:
        latest_date = connection.execute(
            "SELECT MAX(trade_date) FROM daily_k"
        ).fetchone()[0]
        if latest_date is None:
            raise RuntimeError("daily_k 表为空")
        start_date = connection.execute(
            """
            SELECT MIN(trade_date)
            FROM (
                SELECT DISTINCT trade_date
                FROM daily_k
                ORDER BY trade_date DESC
                LIMIT ?
            )
            """,
            (trading_days,),
        ).fetchone()[0]
        history = pd.read_sql_query(
            f"""
            SELECT {", ".join(PRICE_COLUMNS)}
            FROM daily_k
            WHERE trade_date >= ?
            ORDER BY thscode, trade_date
            """,
            connection,
            params=(start_date,),
        )
    finally:
        connection.close()
    return history, str(latest_date)


def rolling_group(
    frame: pd.DataFrame, column: str, window: int, operation: str
) -> pd.Series:
    grouped = frame.groupby("thscode", sort=False)[column]
    rolling = grouped.rolling(window, min_periods=window)
    result = getattr(rolling, operation)()
    return result.reset_index(level=0, drop=True).sort_index()


def build_features(history: pd.DataFrame) -> pd.DataFrame:
    """Build the shared features used by the B/C strategy adapters."""
    data = history.copy()
    grouped = data.groupby("thscode", sort=False)
    data["daily_return"] = grouped["close_price"].pct_change(fill_method=None)
    data["ret_20"] = grouped["close_price"].pct_change(20, fill_method=None)
    data["ret_120"] = grouped["close_price"].pct_change(120, fill_method=None)
    data["ma_5"] = rolling_group(data, "close_price", 5, "mean")
    data["ma_20"] = rolling_group(data, "close_price", 20, "mean")
    data["vol_120"] = rolling_group(data, "daily_return", 120, "std")
    data["vol_20"] = rolling_group(data, "daily_return", 20, "std")
    data["year_high"] = (
        grouped["high_price"]
        .rolling(250, min_periods=1)
        .max()
        .reset_index(level=0, drop=True)
        .sort_index()
    )
    data["year_low"] = (
        grouped["low_price"]
        .rolling(250, min_periods=1)
        .min()
        .reset_index(level=0, drop=True)
        .sort_index()
    )
    data["avg_turnover_20"] = rolling_group(data, "turnover", 20, "mean")

    candle_range = (data["high_price"] - data["low_price"]).replace(0, np.nan)
    upper_shadow = data["high_price"] - data[
        ["open_price", "close_price"]
    ].max(axis=1)
    lower_shadow = data[["open_price", "close_price"]].min(axis=1) - data[
        "low_price"
    ]
    data["lower_shadow_dominance"] = (lower_shadow - upper_shadow) / candle_range
    data["shadow_factor_20"] = rolling_group(
        data, "lower_shadow_dominance", 20, "mean"
    )

    # The existing alligator implementation uses SMA(13/8/5), shifted by
    # 8/5/3 days respectively.  Reproduce those exact lines without requiring
    # TA-Lib, which is optional in the original project.
    data["jaw"] = grouped["close_price"].transform(
        lambda values: values.rolling(13, min_periods=13).mean().shift(8)
    )
    data["teeth"] = grouped["close_price"].transform(
        lambda values: values.rolling(8, min_periods=8).mean().shift(5)
    )
    data["lips"] = grouped["close_price"].transform(
        lambda values: values.rolling(5, min_periods=5).mean().shift(3)
    )
    data["alligator_bullish"] = (
        (data["jaw"] < data["teeth"]) & (data["teeth"] < data["lips"])
    )
    data["alligator_bullish_previous"] = grouped["alligator_bullish"].shift(1)
    return add_extended_features(data)


def apply_icu_feature(history: pd.DataFrame, latest: pd.DataFrame) -> pd.DataFrame:
    """Call the ICU implementation in C-择时类 and return its latest value."""
    module = load_module("quantsplaybook_icu_ma", ICU_MODULE)
    rows: list[tuple[str, float, float]] = []
    for symbol, group in history.groupby("thscode", sort=False):
        closes = group["close_price"].dropna().to_numpy(dtype=float)
        if len(closes) < 21:
            continue
        current = float(module.siegelslopes_ma(closes[-20:]))
        previous = float(module.siegelslopes_ma(closes[-21:-1]))
        rows.append((symbol, current, previous))
    icu = pd.DataFrame(rows, columns=["thscode", "icu_20", "icu_20_previous"])
    return latest.merge(icu, on="thscode", how="left")


def select_candidates(
    features: pd.DataFrame,
    latest_date: str,
    fundamental_signals: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    latest = features.loc[features["trade_date"].eq(latest_date)].copy()
    if latest.empty:
        raise RuntimeError(f"{latest_date} 没有行情")

    # Exclude effectively illiquid names.  The threshold is deliberately low
    # enough to retain small-cap A shares while avoiding unusable candidates.
    liquid = latest["avg_turnover_20"].ge(20_000_000)

    quality_momentum = latest["ret_120"] / (
        latest["vol_120"] * math.sqrt(120)
    ).replace(0, np.nan)
    latest["_liquid"] = liquid
    latest["_momentum_score"] = quality_momentum
    momentum_cutoff = quality_momentum[liquid].quantile(0.95)
    latest["B/高质量动量因子选股"] = (
        liquid
        & quality_momentum.ge(momentum_cutoff)
        & latest["ret_20"].gt(0)
        & latest["close_price"].gt(latest["ma_20"])
    )

    shadow_cutoff = latest.loc[liquid, "shadow_factor_20"].quantile(0.95)
    latest["B/上下影线因子"] = (
        liquid
        & latest["shadow_factor_20"].ge(shadow_cutoff)
        & latest["close_price"].gt(latest["ma_20"])
    )

    latest = apply_icu_feature(features, latest)
    liquid = latest["_liquid"].fillna(False).astype(bool)
    latest["C/ICU均线"] = (
        liquid
        & latest["close_price"].gt(latest["icu_20"])
        & latest["icu_20"].gt(latest["icu_20_previous"])
        & latest["ma_5"].gt(latest["ma_20"])
    )
    latest["C/鳄鱼线"] = (
        liquid
        & latest["alligator_bullish"]
        & ~latest["alligator_bullish_previous"].eq(True)
    )

    extended, extended_strategy_columns = technical_signals(
        features, latest_date
    )
    extended = extended.set_index("thscode")
    for name in extended_strategy_columns:
        latest[name] = latest["thscode"].map(extended[name]).fillna(False)

    fundamental_signals = fundamental_signals or {}
    for name in FUNDAMENTAL_POINTS:
        latest[name] = latest["thscode"].map(
            lambda symbol: name in fundamental_signals.get(symbol, [])
        )

    strategy_columns = [
        *FUNDAMENTAL_POINTS,
        "B/高质量动量因子选股",
        "B/上下影线因子",
        *extended_strategy_columns,
        "C/ICU均线",
        "C/鳄鱼线",
    ]
    latest["strategies"] = latest.apply(
        lambda row: [name for name in strategy_columns if bool(row[name])], axis=1
    )
    candidates = latest.loc[latest["strategies"].str.len().gt(0)].copy()
    candidates["signal_count"] = candidates["strategies"].str.len()
    candidates["strategy_score"] = candidates["strategies"].map(
        lambda names: sum(STRATEGY_POINTS[name] for name in names)
    )
    candidates["momentum_score"] = candidates["_momentum_score"]
    return candidates


def load_de_class():
    """Load and repair the repository's DE helper's invalid index sampler."""
    module = load_module("quantsplaybook_de", DE_MODULE)

    def check_random_indices(random_idx: np.ndarray):
        size = len(random_idx)
        for row in range(size):
            while len(set(int(value) for value in random_idx[row])) < 3:
                random_idx[row] = np.random.randint(0, size, size=3)
        return random_idx[:, 0], random_idx[:, 1], random_idx[:, 2]

    module.DE._check_r = staticmethod(check_random_indices)
    return module.DE


def apply_de_optimization(
    candidates: pd.DataFrame,
    features: pd.DataFrame,
    max_assets: int = 40,
    selected_assets: int = 20,
) -> pd.DataFrame:
    """Use D/DE算法下的组合优化 to rank the strongest signal candidates."""
    if candidates.empty:
        candidates["de_weight"] = pd.Series(dtype=float)
        return candidates

    pool = candidates.sort_values(
        ["signal_count", "momentum_score", "avg_turnover_20"],
        ascending=False,
    ).head(max_assets)
    prices = (
        features.loc[features["thscode"].isin(pool["thscode"])]
        .pivot(index="trade_date", columns="thscode", values="close_price")
        .sort_index()
    )
    returns = prices.pct_change(fill_method=None).tail(120)
    usable = returns.columns[returns.notna().mean().ge(0.80)]
    returns = returns[usable].fillna(0.0)
    if returns.shape[1] < 2:
        candidates["de_weight"] = 0.0
        return candidates

    means = returns.mean().to_numpy(dtype=float) * 252
    covariance = returns.cov().to_numpy(dtype=float) * 252

    def objective(raw_weights: np.ndarray) -> float:
        weights = np.clip(raw_weights, 1e-12, None)
        weights = weights / weights.sum()
        expected_return = float(means @ weights)
        volatility = float(np.sqrt(max(weights @ covariance @ weights, 1e-12)))
        concentration = float(np.square(weights).sum())
        return -(expected_return / volatility) + 0.02 * concentration

    np.random.seed(42)
    de_class = load_de_class()
    optimizer = de_class(
        objective,
        n_dim=returns.shape[1],
        size_pop=40,
        max_iter=80,
        prob_mut=0.5,
        lb=0.0,
        ub=1.0,
    )
    raw_weights, _ = optimizer.run()
    weights = np.clip(np.asarray(raw_weights, dtype=float), 0, None)
    weights = weights / weights.sum()
    weight_map = dict(zip(returns.columns, weights))
    candidates["de_weight"] = candidates["thscode"].map(weight_map).fillna(0.0)

    optimized = set(
        candidates.nlargest(min(selected_assets, len(candidates)), "de_weight")[
            "thscode"
        ]
    )
    candidates.loc[candidates["thscode"].isin(optimized), "strategies"] = (
        candidates.loc[candidates["thscode"].isin(optimized), "strategies"].apply(
            lambda names: [*names, "D/DE组合优化"]
        )
    )
    candidates["signal_count"] = candidates["strategies"].str.len()
    candidates["strategy_score"] = candidates["strategies"].map(
        lambda names: sum(STRATEGY_POINTS[name] for name in names)
    )
    return candidates


def xueqiu_symbol(thscode: str) -> str:
    ticker, exchange = thscode.split(".", maxsplit=1)
    return f"{exchange.upper()}{ticker}"


def request_json(
    url: str,
    headers: dict[str, str] | None = None,
    timeout: int = 20,
) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 QuantsPlaybook/1.0",
            "Connection": "close",
            **(headers or {}),
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_dragon_tiger_list(
    api_config: dict[str, Any],
    trade_date: str,
    timeout: int = 30,
) -> dict[str, Any]:
    """Fetch the full A-share dragon-tiger list for one trading day."""
    api_key = str(api_config.get("x_api_key") or "").strip()
    if not api_key:
        raise ValueError(
            "龙虎榜查询需要 financial_api.x_api_key；"
            "如需离线运行请使用 --no-dragon-tiger"
        )
    base_url = str(
        api_config.get("base_url") or "https://fuyao.aicubes.cn"
    ).rstrip("/")
    query = urllib.parse.urlencode(
        {"board_type": DRAGON_TIGER_BOARD_TYPE, "date": trade_date}
    )
    url = (
        f"{base_url}/api/a-share/special-data/dragon-tiger-list?{query}"
    )
    last_error: BaseException | None = None
    for attempt in range(4):
        try:
            payload = request_json(
                url,
                headers={"X-api-key": api_key},
                timeout=timeout,
            )
            code = payload.get("code")
            if code == 0:
                data = payload.get("data")
                if not isinstance(data, dict):
                    raise RuntimeError("龙虎榜接口成功响应缺少 data object")
                actual_date = str(data.get("trade_date") or "")
                if actual_date != trade_date:
                    raise RuntimeError(
                        f"龙虎榜接口返回交易日 {actual_date or '空'}，"
                        f"预期 {trade_date}"
                    )
                return data
            message = payload.get("message")
            if code == 4001 or code in {5001, 5002, 5003}:
                last_error = RuntimeError(
                    f"code={code}, message={message}, "
                    f"request_id={payload.get('request_id')}"
                )
            else:
                raise RuntimeError(
                    f"龙虎榜查询失败: code={code}, message={message}, "
                    f"request_id={payload.get('request_id')}"
                )
        except (OSError, ValueError, urllib.error.URLError) as error:
            last_error = error
        if attempt < 3:
            time.sleep(2**attempt)
    raise RuntimeError(f"龙虎榜查询重试后仍失败: {last_error}") from last_error


DRAGON_TIGER_API_COLUMNS = [
    "thscode",
    "ticker",
    "stock_name",
    "xueqiu_url",
    "concept_list",
    "change",
    "net_value",
    "net_rate",
    "hot_rank",
    "buy_value",
    "sell_value",
    "limit_reason",
    "range_days",
    "org_net_value",
    "org_net_rate",
    "org_buy_num",
    "org_sell_num",
    "amount",
    "hot_money_net_value",
    "hot_money_net_rate",
    "hot_money_item_net_value",
    "hot_money_item_net_rate",
]


def prepare_dragon_tiger_results(data: dict[str, Any]) -> pd.DataFrame:
    """Normalize Financial-API dragon-tiger stock rows for storage."""
    raw_items = data.get("stock_items") or []
    if not isinstance(raw_items, list):
        raise RuntimeError("龙虎榜接口 stock_items 不是 array")
    rows: list[dict[str, Any]] = []
    for item in raw_items:
        if not isinstance(item, dict) or not item.get("thscode"):
            continue
        symbol = str(item["thscode"])
        concepts = item.get("concept_list") or []
        rows.append(
            {
                "thscode": symbol,
                "ticker": str(
                    item.get("ticker") or symbol.split(".", maxsplit=1)[0]
                ),
                "stock_name": str(
                    item.get("name")
                    or symbol.split(".", maxsplit=1)[0]
                ),
                "xueqiu_url": (
                    f"https://xueqiu.com/S/{xueqiu_symbol(symbol)}"
                ),
                "concept_list": json.dumps(
                    concepts, ensure_ascii=False, separators=(",", ":")
                ),
                **{
                    column: item.get(column)
                    for column in DRAGON_TIGER_API_COLUMNS
                    if column
                    not in {
                        "thscode",
                        "ticker",
                        "stock_name",
                        "xueqiu_url",
                        "concept_list",
                    }
                },
            }
        )
    results = pd.DataFrame(rows, columns=DRAGON_TIGER_API_COLUMNS)
    if results.empty:
        return results
    results["_net_value_sort"] = pd.to_numeric(
        results["net_value"], errors="coerce"
    ).fillna(-math.inf)
    results.sort_values(
        ["_net_value_sort", "thscode"],
        ascending=[False, True],
        inplace=True,
    )
    results.drop(columns="_net_value_sort", inplace=True)
    return results.reset_index(drop=True)


def resolve_stock_names_funyao(
    symbols: Iterable[str],
    api_config: dict[str, Any],
    timeout: int = 20,
) -> dict[str, str]:
    """Resolve names with Financial-API's A-share valuation snapshot."""
    api_key = str(api_config.get("x_api_key") or "").strip()
    if not api_key:
        return {}
    base_url = str(
        api_config.get("base_url") or "https://fuyao.aicubes.cn"
    ).rstrip("/")
    names: dict[str, str] = {}
    symbol_list = list(symbols)
    for start in range(0, len(symbol_list), 100):
        batch = symbol_list[start : start + 100]
        query = urllib.parse.urlencode({"thscodes": ",".join(batch)})
        url = f"{base_url}/api/a-share/valuations/snapshot?{query}"
        try:
            payload = request_json(
                url, headers={"X-api-key": api_key}, timeout=timeout
            )
            if payload.get("code") != 0:
                raise RuntimeError(
                    f"code={payload.get('code')}, message={payload.get('message')}"
                )
            for item in (payload.get("data") or {}).get("item") or []:
                if item.get("thscode") and item.get("name"):
                    names[str(item["thscode"])] = str(item["name"])
        except (OSError, ValueError, RuntimeError, urllib.error.URLError) as error:
            print(f"Financial-API 股票名称查询失败: {error}", file=sys.stderr)
            break
    return names


def resolve_stock_names(
    symbols: Iterable[str],
    api_config: dict[str, Any] | None = None,
    timeout: int = 15,
) -> dict[str, str]:
    """Resolve names, preferring the configured Financial-API."""
    symbol_list = list(symbols)
    names = resolve_stock_names_funyao(
        symbol_list, api_config or {}, timeout=max(timeout, 20)
    )
    unresolved = [symbol for symbol in symbol_list if symbol not in names]
    if not unresolved:
        return names

    # Fallback providers keep notifications usable during an API outage.
    for start in range(0, len(unresolved), 50):
        batch = unresolved[start : start + 50]
        query_codes = [
            f"{symbol.split('.')[1].lower()}{symbol.split('.')[0]}" for symbol in batch
        ]
        url = "https://qt.gtimg.cn/q=" + ",".join(query_codes)
        request = urllib.request.Request(
            url, headers={"User-Agent": "Mozilla/5.0 QuantsPlaybook/1.0"}
        )
        content = ""
        for attempt in range(3):
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    content = response.read().decode("gbk", errors="replace")
                break
            except (OSError, urllib.error.URLError) as error:
                if attempt == 2:
                    print(f"腾讯股票名称查询失败: {error}", file=sys.stderr)
                else:
                    time.sleep(1 + attempt * 2)
        if not content:
            break
        for line in content.splitlines():
            if '="' not in line:
                continue
            key, value = line.split('="', maxsplit=1)
            fields = value.rstrip('";').split("~")
            if len(fields) < 3 or not fields[1] or not fields[2]:
                continue
            exchange_code = key.rsplit("_", maxsplit=1)[-1]
            exchange = exchange_code[:2].upper()
            names[f"{fields[2]}.{exchange}"] = fields[1]
    missing = [symbol for symbol in symbol_list if symbol not in names]
    if missing:
        names.update(resolve_stock_names_eastmoney(missing, timeout))
    if len(names) < len(symbol_list):
        print(
            f"仍有 {len(symbol_list) - len(names)} 只股票未解析名称，使用代码代替。",
            file=sys.stderr,
        )
    return names


def resolve_stock_names_eastmoney(
    symbols: Iterable[str], timeout: int = 15
) -> dict[str, str]:
    """Fallback name resolver using Eastmoney's multi-security endpoint."""
    symbol_list = list(symbols)
    secids = []
    reverse: dict[str, str] = {}
    for symbol in symbol_list:
        ticker, exchange = symbol.split(".", maxsplit=1)
        market = "1" if exchange == "SH" else "0"
        secid = f"{market}.{ticker}"
        secids.append(secid)
        reverse[ticker] = symbol
    query = urllib.parse.urlencode(
        {
            "fltt": "2",
            "invt": "2",
            "fields": "f12,f14",
            "secids": ",".join(secids),
        }
    )
    request = urllib.request.Request(
        f"https://push2.eastmoney.com/api/qt/ulist.np/get?{query}",
        headers={
            "User-Agent": "Mozilla/5.0 QuantsPlaybook/1.0",
            "Connection": "close",
        },
    )
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
            rows = (payload.get("data") or {}).get("diff") or []
            return {
                reverse[str(row["f12"])]: str(row["f14"])
                for row in rows
                if str(row.get("f12")) in reverse and row.get("f14")
            }
        except (OSError, ValueError, KeyError, urllib.error.URLError) as error:
            if attempt == 2:
                print(f"东方财富股票名称查询失败: {error}", file=sys.stderr)
                return {}
            time.sleep(1 + attempt * 2)
    return {}


def prepare_results(
    candidates: pd.DataFrame,
    max_results: int | None,
    lookup_names: bool,
    api_config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    # 在线权重总分从高到低选 Top 30；--max-results 仅用于调试覆盖。
    ranked = candidates.sort_values(
        [
            "strategy_score",
            "signal_count",
            "de_weight",
            "momentum_score",
            "avg_turnover_20",
        ],
        ascending=False,
    )
    ranked = ranked.head(max_results or 30)
    selected = ranked.sort_values(
        ["strategy_score", "thscode"],
        ascending=[True, True],
    )
    names = (
        resolve_stock_names(selected["thscode"], api_config)
        if lookup_names
        else {}
    )
    selected = selected.copy()
    selected["stock_name"] = selected["thscode"].map(names)
    selected["stock_name"] = selected["stock_name"].where(
        selected["stock_name"].notna(),
        selected["thscode"].str.split(".").str[0],
    )
    selected["xueqiu_url"] = selected["thscode"].map(
        lambda symbol: f"https://xueqiu.com/S/{xueqiu_symbol(symbol)}"
    )
    selected["strategy"] = selected["strategies"].str.join("、")
    return selected[
        [
            "thscode",
            "stock_name",
            "xueqiu_url",
            "close_price",
            "year_high",
            "year_low",
            "strategy",
            "strategy_score",
        ]
    ].reset_index(drop=True)


def prepare_ml_results(
    history: pd.DataFrame,
    features: pd.DataFrame,
    latest_date: str,
    model_path: Path,
    lookup_names: bool,
    api_config: dict[str, Any] | None = None,
    top: int = 30,
) -> pd.DataFrame:
    """使用日 K 强化学习模型生成独立 Top 30 候选榜单。"""
    if not model_path.is_file():
        raise FileNotFoundError(f"强化学习模型不存在: {model_path}")
    try:
        import torch

        from ML.data import (
            FEATURE_COLUMNS as ML_FEATURE_COLUMNS,
            FeatureScaler,
            build_features as build_ml_features,
        )
        from ML.model import ActorCritic
    except ImportError as error:
        raise ImportError(
            "加载强化学习模型需要 PyTorch；请执行 "
            "python -m pip install -r ML/requirements.txt"
        ) from error

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        checkpoint = torch.load(
            model_path, map_location=device, weights_only=False
        )
        checkpoint_features = list(checkpoint["feature_columns"])
        if checkpoint_features != list(ML_FEATURE_COLUMNS):
            raise RuntimeError("模型特征定义与当前 ML/data.py 不一致")
        model = ActorCritic(len(ML_FEATURE_COLUMNS)).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
    except (KeyError, OSError, RuntimeError, ValueError) as error:
        raise RuntimeError(f"强化学习模型无法加载: {model_path}") from error
    model.eval()

    universe_size = int(checkpoint.get("universe_size", 800))
    universe = (
        history.groupby("thscode", sort=False)["turnover"]
        .mean()
        .nlargest(universe_size)
        .index
    )
    model_history = history.loc[history["thscode"].isin(universe)].copy()
    model_features = build_ml_features(
        model_history,
        str(model_history["trade_date"].min()),
        latest_date,
    )
    scaler = FeatureScaler(
        mean=dict(checkpoint["scaler_mean"]),
        std=dict(checkpoint["scaler_std"]),
    )
    model_features = scaler.transform(model_features)
    latest = model_features.loc[
        model_features["trade_date"].eq(latest_date)
        & model_features["avg_turnover_20"].ge(20_000_000)
        & model_features["feature_ready"]
    ].copy()
    if latest.empty:
        raise RuntimeError(f"{latest_date} 没有可供模型评分的股票")

    observations = torch.as_tensor(
        latest[ML_FEATURE_COLUMNS].to_numpy(dtype=np.float32),
        dtype=torch.float32,
        device=device,
    )
    with torch.no_grad():
        logits, _ = model(observations)
        probabilities = torch.softmax(
            torch.cat([logits, model.cash_logit.reshape(1)]), dim=0
        )[:-1]
    latest["strategy_score"] = logits.cpu().numpy()
    latest["selection_probability"] = probabilities.cpu().numpy()
    selected = latest.nlargest(top, "strategy_score").copy()

    market_latest = (
        features.loc[features["trade_date"].eq(latest_date)]
        .set_index("thscode")
    )
    selected["year_high"] = selected["thscode"].map(
        market_latest["year_high"]
    )
    selected["year_low"] = selected["thscode"].map(
        market_latest["year_low"]
    )
    selected.dropna(subset=["year_high", "year_low"], inplace=True)
    names = (
        resolve_stock_names(selected["thscode"], api_config)
        if lookup_names
        else {}
    )
    selected["stock_name"] = selected["thscode"].map(names)
    selected["stock_name"] = selected["stock_name"].where(
        selected["stock_name"].notna(),
        selected["thscode"].str.split(".").str[0],
    )
    selected["xueqiu_url"] = selected["thscode"].map(
        lambda symbol: f"https://xueqiu.com/S/{xueqiu_symbol(symbol)}"
    )
    selected.sort_values(
        ["strategy_score", "thscode"],
        ascending=[True, True],
        inplace=True,
    )
    return selected[
        [
            "thscode",
            "stock_name",
            "xueqiu_url",
            "close_price",
            "year_high",
            "year_low",
            "strategy_score",
            "selection_probability",
        ]
    ].reset_index(drop=True)


HISTORY_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS selection_runs (
    run_id          INTEGER PRIMARY KEY,
    trade_date      TEXT NOT NULL UNIQUE,
    created_at      TEXT NOT NULL,
    selected_count  INTEGER NOT NULL,
    event_count     INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS selection_events (
    event_id        INTEGER PRIMARY KEY,
    run_id          INTEGER NOT NULL,
    thscode         TEXT NOT NULL,
    stock_name      TEXT NOT NULL,
    xueqiu_url      TEXT NOT NULL,
    status          TEXT NOT NULL CHECK (status IN ('买入', '持有', '卖出')),
    current_price   REAL NOT NULL,
    buy_price       REAL NOT NULL,
    sell_price      REAL,
    year_high       REAL NOT NULL,
    year_low        REAL NOT NULL,
    strategy        TEXT NOT NULL,
    strategy_score  REAL NOT NULL DEFAULT 0,
    quantity        INTEGER NOT NULL DEFAULT 0,
    trade_amount    REAL NOT NULL DEFAULT 0,
    fee             REAL NOT NULL DEFAULT 0,
    cash_amount     REAL NOT NULL DEFAULT 0,
    exit_reason     TEXT NOT NULL DEFAULT '',
    holding_days    INTEGER NOT NULL DEFAULT 0,
    return_rate     REAL,
    peak_drawdown   REAL,
    FOREIGN KEY (run_id) REFERENCES selection_runs(run_id),
    UNIQUE (run_id, thscode)
);

CREATE INDEX IF NOT EXISTS idx_selection_events_symbol
ON selection_events (thscode, run_id);

CREATE TABLE IF NOT EXISTS open_positions (
    thscode         TEXT PRIMARY KEY,
    stock_name      TEXT NOT NULL,
    xueqiu_url      TEXT NOT NULL,
    buy_date        TEXT NOT NULL,
    buy_price       REAL NOT NULL,
    last_seen_date  TEXT NOT NULL,
    entry_strategy  TEXT NOT NULL DEFAULT '',
    last_strategy   TEXT NOT NULL,
    last_strategy_score REAL NOT NULL DEFAULT 0,
    quantity        INTEGER NOT NULL DEFAULT 0,
    buy_fee         REAL NOT NULL DEFAULT 0,
    invested_amount REAL NOT NULL DEFAULT 0,
    peak_price      REAL NOT NULL DEFAULT 0,
    score_decline_count INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS exit_cooldowns (
    thscode         TEXT PRIMARY KEY,
    remaining_days  INTEGER NOT NULL,
    exit_date       TEXT NOT NULL,
    exit_reason     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS strategy_weights (
    strategy          TEXT PRIMARY KEY,
    weight            REAL NOT NULL,
    trade_count       INTEGER NOT NULL DEFAULT 0,
    win_count         INTEGER NOT NULL DEFAULT 0,
    cumulative_return REAL NOT NULL DEFAULT 0,
    mean_return       REAL NOT NULL DEFAULT 0,
    last_return       REAL,
    updated_at        TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS strategy_weight_updates (
    update_id      INTEGER PRIMARY KEY,
    run_id         INTEGER NOT NULL,
    trade_date     TEXT NOT NULL,
    thscode        TEXT NOT NULL,
    strategy       TEXT NOT NULL,
    return_rate    REAL NOT NULL,
    old_weight     REAL NOT NULL,
    new_weight     REAL NOT NULL,
    FOREIGN KEY (run_id) REFERENCES selection_runs(run_id),
    UNIQUE (run_id, thscode, strategy)
);

CREATE INDEX IF NOT EXISTS idx_strategy_weight_updates_date
ON strategy_weight_updates (trade_date, strategy);

CREATE TABLE IF NOT EXISTS teams_recommendation_runs (
    recommendation_run_id INTEGER PRIMARY KEY,
    source                TEXT NOT NULL,
    trade_date            TEXT NOT NULL,
    created_at            TEXT NOT NULL,
    recommended_count     INTEGER NOT NULL,
    event_count           INTEGER NOT NULL,
    UNIQUE (source, trade_date)
);

CREATE TABLE IF NOT EXISTS teams_recommendation_events (
    recommendation_event_id INTEGER PRIMARY KEY,
    recommendation_run_id   INTEGER NOT NULL,
    thscode                 TEXT NOT NULL,
    stock_name              TEXT NOT NULL,
    xueqiu_url              TEXT NOT NULL,
    status                  TEXT NOT NULL
                            CHECK (status IN ('买进', '持有', '卖出')),
    close_price             REAL NOT NULL,
    year_high               REAL NOT NULL,
    year_low                REAL NOT NULL,
    strategy_score          REAL NOT NULL,
    sort_order              INTEGER NOT NULL,
    FOREIGN KEY (recommendation_run_id)
        REFERENCES teams_recommendation_runs(recommendation_run_id),
    UNIQUE (recommendation_run_id, thscode)
);

CREATE INDEX IF NOT EXISTS idx_teams_recommendation_events_symbol
ON teams_recommendation_events (thscode, recommendation_run_id);

CREATE TABLE IF NOT EXISTS dragon_tiger_runs (
    dragon_tiger_run_id INTEGER PRIMARY KEY,
    trade_date          TEXT NOT NULL,
    board_type          TEXT NOT NULL,
    source_timestamp    INTEGER,
    source_count        INTEGER NOT NULL,
    stock_count         INTEGER NOT NULL,
    created_at          TEXT NOT NULL,
    event_count         INTEGER NOT NULL,
    UNIQUE (board_type, trade_date)
);

CREATE TABLE IF NOT EXISTS dragon_tiger_events (
    dragon_tiger_event_id INTEGER PRIMARY KEY,
    dragon_tiger_run_id   INTEGER NOT NULL,
    thscode               TEXT NOT NULL,
    ticker                TEXT NOT NULL,
    stock_name            TEXT NOT NULL,
    xueqiu_url             TEXT NOT NULL,
    status                 TEXT NOT NULL
                           CHECK (status IN ('买入', '持有', '卖出')),
    concept_list           TEXT NOT NULL DEFAULT '[]',
    change                 REAL,
    net_value              REAL,
    net_rate               REAL,
    hot_rank               INTEGER,
    buy_value              REAL,
    sell_value             REAL,
    limit_reason           TEXT,
    range_days             INTEGER,
    org_net_value          REAL,
    org_net_rate           REAL,
    org_buy_num            INTEGER,
    org_sell_num           INTEGER,
    amount                 REAL,
    hot_money_net_value    REAL,
    hot_money_net_rate     REAL,
    hot_money_item_net_value REAL,
    hot_money_item_net_rate  REAL,
    sort_order             INTEGER NOT NULL,
    FOREIGN KEY (dragon_tiger_run_id)
        REFERENCES dragon_tiger_runs(dragon_tiger_run_id),
    UNIQUE (dragon_tiger_run_id, thscode)
);

CREATE INDEX IF NOT EXISTS idx_dragon_tiger_events_symbol
ON dragon_tiger_events (thscode, dragon_tiger_run_id);
"""

EVENT_COLUMNS = [
    "thscode",
    "stock_name",
    "xueqiu_url",
    "status",
    "current_price",
    "buy_price",
    "sell_price",
    "year_high",
    "year_low",
    "strategy",
    "strategy_score",
    "quantity",
    "trade_amount",
    "fee",
    "cash_amount",
    "exit_reason",
    "holding_days",
    "return_rate",
    "peak_drawdown",
]


def ensure_history_columns(
    connection: sqlite3.Connection,
    trading: TradingSettings,
) -> None:
    """Migrate history databases created by earlier versions of main.py."""
    event_columns = {
        row[1] for row in connection.execute("PRAGMA table_info(selection_events)")
    }
    if "strategy_score" not in event_columns:
        connection.execute(
            "ALTER TABLE selection_events "
            "ADD COLUMN strategy_score REAL NOT NULL DEFAULT 0"
        )
    for name, definition in (
        ("quantity", "INTEGER NOT NULL DEFAULT 0"),
        ("trade_amount", "REAL NOT NULL DEFAULT 0"),
        ("fee", "REAL NOT NULL DEFAULT 0"),
        ("cash_amount", "REAL NOT NULL DEFAULT 0"),
        ("exit_reason", "TEXT NOT NULL DEFAULT ''"),
        ("holding_days", "INTEGER NOT NULL DEFAULT 0"),
        ("return_rate", "REAL"),
        ("peak_drawdown", "REAL"),
    ):
        if name not in event_columns:
            connection.execute(
                f"ALTER TABLE selection_events ADD COLUMN {name} {definition}"
            )
    legacy_events = connection.execute(
        """
        SELECT event_id, status, current_price, buy_price, sell_price
        FROM selection_events
        WHERE quantity <= 0
        """
    ).fetchall()
    for event_id, status, current_price, buy_price, sell_price in legacy_events:
        quantity = calculate_buy_quantity(float(buy_price), trading)
        if status == "买入":
            trade_amount, fee, cash_amount = transaction_values(
                float(buy_price),
                quantity,
                trading.buy_fee_rate,
                "buy",
            )
        elif status == "卖出" and sell_price is not None:
            trade_amount, fee, cash_amount = transaction_values(
                float(sell_price),
                quantity,
                trading.sell_fee_rate,
                "sell",
            )
        else:
            trade_amount = float(current_price) * quantity
            fee = 0.0
            cash_amount = trade_amount
        connection.execute(
            """
            UPDATE selection_events
            SET quantity = ?, trade_amount = ?, fee = ?, cash_amount = ?
            WHERE event_id = ?
            """,
            (quantity, trade_amount, fee, cash_amount, event_id),
        )
    position_columns = {
        row[1] for row in connection.execute("PRAGMA table_info(open_positions)")
    }
    if "entry_strategy" not in position_columns:
        connection.execute(
            "ALTER TABLE open_positions "
            "ADD COLUMN entry_strategy TEXT NOT NULL DEFAULT ''"
        )
    if "last_strategy_score" not in position_columns:
        connection.execute(
            "ALTER TABLE open_positions "
            "ADD COLUMN last_strategy_score REAL NOT NULL DEFAULT 0"
        )
    for name, definition in (
        ("quantity", "INTEGER NOT NULL DEFAULT 0"),
        ("buy_fee", "REAL NOT NULL DEFAULT 0"),
        ("invested_amount", "REAL NOT NULL DEFAULT 0"),
        ("peak_price", "REAL NOT NULL DEFAULT 0"),
        ("score_decline_count", "INTEGER NOT NULL DEFAULT 0"),
    ):
        if name not in position_columns:
            connection.execute(
                f"ALTER TABLE open_positions ADD COLUMN {name} {definition}"
            )
    legacy_positions = connection.execute(
        "SELECT thscode, buy_price FROM open_positions WHERE quantity <= 0"
    ).fetchall()
    for symbol, buy_price in legacy_positions:
        quantity = calculate_buy_quantity(float(buy_price), trading)
        _, fee, invested = transaction_values(
            float(buy_price),
            quantity,
            trading.buy_fee_rate,
            "buy",
        )
        connection.execute(
            """
            UPDATE open_positions
            SET quantity = ?, buy_fee = ?, invested_amount = ?
            WHERE thscode = ?
            """,
            (quantity, fee, invested, symbol),
        )
    connection.execute(
        """
        UPDATE open_positions
        SET peak_price = buy_price
        WHERE peak_price <= 0
        """
    )
    connection.execute(
        """
        UPDATE open_positions
        SET entry_strategy = last_strategy
        WHERE entry_strategy = ''
        """
    )
    connection.commit()


def initialize_strategy_weights(
    connection: sqlite3.Connection,
    settings: StrategyWeightSettings,
) -> dict[str, float]:
    now = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")
    connection.executemany(
        """
        INSERT OR IGNORE INTO strategy_weights (
            strategy, weight, updated_at
        ) VALUES (?, ?, ?)
        """,
        [
            (strategy, settings.initial_weight, now)
            for strategy in STRATEGY_NAMES
        ],
    )
    connection.commit()
    rows = connection.execute(
        "SELECT strategy, weight FROM strategy_weights"
    ).fetchall()
    stored = {str(row[0]): float(row[1]) for row in rows}
    weights: dict[str, float] = {}
    for strategy in STRATEGY_NAMES:
        weight = stored[strategy]
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError(f"策略 {strategy} 的在线权重必须是正有限数")
        weights[strategy] = weight
    return weights


def load_online_strategy_weights(
    history_database: Path,
    settings: StrategyWeightSettings,
) -> dict[str, float]:
    history_database.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(history_database)
    try:
        connection.executescript(HISTORY_SCHEMA_SQL)
        return initialize_strategy_weights(connection, settings)
    finally:
        connection.close()


def parse_strategy_names(value: str) -> list[str]:
    return [
        name
        for name in dict.fromkeys(part.strip() for part in value.split("、"))
        if name in STRATEGY_NAMES
    ]


def update_strategy_weights(
    connection: sqlite3.Connection,
    run_id: int,
    trade_date: str,
    events: list[dict[str, Any]],
    settings: StrategyWeightSettings,
) -> None:
    """使用已平仓、扣除双边手续费后的收益更新入场策略权重。"""
    now = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")
    for event in events:
        if event["status"] != "卖出":
            continue
        return_rate = float(event["return_rate"])
        if not math.isfinite(return_rate):
            continue
        signal = math.tanh(return_rate / settings.return_scale)
        for strategy in parse_strategy_names(event["_entry_strategy"]):
            row = connection.execute(
                """
                SELECT weight, trade_count, win_count,
                       cumulative_return
                FROM strategy_weights
                WHERE strategy = ?
                """,
                (strategy,),
            ).fetchone()
            if row is None:
                continue
            old_weight = float(row[0])
            new_weight = old_weight * math.exp(
                settings.learning_rate * signal
            )
            new_weight = min(
                max(new_weight, settings.min_weight),
                settings.max_weight,
            )
            trade_count = int(row[1]) + 1
            win_count = int(row[2]) + int(return_rate > 0)
            cumulative_return = float(row[3]) + return_rate
            mean_return = cumulative_return / trade_count
            connection.execute(
                """
                UPDATE strategy_weights
                SET weight = ?, trade_count = ?, win_count = ?,
                    cumulative_return = ?, mean_return = ?,
                    last_return = ?, updated_at = ?
                WHERE strategy = ?
                """,
                (
                    new_weight,
                    trade_count,
                    win_count,
                    cumulative_return,
                    mean_return,
                    return_rate,
                    now,
                    strategy,
                ),
            )
            connection.execute(
                """
                INSERT OR IGNORE INTO strategy_weight_updates (
                    run_id, trade_date, thscode, strategy,
                    return_rate, old_weight, new_weight
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    trade_date,
                    event["thscode"],
                    strategy,
                    return_rate,
                    old_weight,
                    new_weight,
                ),
            )


def reset_history_database(
    history_database: Path,
    settings: StrategyWeightSettings,
) -> Path | None:
    """备份后清空旧持仓、事件、冷却期和在线权重。"""
    history_database.parent.mkdir(parents=True, exist_ok=True)
    backup_path: Path | None = None
    connection = sqlite3.connect(history_database)
    try:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.executescript(HISTORY_SCHEMA_SQL)
        if history_database.is_file() and history_database.stat().st_size > 0:
            stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime(
                "%Y%m%d_%H%M%S"
            )
            backup_path = history_database.with_name(
                f"{history_database.stem}_before_reset_{stamp}"
                f"{history_database.suffix}"
            )
            backup = sqlite3.connect(backup_path)
            try:
                connection.backup(backup)
            finally:
                backup.close()
        connection.execute("BEGIN IMMEDIATE")
        connection.execute("DELETE FROM strategy_weight_updates")
        connection.execute("DELETE FROM strategy_weights")
        connection.execute("DELETE FROM exit_cooldowns")
        connection.execute("DELETE FROM open_positions")
        connection.execute("DELETE FROM selection_events")
        connection.execute("DELETE FROM selection_runs")
        connection.execute("DELETE FROM teams_recommendation_events")
        connection.execute("DELETE FROM teams_recommendation_runs")
        connection.execute("DELETE FROM dragon_tiger_events")
        connection.execute("DELETE FROM dragon_tiger_runs")
        connection.commit()
        initialize_strategy_weights(connection, settings)
    except BaseException:
        connection.rollback()
        raise
    finally:
        connection.close()
    return backup_path


def read_recorded_run(
    connection: sqlite3.Connection, trade_date: str
) -> pd.DataFrame | None:
    run = connection.execute(
        "SELECT run_id FROM selection_runs WHERE trade_date = ?", (trade_date,)
    ).fetchone()
    if run is None:
        return None
    return pd.read_sql_query(
        f"""
        SELECT {", ".join(EVENT_COLUMNS)}
        FROM selection_events
        WHERE run_id = ?
        ORDER BY strategy_score DESC, thscode
        """,
        connection,
        params=(run[0],),
    )


TEAMS_RECOMMENDATION_COLUMNS = [
    "thscode",
    "stock_name",
    "xueqiu_url",
    "status",
    "close_price",
    "year_high",
    "year_low",
    "strategy_score",
]


def record_teams_recommendations(
    candidates: pd.DataFrame,
    features: pd.DataFrame,
    trade_date: str,
    history_database: Path,
    source: str,
) -> tuple[pd.DataFrame, bool]:
    """按上一交易日推荐集合生成买进/持有/卖出状态并写入 SQLite。"""
    history_database.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(history_database)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA journal_mode=WAL")
        connection.executescript(HISTORY_SCHEMA_SQL)
        existing_run = connection.execute(
            """
            SELECT recommendation_run_id
            FROM teams_recommendation_runs
            WHERE source = ? AND trade_date = ?
            """,
            (source, trade_date),
        ).fetchone()
        if existing_run is not None:
            recorded = pd.read_sql_query(
                f"""
                SELECT {", ".join(TEAMS_RECOMMENDATION_COLUMNS)}
                FROM teams_recommendation_events
                WHERE recommendation_run_id = ?
                ORDER BY sort_order
                """,
                connection,
                params=(int(existing_run[0]),),
            )
            return recorded, False

        previous_run = connection.execute(
            """
            SELECT recommendation_run_id
            FROM teams_recommendation_runs
            WHERE source = ? AND trade_date < ?
            ORDER BY trade_date DESC
            LIMIT 1
            """,
            (source, trade_date),
        ).fetchone()
        previous: dict[str, dict[str, Any]] = {}
        if previous_run is not None:
            previous = {
                str(row["thscode"]): dict(row)
                for row in connection.execute(
                    """
                    SELECT thscode, stock_name, xueqiu_url, close_price,
                           year_high, year_low, strategy_score
                    FROM teams_recommendation_events
                    WHERE recommendation_run_id = ? AND status != '卖出'
                    """,
                    (int(previous_run[0]),),
                )
            }

        latest_snapshots = (
            features.loc[features["trade_date"].astype(str).eq(trade_date)]
            .drop_duplicates("thscode", keep="last")
            .set_index("thscode")
        )
        current_symbols = set(candidates["thscode"].astype(str))
        events: list[dict[str, Any]] = []
        for sort_order, row in enumerate(
            candidates.itertuples(index=False)
        ):
            events.append(
                {
                    "thscode": row.thscode,
                    "stock_name": row.stock_name,
                    "xueqiu_url": row.xueqiu_url,
                    "status": (
                        "持有" if row.thscode in previous else "买进"
                    ),
                    "close_price": float(row.close_price),
                    "year_high": float(row.year_high),
                    "year_low": float(row.year_low),
                    "strategy_score": float(row.strategy_score),
                    "sort_order": sort_order,
                }
            )

        next_order = len(events)
        for symbol in sorted(previous.keys() - current_symbols):
            old = previous[symbol]
            if symbol in latest_snapshots.index:
                snapshot = latest_snapshots.loc[symbol]
                close_price = float(snapshot["close_price"])
                year_high = float(snapshot["year_high"])
                year_low = float(snapshot["year_low"])
            else:
                close_price = float(old["close_price"])
                year_high = float(old["year_high"])
                year_low = float(old["year_low"])
            events.append(
                {
                    "thscode": symbol,
                    "stock_name": old["stock_name"],
                    "xueqiu_url": old["xueqiu_url"],
                    "status": "卖出",
                    "close_price": close_price,
                    "year_high": year_high,
                    "year_low": year_low,
                    "strategy_score": float(old["strategy_score"]),
                    "sort_order": next_order,
                }
            )
            next_order += 1

        now = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(
            timespec="seconds"
        )
        connection.execute("BEGIN IMMEDIATE")
        cursor = connection.execute(
            """
            INSERT INTO teams_recommendation_runs (
                source, trade_date, created_at, recommended_count, event_count
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (source, trade_date, now, len(candidates), len(events)),
        )
        run_id = int(cursor.lastrowid)
        connection.executemany(
            """
            INSERT INTO teams_recommendation_events (
                recommendation_run_id, thscode, stock_name, xueqiu_url,
                status, close_price, year_high, year_low, strategy_score,
                sort_order
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    run_id,
                    event["thscode"],
                    event["stock_name"],
                    event["xueqiu_url"],
                    event["status"],
                    event["close_price"],
                    event["year_high"],
                    event["year_low"],
                    event["strategy_score"],
                    event["sort_order"],
                )
                for event in events
            ],
        )
        connection.commit()
        return pd.DataFrame(
            events, columns=TEAMS_RECOMMENDATION_COLUMNS
        ), True
    except BaseException:
        connection.rollback()
        raise
    finally:
        connection.close()


DRAGON_TIGER_EVENT_COLUMNS = [
    "thscode",
    "ticker",
    "stock_name",
    "xueqiu_url",
    "status",
    "concept_list",
    "change",
    "net_value",
    "net_rate",
    "hot_rank",
    "buy_value",
    "sell_value",
    "limit_reason",
    "range_days",
    "org_net_value",
    "org_net_rate",
    "org_buy_num",
    "org_sell_num",
    "amount",
    "hot_money_net_value",
    "hot_money_net_rate",
    "hot_money_item_net_value",
    "hot_money_item_net_rate",
]


def record_dragon_tiger_results(
    candidates: pd.DataFrame,
    data: dict[str, Any],
    history_database: Path,
) -> tuple[pd.DataFrame, bool]:
    """Store raw dragon-tiger fields and derive buy/hold/sell membership."""
    trade_date = str(data["trade_date"])
    board_type = str(data.get("board_type") or DRAGON_TIGER_BOARD_TYPE)
    history_database.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(history_database)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA journal_mode=WAL")
        connection.executescript(HISTORY_SCHEMA_SQL)
        existing_run = connection.execute(
            """
            SELECT dragon_tiger_run_id
            FROM dragon_tiger_runs
            WHERE board_type = ? AND trade_date = ?
            """,
            (board_type, trade_date),
        ).fetchone()
        if existing_run is not None:
            recorded = pd.read_sql_query(
                f"""
                SELECT {", ".join(DRAGON_TIGER_EVENT_COLUMNS)}
                FROM dragon_tiger_events
                WHERE dragon_tiger_run_id = ?
                ORDER BY sort_order
                """,
                connection,
                params=(int(existing_run[0]),),
            )
            return recorded, False

        previous_run = connection.execute(
            """
            SELECT dragon_tiger_run_id
            FROM dragon_tiger_runs
            WHERE board_type = ? AND trade_date < ?
            ORDER BY trade_date DESC
            LIMIT 1
            """,
            (board_type, trade_date),
        ).fetchone()
        previous: dict[str, dict[str, Any]] = {}
        if previous_run is not None:
            previous = {
                str(row["thscode"]): dict(row)
                for row in connection.execute(
                    f"""
                    SELECT {", ".join(DRAGON_TIGER_EVENT_COLUMNS)}
                    FROM dragon_tiger_events
                    WHERE dragon_tiger_run_id = ? AND status != '卖出'
                    """,
                    (int(previous_run[0]),),
                )
            }

        current_symbols = set(candidates["thscode"].astype(str))
        events: list[dict[str, Any]] = []
        for row in candidates.to_dict(orient="records"):
            event = {
                column: row.get(column)
                for column in DRAGON_TIGER_EVENT_COLUMNS
                if column != "status"
            }
            event["status"] = (
                "持有" if str(row["thscode"]) in previous else "买入"
            )
            events.append(event)
        for symbol in sorted(previous.keys() - current_symbols):
            event = previous[symbol]
            event["status"] = "卖出"
            events.append(event)

        now = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(
            timespec="seconds"
        )
        connection.execute("BEGIN IMMEDIATE")
        cursor = connection.execute(
            """
            INSERT INTO dragon_tiger_runs (
                trade_date, board_type, source_timestamp, source_count,
                stock_count, created_at, event_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                trade_date,
                board_type,
                data.get("timestamp"),
                int(data.get("count") or 0),
                int(data.get("stock_count") or len(candidates)),
                now,
                len(events),
            ),
        )
        run_id = int(cursor.lastrowid)
        insert_columns = [
            "dragon_tiger_run_id",
            *DRAGON_TIGER_EVENT_COLUMNS,
            "sort_order",
        ]
        placeholders = ", ".join("?" for _ in insert_columns)
        connection.executemany(
            f"""
            INSERT INTO dragon_tiger_events (
                {", ".join(insert_columns)}
            ) VALUES ({placeholders})
            """,
            [
                (
                    run_id,
                    *(event.get(column) for column in DRAGON_TIGER_EVENT_COLUMNS),
                    sort_order,
                )
                for sort_order, event in enumerate(events)
            ],
        )
        connection.commit()
        return pd.DataFrame(
            events, columns=DRAGON_TIGER_EVENT_COLUMNS
        ), True
    except BaseException:
        connection.rollback()
        raise
    finally:
        connection.close()


def reconcile_and_record(
    selected: pd.DataFrame,
    features: pd.DataFrame,
    trade_date: str,
    history_database: Path,
    trading: TradingSettings,
    exit_rules: ExitSettings,
    weight_settings: StrategyWeightSettings,
) -> tuple[pd.DataFrame, bool]:
    """应用退出规则，对目标名单和当前持仓进行对账并原子落库。"""
    history_database.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(history_database)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA journal_mode=WAL")
        connection.executescript(HISTORY_SCHEMA_SQL)
        ensure_history_columns(connection, trading)
        initialize_strategy_weights(connection, weight_settings)
        existing_run = connection.execute(
            "SELECT run_id FROM selection_runs WHERE trade_date = ?", (trade_date,)
        ).fetchone()
        if existing_run is not None:
            run_id = int(existing_run[0])
            connection.executemany(
                """
                UPDATE selection_events
                SET strategy = ?, strategy_score = ?
                WHERE run_id = ? AND thscode = ?
                """,
                [
                    (
                        row.strategy,
                        float(row.strategy_score),
                        run_id,
                        row.thscode,
                    )
                    for row in selected.itertuples(index=False)
                ],
            )
            connection.executemany(
                """
                UPDATE open_positions
                SET last_strategy = ?, last_strategy_score = ?
                WHERE thscode = ?
                """,
                [
                    (
                        row.strategy,
                        float(row.strategy_score),
                        row.thscode,
                    )
                    for row in selected.itertuples(index=False)
                ],
            )
            connection.commit()
        recorded = read_recorded_run(connection, trade_date)
        if recorded is not None:
            return recorded, False

        open_positions = {
            row["thscode"]: dict(row)
            for row in connection.execute("SELECT * FROM open_positions")
        }
        cooldowns = {
            row["thscode"]: int(row["remaining_days"])
            for row in connection.execute(
                "SELECT thscode, remaining_days FROM exit_cooldowns"
            )
        }
        selected_by_symbol = {
            row.thscode: row for row in selected.itertuples(index=False)
        }
        latest_snapshots = (
            features.sort_values(["thscode", "trade_date"])
            .groupby("thscode", sort=False)
            .tail(1)
            .set_index("thscode")
        )
        trading_dates = sorted(features["trade_date"].astype(str).unique())
        date_positions = {date: index for index, date in enumerate(trading_dates)}
        current_date_index = date_positions.get(trade_date, len(trading_dates) - 1)
        position_histories = {
            symbol: group
            for symbol, group in features.loc[
                features["thscode"].isin(open_positions)
            ].groupby("thscode", sort=False)
        }
        risk_off, market_summary = market_risk_status(features, exit_rules)
        if risk_off:
            print(f"大盘风险控制已触发：{market_summary}")
        events: list[dict[str, Any]] = []

        for symbol, position in open_positions.items():
            if symbol not in latest_snapshots.index:
                continue
            snapshot = latest_snapshots.loc[symbol]
            has_current_bar = str(snapshot["trade_date"]) == trade_date
            row = selected_by_symbol.get(symbol)
            quantity = int(position["quantity"])
            current_price = float(snapshot["close_price"])
            buy_price = float(position["buy_price"])
            buy_date = str(position["buy_date"])
            buy_date_index = date_positions.get(buy_date, current_date_index)
            holding_days = max(current_date_index - buy_date_index + 1, 1)

            symbol_history = position_histories.get(symbol)
            if symbol_history is None:
                position_history = features.iloc[0:0]
            else:
                symbol_dates = symbol_history["trade_date"].astype(str)
                position_history = symbol_history.loc[
                    symbol_dates.ge(buy_date) & symbol_dates.le(trade_date)
                ]
            historical_peak = (
                float(position_history["high_price"].max())
                if not position_history.empty
                else current_price
            )
            peak_price = max(
                float(position.get("peak_price") or buy_price),
                historical_peak,
            )
            peak_drawdown = max(1.0 - current_price / peak_price, 0.0)

            _, estimated_sell_fee, estimated_proceeds = transaction_values(
                current_price,
                quantity,
                trading.sell_fee_rate,
                "sell",
            )
            invested_amount = float(position["invested_amount"])
            return_rate = (
                estimated_proceeds / invested_amount - 1.0
                if invested_amount > 0
                else current_price / buy_price - 1.0
            )

            previous_declines = int(position.get("score_decline_count") or 0)
            if row is None:
                decline_count = previous_declines + 1
            else:
                current_score = float(row.strategy_score)
                decline_count = (
                    previous_declines + 1
                    if current_score < float(position["last_strategy_score"])
                    else 0
                )

            exit_reason = ""
            if has_current_bar:
                if risk_off:
                    exit_reason = f"大盘风险控制：{market_summary}"
                elif return_rate <= -exit_rules.stop_loss_rate:
                    exit_reason = (
                        f"止损：扣费后收益{return_rate:.2%} ≤ "
                        f"-{exit_rules.stop_loss_rate:.2%}"
                    )
                elif peak_drawdown >= exit_rules.trailing_drawdown_rate:
                    exit_reason = (
                        f"回撤止损：距持仓期高点回撤{peak_drawdown:.2%} ≥ "
                        f"{exit_rules.trailing_drawdown_rate:.2%}"
                    )
                elif return_rate >= exit_rules.take_profit_rate:
                    exit_reason = (
                        f"止盈：扣费后收益{return_rate:.2%} ≥ "
                        f"{exit_rules.take_profit_rate:.2%}"
                    )
                elif holding_days >= exit_rules.max_holding_days:
                    exit_reason = (
                        f"最大持仓期限：已持有{holding_days}个交易日"
                    )
                elif (
                    decline_count
                    >= exit_rules.score_decline_confirmation_days
                ):
                    exit_reason = (
                        f"策略分数下降确认：连续{decline_count}个交易日"
                        + (
                            "下降"
                            if row is not None
                            else "未恢复目标名单"
                        )
                    )

            if exit_reason:
                status = "卖出"
                sell_price: float | None = current_price
                trade_amount, fee, cash_amount = transaction_values(
                    current_price,
                    quantity,
                    trading.sell_fee_rate,
                    "sell",
                )
            else:
                status = "持有"
                sell_price = None
                trade_amount = current_price * quantity
                fee = 0.0
                cash_amount = trade_amount

            events.append(
                {
                    "thscode": symbol,
                    "stock_name": (
                        position["stock_name"]
                        if row is None
                        or row.stock_name == symbol.split(".")[0]
                        else row.stock_name
                    ),
                    "xueqiu_url": (
                        position["xueqiu_url"] if row is None else row.xueqiu_url
                    ),
                    "status": status,
                    "current_price": current_price,
                    "buy_price": buy_price,
                    "sell_price": sell_price,
                    "year_high": float(snapshot["year_high"]),
                    "year_low": float(snapshot["year_low"]),
                    "strategy": (
                        position["last_strategy"] if row is None else row.strategy
                    ),
                    "strategy_score": (
                        float(position["last_strategy_score"])
                        if row is None
                        else float(row.strategy_score)
                    ),
                    "quantity": quantity,
                    "trade_amount": trade_amount,
                    "fee": fee,
                    "cash_amount": cash_amount,
                    "exit_reason": exit_reason,
                    "holding_days": holding_days,
                    "return_rate": return_rate,
                    "peak_drawdown": peak_drawdown,
                    "_peak_price": peak_price,
                    "_decline_count": decline_count,
                    "_entry_strategy": (
                        position.get("entry_strategy")
                        or position["last_strategy"]
                    ),
                }
            )

        for symbol, row in selected_by_symbol.items():
            if symbol in open_positions:
                continue
            if risk_off:
                continue
            remaining_cooldown = cooldowns.get(symbol, 0)
            if remaining_cooldown > 0:
                print(
                    f"跳过 {symbol}: 卖出冷却期还剩 "
                    f"{remaining_cooldown} 个交易日"
                )
                continue
            quantity = calculate_buy_quantity(float(row.close_price), trading)
            if quantity < trading.lot_size:
                print(
                    f"跳过 {symbol}: 每股 {float(row.close_price):.2f} 元，"
                    f"{trading.budget_per_stock:.2f} 元不足买入 "
                    f"{trading.lot_size} 股"
                )
                continue
            trade_amount, fee, cash_amount = transaction_values(
                float(row.close_price),
                quantity,
                trading.buy_fee_rate,
                "buy",
            )
            events.append(
                {
                    "thscode": symbol,
                    "stock_name": row.stock_name,
                    "xueqiu_url": row.xueqiu_url,
                    "status": "买入",
                    "current_price": float(row.close_price),
                    "buy_price": float(row.close_price),
                    "sell_price": None,
                    "year_high": float(row.year_high),
                    "year_low": float(row.year_low),
                    "strategy": row.strategy,
                    "strategy_score": float(row.strategy_score),
                    "quantity": quantity,
                    "trade_amount": trade_amount,
                    "fee": fee,
                    "cash_amount": cash_amount,
                    "exit_reason": "",
                    "holding_days": 1,
                    "return_rate": None,
                    "peak_drawdown": 0.0,
                    "_peak_price": float(row.close_price),
                    "_decline_count": 0,
                    "_entry_strategy": row.strategy,
                }
            )

        now = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")
        connection.execute("BEGIN IMMEDIATE")
        cursor = connection.execute(
            """
            INSERT INTO selection_runs
                (trade_date, created_at, selected_count, event_count)
            VALUES (?, ?, ?, ?)
            """,
            (
                trade_date,
                now,
                sum(event["status"] != "卖出" for event in events),
                len(events),
            ),
        )
        run_id = int(cursor.lastrowid)
        connection.executemany(
            """
            INSERT INTO selection_events (
                run_id, thscode, stock_name, xueqiu_url, status,
                current_price, buy_price, sell_price, year_high, year_low, strategy
                , strategy_score, quantity, trade_amount, fee, cash_amount,
                exit_reason, holding_days, return_rate, peak_drawdown
            ) VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            [
                (
                    run_id,
                    event["thscode"],
                    event["stock_name"],
                    event["xueqiu_url"],
                    event["status"],
                    event["current_price"],
                    event["buy_price"],
                    event["sell_price"],
                    event["year_high"],
                    event["year_low"],
                    event["strategy"],
                    event["strategy_score"],
                    event["quantity"],
                    event["trade_amount"],
                    event["fee"],
                    event["cash_amount"],
                    event["exit_reason"],
                    event["holding_days"],
                    event["return_rate"],
                    event["peak_drawdown"],
                )
                for event in events
            ],
        )
        update_strategy_weights(
            connection,
            run_id,
            trade_date,
            events,
            weight_settings,
        )

        connection.execute(
            """
            UPDATE exit_cooldowns
            SET remaining_days = remaining_days - 1
            WHERE remaining_days > 0
            """
        )
        connection.execute(
            "DELETE FROM exit_cooldowns WHERE remaining_days <= 0"
        )
        for event in events:
            if event["status"] == "卖出":
                connection.execute(
                    "DELETE FROM open_positions WHERE thscode = ?",
                    (event["thscode"],),
                )
                if exit_rules.reentry_cooldown_days > 0:
                    connection.execute(
                        """
                        INSERT INTO exit_cooldowns (
                            thscode, remaining_days, exit_date, exit_reason
                        ) VALUES (?, ?, ?, ?)
                        ON CONFLICT(thscode) DO UPDATE SET
                            remaining_days=excluded.remaining_days,
                            exit_date=excluded.exit_date,
                            exit_reason=excluded.exit_reason
                        """,
                        (
                            event["thscode"],
                            exit_rules.reentry_cooldown_days,
                            trade_date,
                            event["exit_reason"],
                        ),
                    )
                continue
            existing = open_positions.get(event["thscode"])
            buy_date = existing["buy_date"] if existing else trade_date
            connection.execute(
                """
                INSERT INTO open_positions (
                    thscode, stock_name, xueqiu_url, buy_date, buy_price,
                    last_seen_date, entry_strategy, last_strategy,
                    last_strategy_score,
                    quantity, buy_fee, invested_amount, peak_price,
                    score_decline_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(thscode) DO UPDATE SET
                    stock_name=excluded.stock_name,
                    xueqiu_url=excluded.xueqiu_url,
                    last_seen_date=excluded.last_seen_date,
                    last_strategy=excluded.last_strategy,
                    last_strategy_score=excluded.last_strategy_score,
                    quantity=excluded.quantity,
                    buy_fee=excluded.buy_fee,
                    invested_amount=excluded.invested_amount,
                    peak_price=excluded.peak_price,
                    score_decline_count=excluded.score_decline_count
                """,
                (
                    event["thscode"],
                    event["stock_name"],
                    event["xueqiu_url"],
                    buy_date,
                    event["buy_price"],
                    trade_date,
                    (
                        event["_entry_strategy"]
                        if existing is None
                        else existing["entry_strategy"]
                    ),
                    event["strategy"],
                    event["strategy_score"],
                    event["quantity"],
                    (
                        event["fee"]
                        if existing is None
                        else float(existing["buy_fee"])
                    ),
                    (
                        event["cash_amount"]
                        if existing is None
                        else float(existing["invested_amount"])
                    ),
                    event["_peak_price"],
                    event["_decline_count"],
                ),
            )
        connection.commit()
        frame = pd.DataFrame(events, columns=EVENT_COLUMNS)
        frame = (
            frame.sort_values(
                ["strategy_score", "thscode"],
                ascending=[False, True],
            )
            .reset_index(drop=True)
        )
        return frame, True
    except BaseException:
        connection.rollback()
        raise
    finally:
        connection.close()


def text_block(text: str, **kwargs: Any) -> dict[str, Any]:
    return {"type": "TextBlock", "text": text, "wrap": True, **kwargs}


def card_row(values: list[str], header: bool = False) -> dict[str, Any]:
    widths = [
        "auto",
        "stretch",
        "auto",
        "auto",
        "auto",
        "auto",
        "auto",
        "auto",
        "auto",
        "auto",
        "auto",
        "stretch",
        "stretch",
    ]
    columns = []
    for value, width in zip(values, widths):
        columns.append(
            {
                "type": "Column",
                "width": width,
                "items": [
                    text_block(
                        value,
                        weight="Bolder" if header else "Default",
                        size="Small",
                    )
                ],
            }
        )
    return {
        "type": "ColumnSet",
        "separator": not header,
        "spacing": "Small",
        "columns": columns,
    }


def build_teams_payload(
    results: pd.DataFrame,
    trade_date: str,
    page: int = 1,
    pages: int = 1,
) -> dict[str, Any]:
    body: list[dict[str, Any]] = [
        text_block("A 股在线策略权重选股", weight="Bolder", size="Large"),
        text_block(
            f"交易日：{trade_date}　状态记录：{len(results)}"
            + (f"　第 {page}/{pages} 页" if pages > 1 else ""),
            isSubtle=True,
            spacing="None",
        ),
        card_row(
            [
                "股票代码",
                "股票名",
                "状态",
                "今日价格",
                "买入价格",
                "卖出价格",
                "股数",
                "资金额",
                "收益率",
                "卖出原因",
                "一年最高",
                "一年最低",
                "策略总分",
            ],
            header=True,
        ),
    ]
    for row in results.itertuples(index=False):
        stock_link = f"[{row.stock_name}]({row.xueqiu_url})"
        sell_price = "-" if pd.isna(row.sell_price) else f"{row.sell_price:.2f}"
        return_rate = (
            "-"
            if pd.isna(row.return_rate)
            else f"{float(row.return_rate):.2%}"
        )
        body.append(
            card_row(
                [
                    row.thscode,
                    stock_link,
                    row.status,
                    f"{row.current_price:.2f}",
                    f"{row.buy_price:.2f}",
                    sell_price,
                    f"{int(row.quantity)}",
                    f"{row.cash_amount:.2f}",
                    return_rate,
                    row.exit_reason or "-",
                    f"{row.year_high:.2f}",
                    f"{row.year_low:.2f}",
                    f"{row.strategy_score:.3f}",
                ]
            )
        )
    return {
        "type": "message",
        "attachments": [
            {
                "contentType": "application/vnd.microsoft.card.adaptive",
                "contentUrl": None,
                "content": {
                    "$schema": "https://adaptivecards.io/schemas/adaptive-card.json",
                    "type": "AdaptiveCard",
                    "version": "1.4",
                    "body": body,
                },
            }
        ],
    }


def build_ranked_teams_payload(
    candidates: pd.DataFrame,
    trade_date: str,
    page: int = 1,
    pages: int = 1,
    title: str = "A 股在线权重 Top 30",
    score_label: str = "权重总分",
) -> dict[str, Any]:
    body: list[dict[str, Any]] = [
        text_block(title, weight="Bolder", size="Large"),
        text_block(
            f"交易日：{trade_date}　候选数：{len(candidates)}"
            + (f"　第 {page}/{pages} 页" if pages > 1 else ""),
            isSubtle=True,
            spacing="None",
        ),
        {
            "type": "ColumnSet",
            "columns": [
                {
                    "type": "Column",
                    "width": "auto",
                    "items": [text_block("股票代码", weight="Bolder", size="Small")],
                },
                {
                    "type": "Column",
                    "width": "stretch",
                    "items": [text_block("股票名", weight="Bolder", size="Small")],
                },
                {
                    "type": "Column",
                    "width": "auto",
                    "items": [text_block("状态", weight="Bolder", size="Small")],
                },
                {
                    "type": "Column",
                    "width": "auto",
                    "items": [text_block("今日价格", weight="Bolder", size="Small")],
                },
                {
                    "type": "Column",
                    "width": "auto",
                    "items": [text_block("一年最高", weight="Bolder", size="Small")],
                },
                {
                    "type": "Column",
                    "width": "auto",
                    "items": [text_block("一年最低", weight="Bolder", size="Small")],
                },
                {
                    "type": "Column",
                    "width": "auto",
                    "items": [text_block(score_label, weight="Bolder", size="Small")],
                },
            ],
        },
    ]
    for row in candidates.itertuples(index=False):
        stock_name = f"[{row.stock_name}]({row.xueqiu_url})"
        values = [
            row.thscode,
            stock_name,
            row.status,
            f"{row.close_price:.2f}",
            f"{row.year_high:.2f}",
            f"{row.year_low:.2f}",
            f"{row.strategy_score:.3f}",
        ]
        widths = ["auto", "stretch", "auto", "auto", "auto", "auto", "auto"]
        body.append(
            {
                "type": "ColumnSet",
                "separator": True,
                "spacing": "Small",
                "columns": [
                    {
                        "type": "Column",
                        "width": width,
                        "items": [text_block(value, size="Small")],
                    }
                    for value, width in zip(values, widths)
                ],
            }
        )
    return {
        "type": "message",
        "attachments": [
            {
                "contentType": "application/vnd.microsoft.card.adaptive",
                "contentUrl": None,
                "content": {
                    "$schema": "https://adaptivecards.io/schemas/adaptive-card.json",
                    "type": "AdaptiveCard",
                    "version": "1.4",
                    "body": body,
                },
            }
        ],
    }


def format_dragon_tiger_amount(value: Any) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value) / 10_000:.2f}"


def build_dragon_tiger_teams_payload(
    results: pd.DataFrame,
    trade_date: str,
    page: int = 1,
    pages: int = 1,
) -> dict[str, Any]:
    headers = [
        "股票代码",
        "股票名",
        "状态",
        "涨跌幅",
        "净买入(万)",
        "买入额(万)",
        "卖出额(万)",
        "机构净买入(万)",
        "区间",
    ]
    widths = [
        "auto",
        "stretch",
        "auto",
        "auto",
        "auto",
        "auto",
        "auto",
        "auto",
        "auto",
    ]
    body: list[dict[str, Any]] = [
        text_block("A 股龙虎榜", weight="Bolder", size="Large"),
        text_block(
            f"交易日：{trade_date}　状态记录：{len(results)}"
            + (f"　第 {page}/{pages} 页" if pages > 1 else ""),
            isSubtle=True,
            spacing="None",
        ),
        {
            "type": "ColumnSet",
            "columns": [
                {
                    "type": "Column",
                    "width": width,
                    "items": [
                        text_block(header, weight="Bolder", size="Small")
                    ],
                }
                for header, width in zip(headers, widths)
            ],
        },
    ]
    for row in results.itertuples(index=False):
        change = "-" if pd.isna(row.change) else f"{float(row.change):.2%}"
        range_days = (
            "-" if pd.isna(row.range_days) else f"{int(row.range_days)}日"
        )
        values = [
            row.thscode,
            f"[{row.stock_name}]({row.xueqiu_url})",
            row.status,
            change,
            format_dragon_tiger_amount(row.net_value),
            format_dragon_tiger_amount(row.buy_value),
            format_dragon_tiger_amount(row.sell_value),
            format_dragon_tiger_amount(row.org_net_value),
            range_days,
        ]
        body.append(
            {
                "type": "ColumnSet",
                "separator": True,
                "spacing": "Small",
                "columns": [
                    {
                        "type": "Column",
                        "width": width,
                        "items": [text_block(str(value), size="Small")],
                    }
                    for value, width in zip(values, widths)
                ],
            }
        )
    return {
        "type": "message",
        "attachments": [
            {
                "contentType": "application/vnd.microsoft.card.adaptive",
                "contentUrl": None,
                "content": {
                    "$schema": (
                        "https://adaptivecards.io/schemas/adaptive-card.json"
                    ),
                    "type": "AdaptiveCard",
                    "version": "1.4",
                    "body": body,
                },
            }
        ],
    }


def post_teams_payload(
    webhook_url: str,
    payload: dict[str, Any],
    description: str,
) -> None:
    request = urllib.request.Request(
        webhook_url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Content-Type": "application/json; charset=utf-8",
            "Connection": "close",
        },
        method="POST",
    )
    last_error: BaseException | None = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                response_body = response.read().decode(
                    "utf-8", errors="replace"
                )
                if not 200 <= response.status < 300:
                    raise RuntimeError(
                        f"Teams Webhook 返回 {response.status}: "
                        f"{response_body}"
                    )
            return
        except urllib.error.HTTPError as error:
            if 400 <= error.code < 500:
                detail = error.read().decode("utf-8", errors="replace")
                raise RuntimeError(
                    f"Teams Webhook 拒绝请求 {error.code}: {detail}"
                ) from error
            last_error = error
        except (OSError, urllib.error.URLError) as error:
            last_error = error
        if attempt < 3:
            time.sleep(2**attempt)
    raise RuntimeError(
        f"Teams Webhook 发送{description}失败: {last_error}"
    ) from last_error


def send_ranked_to_teams(
    webhook_url: str,
    candidates: pd.DataFrame,
    trade_date: str,
    rows_per_message: int = 10,
    title: str = "A 股在线权重 Top 30",
    score_label: str = "权重总分",
) -> None:
    if not webhook_url.startswith("https://"):
        raise ValueError("Teams Webhook 必须是 HTTPS 地址")
    chunks = [
        candidates.iloc[start : start + rows_per_message]
        for start in range(0, len(candidates), rows_per_message)
    ] or [candidates]
    for page, chunk in enumerate(chunks, start=1):
        payload = build_ranked_teams_payload(
            chunk,
            trade_date,
            page,
            len(chunks),
            title,
            score_label,
        )
        post_teams_payload(
            webhook_url,
            payload,
            f"{title} 第 {page}/{len(chunks)} 页",
        )


def send_dragon_tiger_to_teams(
    webhook_url: str,
    results: pd.DataFrame,
    trade_date: str,
    rows_per_message: int = 10,
) -> None:
    if not webhook_url.startswith("https://"):
        raise ValueError("Teams Webhook 必须是 HTTPS 地址")
    chunks = [
        results.iloc[start : start + rows_per_message]
        for start in range(0, len(results), rows_per_message)
    ] or [results]
    for page, chunk in enumerate(chunks, start=1):
        payload = build_dragon_tiger_teams_payload(
            chunk, trade_date, page, len(chunks)
        )
        post_teams_payload(
            webhook_url,
            payload,
            f"龙虎榜第 {page}/{len(chunks)} 页",
        )


def print_dragon_tiger_results(
    results: pd.DataFrame, trade_date: str
) -> None:
    print(f"\nA 股龙虎榜（交易日: {trade_date}）")
    if results.empty:
        print("龙虎榜没有股票记录。")
        return
    display = results[
        [
            "thscode",
            "stock_name",
            "status",
            "change",
            "net_value",
            "buy_value",
            "sell_value",
            "org_net_value",
            "range_days",
        ]
    ].copy()
    display.columns = [
        "股票代码",
        "股票名",
        "状态",
        "涨跌幅",
        "净买入(万)",
        "买入额(万)",
        "卖出额(万)",
        "机构净买入(万)",
        "区间(天)",
    ]
    display["涨跌幅"] = display["涨跌幅"].map(
        lambda value: "-" if pd.isna(value) else f"{float(value):.2%}"
    )
    for column in ["净买入(万)", "买入额(万)", "卖出额(万)", "机构净买入(万)"]:
        display[column] = display[column].map(format_dragon_tiger_amount)
    print(display.to_string(index=False))


def print_ranked_candidates(
    candidates: pd.DataFrame,
    trade_date: str,
    title: str,
    score_label: str,
) -> None:
    print(f"\n{title}（交易日: {trade_date}）")
    if candidates.empty:
        print("没有候选股票。")
        return
    display = candidates.rename(
        columns={
            "thscode": "股票代码",
            "stock_name": "股票名",
            "status": "状态",
            "close_price": "今日价格",
            "year_high": "一年最高",
            "year_low": "一年最低",
            "strategy_score": score_label,
        }
    )
    display[score_label] = display[score_label].map(
        lambda value: f"{float(value):.3f}"
    )
    columns = ["股票代码", "股票名"]
    if "状态" in display.columns:
        columns.append("状态")
    columns.extend(
        ["今日价格", "一年最高", "一年最低", score_label]
    )
    print(
        display[columns].to_string(index=False)
    )


def send_to_teams(
    webhook_url: str,
    results: pd.DataFrame,
    trade_date: str,
    rows_per_message: int = 6,
) -> None:
    if not webhook_url.startswith("https://"):
        raise ValueError("Teams Webhook 必须是 HTTPS 地址")
    chunks = [
        results.iloc[start : start + rows_per_message]
        for start in range(0, len(results), rows_per_message)
    ] or [results]
    for page, chunk in enumerate(chunks, start=1):
        payload = build_teams_payload(chunk, trade_date, page, len(chunks))
        post_teams_payload(
            webhook_url,
            payload,
            f"交易事件第 {page}/{len(chunks)} 页",
        )


def print_results(results: pd.DataFrame, trade_date: str) -> None:
    print(f"\n交易日: {trade_date}")
    if results.empty:
        print("没有股票同时满足流动性和策略条件。")
        return
    display = results.rename(
        columns={
            "thscode": "股票代码",
            "stock_name": "股票名",
            "status": "状态",
            "current_price": "今日价格",
            "buy_price": "买入价格",
            "sell_price": "卖出价格",
            "quantity": "股数",
            "cash_amount": "资金额",
            "return_rate": "收益率",
            "exit_reason": "卖出原因",
            "year_high": "一年最高",
            "year_low": "一年最低",
            "strategy_score": "策略总分",
        }
    )
    display["卖出价格"] = display["卖出价格"].map(
        lambda value: "-" if pd.isna(value) else f"{float(value):.2f}"
    )
    display["策略总分"] = display["策略总分"].map(
        lambda value: f"{float(value):.3f}"
    )
    display["收益率"] = display["收益率"].map(
        lambda value: "-" if pd.isna(value) else f"{float(value):.2%}"
    )
    display["卖出原因"] = display["卖出原因"].replace("", "-")
    print(
        display[
            [
                "股票代码",
                "股票名",
                "状态",
                "今日价格",
                "买入价格",
                "卖出价格",
                "股数",
                "资金额",
                "收益率",
                "卖出原因",
                "一年最高",
                "一年最低",
                "策略总分",
            ]
        ].to_string(index=False)
    )


def load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"配置文件不存在: {path}；请复制 config.example.json 后填写"
        )
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"配置文件无法读取: {path}: {error}") from error
    if not isinstance(config, dict):
        raise ValueError("配置文件根节点必须是 JSON object")
    return config


def load_strategy_weight_settings(
    config: dict[str, Any],
) -> StrategyWeightSettings:
    raw = config.get("strategy_weights") or {}
    if not isinstance(raw, dict):
        raise ValueError("strategy_weights 配置必须是 JSON object")
    settings = StrategyWeightSettings(
        initial_weight=float(raw.get("initial_weight", 1.0)),
        learning_rate=float(raw.get("learning_rate", 0.25)),
        return_scale=float(raw.get("return_scale", 0.10)),
        min_weight=float(raw.get("min_weight", 0.20)),
        max_weight=float(raw.get("max_weight", 5.0)),
    )
    settings.validate()
    return settings


def load_trading_settings(config: dict[str, Any]) -> TradingSettings:
    raw = config.get("trading") or {}
    if not isinstance(raw, dict):
        raise ValueError("trading 配置必须是 JSON object")
    settings = TradingSettings(
        budget_per_stock=float(raw.get("budget_per_stock", 100_000)),
        lot_size=int(raw.get("lot_size", 100)),
        buy_fee_rate=float(raw.get("buy_fee_rate", 0.00025)),
        sell_fee_rate=float(raw.get("sell_fee_rate", 0.001)),
    )
    settings.validate()
    return settings


def load_exit_settings(config: dict[str, Any]) -> ExitSettings:
    raw = config.get("exit_rules") or {}
    if not isinstance(raw, dict):
        raise ValueError("exit_rules 配置必须是 JSON object")
    settings = ExitSettings(
        take_profit_rate=float(raw.get("take_profit_rate", 0.20)),
        stop_loss_rate=float(raw.get("stop_loss_rate", 0.10)),
        max_holding_days=int(raw.get("max_holding_days", 60)),
        trailing_drawdown_rate=float(
            raw.get("trailing_drawdown_rate", 0.12)
        ),
        score_decline_confirmation_days=int(
            raw.get("score_decline_confirmation_days", 3)
        ),
        market_ma_window=int(raw.get("market_ma_window", 20)),
        market_return_window=int(raw.get("market_return_window", 20)),
        market_return_threshold=float(
            raw.get("market_return_threshold", -0.05)
        ),
        reentry_cooldown_days=int(raw.get("reentry_cooldown_days", 5)),
    )
    settings.validate()
    return settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument(
        "--history-database", type=Path, default=DEFAULT_HISTORY_DATABASE
    )
    parser.add_argument(
        "--fundamental-database",
        type=Path,
        default=DEFAULT_FUNDAMENTAL_DATABASE,
        help="A 类策略使用的基本面缓存数据库",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=None,
        help="调试覆盖选股数量；默认按在线权重总分取 Top 30",
    )
    parser.add_argument(
        "--ml-model",
        type=Path,
        default=DEFAULT_ML_MODEL,
        help="强化学习模型 checkpoint，默认 ML/artifacts/actor_critic.pt",
    )
    parser.add_argument(
        "--no-ml",
        action="store_true",
        help="跳过强化学习模型 Top 30 生成与发送",
    )
    parser.add_argument(
        "--reset-history",
        action="store_true",
        help="备份后清空旧持仓、事件和权重，并从权重 1 重新开始",
    )
    parser.add_argument(
        "--backfill",
        action="store_true",
        help=(
            "选股前增量更新日 K、复权事件、估值快照和轮换财报"
        ),
    )
    parser.add_argument(
        "--send",
        action="store_true",
        help="实际发送到 Teams；不指定时只在终端预览",
    )
    parser.add_argument(
        "--webhook-url",
        default=None,
        help=(
            "覆盖配置文件中的 Teams Webhook；也可使用环境变量 "
            f"{TEAMS_WEBHOOK_ENV}"
        ),
    )
    parser.add_argument(
        "--no-name-lookup",
        action="store_true",
        help="不联网查询股票名，使用证券代码代替",
    )
    parser.add_argument(
        "--no-dragon-tiger",
        action="store_true",
        help="跳过龙虎榜查询、SQLite 状态记录和 Teams 推送",
    )
    return parser.parse_args()


def main() -> int:
    global STRATEGY_POINTS
    args = parse_args()
    if args.max_results is not None and args.max_results <= 0:
        raise ValueError("--max-results 必须大于 0")
    config = load_config(args.config.resolve())
    weight_settings = load_strategy_weight_settings(config)
    history_database = args.history_database.resolve()
    if args.reset_history:
        backup = reset_history_database(history_database, weight_settings)
        print(
            f"旧持仓与评分记录已清空；备份: "
            f"{backup if backup is not None else '无'}"
        )
    STRATEGY_POINTS = load_online_strategy_weights(
        history_database,
        weight_settings,
    )
    print(
        "已加载在线策略权重: "
        + "，".join(
            f"{name}={STRATEGY_POINTS[name]:.3f}"
            for name in STRATEGY_NAMES
        )
    )
    trading_settings = load_trading_settings(config)
    exit_settings = load_exit_settings(config)
    financial_api_config = config.get("financial_api") or {}
    teams_config = config.get("teams") or {}
    if not isinstance(financial_api_config, dict) or not isinstance(
        teams_config, dict
    ):
        raise ValueError("financial_api 和 teams 配置必须是 JSON object")
    webhook_url = (
        args.webhook_url
        or teams_config.get("webhook_url")
        or os.environ.get(TEAMS_WEBHOOK_ENV)
    )

    if args.backfill:
        try:
            from Sequoia.update_daily_k import incremental_update_daily_k
        except SystemExit as error:
            raise ImportError(
                "执行 --backfill 需要 pyarrow；请先运行 "
                "python -m pip install -r Sequoia/requirements.txt"
            ) from error
        update = incremental_update_daily_k(
            args.database.resolve(),
            financial_api_config,
            fundamental_database=args.fundamental_database.resolve(),
        )
        print(
            f"行情增量更新完成：{update['before_date']} -> "
            f"{update['after_date']}，新增 {update['rows_added']} 行；"
            f"复权事件 {update['event_rows_after']} 行；"
            f"估值 {update['valuation_rows']} 行；"
            f"财报刷新成功 "
            f"{update['report_symbols_succeeded']}/"
            f"{update['report_symbols_requested']} 只"
        )
    
    history, latest_date = load_history(args.database.resolve())
    features = build_features(history)
    latest = features.loc[features["trade_date"].eq(latest_date)]
    fundamental_signals, fundamental_summary = load_fundamental_signals(
        args.fundamental_database.resolve(),
        args.database.resolve(),
        latest,
        latest_date,
    )
    candidates = select_candidates(
        features, latest_date, fundamental_signals=fundamental_signals
    )
    candidates = apply_risk_parity(candidates)
    candidates["signal_count"] = candidates["strategies"].str.len()
    candidates["strategy_score"] = candidates["strategies"].map(
        lambda names: sum(STRATEGY_POINTS[name] for name in names)
    )
    candidates = apply_de_optimization(candidates, features)
    selected = prepare_results(
        candidates,
        max_results=args.max_results,
        lookup_names=not args.no_name_lookup,
        api_config=financial_api_config,
    )
    dragon_tiger_results = pd.DataFrame()
    dragon_tiger_is_new = False
    if not args.no_dragon_tiger:
        dragon_tiger_data = fetch_dragon_tiger_list(
            financial_api_config, latest_date
        )
        dragon_tiger_candidates = prepare_dragon_tiger_results(
            dragon_tiger_data
        )
        dragon_tiger_results, dragon_tiger_is_new = (
            record_dragon_tiger_results(
                dragon_tiger_candidates,
                dragon_tiger_data,
                history_database,
            )
        )
        print_dragon_tiger_results(dragon_tiger_results, latest_date)
    ml_selected = pd.DataFrame()
    if not args.no_ml:
        try:
            ml_selected = prepare_ml_results(
                history=history,
                features=features,
                latest_date=latest_date,
                model_path=args.ml_model.resolve(),
                lookup_names=not args.no_name_lookup,
                api_config=financial_api_config,
            )
            print_ranked_candidates(
                ml_selected,
                latest_date,
                "A 股强化学习模型 Top 30",
                "模型分数",
            )
        except (FileNotFoundError, ImportError, RuntimeError, ValueError) as error:
            print(f"强化学习 Top 30 已跳过: {error}", file=sys.stderr)
    results, is_new_run = reconcile_and_record(
        selected,
        features,
        latest_date,
        history_database,
        trading_settings,
        exit_settings,
        weight_settings,
    )
    selected, online_recommendations_are_new = record_teams_recommendations(
        selected,
        features,
        latest_date,
        history_database,
        "online_weight",
    )
    ml_recommendations_are_new = False
    if not ml_selected.empty:
        ml_selected, ml_recommendations_are_new = (
            record_teams_recommendations(
                ml_selected,
                features,
                latest_date,
                history_database,
                "reinforcement_learning",
            )
        )
    print_results(results, latest_date)
    print(
        f"\n运行摘要: {fundamental_summary}；"
        "B/C 类运行 12 个日线策略；"
        "D 类完成低波动、风险平价及 DE 候选组合优化。"
    )
    print(
        f"历史记录: {args.history_database.resolve()} "
        f"（{'新增' if is_new_run else '复用同交易日记录'}）"
    )
    print(
        "Teams 推荐记录: 在线权重"
        f"{'新增' if online_recommendations_are_new else '复用'}"
        + (
            "，强化学习"
            f"{'新增' if ml_recommendations_are_new else '复用'}"
            if not ml_selected.empty
            else ""
        )
    )
    if not args.no_dragon_tiger:
        print(
            "龙虎榜记录: "
            f"{'新增' if dragon_tiger_is_new else '复用'}，"
            f"状态事件 {len(dragon_tiger_results)} 条"
        )

    if args.send:
        if not webhook_url:
            raise ValueError(
                "未提供 Webhook；请填写配置文件 teams.webhook_url，"
                f"或设置 {TEAMS_WEBHOOK_ENV}"
            )
        send_ranked_to_teams(str(webhook_url), selected, latest_date)
        if not ml_selected.empty:
            send_ranked_to_teams(
                str(webhook_url),
                ml_selected,
                latest_date,
                title="A 股强化学习模型 Top 30",
                score_label="模型分数",
            )
        if not args.no_dragon_tiger:
            send_dragon_tiger_to_teams(
                str(webhook_url), dragon_tiger_results, latest_date
            )
        sell_events = results.loc[results["status"].eq("卖出")]
        if not sell_events.empty:
            send_to_teams(str(webhook_url), sell_events, latest_date)
        print(
            f"已向 Teams 发送在线权重 Top {len(selected)} 候选"
            + (
                f"、强化学习 Top {len(ml_selected)} 候选"
                if not ml_selected.empty
                else ""
            )
            + (
                f"、龙虎榜 {len(dragon_tiger_results)} 条状态"
                if not args.no_dragon_tiger
                else ""
            )
            + (
                f"及 {len(sell_events)} 条卖出通知。"
                if not sell_events.empty
                else "。"
            )
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ImportError, RuntimeError, ValueError) as error:
        print(f"运行失败: {error}", file=sys.stderr)
        raise SystemExit(1) from error
