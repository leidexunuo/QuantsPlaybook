"""组合仿真与按月 Actor-Critic 更新。"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from .data import FEATURE_COLUMNS
from .model import ActorCritic


@dataclass(frozen=True)
class TradingSettings:
    budget_per_stock: float = 100_000.0
    lot_size: int = 100
    buy_fee_rate: float = 0.00025
    sell_fee_rate: float = 0.001


@dataclass(frozen=True)
class ExitSettings:
    take_profit_rate: float = 0.20
    stop_loss_rate: float = 0.10
    max_holding_days: int = 60
    trailing_drawdown_rate: float = 0.12
    score_decline_confirmation_days: int = 3
    market_return_threshold: float = -0.05
    reentry_cooldown_days: int = 5


@dataclass
class Position:
    symbol: str
    quantity: int
    buy_price: float
    invested_amount: float
    buy_day_index: int
    peak_price: float
    last_score: float
    score_decline_count: int = 0


def load_main_settings(config_path: Path) -> tuple[TradingSettings, ExitSettings]:
    """读取 main.py 同一配置；缺失字段沿用 main.py 默认值。"""
    config: dict[str, Any] = {}
    if config_path.is_file():
        config = json.loads(config_path.read_text(encoding="utf-8"))
    trading_raw = config.get("trading") or {}
    exit_raw = config.get("exit_rules") or {}
    trading = TradingSettings(
        budget_per_stock=float(
            trading_raw.get("budget_per_stock", 100_000)
        ),
        lot_size=int(trading_raw.get("lot_size", 100)),
        buy_fee_rate=float(trading_raw.get("buy_fee_rate", 0.00025)),
        sell_fee_rate=float(trading_raw.get("sell_fee_rate", 0.001)),
    )
    exits = ExitSettings(
        take_profit_rate=float(exit_raw.get("take_profit_rate", 0.20)),
        stop_loss_rate=float(exit_raw.get("stop_loss_rate", 0.10)),
        max_holding_days=int(exit_raw.get("max_holding_days", 60)),
        trailing_drawdown_rate=float(
            exit_raw.get("trailing_drawdown_rate", 0.12)
        ),
        score_decline_confirmation_days=int(
            exit_raw.get("score_decline_confirmation_days", 3)
        ),
        market_return_threshold=float(
            exit_raw.get("market_return_threshold", -0.05)
        ),
        reentry_cooldown_days=int(
            exit_raw.get("reentry_cooldown_days", 5)
        ),
    )
    return trading, exits


def shaped_monthly_reward(
    monthly_return: float, negative_penalty: float = 1.5
) -> float:
    """正月收益为正奖励，负月收益按倍数处罚。"""
    if monthly_return >= 0:
        return float(monthly_return)
    return float(monthly_return * negative_penalty)


class MonthlyTradingTrainer:
    def __init__(
        self,
        model: ActorCritic,
        optimizer: torch.optim.Optimizer,
        trading: TradingSettings,
        exits: ExitSettings,
        device: torch.device,
        max_positions: int = 20,
        daily_candidates: int = 128,
        max_buys_per_day: int = 3,
        negative_penalty: float = 1.5,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        gradient_clip: float = 1.0,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.trading = trading
        self.exits = exits
        self.device = device
        self.max_positions = max_positions
        self.daily_candidates = daily_candidates
        self.max_buys_per_day = max_buys_per_day
        self.negative_penalty = negative_penalty
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.gradient_clip = gradient_clip

    def _observations(self, frame: pd.DataFrame) -> torch.Tensor:
        return torch.as_tensor(
            frame[FEATURE_COLUMNS].to_numpy(dtype=np.float32),
            dtype=torch.float32,
            device=self.device,
        )

    def _portfolio_value(
        self,
        cash: float,
        positions: dict[str, Position],
        prices: dict[str, float],
    ) -> float:
        value = cash
        for symbol, position in positions.items():
            price = prices.get(symbol, position.buy_price)
            value += position.quantity * price
        return float(value)

    def _update_month(
        self,
        log_probs: list[torch.Tensor],
        values: list[torch.Tensor],
        entropies: list[torch.Tensor],
        monthly_return: float,
    ) -> dict[str, float]:
        reward = shaped_monthly_reward(
            monthly_return, self.negative_penalty
        )
        if not log_probs:
            return {
                "monthly_return": monthly_return,
                "reward": reward,
                "loss": 0.0,
            }
        policy_log_prob = torch.stack(log_probs).mean()
        value = torch.stack(values).mean()
        entropy = torch.stack(entropies).mean()
        target = torch.tensor(reward, dtype=torch.float32, device=self.device)
        advantage = target - value.detach()
        policy_loss = -(policy_log_prob * advantage)
        value_loss = (value - target).pow(2)
        loss = (
            policy_loss
            + self.value_coef * value_loss
            - self.entropy_coef * entropy
        )
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
        self.optimizer.step()
        return {
            "monthly_return": monthly_return,
            "reward": reward,
            "loss": float(loss.detach().cpu()),
        }

    def run_epoch(
        self, features: pd.DataFrame, epoch: int
    ) -> list[dict[str, Any]]:
        self.model.train()
        initial_cash = self.max_positions * self.trading.budget_per_stock
        cash = float(initial_cash)
        positions: dict[str, Position] = {}
        cooldown_until: dict[str, int] = {}
        last_prices: dict[str, float] = {}
        month_start_value = initial_cash
        current_month: str | None = None
        month_log_probs: list[torch.Tensor] = []
        month_values: list[torch.Tensor] = []
        month_entropies: list[torch.Tensor] = []
        history: list[dict[str, Any]] = []

        grouped = features.groupby("trade_date", sort=True)
        total_days = features["trade_date"].nunique()
        for day_index, (trade_date, day) in enumerate(grouped):
            month = str(trade_date)[:7]
            if current_month is None:
                current_month = month
                month_start_value = self._portfolio_value(
                    cash, positions, last_prices
                )
            elif month != current_month:
                end_value = self._portfolio_value(
                    cash, positions, last_prices
                )
                monthly_return = end_value / month_start_value - 1.0
                metrics = self._update_month(
                    month_log_probs,
                    month_values,
                    month_entropies,
                    monthly_return,
                )
                history.append(
                    {
                        "epoch": epoch,
                        "month": current_month,
                        "nav": end_value / initial_cash,
                        "positions": len(positions),
                        **metrics,
                    }
                )
                current_month = month
                month_start_value = end_value
                month_log_probs = []
                month_values = []
                month_entropies = []

            prices = dict(
                zip(day["thscode"], day["close_price"].astype(float))
            )
            last_prices.update(prices)
            day_by_symbol = day.set_index("thscode", drop=False)
            market_return = day["market_return_20_raw"].dropna()
            market_below = day["market_below_ma_20_raw"].dropna()
            risk_off = bool(
                not market_return.empty
                and not market_below.empty
                and float(market_below.iloc[0]) > 0.5
                and float(market_return.iloc[0])
                <= self.exits.market_return_threshold
            )

            # 先按 main.py 的优先级处理强制卖出。
            for symbol, position in list(positions.items()):
                if symbol not in day_by_symbol.index:
                    continue
                row = day_by_symbol.loc[symbol]
                current_price = float(row["close_price"])
                position.peak_price = max(
                    position.peak_price, float(row["high_price"])
                )
                peak_drawdown = max(
                    1.0 - current_price / position.peak_price, 0.0
                )
                proceeds = (
                    current_price
                    * position.quantity
                    * (1.0 - self.trading.sell_fee_rate)
                )
                return_rate = proceeds / position.invested_amount - 1.0
                holding_days = day_index - position.buy_day_index + 1

                with torch.no_grad():
                    score = float(
                        self.model(self._observations(row.to_frame().T))[0]
                        .squeeze()
                        .cpu()
                    )
                if score < position.last_score:
                    position.score_decline_count += 1
                else:
                    position.score_decline_count = 0
                position.last_score = score

                should_sell = (
                    risk_off
                    or return_rate <= -self.exits.stop_loss_rate
                    or peak_drawdown >= self.exits.trailing_drawdown_rate
                    or return_rate >= self.exits.take_profit_rate
                    or holding_days >= self.exits.max_holding_days
                    or position.score_decline_count
                    >= self.exits.score_decline_confirmation_days
                )
                if should_sell:
                    cash += proceeds
                    del positions[symbol]
                    cooldown_until[symbol] = (
                        day_index + self.exits.reentry_cooldown_days
                    )

            if not risk_off and len(positions) < self.max_positions:
                eligible = day.loc[
                    day["avg_turnover_20"].ge(20_000_000)
                    & day["open_price"].gt(0)
                    & day["feature_ready"]
                    & ~day["thscode"].isin(positions)
                ].copy()
                eligible = eligible.loc[
                    eligible["thscode"].map(
                        lambda symbol: cooldown_until.get(symbol, -1)
                        < day_index
                    )
                ]
                eligible.sort_values(
                    "avg_turnover_20", ascending=False, inplace=True
                )
                eligible = eligible.head(self.daily_candidates)

                buys_left = min(
                    self.max_buys_per_day,
                    self.max_positions - len(positions),
                )
                while buys_left > 0 and not eligible.empty:
                    observations = self._observations(eligible)
                    logits, values = self.model(observations)
                    action_logits = torch.cat(
                        [logits, self.model.cash_logit.reshape(1)]
                    )
                    distribution = torch.distributions.Categorical(
                        logits=action_logits
                    )
                    action = distribution.sample()
                    month_log_probs.append(distribution.log_prob(action))
                    month_values.append(values.mean())
                    month_entropies.append(distribution.entropy())
                    action_index = int(action.detach().cpu())
                    if action_index == len(eligible):
                        break

                    row = eligible.iloc[action_index]
                    price = float(row["open_price"])
                    affordable = self.trading.budget_per_stock / (
                        price * (1.0 + self.trading.buy_fee_rate)
                    )
                    quantity = (
                        int(affordable // self.trading.lot_size)
                        * self.trading.lot_size
                    )
                    invested = (
                        price
                        * quantity
                        * (1.0 + self.trading.buy_fee_rate)
                    )
                    if quantity > 0 and invested <= cash:
                        symbol = str(row["thscode"])
                        score = float(logits[action_index].detach().cpu())
                        cash -= invested
                        positions[symbol] = Position(
                            symbol=symbol,
                            quantity=quantity,
                            buy_price=price,
                            invested_amount=invested,
                            buy_day_index=day_index,
                            peak_price=max(price, float(row["high_price"])),
                            last_score=score,
                        )
                        buys_left -= 1
                    eligible = eligible.drop(eligible.index[action_index])

            if day_index + 1 == total_days and current_month is not None:
                end_value = self._portfolio_value(
                    cash, positions, last_prices
                )
                monthly_return = end_value / month_start_value - 1.0
                metrics = self._update_month(
                    month_log_probs,
                    month_values,
                    month_entropies,
                    monthly_return,
                )
                history.append(
                    {
                        "epoch": epoch,
                        "month": current_month,
                        "nav": end_value / initial_cash,
                        "positions": len(positions),
                        **metrics,
                    }
                )
        return history


def settings_dict(
    trading: TradingSettings, exits: ExitSettings
) -> dict[str, dict[str, Any]]:
    return {"trading": asdict(trading), "exit_rules": asdict(exits)}
