"""在十年 A 股日 K 上训练 2 x 256 Actor-Critic 选股模型。"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ML.data import FEATURE_COLUMNS, FeatureScaler, build_features, load_daily_bars
from ML.environment import MonthlyTradingTrainer, load_main_settings, settings_dict
from ML.model import ActorCritic


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database",
        type=Path,
        default=ROOT / "Sequoia" / "a_share_market.sqlite3",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "Sequoia" / "config.json",
    )
    parser.add_argument("--years", type=int, default=10)
    parser.add_argument("--universe-size", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--max-positions", type=int, default=20)
    parser.add_argument("--daily-candidates", type=int, default=128)
    parser.add_argument("--max-buys-per-day", type=int, default=10)
    parser.add_argument("--negative-penalty", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "ML" / "artifacts" / "actor_critic.pt",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("读取十年日 K 数据...")
    bars, start_date, end_date = load_daily_bars(
        args.database,
        years=args.years,
        universe_size=args.universe_size,
    )
    print(
        f"行情: {start_date} 至 {end_date}, "
        f"{bars['thscode'].nunique()} 只股票, {len(bars):,} 行"
    )
    features = build_features(bars, start_date, end_date)
    scaler_fit_rows = features[FEATURE_COLUMNS].dropna()
    if scaler_fit_rows.empty:
        raise RuntimeError("有效特征为空，请检查行情覆盖范围")
    scaler = FeatureScaler.fit(scaler_fit_rows)
    features = scaler.transform(features)

    trading, exits = load_main_settings(args.config)
    model = ActorCritic(len(FEATURE_COLUMNS)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    trainer = MonthlyTradingTrainer(
        model=model,
        optimizer=optimizer,
        trading=trading,
        exits=exits,
        device=device,
        max_positions=args.max_positions,
        daily_candidates=args.daily_candidates,
        max_buys_per_day=args.max_buys_per_day,
        negative_penalty=args.negative_penalty,
    )

    all_history: list[dict] = []
    for epoch in range(1, args.epochs + 1):
        history = trainer.run_epoch(features, epoch)
        all_history.extend(history)
        returns = pd.Series(
            [row["monthly_return"] for row in history], dtype=float
        )
        print(
            f"epoch={epoch} 月数={len(history)} "
            f"平均月收益={returns.mean():.2%} "
            f"正收益月份={(returns > 0).mean():.1%}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_columns": FEATURE_COLUMNS,
            "scaler_mean": scaler.mean,
            "scaler_std": scaler.std,
            "start_date": start_date,
            "end_date": end_date,
            "years": args.years,
            "universe_size": args.universe_size,
            "max_positions": args.max_positions,
            "negative_penalty": args.negative_penalty,
            **settings_dict(trading, exits),
        },
        args.output,
    )
    history_path = args.output.with_suffix(".monthly.csv")
    pd.DataFrame(all_history).to_csv(
        history_path, index=False, encoding="utf-8-sig"
    )
    print(f"模型已保存: {args.output}")
    print(f"月度训练记录: {history_path}")


if __name__ == "__main__":
    main()
