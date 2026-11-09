"""使用训练好的模型对最新日 K 生成下一交易日候选股票。"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ML.data import FEATURE_COLUMNS, FeatureScaler, build_features, load_daily_bars
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
        "--model",
        type=Path,
        default=ROOT / "ML" / "artifacts" / "actor_critic.pt",
    )
    parser.add_argument("--universe-size", type=int, default=800)
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "ML" / "artifacts" / "latest_candidates.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(
        args.model, map_location=device, weights_only=False
    )
    model = ActorCritic(len(FEATURE_COLUMNS)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    bars, start_date, end_date = load_daily_bars(
        args.database,
        years=1,
        universe_size=args.universe_size,
    )
    features = build_features(bars, start_date, end_date)
    scaler = FeatureScaler(
        mean=checkpoint["scaler_mean"],
        std=checkpoint["scaler_std"],
    )
    features = scaler.transform(features)
    latest_date = str(features["trade_date"].max())
    latest = features.loc[
        features["trade_date"].eq(latest_date)
        & features["avg_turnover_20"].ge(20_000_000)
        & features["feature_ready"]
    ].copy()
    observations = torch.as_tensor(
        latest[FEATURE_COLUMNS].to_numpy(dtype="float32"),
        dtype=torch.float32,
        device=device,
    )
    with torch.no_grad():
        logits, _ = model(observations)
        probabilities = torch.softmax(
            torch.cat([logits, model.cash_logit.reshape(1)]), dim=0
        )[:-1]
        latest["rl_score"] = logits.cpu().numpy()
        latest["selection_probability"] = probabilities.cpu().numpy()
    selected = latest.nlargest(args.top, "rl_score")[
        [
            "trade_date",
            "thscode",
            "close_price",
            "avg_turnover_20",
            "rl_score",
            "selection_probability",
        ]
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(selected.to_string(index=False))
    print(f"\n信号基于 {latest_date} 收盘数据，供下一交易日参考。")
    print(f"候选名单已保存: {args.output}")


if __name__ == "__main__":
    main()
