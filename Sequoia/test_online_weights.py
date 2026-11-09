"""在线策略权重的最小回归测试。"""

from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

import main
import pandas as pd


class OnlineStrategyWeightTest(unittest.TestCase):
    def test_ranked_teams_payload_has_requested_columns(self) -> None:
        candidates = pd.DataFrame(
            [
                {
                    "thscode": "000001.SZ",
                    "stock_name": "平安银行",
                    "xueqiu_url": "https://xueqiu.com/S/SZ000001",
                    "close_price": 10.0,
                    "year_high": 12.0,
                    "year_low": 8.0,
                    "strategy_score": 2.5,
                    "status": "买进",
                }
            ]
        )
        payload = main.build_ranked_teams_payload(
            candidates,
            "2026-11-03",
        )
        body = payload["attachments"][0]["content"]["body"]
        headers = [
            column["items"][0]["text"]
            for column in body[2]["columns"]
        ]
        self.assertEqual(
            headers,
            [
                "股票代码",
                "股票名",
                "状态",
                "今日价格",
                "一年最高",
                "一年最低",
                "权重总分",
            ],
        )

    def test_teams_recommendation_statuses_follow_previous_run(self) -> None:
        features = pd.DataFrame(
            [
                {
                    "trade_date": "2026-11-03",
                    "thscode": "000001.SZ",
                    "close_price": 10.0,
                    "year_high": 12.0,
                    "year_low": 8.0,
                },
                {
                    "trade_date": "2026-11-04",
                    "thscode": "000001.SZ",
                    "close_price": 10.5,
                    "year_high": 12.0,
                    "year_low": 8.0,
                },
                {
                    "trade_date": "2026-11-04",
                    "thscode": "000002.SZ",
                    "close_price": 20.0,
                    "year_high": 22.0,
                    "year_low": 18.0,
                },
                {
                    "trade_date": "2026-11-05",
                    "thscode": "000002.SZ",
                    "close_price": 20.5,
                    "year_high": 22.0,
                    "year_low": 18.0,
                },
            ]
        )

        def candidates(symbols: list[str], trade_date: str) -> pd.DataFrame:
            rows = []
            for symbol in symbols:
                market = features.loc[
                    features["trade_date"].eq(trade_date)
                    & features["thscode"].eq(symbol)
                ].iloc[0]
                rows.append(
                    {
                        "thscode": symbol,
                        "stock_name": symbol[:6],
                        "xueqiu_url": f"https://xueqiu.com/S/{symbol}",
                        "close_price": market["close_price"],
                        "year_high": market["year_high"],
                        "year_low": market["year_low"],
                        "strategy_score": 1.0,
                    }
                )
            return pd.DataFrame(rows)

        with tempfile.TemporaryDirectory() as temp:
            database = Path(temp) / "history.sqlite3"
            first, is_new = main.record_teams_recommendations(
                candidates(["000001.SZ"], "2026-11-03"),
                features,
                "2026-11-03",
                database,
                "online_weight",
            )
            self.assertTrue(is_new)
            self.assertEqual(
                dict(zip(first.thscode, first.status)),
                {"000001.SZ": "买进"},
            )

            second, is_new = main.record_teams_recommendations(
                candidates(["000002.SZ"], "2026-11-04"),
                features,
                "2026-11-04",
                database,
                "online_weight",
            )
            self.assertTrue(is_new)
            self.assertEqual(
                dict(zip(second.thscode, second.status)),
                {
                    "000002.SZ": "买进",
                    "000001.SZ": "卖出",
                },
            )

            connection = sqlite3.connect(database)
            try:
                statuses = dict(
                    connection.execute(
                        """
                        SELECT thscode, status
                        FROM teams_recommendation_events
                        WHERE recommendation_run_id = (
                            SELECT recommendation_run_id
                            FROM teams_recommendation_runs
                            WHERE source = ? AND trade_date = ?
                        )
                        """,
                        ("online_weight", "2026-11-04"),
                    )
                )
            finally:
                connection.close()
            self.assertEqual(statuses, {
                "000002.SZ": "买进",
                "000001.SZ": "卖出",
            })

            third, is_new = main.record_teams_recommendations(
                candidates(["000002.SZ"], "2026-11-05"),
                features,
                "2026-11-05",
                database,
                "online_weight",
            )
            self.assertTrue(is_new)
            self.assertEqual(
                dict(zip(third.thscode, third.status)),
                {"000002.SZ": "持有"},
            )

    def test_profitable_close_increases_entry_strategy_weights(self) -> None:
        settings = main.StrategyWeightSettings()
        with tempfile.TemporaryDirectory() as temp:
            database = Path(temp) / "history.sqlite3"
            weights = main.load_online_strategy_weights(database, settings)
            self.assertEqual(len(weights), len(main.STRATEGY_NAMES))
            self.assertEqual(set(weights.values()), {1.0})

            connection = sqlite3.connect(database)
            try:
                connection.executescript(main.HISTORY_SCHEMA_SQL)
                cursor = connection.execute(
                    """
                    INSERT INTO selection_runs (
                        trade_date, created_at, selected_count, event_count
                    ) VALUES (?, ?, ?, ?)
                    """,
                    ("2026-11-04", "test", 0, 1),
                )
                events = [
                    {
                        "status": "卖出",
                        "return_rate": 0.10,
                        "_entry_strategy": (
                            "A/华泰FFScore、C/RSRS择时"
                        ),
                        "thscode": "000001.SZ",
                    }
                ]
                main.update_strategy_weights(
                    connection,
                    int(cursor.lastrowid),
                    "2026-11-04",
                    events,
                    settings,
                )
                connection.commit()
                updated = dict(
                    connection.execute(
                        """
                        SELECT strategy, weight
                        FROM strategy_weights
                        WHERE strategy IN (?, ?)
                        """,
                        ("A/华泰FFScore", "C/RSRS择时"),
                    )
                )
                self.assertTrue(
                    all(weight > 1.0 for weight in updated.values())
                )
                count = connection.execute(
                    "SELECT COUNT(*) FROM strategy_weight_updates"
                ).fetchone()[0]
                self.assertEqual(count, 2)
            finally:
                connection.close()


if __name__ == "__main__":
    unittest.main()
