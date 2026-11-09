from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import main


def dragon_tiger_data(
    trade_date: str, symbols: list[tuple[str, str, float]]
) -> dict[str, object]:
    return {
        "timestamp": 1_800_000_000_000,
        "board_type": "all",
        "trade_date": trade_date,
        "count": len(symbols),
        "stock_count": len(symbols),
        "stock_items": [
            {
                "thscode": symbol,
                "ticker": symbol.split(".")[0],
                "name": name,
                "concept_list": [{"name": "测试概念"}],
                "change": 0.05,
                "net_value": net_value,
                "net_rate": 0.1,
                "hot_rank": 10,
                "buy_value": net_value + 1_000_000,
                "sell_value": 1_000_000,
                "limit_reason": "测试原因",
                "range_days": 1,
                "org_net_value": net_value / 2,
                "org_net_rate": 0.05,
                "org_buy_num": 2,
                "org_sell_num": 1,
                "amount": 20_000_000,
                "hot_money_net_value": net_value / 4,
                "hot_money_net_rate": 0.025,
            }
            for symbol, name, net_value in symbols
        ],
    }


class DragonTigerTests(unittest.TestCase):
    def test_fetch_validates_trade_date_and_uses_api_key_header(self) -> None:
        response = dragon_tiger_data(
            "2026-11-06", [("600001.SH", "测试一", 2_000_000)]
        )
        with patch(
            "main.request_json",
            return_value={"code": 0, "message": "success", "data": response},
        ) as request:
            result = main.fetch_dragon_tiger_list(
                {
                    "base_url": "https://example.test",
                    "x_api_key": "secret",
                },
                "2026-11-06",
            )
        self.assertEqual(result["trade_date"], "2026-11-06")
        url = request.call_args.args[0]
        self.assertIn("board_type=all", url)
        self.assertIn("date=2026-11-06", url)
        self.assertEqual(
            request.call_args.kwargs["headers"], {"X-api-key": "secret"}
        )

    def test_records_buy_hold_sell_and_reuses_same_day(self) -> None:
        day_one = dragon_tiger_data(
            "2026-11-05",
            [
                ("600001.SH", "测试一", 2_000_000),
                ("000002.SZ", "测试二", 1_000_000),
            ],
        )
        day_two = dragon_tiger_data(
            "2026-11-06",
            [
                ("600001.SH", "测试一", 3_000_000),
                ("000003.SZ", "测试三", 500_000),
            ],
        )
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "history.sqlite3"
            first, first_is_new = main.record_dragon_tiger_results(
                main.prepare_dragon_tiger_results(day_one),
                day_one,
                database,
            )
            second, second_is_new = main.record_dragon_tiger_results(
                main.prepare_dragon_tiger_results(day_two),
                day_two,
                database,
            )
            repeated, repeated_is_new = main.record_dragon_tiger_results(
                main.prepare_dragon_tiger_results(day_two),
                day_two,
                database,
            )

            self.assertTrue(first_is_new)
            self.assertEqual(set(first["status"]), {"买入"})
            self.assertTrue(second_is_new)
            statuses = dict(zip(second["thscode"], second["status"]))
            self.assertEqual(statuses["600001.SH"], "持有")
            self.assertEqual(statuses["000003.SZ"], "买入")
            self.assertEqual(statuses["000002.SZ"], "卖出")
            self.assertFalse(repeated_is_new)
            self.assertEqual(
                repeated[["thscode", "status"]].to_dict("records"),
                second[["thscode", "status"]].to_dict("records"),
            )

            connection = sqlite3.connect(database)
            try:
                run_count = connection.execute(
                    "SELECT COUNT(*) FROM dragon_tiger_runs"
                ).fetchone()[0]
                concept_list, limit_reason = connection.execute(
                    """
                    SELECT concept_list, limit_reason
                    FROM dragon_tiger_events
                    WHERE thscode = '600001.SH'
                    ORDER BY dragon_tiger_event_id DESC
                    LIMIT 1
                    """
                ).fetchone()
            finally:
                connection.close()
            self.assertEqual(run_count, 2)
            self.assertIn("测试概念", concept_list)
            self.assertEqual(limit_reason, "测试原因")

            payload = main.build_dragon_tiger_teams_payload(
                second, "2026-11-06"
            )
            body = payload["attachments"][0]["content"]["body"]
            self.assertEqual(body[0]["text"], "A 股龙虎榜")
            self.assertEqual(len(body), len(second) + 3)


if __name__ == "__main__":
    unittest.main()
