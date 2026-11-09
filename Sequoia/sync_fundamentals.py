"""从 Financial-API 增量同步 A 股年报与估值快照到 SQLite。

默认同步行情库内全部证券；中断后重新运行会跳过已有完整缓存的证券。首次全量同步
会产生较多 API 请求，可用 ``--symbols`` 或 ``--limit-symbols`` 分批执行。
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "Sequoia" / "config.json"
DEFAULT_MARKET_DATABASE = ROOT / "Sequoia" / "a_share_market.sqlite3"
DEFAULT_DATABASE = ROOT / "Sequoia" / "fundamentals.sqlite3"

SCHEMA = """
CREATE TABLE IF NOT EXISTS valuation_snapshots (
    thscode TEXT NOT NULL,
    snapshot_at TEXT NOT NULL,
    name TEXT,
    pe_ttm REAL,
    pe_mrq REAL,
    pb_mrq REAL,
    ps_ttm REAL,
    pcf_ttm REAL,
    PRIMARY KEY (thscode, snapshot_at)
);
CREATE TABLE IF NOT EXISTS fundamental_reports (
    thscode TEXT NOT NULL,
    statement_type TEXT NOT NULL,
    period TEXT NOT NULL,
    period_end_ms INTEGER NOT NULL,
    report_date_ms INTEGER,
    payload_json TEXT NOT NULL,
    synced_at TEXT NOT NULL,
    PRIMARY KEY (thscode, statement_type, period, period_end_ms)
);
CREATE INDEX IF NOT EXISTS idx_fundamental_reports_symbol
ON fundamental_reports (thscode, period, period_end_ms);
CREATE TABLE IF NOT EXISTS fundamental_sync_state (
    state_key TEXT PRIMARY KEY,
    state_value TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""

ENDPOINTS = {
    "income": "income-statements",
    "balance": "balance-sheets",
    "cash_flow": "cash-flow-statements",
}


def request_json(
    url: str, api_key: str, timeout: int = 30
) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={
            "X-api-key": api_key,
            "User-Agent": "QuantsPlaybook/1.0",
            "Connection": "close",
        },
    )
    last_error: BaseException | None = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if payload.get("code") != 0:
                raise RuntimeError(
                    f"code={payload.get('code')}: {payload.get('message')}"
                )
            return payload
        except (
            OSError,
            ValueError,
            RuntimeError,
            urllib.error.URLError,
        ) as error:
            last_error = error
            if attempt < 3:
                time.sleep(2**attempt)
    raise RuntimeError(f"API 请求失败: {last_error}") from last_error


def load_symbols(
    market_database: Path, explicit: str | None
) -> list[str]:
    if explicit:
        return sorted(
            {
                value.strip().upper()
                for value in explicit.split(",")
                if value.strip()
            }
        )
    connection = sqlite3.connect(market_database)
    try:
        return [
            row[0]
            for row in connection.execute(
                "SELECT DISTINCT thscode FROM daily_k ORDER BY thscode"
            )
        ]
    finally:
        connection.close()


def sync_valuations(
    connection: sqlite3.Connection,
    symbols: list[str],
    base_url: str,
    api_key: str,
    snapshot_at: str,
) -> int:
    rows_written = 0
    for start in range(0, len(symbols), 100):
        batch = symbols[start : start + 100]
        query = urllib.parse.urlencode({"thscodes": ",".join(batch)})
        payload = request_json(
            f"{base_url}/api/a-share/valuations/snapshot?{query}", api_key
        )
        rows = (payload.get("data") or {}).get("item") or []
        connection.executemany(
            """
            INSERT OR REPLACE INTO valuation_snapshots (
                thscode, snapshot_at, name, pe_ttm, pe_mrq,
                pb_mrq, ps_ttm, pcf_ttm
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    row.get("thscode"),
                    snapshot_at,
                    row.get("name"),
                    row.get("pe_ttm"),
                    row.get("pe_mrq"),
                    row.get("pb_mrq"),
                    row.get("ps_ttm"),
                    row.get("pcf_ttm"),
                )
                for row in rows
                if row.get("thscode")
            ],
        )
        connection.commit()
        rows_written += len(rows)
    return rows_written


def has_reports(connection: sqlite3.Connection, symbol: str) -> bool:
    counts = connection.execute(
        """
        SELECT statement_type, COUNT(*)
        FROM fundamental_reports
        WHERE thscode = ? AND period = 'annual'
        GROUP BY statement_type
        """,
        (symbol,),
    ).fetchall()
    return (
        len(counts) == len(ENDPOINTS)
        and all(int(count) >= 2 for _, count in counts)
    )


def sync_reports(
    connection: sqlite3.Connection,
    symbol: str,
    base_url: str,
    api_key: str,
    synced_at: str,
) -> int:
    rows_written = 0
    for statement_type, endpoint in ENDPOINTS.items():
        query = urllib.parse.urlencode(
            {"thscode": symbol, "period": "annual", "limit": 8}
        )
        payload = request_json(
            f"{base_url}/api/a-share/financials/{endpoint}?{query}", api_key
        )
        rows = (payload.get("data") or {}).get("item") or []
        connection.executemany(
            """
            INSERT OR REPLACE INTO fundamental_reports (
                thscode, statement_type, period, period_end_ms,
                report_date_ms, payload_json, synced_at
            ) VALUES (?, ?, 'annual', ?, ?, ?, ?)
            """,
            [
                (
                    symbol,
                    statement_type,
                    int(row["period_end_ms"]),
                    row.get("report_date_ms"),
                    json.dumps(row, ensure_ascii=False, separators=(",", ":")),
                    synced_at,
                )
                for row in rows
                if row.get("period_end_ms") is not None
            ],
        )
        connection.commit()
        rows_written += len(rows)
    return rows_written


def complete_report_symbols(
    connection: sqlite3.Connection,
) -> set[str]:
    return {
        str(row[0])
        for row in connection.execute(
            """
            SELECT thscode
            FROM fundamental_reports
            WHERE period = 'annual'
            GROUP BY thscode
            HAVING SUM(CASE WHEN statement_type = 'income' THEN 1 ELSE 0 END) >= 2
               AND SUM(CASE WHEN statement_type = 'balance' THEN 1 ELSE 0 END) >= 2
               AND SUM(CASE WHEN statement_type = 'cash_flow' THEN 1 ELSE 0 END) >= 2
            """,
        )
    }


def incremental_sync_fundamentals(
    market_database: Path,
    database: Path,
    base_url: str,
    api_key: str,
    report_batch_size: int = 50,
) -> dict[str, Any]:
    """刷新全市场估值，并轮换更新一批股票的三张年报。"""
    if report_batch_size < 0:
        raise ValueError("report_batch_size 不能为负")
    symbols = load_symbols(market_database, None)
    now = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")
    snapshot_at = now[:10]
    database.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database)
    try:
        connection.executescript(SCHEMA)
        valuation_rows = sync_valuations(
            connection,
            symbols,
            base_url,
            api_key,
            snapshot_at,
        )
        if not symbols or report_batch_size == 0:
            return {
                "valuation_rows": valuation_rows,
                "report_symbols_requested": 0,
                "report_symbols_succeeded": 0,
                "report_symbols_failed": 0,
                "report_rows": 0,
            }

        completed = complete_report_symbols(connection)
        incomplete = [
            symbol for symbol in symbols if symbol not in completed
        ]
        state = connection.execute(
            """
            SELECT state_value
            FROM fundamental_sync_state
            WHERE state_key = 'annual_report_cursor'
            """
        ).fetchone()
        cursor = int(state[0]) if state and str(state[0]).isdigit() else 0
        cursor %= len(symbols)
        rotated = symbols[cursor:] + symbols[:cursor]
        pending = incomplete[:report_batch_size]
        pending_set = set(pending)
        rotation_slots = report_batch_size - len(pending)
        rotation_batch = [
            symbol for symbol in rotated if symbol not in pending_set
        ][:rotation_slots]
        pending.extend(rotation_batch)

        report_rows = 0
        succeeded = 0
        failed = 0
        for symbol in pending:
            try:
                report_rows += sync_reports(
                    connection,
                    symbol,
                    base_url,
                    api_key,
                    now,
                )
                succeeded += 1
            except RuntimeError as error:
                failed += 1
                print(f"{symbol} 财报增量同步失败: {error}", file=sys.stderr)

        next_cursor = (cursor + max(len(rotation_batch), 1)) % len(symbols)
        connection.execute(
            """
            INSERT INTO fundamental_sync_state (
                state_key, state_value, updated_at
            ) VALUES ('annual_report_cursor', ?, ?)
            ON CONFLICT(state_key) DO UPDATE SET
                state_value=excluded.state_value,
                updated_at=excluded.updated_at
            """,
            (str(next_cursor), now),
        )
        connection.commit()
        return {
            "valuation_rows": valuation_rows,
            "report_symbols_requested": len(pending),
            "report_symbols_succeeded": succeeded,
            "report_symbols_failed": failed,
            "report_rows": report_rows,
        }
    finally:
        connection.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--market-database", type=Path, default=DEFAULT_MARKET_DATABASE)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument(
        "--symbols",
        help="逗号分隔的 thscode；不指定时读取行情库全部股票",
    )
    parser.add_argument(
        "--limit-symbols",
        type=int,
        help="只同步前 N 个尚未完整缓存的股票，便于分批执行",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="即使已有三类年报缓存也重新请求",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    api_config = config.get("financial_api") or {}
    api_key = str(api_config.get("x_api_key") or "").strip()
    if not api_key:
        raise ValueError("配置文件缺少 financial_api.x_api_key")
    base_url = str(
        api_config.get("base_url") or "https://fuyao.aicubes.cn"
    ).rstrip("/")
    symbols = load_symbols(args.market_database.resolve(), args.symbols)
    now = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")
    database = args.database.resolve()
    database.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database)
    try:
        connection.executescript(SCHEMA)
        pending = [
            symbol
            for symbol in symbols
            if args.refresh or not has_reports(connection, symbol)
        ]
        if args.limit_symbols is not None:
            if args.limit_symbols <= 0:
                raise ValueError("--limit-symbols 必须大于 0")
            pending = pending[: args.limit_symbols]
        sync_valuations(connection, symbols, base_url, api_key, now)
        total = len(pending)
        for index, symbol in enumerate(pending, start=1):
            try:
                sync_reports(connection, symbol, base_url, api_key, now)
                print(f"[{index}/{total}] {symbol} 完成")
            except RuntimeError as error:
                print(f"[{index}/{total}] {symbol} 失败: {error}", file=sys.stderr)
        print(f"基本面缓存已更新: {database}")
    finally:
        connection.close()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        print(f"同步失败: {error}", file=sys.stderr)
        raise SystemExit(1) from error
