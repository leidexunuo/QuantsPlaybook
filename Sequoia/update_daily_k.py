"""增量更新主流程依赖的日 K、复权事件、估值快照和财务报表。"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import tempfile
import time
import urllib.error
import urllib.request
from contextlib import closing
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Sequoia.build_sqlite import (
    DAILY_UPSERT_SQL,
    EVENT_INSERT_SQL,
    INDEX_SQL,
    configure_connection,
    import_parquet,
)
from Sequoia.sync_fundamentals import incremental_sync_fundamentals


DEFAULT_CONFIG = ROOT / "Sequoia" / "config.json"
DEFAULT_DATABASE = ROOT / "Sequoia" / "a_share_market.sqlite3"
DEFAULT_FUNDAMENTAL_DATABASE = ROOT / "Sequoia" / "fundamentals.sqlite3"
DAILY_DUMP_ENDPOINT = "/api/dump/market-dumps/daily-k-10d/download-url"
ADJUSTMENT_DUMP_ENDPOINT = (
    "/api/dump/market-dumps/adjustment-factors/download-url"
)
DAILY_COLUMNS = [
    "thscode",
    "currency",
    "interval",
    "adjusted",
    "date_ms",
    "open_price",
    "high_price",
    "low_price",
    "close_price",
    "volume",
    "turnover",
]
EVENT_COLUMNS = [
    "thscode",
    "ticker",
    "ex_date_ms",
    "dividend_per_share",
    "per_share_bonus",
    "allotment_ratio",
    "allotment_price",
    "currency",
]


def _request_json(
    url: str,
    api_key: str,
    description: str,
    timeout: int = 30,
) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={
            "X-api-key": api_key,
            "User-Agent": "QuantsPlaybook/1.0",
            "Connection": "close",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, ValueError, urllib.error.URLError) as error:
        raise RuntimeError(
            f"无法获取{description}下载地址: {error}"
        ) from error
    if payload.get("code") != 0:
        raise RuntimeError(
            f"{description}接口返回错误: "
            f"code={payload.get('code')}, message={payload.get('message')}"
        )
    return payload


def _download_file(
    url: str,
    destination: Path,
    description: str,
    timeout: int = 180,
) -> None:
    # 预签名 URL 是短期凭据，不写入日志或异常消息。
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "QuantsPlaybook/1.0",
            "Connection": "close",
        },
    )
    try:
        with (
            urllib.request.urlopen(request, timeout=timeout) as response,
            destination.open("wb") as output,
        ):
            while block := response.read(4 * 1024 * 1024):
                output.write(block)
    except (OSError, urllib.error.URLError) as error:
        destination.unlink(missing_ok=True)
        raise RuntimeError(f"{description}下载失败") from error
    if not destination.is_file() or destination.stat().st_size == 0:
        raise RuntimeError(f"下载的{description}文件为空")


def _download_dump(
    base_url: str,
    endpoint: str,
    api_key: str,
    destination: Path,
    description: str,
) -> None:
    last_error: RuntimeError | None = None
    for attempt in range(3):
        try:
            payload = _request_json(
                f"{base_url}{endpoint}",
                api_key,
                description,
            )
            presigned_url = str(
                (payload.get("data") or {}).get("presigned_url") or ""
            )
            if not presigned_url.startswith("https://"):
                raise RuntimeError(
                    f"{description}响应缺少有效下载地址"
                )
            _download_file(
                presigned_url,
                destination,
                description,
            )
            presigned_url = ""
            return
        except RuntimeError as error:
            last_error = error
            destination.unlink(missing_ok=True)
            if attempt < 2:
                time.sleep(2**attempt)
    raise RuntimeError(
        f"{description}连续 3 次下载失败"
    ) from last_error


def incremental_update_daily_k(
    database: Path,
    api_config: dict[str, Any],
    batch_size: int = 100_000,
    fundamental_database: Path | None = None,
) -> dict[str, Any]:
    """更新行情、复权事件、最新估值和轮换财报缓存。"""
    api_key = str(api_config.get("x_api_key") or "").strip()
    if not api_key:
        raise ValueError("配置文件缺少 financial_api.x_api_key")
    base_url = str(
        api_config.get("base_url") or "https://fuyao.aicubes.cn"
    ).rstrip("/")

    database = database.resolve()
    if fundamental_database is None:
        fundamental_database = database.with_name("fundamentals.sqlite3")
    fundamental_database = fundamental_database.resolve()
    database.parent.mkdir(parents=True, exist_ok=True)
    before_rows = 0
    before_date: str | None = None
    before_events = 0
    probe = configure_connection(database)
    try:
        before_rows, before_date = probe.execute(
            "SELECT COUNT(*), MAX(trade_date) FROM daily_k"
        ).fetchone()
        before_events = int(
            probe.execute(
                "SELECT COUNT(*) FROM adjustment_events"
            ).fetchone()[0]
        )
    finally:
        probe.close()

    with tempfile.TemporaryDirectory(prefix="quantsplaybook_daily_k_") as temp:
        temp_dir = Path(temp)
        daily_path = temp_dir / "a_share_daily_k_1d_none_10d.parquet"
        event_path = temp_dir / "a_share_adjustment_factors_event_none_all.parquet"
        _download_dump(
            base_url,
            DAILY_DUMP_ENDPOINT,
            api_key,
            daily_path,
            "全市场最近 10 个交易日日 K",
        )
        # 预签名 URL 仅短时有效，前一个文件完成后再申请复权事件链接。
        _download_dump(
            base_url,
            ADJUSTMENT_DUMP_ENDPOINT,
            api_key,
            event_path,
            "全市场复权事件",
        )
        with closing(configure_connection(database)) as connection:
            import_parquet(
                connection,
                daily_path,
                "daily_10d",
                DAILY_COLUMNS,
                DAILY_UPSERT_SQL,
                batch_size,
            )
            import_parquet(
                connection,
                event_path,
                "adjustment_events",
                EVENT_COLUMNS,
                EVENT_INSERT_SQL,
                batch_size,
                replace_events=True,
            )
            connection.executescript(INDEX_SQL)
            connection.execute("PRAGMA optimize")
            after_rows, after_date = connection.execute(
                "SELECT COUNT(*), MAX(trade_date) FROM daily_k"
            ).fetchone()
            after_events = int(
                connection.execute(
                    "SELECT COUNT(*) FROM adjustment_events"
                ).fetchone()[0]
            )

    update_fundamentals = bool(
        api_config.get("update_fundamentals", True)
    )
    fundamental_result: dict[str, Any] = {
        "valuation_rows": 0,
        "report_symbols_requested": 0,
        "report_symbols_succeeded": 0,
        "report_symbols_failed": 0,
        "report_rows": 0,
    }
    if update_fundamentals:
        report_batch_size = int(
            api_config.get("incremental_report_batch_size", 50)
        )
        fundamental_result = incremental_sync_fundamentals(
            database,
            fundamental_database,
            base_url,
            api_key,
            report_batch_size=report_batch_size,
        )

    return {
        "before_rows": int(before_rows),
        "after_rows": int(after_rows),
        "rows_added": int(after_rows) - int(before_rows),
        "before_date": before_date,
        "after_date": after_date,
        "event_rows_before": before_events,
        "event_rows_after": after_events,
        "event_rows_changed": after_events - before_events,
        "fundamental_database": str(fundamental_database),
        **fundamental_result,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument(
        "--fundamental-database",
        type=Path,
        default=DEFAULT_FUNDAMENTAL_DATABASE,
    )
    parser.add_argument("--batch-size", type=int, default=100_000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    api_config = config.get("financial_api") or {}
    if not isinstance(api_config, dict):
        raise ValueError("financial_api 配置必须是 JSON object")
    result = incremental_update_daily_k(
        args.database,
        api_config,
        batch_size=args.batch_size,
        fundamental_database=args.fundamental_database,
    )
    print(
        f"日 K: {result['before_date']} -> {result['after_date']}，"
        f"新增 {result['rows_added']:,} 行"
    )
    print(
        f"复权事件: {result['event_rows_after']:,} 行；"
        f"估值快照: {result['valuation_rows']:,} 行"
    )
    print(
        "财报轮换: "
        f"请求 {result['report_symbols_requested']} 只，"
        f"成功 {result['report_symbols_succeeded']} 只，"
        f"失败 {result['report_symbols_failed']} 只，"
        f"写入 {result['report_rows']:,} 行"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        FileNotFoundError,
        json.JSONDecodeError,
        RuntimeError,
        sqlite3.Error,
        ValueError,
    ) as error:
        print(f"增量更新失败: {error}", file=sys.stderr)
        raise SystemExit(1) from error
