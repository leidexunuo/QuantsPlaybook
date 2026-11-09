"""通过 Financial-API 获取最近三年的 A 股行情与复权事件并写入 SQLite。

Financial-API 提供约 10 年的全市场日 K dump，本脚本下载该 dump 后按日期过滤，
只把指定窗口的数据写入行情库。下载链接是短期预签名地址，不会写入日志或数据库。

注意：该 API 当前只提供“最新估值快照”，没有历史逐日 PE/PB/PCF 接口。本脚本
不会使用当前估值回填历史，避免回测中的未来函数。
"""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import sys
import tempfile
import time
import urllib.error
import urllib.request
from contextlib import closing
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

try:
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "缺少下载处理依赖，请先执行: "
        "python -m pip install -r Sequoia/requirements.txt"
    ) from exc

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


DEFAULT_CONFIG = ROOT / "Sequoia" / "config.json"
DEFAULT_DATABASE = ROOT / "Sequoia" / "a_share_market.sqlite3"

DAILY_DUMP_ENDPOINT = "/api/dump/market-dumps/daily-k/download-url"
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


def subtract_years(value: date, years: int) -> date:
    """按日历年回退，处理 2 月 29 日。"""
    try:
        return value.replace(year=value.year - years)
    except ValueError:
        return value.replace(year=value.year - years, day=28)


def shanghai_midnight_ms(value: date) -> int:
    timestamp = datetime(
        value.year,
        value.month,
        value.day,
        tzinfo=ZoneInfo("Asia/Shanghai"),
    )
    return int(timestamp.timestamp() * 1000)


def request_download_url(
    base_url: str,
    endpoint: str,
    api_key: str,
    timeout: int,
) -> str:
    request = urllib.request.Request(
        f"{base_url}{endpoint}",
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
            url = str(
                (payload.get("data") or {}).get("presigned_url") or ""
            )
            if not url.startswith("https://"):
                raise RuntimeError("响应缺少有效的 presigned_url")
            return url
        except (
            OSError,
            ValueError,
            RuntimeError,
            urllib.error.URLError,
        ) as error:
            last_error = error
            if attempt < 3:
                time.sleep(2**attempt)
    raise RuntimeError(f"无法获取下载地址: {last_error}") from last_error


def download_file(url: str, destination: Path, timeout: int) -> None:
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
        raise RuntimeError(f"数据文件下载失败: {destination.name}") from error
    if not destination.is_file() or destination.stat().st_size == 0:
        raise RuntimeError(f"下载的数据文件为空: {destination.name}")


def filter_parquet_by_date(
    source: Path,
    destination: Path,
    date_column: str,
    start_ms: int,
    end_ms: int,
    batch_size: int,
) -> int:
    parquet = pq.ParquetFile(source)
    if date_column not in parquet.schema_arrow.names:
        raise ValueError(f"{source.name} 缺少日期字段 {date_column}")

    writer: pq.ParquetWriter | None = None
    rows_written = 0
    try:
        for batch in parquet.iter_batches(batch_size=batch_size):
            table = pa.Table.from_batches([batch])
            values = table[date_column]
            mask = pc.and_(
                pc.greater_equal(values, pa.scalar(start_ms, pa.int64())),
                pc.less_equal(values, pa.scalar(end_ms, pa.int64())),
            )
            filtered = table.filter(mask)
            if filtered.num_rows == 0:
                continue
            if writer is None:
                writer = pq.ParquetWriter(
                    destination,
                    filtered.schema,
                    compression="zstd",
                )
            writer.write_table(filtered)
            rows_written += filtered.num_rows
    finally:
        if writer is not None:
            writer.close()

    if rows_written == 0:
        destination.unlink(missing_ok=True)
        raise RuntimeError(
            f"{source.name} 在指定日期范围内没有数据"
        )
    return rows_written


def copy_snapshots(
    daily_path: Path,
    event_path: Path,
    output_dir: Path,
    start_date: date,
    end_date: date,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"{start_date:%Y%m%d}_{end_date:%Y%m%d}"
    daily_output = output_dir / f"a_share_daily_k_3y_{suffix}.parquet"
    event_output = (
        output_dir / f"a_share_adjustment_factors_3y_{suffix}.parquet"
    )
    shutil.copy2(daily_path, daily_output)
    shutil.copy2(event_path, event_output)
    return daily_output, event_output


def import_snapshots(
    database: Path,
    daily_path: Path,
    event_path: Path,
    batch_size: int,
) -> dict[str, Any]:
    database.parent.mkdir(parents=True, exist_ok=True)
    with closing(configure_connection(database)) as connection:
        before_daily = int(
            connection.execute("SELECT COUNT(*) FROM daily_k").fetchone()[0]
        )
        before_events = int(
            connection.execute(
                "SELECT COUNT(*) FROM adjustment_events"
            ).fetchone()[0]
        )
        import_parquet(
            connection,
            daily_path,
            "daily_10y",
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
            replace_events=False,
        )
        connection.executescript(INDEX_SQL)
        connection.execute("PRAGMA optimize")
        connection.commit()
        after_daily, min_date, max_date = connection.execute(
            "SELECT COUNT(*), MIN(trade_date), MAX(trade_date) FROM daily_k"
        ).fetchone()
        after_events = int(
            connection.execute(
                "SELECT COUNT(*) FROM adjustment_events"
            ).fetchone()[0]
        )
    return {
        "daily_rows_before": before_daily,
        "daily_rows_after": int(after_daily),
        "daily_rows_added": int(after_daily) - before_daily,
        "event_rows_before": before_events,
        "event_rows_after": after_events,
        "event_rows_added": after_events - before_events,
        "database_min_date": min_date,
        "database_max_date": max_date,
    }


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument(
        "--years",
        type=int,
        default=3,
        help="获取最近 N 个日历年（默认: 3）",
    )
    parser.add_argument(
        "--as-of",
        type=date.fromisoformat,
        help="截止日期 YYYY-MM-DD，默认使用上海时区当天",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="可选：保存过滤后的三年 Parquet 快照",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100_000,
    )
    parser.add_argument(
        "--request-timeout",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--download-timeout",
        type=int,
        default=1800,
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="只验证两个下载接口，不下载大文件",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.years <= 0:
        raise ValueError("--years 必须大于 0")
    if args.batch_size <= 0:
        raise ValueError("--batch-size 必须大于 0")

    config = json.loads(args.config.read_text(encoding="utf-8"))
    api_config = config.get("financial_api") or {}
    api_key = str(api_config.get("x_api_key") or "").strip()
    if not api_key or api_key.startswith("<"):
        raise ValueError("配置文件缺少有效的 financial_api.x_api_key")
    base_url = str(
        api_config.get("base_url") or "https://fuyao.aicubes.cn"
    ).rstrip("/")

    end_date = args.as_of or datetime.now(
        ZoneInfo("Asia/Shanghai")
    ).date()
    start_date = subtract_years(end_date, args.years)
    start_ms = shanghai_midnight_ms(start_date)
    end_ms = shanghai_midnight_ms(end_date)

    print("验证 Financial-API 下载接口...", flush=True)
    daily_url = request_download_url(
        base_url,
        DAILY_DUMP_ENDPOINT,
        api_key,
        args.request_timeout,
    )
    if args.check_only:
        request_download_url(
            base_url,
            ADJUSTMENT_DUMP_ENDPOINT,
            api_key,
            args.request_timeout,
        )
        print("接口验证成功：日 K 与复权事件下载地址均可获取。")
        return 0

    with tempfile.TemporaryDirectory(
        prefix="quantsplaybook_3y_"
    ) as temp:
        temp_dir = Path(temp)
        raw_daily = temp_dir / "daily_k_10y.parquet"
        raw_events = temp_dir / "adjustment_factors_all.parquet"
        filtered_daily = temp_dir / "daily_k_3y.parquet"
        filtered_events = temp_dir / "adjustment_factors_3y.parquet"

        print("下载全市场约 10 年日 K...", flush=True)
        download_file(daily_url, raw_daily, args.download_timeout)
        daily_url = ""
        print("下载全市场复权事件...", flush=True)
        # 预签名链接通常仅有效数分钟。日 K 文件较大，因此要在其下载完成后
        # 再为复权事件申请新链接，避免提前申请后等待过期。
        event_url = request_download_url(
            base_url,
            ADJUSTMENT_DUMP_ENDPOINT,
            api_key,
            args.request_timeout,
        )
        download_file(event_url, raw_events, args.download_timeout)
        event_url = ""

        print(
            f"过滤日期范围 {start_date} 至 {end_date}...",
            flush=True,
        )
        daily_rows = filter_parquet_by_date(
            raw_daily,
            filtered_daily,
            "date_ms",
            start_ms,
            end_ms,
            args.batch_size,
        )
        event_rows = filter_parquet_by_date(
            raw_events,
            filtered_events,
            "ex_date_ms",
            start_ms,
            end_ms,
            args.batch_size,
        )
        print(
            f"过滤完成：日 K {daily_rows:,} 行，"
            f"复权事件 {event_rows:,} 行。"
        )

        result = import_snapshots(
            args.database.resolve(),
            filtered_daily,
            filtered_events,
            args.batch_size,
        )
        if args.output_dir:
            daily_output, event_output = copy_snapshots(
                filtered_daily,
                filtered_events,
                args.output_dir.resolve(),
                start_date,
                end_date,
            )
            print(f"已保存行情快照: {daily_output}")
            print(f"已保存复权快照: {event_output}")

    print(
        f"行情库已更新: {args.database.resolve()}\n"
        f"日 K 新增 {result['daily_rows_added']:,} 行，"
        f"复权事件新增 {result['event_rows_added']:,} 行；"
        f"库内范围 {result['database_min_date']} 至 "
        f"{result['database_max_date']}。"
    )
    print(
        "提示：Financial-API 当前公开接口仅有最新估值快照，"
        "没有历史逐日估值。本脚本未回填 PE/PB/PCF；"
        "基本面策略回测需另行提供无未来函数的历史估值数据。"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        FileNotFoundError,
        json.JSONDecodeError,
        OSError,
        RuntimeError,
        sqlite3.Error,
        ValueError,
    ) as error:
        print(f"获取三年数据失败: {error}", file=sys.stderr)
        raise SystemExit(1) from error
