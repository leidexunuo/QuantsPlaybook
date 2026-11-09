"""Import Sequoia A-share Parquet snapshots into SQLite.

The importer is idempotent at the source-file level and merges daily bars by
``(thscode, trade_date)``.  A newer 10-day snapshot therefore overwrites the
overlapping rows from the 10-year snapshot.
"""

from __future__ import annotations

import argparse
from contextlib import closing
import hashlib
import sqlite3
import sys
import time
from pathlib import Path
from typing import Iterable, Iterator, Sequence

try:
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError as exc:  # pragma: no cover - exercised by users' environment
    raise SystemExit(
        "缺少导入依赖，请先执行: python -m pip install -r Sequoia/requirements.txt"
    ) from exc


HERE = Path(__file__).resolve().parent
DEFAULT_DATABASE = HERE / "a_share_market.sqlite3"
DAILY_10Y_PATTERN = "a_share_daily_k_1d_none_10y_*.parquet"
DAILY_10D_PATTERN = "a_share_daily_k_1d_none_10d_*.parquet"
EVENT_PATTERN = "a_share_adjustment_factors_event_none_all_*.parquet"

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS source_imports (
    source_id       INTEGER PRIMARY KEY,
    source_name     TEXT NOT NULL,
    source_kind     TEXT NOT NULL
                    CHECK (source_kind IN ('daily_10y', 'daily_10d', 'adjustment_events')),
    sha256          TEXT NOT NULL UNIQUE,
    file_size       INTEGER NOT NULL,
    parquet_rows    INTEGER NOT NULL,
    rows_written    INTEGER NOT NULL DEFAULT 0,
    min_data_date   TEXT,
    max_data_date   TEXT,
    imported_at     TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    status          TEXT NOT NULL DEFAULT 'importing'
                    CHECK (status IN ('importing', 'complete', 'failed'))
);

CREATE TABLE IF NOT EXISTS daily_k (
    thscode     TEXT NOT NULL,
    trade_date  TEXT NOT NULL,
    date_ms     INTEGER NOT NULL,
    open_price  REAL NOT NULL,
    high_price  REAL NOT NULL,
    low_price   REAL NOT NULL,
    close_price REAL NOT NULL,
    volume      REAL NOT NULL,
    turnover    REAL NOT NULL,
    currency    TEXT NOT NULL,
    interval    TEXT NOT NULL,
    adjusted    TEXT NOT NULL,
    source_id   INTEGER NOT NULL,
    PRIMARY KEY (thscode, trade_date),
    FOREIGN KEY (source_id) REFERENCES source_imports(source_id)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS adjustment_events (
    event_id            INTEGER PRIMARY KEY,
    thscode             TEXT NOT NULL,
    ticker              TEXT NOT NULL,
    ex_date             TEXT NOT NULL,
    ex_date_ms          INTEGER NOT NULL,
    dividend_per_share  REAL NOT NULL,
    per_share_bonus     REAL NOT NULL,
    allotment_ratio     REAL NOT NULL,
    allotment_price     REAL NOT NULL,
    currency            TEXT NOT NULL,
    source_id           INTEGER NOT NULL,
    FOREIGN KEY (source_id) REFERENCES source_imports(source_id),
    UNIQUE (
        thscode, ex_date_ms, dividend_per_share, per_share_bonus,
        allotment_ratio, allotment_price, currency
    )
);

CREATE VIEW IF NOT EXISTS v_data_coverage AS
SELECT
    'daily_k' AS dataset,
    COUNT(*) AS row_count,
    COUNT(DISTINCT thscode) AS symbol_count,
    MIN(trade_date) AS min_date,
    MAX(trade_date) AS max_date
FROM daily_k
UNION ALL
SELECT
    'adjustment_events',
    COUNT(*),
    COUNT(DISTINCT thscode),
    MIN(ex_date),
    MAX(ex_date)
FROM adjustment_events;
"""

INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_daily_k_trade_date
ON daily_k (trade_date, thscode);

CREATE INDEX IF NOT EXISTS idx_adjustment_events_symbol_date
ON adjustment_events (thscode, ex_date);

CREATE INDEX IF NOT EXISTS idx_adjustment_events_date
ON adjustment_events (ex_date, thscode);
"""

DAILY_UPSERT_SQL = """
INSERT INTO daily_k (
    thscode, trade_date, date_ms, open_price, high_price, low_price,
    close_price, volume, turnover, currency, interval, adjusted, source_id
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(thscode, trade_date) DO UPDATE SET
    date_ms=excluded.date_ms,
    open_price=excluded.open_price,
    high_price=excluded.high_price,
    low_price=excluded.low_price,
    close_price=excluded.close_price,
    volume=excluded.volume,
    turnover=excluded.turnover,
    currency=excluded.currency,
    interval=excluded.interval,
    adjusted=excluded.adjusted,
    source_id=excluded.source_id
"""

EVENT_INSERT_SQL = """
INSERT OR IGNORE INTO adjustment_events (
    thscode, ticker, ex_date, ex_date_ms, dividend_per_share,
    per_share_bonus, allotment_ratio, allotment_price, currency, source_id
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


def sha256sum(path: Path, block_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def to_dates(milliseconds: pa.Array) -> list[str]:
    # The source epoch represents an instant at local market midnight.  UTC
    # formatting would shift every date back one day; China also observed DST
    # in part of the historical event range, so a fixed +08:00 is insufficient.
    timestamps = pd.to_datetime(milliseconds.to_numpy(), unit="ms", utc=True)
    return timestamps.tz_convert("Asia/Shanghai").strftime("%Y-%m-%d").tolist()


def batched_rows(
    parquet_path: Path,
    columns: Sequence[str],
    source_id: int,
    batch_size: int,
) -> Iterator[tuple[list[tuple], str, str]]:
    parquet = pq.ParquetFile(parquet_path)
    is_daily = "date_ms" in columns
    date_column = "date_ms" if is_daily else "ex_date_ms"

    for batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
        values = batch.to_pydict()
        dates = to_dates(batch.column(batch.schema.get_field_index(date_column)))
        if is_daily:
            rows = list(
                zip(
                    values["thscode"],
                    dates,
                    values["date_ms"],
                    values["open_price"],
                    values["high_price"],
                    values["low_price"],
                    values["close_price"],
                    values["volume"],
                    values["turnover"],
                    values["currency"],
                    values["interval"],
                    values["adjusted"],
                    [source_id] * batch.num_rows,
                )
            )
        else:
            rows = list(
                zip(
                    values["thscode"],
                    values["ticker"],
                    dates,
                    values["ex_date_ms"],
                    values["dividend_per_share"],
                    values["per_share_bonus"],
                    values["allotment_ratio"],
                    values["allotment_price"],
                    values["currency"],
                    [source_id] * batch.num_rows,
                )
            )
        yield rows, min(dates), max(dates)


def start_import(
    connection: sqlite3.Connection, path: Path, source_kind: str, digest: str
) -> tuple[int, bool]:
    existing = connection.execute(
        "SELECT source_id, status FROM source_imports WHERE sha256 = ?", (digest,)
    ).fetchone()
    if existing and existing[1] == "complete":
        return int(existing[0]), True
    if existing:
        connection.execute(
            "DELETE FROM daily_k WHERE source_id = ?", (existing[0],)
        )
        connection.execute(
            "DELETE FROM adjustment_events WHERE source_id = ?", (existing[0],)
        )
        connection.execute(
            "DELETE FROM source_imports WHERE source_id = ?", (existing[0],)
        )
        connection.commit()

    parquet_rows = pq.ParquetFile(path).metadata.num_rows
    cursor = connection.execute(
        """
        INSERT INTO source_imports
            (source_name, source_kind, sha256, file_size, parquet_rows)
        VALUES (?, ?, ?, ?, ?)
        """,
        (path.name, source_kind, digest, path.stat().st_size, parquet_rows),
    )
    connection.commit()
    return int(cursor.lastrowid), False


def import_parquet(
    connection: sqlite3.Connection,
    path: Path,
    source_kind: str,
    columns: Sequence[str],
    sql: str,
    batch_size: int,
    replace_events: bool = False,
) -> None:
    print(f"计算校验值: {path.name}", flush=True)
    digest = sha256sum(path)
    source_id, already_imported = start_import(connection, path, source_kind, digest)
    if already_imported:
        print(f"跳过已导入文件: {path.name}", flush=True)
        return

    if replace_events:
        connection.execute("DELETE FROM adjustment_events")

    started = time.monotonic()
    rows_seen = 0
    min_date: str | None = None
    max_date: str | None = None
    try:
        for rows, batch_min, batch_max in batched_rows(
            path, columns, source_id, batch_size
        ):
            connection.executemany(sql, rows)
            if not replace_events:
                connection.commit()
            rows_seen += len(rows)
            min_date = batch_min if min_date is None else min(min_date, batch_min)
            max_date = batch_max if max_date is None else max(max_date, batch_max)
            elapsed = max(time.monotonic() - started, 0.001)
            print(
                f"\r{path.name}: {rows_seen:,} 行 "
                f"({rows_seen / elapsed:,.0f} 行/秒)",
                end="",
                flush=True,
            )
        connection.execute(
            """
            UPDATE source_imports
            SET rows_written = ?, min_data_date = ?, max_data_date = ?,
                status = 'complete'
            WHERE source_id = ?
            """,
            (rows_seen, min_date, max_date, source_id),
        )
        connection.commit()
        print()
    except BaseException:
        connection.rollback()
        connection.execute(
            "UPDATE source_imports SET status = 'failed' WHERE source_id = ?",
            (source_id,),
        )
        connection.commit()
        raise


def latest_file(data_dir: Path, pattern: str) -> Path:
    matches = sorted(data_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"没有找到匹配文件: {data_dir / pattern}")
    return matches[-1]


def configure_connection(database: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(database)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("PRAGMA temp_store=MEMORY")
    connection.execute("PRAGMA cache_size=-262144")
    connection.execute("PRAGMA foreign_keys=ON")
    connection.executescript(SCHEMA_SQL)
    connection.commit()
    return connection


def build_database(
    database: Path = DEFAULT_DATABASE,
    data_dir: Path = HERE,
    batch_size: int = 100_000,
) -> None:
    database.parent.mkdir(parents=True, exist_ok=True)
    daily_10y = latest_file(data_dir, DAILY_10Y_PATTERN)
    daily_10d = latest_file(data_dir, DAILY_10D_PATTERN)
    events = latest_file(data_dir, EVENT_PATTERN)

    daily_columns = [
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
    event_columns = [
        "thscode",
        "ticker",
        "ex_date_ms",
        "dividend_per_share",
        "per_share_bonus",
        "allotment_ratio",
        "allotment_price",
        "currency",
    ]

    with closing(configure_connection(database)) as connection:
        import_parquet(
            connection,
            daily_10y,
            "daily_10y",
            daily_columns,
            DAILY_UPSERT_SQL,
            batch_size,
        )
        import_parquet(
            connection,
            daily_10d,
            "daily_10d",
            daily_columns,
            DAILY_UPSERT_SQL,
            batch_size,
        )
        import_parquet(
            connection,
            events,
            "adjustment_events",
            event_columns,
            EVENT_INSERT_SQL,
            batch_size,
            replace_events=True,
        )
        print("创建查询索引...", flush=True)
        connection.executescript(INDEX_SQL)
        connection.execute("PRAGMA optimize")
        connection.commit()

    print(f"SQLite 数据库已生成: {database}")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database",
        type=Path,
        default=DEFAULT_DATABASE,
        help=f"SQLite 输出路径（默认: {DEFAULT_DATABASE}）",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=HERE,
        help=f"Parquet 所在目录（默认: {HERE}）",
    )
    parser.add_argument("--batch-size", type=int, default=100_000)
    return parser.parse_args(argv)


if __name__ == "__main__":
    arguments = parse_args()
    try:
        build_database(
            database=arguments.database.resolve(),
            data_dir=arguments.data_dir.resolve(),
            batch_size=arguments.batch_size,
        )
    except (FileNotFoundError, sqlite3.Error, ValueError) as error:
        print(f"建库失败: {error}", file=sys.stderr)
        raise SystemExit(1) from error
