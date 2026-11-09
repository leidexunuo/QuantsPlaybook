"""Read-only access helpers for the Sequoia A-share SQLite database."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any


DEFAULT_DATABASE = Path(__file__).resolve().parent / "a_share_market.sqlite3"
DAILY_COLUMNS = (
    "thscode",
    "trade_date",
    "date_ms",
    "open_price",
    "high_price",
    "low_price",
    "close_price",
    "volume",
    "turnover",
    "currency",
    "interval",
    "adjusted",
)
EVENT_COLUMNS = (
    "event_id",
    "thscode",
    "ticker",
    "ex_date",
    "ex_date_ms",
    "dividend_per_share",
    "per_share_bonus",
    "allotment_ratio",
    "allotment_price",
    "currency",
)


class AShareDataStore:
    """A small, read-only query interface suitable for other project modules."""

    def __init__(self, database: str | Path = DEFAULT_DATABASE) -> None:
        self.database = Path(database).resolve()
        if not self.database.is_file():
            raise FileNotFoundError(f"SQLite 数据库不存在: {self.database}")
        uri = f"{self.database.as_uri()}?mode=ro"
        self.connection = sqlite3.connect(uri, uri=True)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA query_only=ON")

    def __enter__(self) -> "AShareDataStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        self.connection.close()

    def query(
        self, sql: str, parameters: Sequence[Any] = ()
    ) -> list[dict[str, Any]]:
        """Run a read-only SQL query and return rows as dictionaries."""
        return [dict(row) for row in self.connection.execute(sql, parameters)]

    @staticmethod
    def _symbols_clause(
        symbols: str | Iterable[str] | None, column: str = "thscode"
    ) -> tuple[str, list[str]]:
        if symbols is None:
            return "", []
        symbol_list = [symbols] if isinstance(symbols, str) else list(symbols)
        if not symbol_list:
            return " AND 0", []
        placeholders = ", ".join("?" for _ in symbol_list)
        return f" AND {column} IN ({placeholders})", symbol_list

    def get_daily_k(
        self,
        symbols: str | Iterable[str] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        columns: Sequence[str] = DAILY_COLUMNS,
    ) -> list[dict[str, Any]]:
        """Fetch raw, unadjusted daily bars ordered by symbol and date."""
        unknown = set(columns) - set(DAILY_COLUMNS)
        if unknown:
            raise ValueError(f"不支持的日 K 字段: {sorted(unknown)}")

        conditions = ["1=1"]
        parameters: list[Any] = []
        symbol_sql, symbol_parameters = self._symbols_clause(symbols)
        if start_date:
            conditions.append("trade_date >= ?")
            parameters.append(start_date)
        if end_date:
            conditions.append("trade_date <= ?")
            parameters.append(end_date)
        parameters.extend(symbol_parameters)
        sql = (
            f"SELECT {', '.join(columns)} FROM daily_k "
            f"WHERE {' AND '.join(conditions)}{symbol_sql} "
            "ORDER BY thscode, trade_date"
        )
        return self.query(sql, parameters)

    def get_adjustment_events(
        self,
        symbols: str | Iterable[str] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch dividend, bonus-share, and rights-issue events."""
        conditions = ["1=1"]
        parameters: list[Any] = []
        symbol_sql, symbol_parameters = self._symbols_clause(symbols)
        if start_date:
            conditions.append("ex_date >= ?")
            parameters.append(start_date)
        if end_date:
            conditions.append("ex_date <= ?")
            parameters.append(end_date)
        parameters.extend(symbol_parameters)
        sql = (
            f"SELECT {', '.join(EVENT_COLUMNS)} FROM adjustment_events "
            f"WHERE {' AND '.join(conditions)}{symbol_sql} "
            "ORDER BY thscode, ex_date, event_id"
        )
        return self.query(sql, parameters)

    def coverage(self) -> list[dict[str, Any]]:
        """Return row count, symbol count, and date coverage by dataset."""
        return self.query("SELECT * FROM v_data_coverage ORDER BY dataset")

    def to_dataframe(
        self, rows: Iterable[dict[str, Any]]
    ):  # return type intentionally optional
        """Convert query output to pandas when pandas is available."""
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError("to_dataframe 需要 pandas") from exc
        return pd.DataFrame.from_records(rows)
