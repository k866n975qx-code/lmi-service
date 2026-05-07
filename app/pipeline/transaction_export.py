from __future__ import annotations

import csv
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path


EXPORT_COLUMNS = [
    "Event",
    "Date",
    "Symbol",
    "Price",
    "Quantity",
    "Currency",
    "FeeTax",
    "Exchange",
    "FeeCurrency",
    "DoNotAdjustCash",
    "Note",
]


@dataclass(frozen=True)
class TransactionExportResult:
    output_path: str
    row_count: int


_EXPORT_SQL = """
WITH tx_enriched AS (
    SELECT
        t.*,
        (
            SELECT AVG(p.amount)
            FROM dividend_events_provider p
            WHERE UPPER(p.symbol) = UPPER(t.symbol)
              AND p.ex_date = (
                  SELECT MAX(p2.ex_date)
                  FROM dividend_events_provider p2
                  WHERE UPPER(p2.symbol) = UPPER(t.symbol)
                    AND p2.ex_date <= t.date
              )
        ) AS provider_div_per_share
    FROM investment_transactions t
),
mapped AS (
    SELECT
        CASE
            WHEN lower(transaction_type) IN ('contribution', 'margin_borrow') THEN 'CASH_IN'
            WHEN lower(transaction_type) IN ('withdrawal', 'margin_repay') THEN 'CASH_OUT'
            WHEN lower(transaction_type) IN ('buy', 'buy_shares', 'reinvest', 'reinvestment') THEN 'BUY'
            WHEN lower(transaction_type) IN ('sell', 'sell_shares', 'redemption') THEN 'SELL'
            WHEN lower(transaction_type) = 'dividend' THEN 'DIVIDEND'
            WHEN lower(transaction_type) = 'interest' THEN 'CASH_GAIN'
            WHEN lower(transaction_type) IN ('margin_interest', 'fee') THEN 'FEE'
            ELSE NULL
        END AS Event,
        date AS Date,
        CASE
            WHEN lower(transaction_type) = 'interest' THEN ''
            ELSE UPPER(COALESCE(symbol, ''))
        END AS Symbol,
        CASE
            WHEN lower(transaction_type) IN ('buy', 'buy_shares', 'reinvest', 'reinvestment', 'sell', 'sell_shares', 'redemption')
                THEN ABS(COALESCE(price, CASE WHEN quantity IS NOT NULL AND quantity != 0 THEN amount / quantity ELSE 0 END, 0))
            WHEN lower(transaction_type) = 'dividend'
                THEN ROUND(COALESCE(provider_div_per_share, 0), 6)
            WHEN lower(transaction_type) IN ('contribution', 'withdrawal', 'margin_borrow', 'margin_repay', 'interest')
                THEN 1
            WHEN lower(transaction_type) IN ('margin_interest', 'fee')
                THEN 0
            ELSE 0
        END AS Price,
        CASE
            WHEN lower(transaction_type) IN ('buy', 'buy_shares', 'reinvest', 'reinvestment', 'sell', 'sell_shares', 'redemption')
                THEN ABS(COALESCE(quantity, 0))
            WHEN lower(transaction_type) IN ('contribution', 'withdrawal', 'margin_borrow', 'margin_repay', 'interest')
                THEN ABS(COALESCE(amount, 0))
            WHEN lower(transaction_type) = 'dividend'
                THEN ABS(COALESCE(amount, 0))
            WHEN lower(transaction_type) IN ('margin_interest', 'fee')
                THEN 0
            ELSE 0
        END AS Quantity,
        UPPER(COALESCE(currency, 'USD')) AS Currency,
        CASE
            WHEN lower(transaction_type) IN ('margin_interest', 'fee')
                THEN ABS(COALESCE(NULLIF(fees, 0), amount, 0))
            ELSE ABS(COALESCE(fees, 0))
        END AS FeeTax,
        '' AS Exchange,
        '' AS FeeCurrency,
        '' AS DoNotAdjustCash,
        COALESCE(name, '') AS Note
    FROM tx_enriched
)
SELECT
    Event,
    Date,
    Symbol,
    Price,
    Quantity,
    Currency,
    FeeTax,
    Exchange,
    FeeCurrency,
    DoNotAdjustCash,
    Note
FROM mapped
WHERE Event IS NOT NULL
ORDER BY Date, Event, Symbol, Note
"""


def export_transactions_csv(
    conn: sqlite3.Connection,
    output_path: str | os.PathLike[str],
) -> TransactionExportResult:
    """Export normalized investment transactions in the portfolio import CSV shape."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")

    cur = conn.execute(_EXPORT_SQL)
    rows = cur.fetchall()

    with tmp_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(EXPORT_COLUMNS)
        writer.writerows(rows)

    tmp_path.replace(path)
    return TransactionExportResult(output_path=str(path), row_count=len(rows))
