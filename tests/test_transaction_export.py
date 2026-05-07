import csv
import sqlite3

from app.pipeline.transaction_export import export_transactions_csv


def _make_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute(
        """
        CREATE TABLE investment_transactions (
            lm_transaction_id TEXT PRIMARY KEY,
            date TEXT,
            transaction_type TEXT,
            symbol TEXT,
            amount REAL,
            quantity REAL,
            price REAL,
            currency TEXT,
            fees REAL,
            name TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE dividend_events_provider (
            symbol TEXT,
            ex_date TEXT,
            amount REAL
        )
        """
    )
    return conn


def _read_rows(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_export_transactions_csv_exports_all_symbols_and_cash_events(tmp_path):
    conn = _make_conn()
    conn.executemany(
        """
        INSERT INTO investment_transactions (
            lm_transaction_id, date, transaction_type, symbol, amount, quantity, price, currency, fees, name
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            ("1", "2026-01-01", "buy", "SPYI", 100.0, 2.0, 50.0, "usd", 0.0, "SPYI buy"),
            ("2", "2026-01-02", "sell", "JEPI", -60.0, -1.0, 60.0, "usd", 0.0, "JEPI sell"),
            ("3", "2026-01-03", "contribution", None, -500.0, 0.0, 0.0, "usd", 0.0, "ACH deposit"),
            ("4", "2026-01-04", "withdrawal", None, 25.0, 0.0, 0.0, "usd", 0.0, "ACH withdrawal"),
            ("5", "2026-01-05", "interest", None, -0.11, 0.0, 0.0, "usd", 0.0, "Securities lending"),
            ("6", "2026-01-06", "margin_interest", None, 4.15, 0.0, 0.0, "usd", 0.0, "Margin interest"),
        ],
    )

    output_path = tmp_path / "transactions.csv"
    result = export_transactions_csv(conn, output_path)
    rows = _read_rows(output_path)

    assert result.row_count == 6
    assert {row["Symbol"] for row in rows if row["Symbol"]} == {"JEPI", "SPYI"}
    assert [row["Event"] for row in rows] == ["BUY", "SELL", "CASH_IN", "CASH_OUT", "CASH_GAIN", "FEE"]

    cash_gain = next(row for row in rows if row["Event"] == "CASH_GAIN")
    assert cash_gain["Symbol"] == ""
    assert cash_gain["Price"] == "1"
    assert cash_gain["Quantity"] == "0.11"

    fee = next(row for row in rows if row["Event"] == "FEE")
    assert fee["Price"] == "0"
    assert fee["Quantity"] == "0"
    assert fee["FeeTax"] == "4.15"


def test_export_transactions_csv_uses_cash_amount_for_dividend_quantity(tmp_path):
    conn = _make_conn()
    conn.execute(
        """
        INSERT INTO dividend_events_provider (symbol, ex_date, amount)
        VALUES ('TRIN', '2026-03-13', 0.17)
        """
    )
    conn.execute(
        """
        INSERT INTO investment_transactions (
            lm_transaction_id, date, transaction_type, symbol, amount, quantity, price, currency, fees, name
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "div1",
            "2026-03-31",
            "dividend",
            "TRIN",
            -67.04,
            0.0,
            0.0,
            "usd",
            0.0,
            "Dividend of 896442308 $67.04 received. - DIVIDEND",
        ),
    )

    output_path = tmp_path / "transactions.csv"
    export_transactions_csv(conn, output_path)
    rows = _read_rows(output_path)

    assert rows == [
        {
            "Event": "DIVIDEND",
            "Date": "2026-03-31",
            "Symbol": "TRIN",
            "Price": "0.17",
            "Quantity": "67.04",
            "Currency": "USD",
            "FeeTax": "0.0",
            "Exchange": "",
            "FeeCurrency": "",
            "DoNotAdjustCash": "",
            "Note": "Dividend of 896442308 $67.04 received. - DIVIDEND",
        }
    ]
