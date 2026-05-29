from datetime import date
import sqlite3

from app.pipeline.periods import _generate_period_activity


def _make_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE investment_transactions (
            lm_transaction_id TEXT PRIMARY KEY,
            date TEXT,
            amount REAL,
            transaction_type TEXT,
            symbol TEXT,
            plaid_account_id TEXT,
            name TEXT,
            external_flow_amount REAL,
            income_flow_amount REAL,
            financing_flow_amount REAL,
            economic_bucket TEXT,
            quantity REAL,
            trading_flow_amount REAL,
            fee_flow_amount REAL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE realized_trade_ledger (
            lm_transaction_id TEXT,
            date TEXT,
            symbol TEXT,
            gross_proceeds REAL,
            net_proceeds REAL,
            realized_cost_basis REAL,
            realized_pnl REAL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE daily_holdings (
            as_of_date_local TEXT,
            symbol TEXT,
            shares REAL
        )
        """
    )
    return conn


def test_generate_period_activity_does_not_double_count_trades():
    conn = _make_conn()
    conn.executemany(
        """
        INSERT INTO investment_transactions (
            lm_transaction_id,
            date,
            amount,
            transaction_type,
            symbol,
            plaid_account_id,
            name,
            external_flow_amount,
            income_flow_amount,
            financing_flow_amount,
            economic_bucket,
            quantity,
            trading_flow_amount,
            fee_flow_amount
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                "buy-1",
                "2026-02-03",
                100.0,
                "buy",
                "BTCI",
                "acct-1",
                "BTCI buy",
                0.0,
                0.0,
                0.0,
                "trade",
                2.0,
                -100.0,
                0.0,
            ),
            (
                "sell-1",
                "2026-02-04",
                60.0,
                "sell",
                "BTCI",
                "acct-1",
                "BTCI sell",
                0.0,
                0.0,
                0.0,
                "trade",
                -1.0,
                60.0,
                0.0,
            ),
        ],
    )
    conn.execute(
        """
        INSERT INTO realized_trade_ledger (
            lm_transaction_id,
            date,
            symbol,
            gross_proceeds,
            net_proceeds,
            realized_cost_basis,
            realized_pnl
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        ("sell-1", "2026-02-04", "BTCI", 60.0, 60.0, 40.0, 20.0),
    )

    activity = _generate_period_activity(conn, date(2026, 2, 1), date(2026, 2, 7))
    trades = activity["trades"]
    by_symbol = trades["by_symbol"]["BTCI"]

    assert trades["buy_count"] == 1
    assert trades["sell_count"] == 1
    assert trades["total_count"] == 2
    assert by_symbol["buy_count"] == 1
    assert by_symbol["sell_count"] == 1
    assert by_symbol["shares_bought"] == 2.0
    assert by_symbol["shares_sold"] == 1.0
    assert by_symbol["realized_capital_pnl"] == 20.0
