from __future__ import annotations

import sqlite3
from collections import defaultdict
from datetime import date, datetime, timedelta

from ..config import settings

BUY_TYPES = {"buy", "buy_shares", "reinvest", "reinvestment"}
SELL_TYPES = {"sell", "sell_shares", "redemption"}
INCOME_TYPES = {"dividend", "interest"}
FINANCING_TYPES = {"margin_interest", "margin_borrow", "margin_repay"}
EXTERNAL_TYPES = {"contribution", "withdrawal"}
FEE_TYPES = {"fee"}


def _allowed_plaid_ids():
    raw = settings.lm_plaid_account_ids
    if not raw:
        return None
    return {part.strip() for part in raw.split(",") if part.strip()}


def _coerce_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text).date()
    except ValueError:
        pass
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def _canonical_tx_type(tx_type: str | None, name: str | None, amount) -> str:
    normalized = (tx_type or "").strip().lower()
    text = (name or "").lower()
    amount_value = _coerce_float(amount)

    if normalized == "margin_repay" and "interest" in text:
        return "margin_interest"
    if normalized == "interest":
        if "margin" in text:
            return "margin_interest"
        if amount_value is not None and amount_value > 0:
            return "margin_interest"
    return normalized


def _classify_cashflows(tx_type: str, amount, fees) -> dict[str, object]:
    amount_value = abs(_coerce_float(amount) or 0.0)
    fee_value = abs(_coerce_float(fees) or 0.0)

    bucket = "other"
    cash_flow = 0.0
    external_flow = 0.0
    trading_flow = 0.0
    income_flow = 0.0
    financing_flow = 0.0
    fee_flow = 0.0

    if tx_type in BUY_TYPES:
        bucket = "trade"
        cash_flow = -amount_value
        trading_flow = -amount_value
    elif tx_type in SELL_TYPES:
        bucket = "trade"
        cash_flow = amount_value
        trading_flow = amount_value
    elif tx_type == "dividend":
        bucket = "income"
        cash_flow = amount_value
        income_flow = amount_value
    elif tx_type == "interest":
        bucket = "income"
        cash_flow = amount_value
        income_flow = amount_value
    elif tx_type == "margin_interest":
        bucket = "financing"
        cash_flow = -amount_value
        financing_flow = -amount_value
    elif tx_type == "margin_borrow":
        bucket = "financing"
        cash_flow = amount_value
        financing_flow = amount_value
    elif tx_type == "margin_repay":
        bucket = "financing"
        cash_flow = -amount_value
        financing_flow = -amount_value
    elif tx_type == "contribution":
        bucket = "external"
        cash_flow = amount_value
        external_flow = amount_value
    elif tx_type == "withdrawal":
        bucket = "external"
        cash_flow = -amount_value
        external_flow = -amount_value
    elif tx_type == "fee":
        bucket = "expense"
        cash_flow = -amount_value
        fee_flow = -amount_value

    if fee_value > 0 and tx_type not in BUY_TYPES and tx_type not in SELL_TYPES:
        fee_flow -= fee_value
        cash_flow -= fee_value
        if bucket == "other":
            bucket = "expense"

    if cash_flow > 0:
        direction = "inflow"
    elif cash_flow < 0:
        direction = "outflow"
    else:
        direction = "flat"

    return {
        "economic_bucket": bucket,
        "cash_flow_direction": direction,
        "cash_flow_amount": round(cash_flow, 6),
        "external_flow_amount": round(external_flow, 6),
        "trading_flow_amount": round(trading_flow, 6),
        "income_flow_amount": round(income_flow, 6),
        "financing_flow_amount": round(financing_flow, 6),
        "fee_flow_amount": round(fee_flow, 6),
    }


def backfill_transaction_economics(conn: sqlite3.Connection) -> int:
    cur = conn.cursor()
    rows = cur.execute(
        """
        SELECT lm_transaction_id, transaction_type, name, amount, fees
        FROM investment_transactions
        """
    ).fetchall()

    updates = []
    for lm_transaction_id, tx_type, name, amount, fees in rows:
        normalized = _canonical_tx_type(tx_type, name, amount)
        flows = _classify_cashflows(normalized, amount, fees)
        updates.append(
            (
                normalized,
                flows["economic_bucket"],
                flows["cash_flow_direction"],
                flows["cash_flow_amount"],
                flows["external_flow_amount"],
                flows["trading_flow_amount"],
                flows["income_flow_amount"],
                flows["financing_flow_amount"],
                flows["fee_flow_amount"],
                lm_transaction_id,
            )
        )

    cur.executemany(
        """
        UPDATE investment_transactions
        SET
          transaction_type=?,
          economic_bucket=?,
          cash_flow_direction=?,
          cash_flow_amount=?,
          external_flow_amount=?,
          trading_flow_amount=?,
          income_flow_amount=?,
          financing_flow_amount=?,
          fee_flow_amount=?
        WHERE lm_transaction_id=?
        """,
        updates,
    )
    conn.commit()
    return len(updates)


def rebuild_realized_trade_ledger(conn: sqlite3.Connection, run_id: str) -> tuple[int, int]:
    cur = conn.cursor()
    rows = cur.execute(
        """
        SELECT
          lm_transaction_id,
          plaid_account_id,
          date,
          transaction_datetime,
          amount,
          transaction_type,
          quantity,
          price,
          fees,
          symbol,
          name,
          source,
          pulled_at_utc
        FROM investment_transactions
        ORDER BY COALESCE(transaction_datetime, date) ASC, lm_transaction_id ASC
        """
    ).fetchall()

    allowed = _allowed_plaid_ids()
    lots: dict[str, list[dict[str, object]]] = defaultdict(list)
    realized_rows: list[tuple] = []
    realized_lot_rows: list[tuple] = []

    cur.execute("DELETE FROM realized_trade_lots")
    cur.execute("DELETE FROM realized_trade_ledger")

    for row in rows:
        (
            lm_transaction_id,
            plaid_account_id,
            tx_date_raw,
            tx_datetime,
            amount,
            tx_type_raw,
            quantity,
            price,
            fees,
            symbol,
            name,
            source,
            pulled_at_utc,
        ) = row

        if allowed and (plaid_account_id is None or str(plaid_account_id) not in allowed):
            continue

        tx_type = _canonical_tx_type(tx_type_raw, name, amount)
        if tx_type not in BUY_TYPES and tx_type not in SELL_TYPES:
            continue
        if not symbol:
            continue

        tx_date = _parse_date(tx_datetime) or _parse_date(tx_date_raw)
        if tx_date is None:
            continue

        quantity_value = abs(_coerce_float(quantity) or 0.0)
        if quantity_value <= 0:
            continue

        amount_value = abs(_coerce_float(amount) or 0.0)
        fee_value = abs(_coerce_float(fees) or 0.0)
        price_value = _coerce_float(price)
        symbol_upper = str(symbol).upper()

        if tx_type in BUY_TYPES:
            principal_cost = amount_value if amount_value > 0 else quantity_value * float(price_value or 0.0)
            total_cost = principal_cost + fee_value
            lots[symbol_upper].append(
                {
                    "qty": quantity_value,
                    "cost": total_cost,
                    "date": tx_date.isoformat(),
                    "lm_transaction_id": lm_transaction_id,
                }
            )
            continue

        gross_proceeds = amount_value if amount_value > 0 else quantity_value * float(price_value or 0.0)
        net_proceeds = gross_proceeds - fee_value
        sell_price = price_value if price_value is not None else (gross_proceeds / quantity_value if quantity_value else None)

        remaining = quantity_value
        matched_shares = 0.0
        realized_cost_total = 0.0
        weighted_holding_days = 0.0
        lots_closed_count = 0
        lot_index = 0

        while remaining > 1e-9 and lots[symbol_upper]:
            lot = lots[symbol_upper][0]
            lot_qty = float(lot["qty"])
            lot_cost = float(lot["cost"])
            take = min(remaining, lot_qty)
            if take <= 0:
                lots[symbol_upper].pop(0)
                continue

            cost_per_share = lot_cost / lot_qty if lot_qty else 0.0
            realized_cost_piece = cost_per_share * take
            proceeds_ratio = take / quantity_value if quantity_value else 0.0
            gross_piece = gross_proceeds * proceeds_ratio
            fee_piece = fee_value * proceeds_ratio
            net_piece = net_proceeds * proceeds_ratio
            realized_pnl_piece = net_piece - realized_cost_piece
            acquisition_date = _parse_date(str(lot.get("date")))
            holding_days = (tx_date - acquisition_date).days if acquisition_date else None

            lot_index += 1
            realized_lot_rows.append(
                (
                    lm_transaction_id,
                    lot_index,
                    symbol_upper,
                    lot.get("lm_transaction_id"),
                    acquisition_date.isoformat() if acquisition_date else None,
                    tx_date.isoformat(),
                    round(take, 6),
                    round(cost_per_share, 6) if cost_per_share is not None else None,
                    round(sell_price, 6) if sell_price is not None else None,
                    round(gross_piece, 6),
                    round(fee_piece, 6),
                    round(net_piece, 6),
                    round(realized_cost_piece, 6),
                    round(realized_pnl_piece, 6),
                    float(holding_days) if holding_days is not None else None,
                )
            )

            lot["qty"] = lot_qty - take
            lot["cost"] = lot_cost - realized_cost_piece
            remaining -= take
            matched_shares += take
            realized_cost_total += realized_cost_piece
            lots_closed_count += 1
            if holding_days is not None:
                weighted_holding_days += holding_days * take

            if float(lot["qty"]) <= 1e-9:
                lots[symbol_upper].pop(0)

        unmatched_shares = max(0.0, remaining)
        matched_complete = 1 if unmatched_shares <= 1e-9 else 0
        realized_pnl_total = net_proceeds - realized_cost_total
        realized_pct = (realized_pnl_total / realized_cost_total * 100.0) if realized_cost_total else None
        weighted_holding_period_days = (weighted_holding_days / matched_shares) if matched_shares > 0 else None

        realized_rows.append(
            (
                lm_transaction_id,
                tx_date.isoformat(),
                tx_datetime,
                symbol_upper,
                round(quantity_value, 6),
                round(sell_price, 6) if sell_price is not None else None,
                round(gross_proceeds, 6),
                round(fee_value, 6),
                round(net_proceeds, 6),
                round(realized_cost_total, 6),
                round(realized_pnl_total, 6),
                round(realized_pct, 6) if realized_pct is not None else None,
                round(matched_shares, 6),
                round(unmatched_shares, 6),
                matched_complete,
                lots_closed_count,
                round(weighted_holding_period_days, 6) if weighted_holding_period_days is not None else None,
                source or "derived_fifo",
                run_id,
                pulled_at_utc or tx_date.isoformat(),
            )
        )

    cur.executemany(
        """
        INSERT INTO realized_trade_ledger (
          lm_transaction_id,
          date,
          transaction_datetime,
          symbol,
          shares_sold,
          sell_price,
          gross_proceeds,
          fees,
          net_proceeds,
          realized_cost_basis,
          realized_pnl,
          realized_pnl_pct,
          matched_shares,
          unmatched_shares,
          matched_complete,
          lots_closed_count,
          weighted_holding_period_days,
          source,
          run_id,
          pulled_at_utc
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        realized_rows,
    )
    cur.executemany(
        """
        INSERT INTO realized_trade_lots (
          lm_transaction_id,
          lot_index,
          symbol,
          acquisition_lm_transaction_id,
          acquisition_date,
          disposal_date,
          shares_closed,
          buy_price_effective,
          sell_price,
          gross_proceeds,
          sell_fees,
          net_proceeds,
          realized_cost_basis,
          realized_pnl,
          holding_period_days
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        realized_lot_rows,
    )
    conn.commit()
    return len(realized_rows), len(realized_lot_rows)


def _min_transaction_date(conn: sqlite3.Connection) -> date | None:
    row = conn.execute("SELECT MIN(date) FROM investment_transactions").fetchone()
    return _parse_date(row[0]) if row and row[0] else None


def _window_specs(conn: sqlite3.Connection, as_of_date: date) -> dict[str, tuple[date, date]]:
    min_date = _min_transaction_date(conn) or as_of_date
    return {
        "mtd": (date(as_of_date.year, as_of_date.month, 1), as_of_date),
        "30d": (as_of_date - timedelta(days=30), as_of_date),
        "qtd": (date(as_of_date.year, ((as_of_date.month - 1) // 3) * 3 + 1, 1), as_of_date),
        "ytd": (date(as_of_date.year, 1, 1), as_of_date),
        "ltd": (min_date, as_of_date),
    }


def _cash_window_totals(conn: sqlite3.Connection, start: date, end: date) -> dict[str, float | int | None]:
    row = conn.execute(
        """
        SELECT
          COALESCE(SUM(CASE WHEN external_flow_amount > 0 THEN external_flow_amount ELSE 0 END), 0),
          COALESCE(SUM(CASE WHEN external_flow_amount < 0 THEN ABS(external_flow_amount) ELSE 0 END), 0),
          COALESCE(SUM(CASE WHEN trading_flow_amount < 0 THEN ABS(trading_flow_amount) ELSE 0 END), 0),
          COALESCE(SUM(CASE WHEN trading_flow_amount > 0 THEN trading_flow_amount ELSE 0 END), 0),
          COALESCE(SUM(CASE WHEN transaction_type='dividend' THEN income_flow_amount ELSE 0 END), 0),
          COALESCE(SUM(CASE WHEN transaction_type='interest' THEN income_flow_amount ELSE 0 END), 0),
          COALESCE(SUM(CASE WHEN transaction_type='margin_interest' THEN ABS(financing_flow_amount) ELSE 0 END), 0),
          COALESCE(SUM(CASE WHEN transaction_type='margin_borrow' THEN financing_flow_amount ELSE 0 END), 0),
          COALESCE(SUM(CASE WHEN transaction_type='margin_repay' THEN ABS(financing_flow_amount) ELSE 0 END), 0),
          COALESCE(SUM(CASE WHEN fee_flow_amount < 0 THEN ABS(fee_flow_amount) ELSE 0 END), 0),
          COUNT(*)
        FROM investment_transactions
        WHERE date BETWEEN ? AND ?
        """,
        (start.isoformat(), end.isoformat()),
    ).fetchone()
    (
        contributions_total,
        withdrawals_total,
        buys_total,
        sells_total,
        dividends_total,
        interest_income_total,
        margin_interest_total,
        margin_borrowed_total,
        margin_repaid_total,
        fees_total,
        transaction_count,
    ) = row

    external_net = contributions_total - withdrawals_total
    trading_net_cash = sells_total - buys_total
    financing_net = margin_borrowed_total - margin_repaid_total - margin_interest_total
    portfolio_cash_net = (
        external_net
        + trading_net_cash
        + dividends_total
        + interest_income_total
        + financing_net
        - fees_total
    )

    return {
        "contributions_total": round(contributions_total, 6),
        "withdrawals_total": round(withdrawals_total, 6),
        "external_net": round(external_net, 6),
        "buys_total": round(buys_total, 6),
        "sells_total": round(sells_total, 6),
        "trading_net_cash": round(trading_net_cash, 6),
        "dividends_total": round(dividends_total, 6),
        "interest_income_total": round(interest_income_total, 6),
        "margin_interest_total": round(margin_interest_total, 6),
        "margin_borrowed_total": round(margin_borrowed_total, 6),
        "margin_repaid_total": round(margin_repaid_total, 6),
        "fees_total": round(fees_total, 6),
        "financing_net": round(financing_net, 6),
        "portfolio_cash_net": round(portfolio_cash_net, 6),
        "transaction_count": int(transaction_count or 0),
    }


def _realized_window_totals(conn: sqlite3.Connection, start: date, end: date) -> dict[str, float | int | None]:
    row = conn.execute(
        """
        SELECT
          COALESCE(SUM(gross_proceeds), 0),
          COALESCE(SUM(net_proceeds), 0),
          COALESCE(SUM(realized_cost_basis), 0),
          COALESCE(SUM(realized_pnl), 0),
          COUNT(*),
          COALESCE(SUM(CASE WHEN realized_pnl >= 0 THEN 1 ELSE 0 END), 0),
          COALESCE(SUM(CASE WHEN realized_pnl < 0 THEN 1 ELSE 0 END), 0)
        FROM realized_trade_ledger
        WHERE date BETWEEN ? AND ?
        """,
        (start.isoformat(), end.isoformat()),
    ).fetchone()
    gross_proceeds, net_proceeds, realized_cost_basis, realized_pnl, sale_count, winning_sales, losing_sales = row
    realized_pnl_pct = (realized_pnl / realized_cost_basis * 100.0) if realized_cost_basis else None
    return {
        "gross_proceeds": round(gross_proceeds, 6),
        "net_proceeds": round(net_proceeds, 6),
        "realized_cost_basis": round(realized_cost_basis, 6),
        "realized_pnl": round(realized_pnl, 6),
        "realized_pnl_pct": round(realized_pnl_pct, 6) if realized_pnl_pct is not None else None,
        "sale_count": int(sale_count or 0),
        "winning_sales": int(winning_sales or 0),
        "losing_sales": int(losing_sales or 0),
    }


def build_daily_realized_snapshot(conn: sqlite3.Connection, as_of_date_str: str, recent_limit: int = 10) -> dict[str, object]:
    as_of_date = _parse_date(as_of_date_str)
    if as_of_date is None:
        return {"windows": {}, "recent_sales": [], "cashflow_windows": {}}

    windows = {}
    cashflow_windows = {}
    for window_key, (start, end) in _window_specs(conn, as_of_date).items():
        realized = _realized_window_totals(conn, start, end)
        cashflow = _cash_window_totals(conn, start, end)
        cashflow["realized_capital_pnl"] = realized["realized_pnl"]
        cashflow["realized_total_return"] = round(
            float(realized["realized_pnl"] or 0.0)
            + float(cashflow["dividends_total"] or 0.0)
            + float(cashflow["interest_income_total"] or 0.0)
            - float(cashflow["margin_interest_total"] or 0.0)
            - float(cashflow["fees_total"] or 0.0),
            6,
        )
        windows[window_key] = realized
        cashflow_windows[window_key] = cashflow

    recent_rows = conn.execute(
        """
        SELECT
          lm_transaction_id,
          date,
          symbol,
          shares_sold,
          sell_price,
          gross_proceeds,
          fees,
          net_proceeds,
          realized_cost_basis,
          realized_pnl,
          realized_pnl_pct,
          weighted_holding_period_days,
          matched_complete
        FROM realized_trade_ledger
        WHERE date <= ?
        ORDER BY date DESC, COALESCE(transaction_datetime, date) DESC
        LIMIT ?
        """,
        (as_of_date.isoformat(), int(recent_limit)),
    ).fetchall()

    recent_sales = [
        {
            "lm_transaction_id": row[0],
            "date": row[1],
            "symbol": row[2],
            "shares_sold": row[3],
            "sell_price": row[4],
            "gross_proceeds": row[5],
            "fees": row[6],
            "net_proceeds": row[7],
            "realized_cost_basis": row[8],
            "realized_pnl": row[9],
            "realized_pnl_pct": row[10],
            "weighted_holding_period_days": row[11],
            "matched_complete": bool(row[12]),
        }
        for row in recent_rows
    ]

    return {
        "windows": windows,
        "recent_sales": recent_sales,
        "cashflow_windows": cashflow_windows,
    }


def build_period_realized_summary(conn: sqlite3.Connection, start: date, end: date) -> dict[str, object]:
    trade_rows = conn.execute(
        """
        SELECT
          symbol,
          COALESCE(SUM(CASE WHEN transaction_type IN ('buy', 'buy_shares', 'reinvest', 'reinvestment') THEN 1 ELSE 0 END), 0),
          COALESCE(SUM(CASE WHEN transaction_type IN ('sell', 'sell_shares', 'redemption') THEN 1 ELSE 0 END), 0),
          COALESCE(SUM(CASE WHEN transaction_type IN ('buy', 'buy_shares', 'reinvest', 'reinvestment') THEN ABS(quantity) ELSE 0 END), 0),
          COALESCE(SUM(CASE WHEN transaction_type IN ('sell', 'sell_shares', 'redemption') THEN ABS(quantity) ELSE 0 END), 0),
          COALESCE(SUM(CASE WHEN trading_flow_amount < 0 THEN ABS(trading_flow_amount) ELSE 0 END), 0),
          COALESCE(SUM(CASE WHEN trading_flow_amount > 0 THEN trading_flow_amount ELSE 0 END), 0)
        FROM investment_transactions
        WHERE date BETWEEN ? AND ?
          AND symbol IS NOT NULL
          AND economic_bucket='trade'
        GROUP BY symbol
        ORDER BY symbol
        """,
        (start.isoformat(), end.isoformat()),
    ).fetchall()

    realized_rows = conn.execute(
        """
        SELECT
          symbol,
          COALESCE(SUM(realized_cost_basis), 0),
          COALESCE(SUM(realized_pnl), 0)
        FROM realized_trade_ledger
        WHERE date BETWEEN ? AND ?
        GROUP BY symbol
        ORDER BY symbol
        """,
        (start.isoformat(), end.isoformat()),
    ).fetchall()

    by_symbol: dict[str, dict[str, float | int]] = {}
    for row in trade_rows:
        symbol = row[0]
        by_symbol[symbol] = {
            "buy_count": int(row[1] or 0),
            "sell_count": int(row[2] or 0),
            "shares_bought": float(row[3] or 0.0),
            "shares_sold": float(row[4] or 0.0),
            "buy_amount_total": float(row[5] or 0.0),
            "sell_amount_total": float(row[6] or 0.0),
            "net_trade_cash": float((row[6] or 0.0) - (row[5] or 0.0)),
            "realized_cost_basis": 0.0,
            "realized_capital_pnl": 0.0,
        }

    total_realized_cost = 0.0
    total_realized_pnl = 0.0
    for row in realized_rows:
        symbol = row[0]
        by_symbol.setdefault(
            symbol,
            {
                "buy_count": 0,
                "sell_count": 0,
                "shares_bought": 0.0,
                "shares_sold": 0.0,
                "buy_amount_total": 0.0,
                "sell_amount_total": 0.0,
                "net_trade_cash": 0.0,
                "realized_cost_basis": 0.0,
                "realized_capital_pnl": 0.0,
            },
        )
        by_symbol[symbol]["realized_cost_basis"] = float(row[1] or 0.0)
        by_symbol[symbol]["realized_capital_pnl"] = float(row[2] or 0.0)
        total_realized_cost += float(row[1] or 0.0)
        total_realized_pnl += float(row[2] or 0.0)

    cashflow = _cash_window_totals(conn, start, end)
    realized = _realized_window_totals(conn, start, end)
    cashflow["realized_capital_pnl"] = realized["realized_pnl"]
    cashflow["realized_total_return"] = round(
        float(realized["realized_pnl"] or 0.0)
        + float(cashflow["dividends_total"] or 0.0)
        + float(cashflow["interest_income_total"] or 0.0)
        - float(cashflow["margin_interest_total"] or 0.0)
        - float(cashflow["fees_total"] or 0.0),
        6,
    )

    return {
        "by_symbol": by_symbol,
        "cashflow": cashflow,
        "realized": realized,
    }
