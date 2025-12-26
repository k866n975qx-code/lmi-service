import re
import sqlite3
from collections import defaultdict
from datetime import datetime, date, timezone
from typing import Dict, List, Tuple

from ..config import settings
from ..utils import to_local_date

_CUSIP_RE = re.compile(r"\b([A-Z0-9]{8,9})\b")
_SHARES_OF_RE = re.compile(r"\bshares\s+of\s+([A-Z][A-Z0-9.\-]{0,9})\b", re.IGNORECASE)
_SHARES_RE = re.compile(r"\b([0-9]*\\.?[0-9]+)\\s+shares?\\b", re.IGNORECASE)

BUY_TYPES = {"buy", "buy_shares", "reinvest", "reinvestment"}
SELL_TYPES = {"sell", "sell_shares", "redemption"}


def _allowed_plaid_ids():
    raw = settings.lm_plaid_account_ids
    if not raw:
        return None
    return {part.strip() for part in raw.split(",") if part.strip()}


def _coerce_float(val):
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _extract_cusip(text: str | None):
    if not text:
        return None
    for match in _CUSIP_RE.finditer(text.upper()):
        token = match.group(1)
        if token == "DIVIDEND":
            continue
        if any(ch.isdigit() for ch in token):
            return token
    return None


def _extract_symbol(text: str | None):
    if not text:
        return None
    match = _SHARES_OF_RE.search(text)
    return match.group(1).upper() if match else None


def _extract_quantity(text: str | None):
    if not text:
        return None
    match = _SHARES_RE.search(text)
    return float(match.group(1)) if match else None


def _parse_date(val: str | None):
    if not val:
        return None
    text = str(val).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
        return dt.date()
    except ValueError:
        pass
    try:
        return date.fromisoformat(text)
    except ValueError:
        return None


def _load_splits(conn: sqlite3.Connection):
    cur = conn.cursor()
    rows = cur.execute("SELECT symbol, ex_date, ratio FROM split_events").fetchall()
    splits = defaultdict(list)
    for symbol, ex_date, ratio in rows:
        if not symbol or not ex_date or ratio is None:
            continue
        ex_dt = _parse_date(ex_date)
        if ex_dt is None:
            continue
        try:
            ratio_val = float(ratio)
        except (TypeError, ValueError):
            continue
        if ratio_val <= 0:
            continue
        splits[str(symbol).upper()].append((ex_dt, ratio_val))
    for sym in splits:
        splits[sym].sort(key=lambda item: item[0])
    return splits


def reconstruct_holdings(conn: sqlite3.Connection) -> Tuple[Dict, List[str]]:
    """Build holdings from normalized investment_transactions (earliest -> latest).
    Returns (holdings_dict, needed_symbols).
    """
    cur = conn.cursor()
    rows = cur.execute(
        """
        SELECT
          lm_transaction_id, plaid_account_id, date, transaction_datetime,
          amount, currency, name, category_name, transaction_type,
          quantity, price, fees, security_id, cusip, symbol
        FROM investment_transactions
        ORDER BY COALESCE(transaction_datetime, date) ASC, lm_transaction_id ASC
        """
    ).fetchall()

    allowed = _allowed_plaid_ids()
    cusip_map = {cusip: symbol for cusip, symbol in cur.execute("SELECT cusip, symbol FROM cusip_map").fetchall()}

    symbols = set()
    lots = defaultdict(list)  # symbol -> list of {qty, cost}
    splits = _load_splits(conn)
    split_idx = {sym: 0 for sym in splits}

    def apply_splits(symbol: str, up_to: date | None):
        if up_to is None:
            return
        items = splits.get(symbol)
        if not items:
            return
        idx = split_idx.get(symbol, 0)
        while idx < len(items) and items[idx][0] <= up_to:
            ratio = items[idx][1]
            if ratio <= 0:
                idx += 1
                continue
            for lot in lots[symbol]:
                lot["qty"] *= ratio
            idx += 1
        split_idx[symbol] = idx

    for row in rows:
        (
            _lm_id,
            plaid_account_id,
            _date,
            _dt,
            amount,
            _currency,
            name,
            _category,
            tx_type,
            quantity,
            price,
            fees,
            _security_id,
            cusip,
            symbol,
        ) = row

        if allowed and (plaid_account_id is None or str(plaid_account_id) not in allowed):
            continue

        tx_type = (tx_type or "").lower()
        name = name or ""

        if not symbol:
            symbol = _extract_symbol(name)
        if not symbol and cusip:
            symbol = cusip_map.get(cusip)
        if not symbol:
            cusip = _extract_cusip(name)
            if cusip:
                symbol = cusip_map.get(cusip)
        if not symbol:
            continue
        symbol = symbol.upper()
        symbols.add(symbol)

        tx_date = _parse_date(_dt) or _parse_date(_date)
        apply_splits(symbol, tx_date)

        qty = _coerce_float(quantity)
        if qty is None:
            qty = _extract_quantity(name)
        if qty is None or qty == 0:
            continue

        direction = None
        if qty < 0:
            direction = -1
            qty = abs(qty)
        if direction is None:
            if tx_type in BUY_TYPES:
                direction = 1
            elif tx_type in SELL_TYPES:
                direction = -1
            elif "sell" in name.lower():
                direction = -1
            elif "buy" in name.lower() or "purchased" in name.lower():
                direction = 1
            else:
                direction = 1 if qty > 0 else None

        if direction is None:
            continue

        price_val = _coerce_float(price)
        fee_val = _coerce_float(fees) or 0.0
        amt_val = _coerce_float(amount)

        if direction > 0:
            cost = 0.0
            if price_val is not None:
                cost = qty * price_val
            elif amt_val is not None:
                cost = abs(amt_val)
            cost += fee_val
            lots[symbol].append({"qty": qty, "cost": cost})
        else:
            remaining = qty
            while remaining > 0 and lots[symbol]:
                lot = lots[symbol][0]
                take = min(remaining, lot["qty"])
                cost_per_share = (lot["cost"] / lot["qty"]) if lot["qty"] else 0.0
                lot["qty"] -= take
                lot["cost"] -= cost_per_share * take
                remaining -= take
                if lot["qty"] <= 0:
                    lots[symbol].pop(0)

    # Apply remaining splits up to today's local date.
    as_of_date = to_local_date(datetime.now(timezone.utc), settings.local_tz, settings.daily_cutover)
    for sym in list(lots.keys()):
        apply_splits(sym, as_of_date)

    holdings = {}
    for sym, sym_lots in lots.items():
        shares = sum(l["qty"] for l in sym_lots)
        cost_basis = sum(l["cost"] for l in sym_lots)
        if abs(shares) < 1e-9:
            continue
        holdings[sym] = {"symbol": sym, "shares": shares, "cost_basis": cost_basis}
    return holdings, sorted(holdings.keys())
