import json
import re
import sqlite3

from ..config import settings

_CUSIP_RE = re.compile(r"\b([A-Z0-9]{8,9})\b")
_SHARES_OF_RE = re.compile(r"\bshares\s+of\s+([A-Z][A-Z0-9.\-]{0,9})\b", re.IGNORECASE)

_ALLOWED_TX_TYPES = {
    "buy",
    "buy_shares",
    "sell",
    "sell_shares",
    "redemption",
    "reinvest",
    "reinvestment",
    "dividend",
    "interest",
    "contribution",
    "withdrawal",
    "margin_borrow",
    "margin_repay",
    "fee",
}


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


def _parse_meta(meta):
    if isinstance(meta, dict):
        return meta
    if isinstance(meta, str):
        try:
            parsed = json.loads(meta)
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            return None
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


def _infer_tx_type(tx: dict, meta: dict | None):
    text = " ".join(
        [
            str(tx.get("category_name") or ""),
            str(tx.get("display_name") or ""),
            str(tx.get("original_name") or ""),
            str(tx.get("payee") or ""),
        ]
    ).lower()
    
    # Get Plaid investment transaction type and subtype
    meta_type = None
    plaid_subtype = None
    if meta:
        for key in ("investment_transaction_type", "type", "subtype"):
            val = meta.get(key)
            if val:
                if key == "subtype":
                    plaid_subtype = str(val).lower()
                else:
                    meta_type = str(val).lower()
                # For buy/sell/dividend, return immediately
                if meta_type in ("buy", "sell", "dividend", "reinvest", "reinvestment", "redemption"):
                    return meta_type
                # For transfer/cash, don't return yet - need to check text for better classification
                break
    
    # Check for dividend explicitly
    if "dividend" in text:
        return "dividend"
    
    # Check for interest (but exclude margin interest)
    if "interest" in text:
        if "margin" in text:
            # This is margin interest expense (repayment to loan account)
            return "margin_repay"
        if "securities lending" in text:
            return "interest"
        # Generic interest - could be margin or securities lending
        # If it's positive, likely margin repayment; if negative, likely securities lending
        amount = tx.get("amount")
        try:
            amount_float = float(amount) if amount is not None else 0.0
            if amount_float > 0:
                return "margin_repay"
            else:
                return "interest"
        except (ValueError, TypeError):
            return "interest"
    
    # Check for fee
    if "fee" in text:
        return "fee"
    
    # Check for transfers (could be contribution or withdrawal)
    if "transfer" in text:
        # For investment account, check amount sign
        # Negative amounts are money coming IN (contribution from checking)
        # Positive amounts are money going OUT (withdrawal to checking)
        amount = tx.get("amount")
        try:
            amount_float = float(amount) if amount is not None else 0.0
            if amount_float < 0:
                # Money coming into investment account = contribution
                return "contribution"
            else:
                # Money going out of investment account = withdrawal
                return "withdrawal"
        except (ValueError, TypeError):
            pass
    
    # Fallback to Plaid transaction type
    if meta_type:
        return meta_type
    
    # Check tx-level type/subtype
    for key in ("subtype", "type"):
        val = tx.get(key)
        if val:
            return str(val).lower()
    
    return "cash" if meta_type == "cash" else "unknown"


def _load_cusip_map(conn: sqlite3.Connection):
    cur = conn.cursor()
    rows = cur.execute("SELECT cusip, symbol FROM cusip_map").fetchall()
    return {cusip: symbol for cusip, symbol in rows}


def normalize_investment_transactions(conn: sqlite3.Connection, run_id: str) -> tuple[int, int]:
    cur = conn.cursor()
    rows = cur.execute(
        "SELECT pulled_at_utc, payload_json FROM lm_raw WHERE run_id=? AND endpoint='transactions'",
        (run_id,),
    ).fetchall()
    if not rows:
        return 0, 0

    allowed = _allowed_plaid_ids()
    cusip_map = _load_cusip_map(conn)
    inserted = 0

    for pulled_at_utc, payload_json in rows:
        try:
            tx = json.loads(payload_json)
        except Exception:
            continue
        if not isinstance(tx, dict):
            continue

        lm_transaction_id = tx.get("id")
        if lm_transaction_id is None:
            continue

        plaid_account_id = tx.get("plaid_account_id")
        if allowed and (plaid_account_id is None or str(plaid_account_id) not in allowed):
            continue

        meta = _parse_meta(tx.get("plaid_metadata"))

        name = tx.get("display_name") or tx.get("original_name") or tx.get("payee")
        meta_name = meta.get("name") if meta else None
        text_for_parse = meta_name or name

        cusip = _extract_cusip(text_for_parse) or _extract_cusip(name)
        symbol = None
        if meta and meta.get("ticker_symbol"):
            symbol = str(meta.get("ticker_symbol")).upper()
        if not symbol:
            symbol = _extract_symbol(text_for_parse) or _extract_symbol(name)
        if not symbol and cusip:
            symbol = cusip_map.get(cusip)

        tx_type = _infer_tx_type(tx, meta)
        if tx_type not in _ALLOWED_TX_TYPES:
            continue

        cur.execute(
            """
            INSERT OR IGNORE INTO investment_transactions (
              lm_transaction_id, plaid_account_id, date, transaction_datetime,
              amount, currency, name, category_name, transaction_type,
              quantity, price, fees, security_id, cusip, symbol,
              plaid_investment_transaction_id, plaid_cancel_transaction_id,
              source, run_id, pulled_at_utc
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                str(lm_transaction_id),
                str(plaid_account_id) if plaid_account_id is not None else None,
                tx.get("date"),
                meta.get("transaction_datetime") if meta else None,
                _coerce_float(tx.get("amount")),
                tx.get("currency") or (meta.get("iso_currency_code") if meta else None),
                name,
                tx.get("category_name"),
                tx_type,
                _coerce_float(meta.get("quantity")) if meta else None,
                _coerce_float(meta.get("price")) if meta else None,
                _coerce_float(meta.get("fees")) if meta else None,
                meta.get("security_id") if meta else None,
                cusip,
                symbol,
                meta.get("investment_transaction_id") if meta else None,
                meta.get("cancel_transaction_id") if meta else None,
                "lunchmoney",
                run_id,
                pulled_at_utc,
            ),
        )
        if cur.rowcount and cur.rowcount > 0:
            inserted += cur.rowcount

    if _ALLOWED_TX_TYPES:
        allowed_types = sorted(_ALLOWED_TX_TYPES)
        type_placeholders = ",".join("?" for _ in allowed_types)
        params: list[str] = []
        where_clauses = [
            f"(transaction_type IS NULL OR lower(transaction_type) NOT IN ({type_placeholders}))"
        ]
        params.extend(allowed_types)
        if allowed:
            acct_placeholders = ",".join("?" for _ in allowed)
            where_clauses.append(f"plaid_account_id IN ({acct_placeholders})")
            params.extend(sorted(allowed))
        cur.execute(
            f"DELETE FROM investment_transactions WHERE {' AND '.join(where_clauses)}",
            params,
        )
    conn.commit()
    return inserted, inserted
