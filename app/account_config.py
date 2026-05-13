from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable

from .config import settings


@dataclass(frozen=True)
class AccountProfile:
    plaid_account_id: str
    short_name: str
    account_role: str
    include_in_portfolio: bool
    include_in_income: bool
    include_in_margin: bool
    is_primary: bool


def _csv_ids(raw: str | None) -> list[str]:
    if not raw:
        return []
    seen: set[str] = set()
    out: list[str] = []
    for part in raw.split(","):
        value = part.strip()
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _ordered_union(*items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for group in items:
        for item in group:
            text = str(item).strip()
            if not text or text in seen:
                continue
            seen.add(text)
            out.append(text)
    return out


def investment_account_ids() -> list[str]:
    explicit = _csv_ids(settings.lm_investment_account_ids)
    if explicit:
        return explicit
    return _csv_ids(settings.lm_plaid_account_ids)


def margin_account_ids() -> list[str]:
    return _csv_ids(settings.lm_margin_account_ids)


def transaction_account_ids() -> list[str]:
    role_based = _ordered_union(investment_account_ids(), margin_account_ids())
    if role_based:
        return role_based
    return _csv_ids(settings.lm_plaid_account_ids)


def primary_investment_account_id() -> str | None:
    explicit = (settings.lm_primary_investment_account_id or "").strip()
    if explicit:
        return explicit
    ids = investment_account_ids()
    return ids[0] if ids else None


def account_short_name(plaid_account_id: str | int | None, fallback: str | None = None) -> str:
    if plaid_account_id is None:
        return fallback or "Account"
    key = str(plaid_account_id).strip()
    env_value = os.getenv(f"LM_ACCOUNT_{key}_LABEL")
    if env_value and env_value.strip():
        return env_value.strip()
    if fallback and fallback.strip():
        return fallback.strip()
    return key


def infer_account_role(
    plaid_account_id: str | int | None,
    *,
    account_type: str | None = None,
    account_subtype: str | None = None,
    name: str | None = None,
) -> str:
    key = str(plaid_account_id).strip() if plaid_account_id is not None else ""
    if key and key in set(margin_account_ids()):
        return "margin"
    if key and key in set(investment_account_ids()):
        return "investment"
    joined = " ".join([account_type or "", account_subtype or "", name or ""]).lower()
    if "loan" in joined or "borrow" in joined or "margin" in joined:
        return "margin"
    if "investment" in joined or "brokerage" in joined:
        return "investment"
    if "cash" in joined or "depository" in joined or "checking" in joined or "savings" in joined:
        return "cash"
    if "credit" in joined:
        return "credit"
    return "unknown"


def account_profile(
    plaid_account_id: str | int,
    *,
    account_type: str | None = None,
    account_subtype: str | None = None,
    name: str | None = None,
) -> AccountProfile:
    key = str(plaid_account_id)
    role = infer_account_role(key, account_type=account_type, account_subtype=account_subtype, name=name)
    primary = key == primary_investment_account_id()
    return AccountProfile(
        plaid_account_id=key,
        short_name=account_short_name(key, name),
        account_role=role,
        include_in_portfolio=role in {"investment", "margin"},
        include_in_income=role == "investment",
        include_in_margin=role == "margin",
        is_primary=primary,
    )


def account_metadata_from_row(row: dict) -> dict:
    plaid_account_id = row.get("plaid_account_id") or row.get("id")
    profile = account_profile(
        plaid_account_id,
        account_type=row.get("type") or row.get("account_type"),
        account_subtype=row.get("subtype"),
        name=row.get("name") or row.get("display_name") or row.get("official_name"),
    )
    return {
        "plaid_account_id": profile.plaid_account_id,
        "short_name": profile.short_name,
        "account_role": profile.account_role,
        "include_in_portfolio": int(profile.include_in_portfolio),
        "include_in_income": int(profile.include_in_income),
        "include_in_margin": int(profile.include_in_margin),
        "is_primary": int(profile.is_primary),
    }
