"""Assemble V5 snapshot JSON from flat tables and strip noise for API responses.

Fields passed through as null (no replacement default string) so null_reasons or
clients can handle them: meta.health_status, meta.schema_version,
goals.pace.current_pace.pct_of_tier_pace, goals.pace.likely_tier.reason,
goals.tiers[].confidence when null or 'medium' (derived from likely_tier: high for
matching tier, low for others).
"""
from __future__ import annotations

import json
import sqlite3
from datetime import date, datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

import math
from .snapshots import (
    _build_goal_progress,
    _build_goal_progress_net,
    _detect_likely_tier,
    _dividend_window,
    _load_dividend_transactions,
    _month_start,
    _quarter_start,
    _year_start,
)
from ..config import settings

SIGNIFICANT_MISSING_PCT = 1.0


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def _enrich_dividends_from_db(
    conn: sqlite3.Connection,
    as_of_str: str,
    holdings_list: list[dict],
    div_realized: dict,
) -> None:
    """Fill dividends.realized from investment_transactions when flat tables have null/missing data."""
    as_of_dt = _parse_date(as_of_str)
    if not as_of_dt:
        return
    holding_symbols = {h.get("symbol") for h in holdings_list if h.get("symbol")}
    div_tx = _load_dividend_transactions(conn)
    if not div_tx:
        return

    # MTD: backfill from DB when missing or by_symbol empty
    realized_mtd = _dividend_window(
        div_tx, _month_start(as_of_dt), as_of_dt, holding_symbols
    )
    mtd = div_realized.get("mtd")
    if mtd is None:
        div_realized["mtd"] = {
            "total_dividends": realized_mtd["total_dividends"],
            "by_symbol": realized_mtd.get("by_symbol") or {},
        }
    elif not mtd.get("by_symbol"):
        mtd["by_symbol"] = realized_mtd.get("by_symbol") or {}
    if mtd and mtd.get("total_dividends") is None:
        mtd["total_dividends"] = realized_mtd["total_dividends"]

    # 30d, ytd, qtd: backfill totals when flat has null; always add by_symbol from DB so dividends appear
    def _needs_fill(d: dict | None, key: str) -> bool:
        return d is None or not isinstance(d, dict) or d.get("total_dividends") is None

    w30 = _dividend_window(
        div_tx, as_of_dt - timedelta(days=30), as_of_dt, holding_symbols
    )
    if _needs_fill(div_realized.get("30d"), "total_dividends"):
        div_realized["30d"] = {"total_dividends": w30["total_dividends"], "by_symbol": w30.get("by_symbol") or {}}
    elif div_realized.get("30d") and "by_symbol" not in div_realized["30d"]:
        div_realized["30d"]["by_symbol"] = w30.get("by_symbol") or {}

    wytd = _dividend_window(
        div_tx, _year_start(as_of_dt), as_of_dt, holding_symbols
    )
    if _needs_fill(div_realized.get("ytd"), "total_dividends"):
        div_realized["ytd"] = {"total_dividends": wytd["total_dividends"], "by_symbol": wytd.get("by_symbol") or {}}
    elif div_realized.get("ytd") and "by_symbol" not in div_realized["ytd"]:
        div_realized["ytd"]["by_symbol"] = wytd.get("by_symbol") or {}

    wqtd = _dividend_window(
        div_tx, _quarter_start(as_of_dt), as_of_dt, holding_symbols
    )
    if _needs_fill(div_realized.get("qtd"), "total_dividends"):
        div_realized["qtd"] = {"total_dividends": wqtd["total_dividends"], "by_symbol": wqtd.get("by_symbol") or {}}
    elif div_realized.get("qtd") and "by_symbol" not in div_realized["qtd"]:
        div_realized["qtd"]["by_symbol"] = wqtd.get("by_symbol") or {}


def _enrich_holding_dividends_from_db(
    conn: sqlite3.Connection,
    as_of_str: str,
    holdings_list: list[dict],
) -> None:
    """Fill holdings[].income.dividends_30d/qtd/ytd from investment_transactions when flat has null."""
    as_of_dt = _parse_date(as_of_str)
    if not as_of_dt:
        return
    div_tx = _load_dividend_transactions(conn)
    if not div_tx:
        return
    by_sym: dict[str, list[dict]] = {}
    for tx in div_tx:
        sym = tx.get("symbol")
        if sym:
            by_sym.setdefault(sym, []).append(tx)

    cut_30d = as_of_dt - timedelta(days=30)
    cut_qtd = _quarter_start(as_of_dt)
    cut_ytd = _year_start(as_of_dt)

    for h in holdings_list:
        inc = h.get("income")
        if not isinstance(inc, dict):
            continue
        sym = h.get("symbol")
        if not sym:
            continue
        events = by_sym.get(sym, [])
        if inc.get("dividends_30d") is None:
            tot = round(sum(e["amount"] for e in events if e["date"] >= cut_30d), 2)
            inc["dividends_30d"] = tot if tot != 0 else 0.0
        if inc.get("dividends_qtd") is None:
            tot = round(sum(e["amount"] for e in events if e["date"] >= cut_qtd), 2)
            inc["dividends_qtd"] = tot if tot != 0 else 0.0
        if inc.get("dividends_ytd") is None:
            tot = round(sum(e["amount"] for e in events if e["date"] >= cut_ytd), 2)
            inc["dividends_ytd"] = tot if tot != 0 else 0.0


def _derive_pace_nulls(pace: dict, r: dict, conn: sqlite3.Connection) -> None:
    """Fill pace.current_pace.pct_of_tier_pace and pace.likely_tier.reason when flat row has nulls."""
    cur = pace.get("current_pace") or {}
    likely = pace.get("likely_tier") or {}
    windows = pace.get("windows") or {}

    if cur.get("pct_of_tier_pace") is None and windows:
        si = windows.get("since_inception") or {}
        cur_pace = si.get("pace") or {}
        pct = cur_pace.get("pct_of_tier_pace")
        if pct is not None:
            cur["pct_of_tier_pace"] = pct

    if likely.get("reason") is None and conn:
        income = r.get("projected_monthly_income")
        mv = r.get("market_value")
        loan = r.get("margin_loan_balance") or 0
        if isinstance(income, (int, float)) and isinstance(mv, (int, float)) and mv and income is not None:
            try:
                detected = _detect_likely_tier(conn, float(income), float(loan), float(mv))
                if detected:
                    reason = (detected.get("detection_basis") or {}).get("notes")
                    if reason:
                        likely["reason"] = reason
            except (TypeError, ValueError):
                pass


def assemble_daily_snapshot(conn: sqlite3.Connection, as_of_date: str | None = None, slim: bool = False) -> dict | None:
    """Reconstruct V5 daily snapshot JSON from flat tables.
    If as_of_date is None, use the latest available date.
    """
    conn.row_factory = sqlite3.Row
    if as_of_date:
        row = conn.execute(
            "SELECT * FROM daily_portfolio WHERE as_of_date_local=?", (as_of_date,)
        ).fetchone()
    else:
        row = conn.execute(
            "SELECT * FROM daily_portfolio ORDER BY as_of_date_local DESC LIMIT 1"
        ).fetchone()
    if not row:
        return None
    r = dict(row)
    as_of = r["as_of_date_local"]
    if as_of_date is not None and as_of != as_of_date:
        raise ValueError(f"assemble_daily_snapshot: requested as_of_date={as_of_date!r} but row has as_of_date_local={as_of!r}")

    # Child rows
    holdings_rows = conn.execute(
        "SELECT * FROM daily_holdings WHERE as_of_date_local=? ORDER BY weight_pct DESC", (as_of,)
    ).fetchall()
    tiers_rows = conn.execute(
        "SELECT * FROM daily_goal_tiers WHERE as_of_date_local=? ORDER BY tier", (as_of,)
    ).fetchall()
    rate_rows = conn.execute(
        "SELECT * FROM daily_margin_rate_scenarios WHERE as_of_date_local=?", (as_of,)
    ).fetchall()
    attr_rows = conn.execute(
        "SELECT * FROM daily_return_attribution WHERE as_of_date_local=?", (as_of,)
    ).fetchall()
    upc_rows = conn.execute(
        "SELECT * FROM daily_dividends_upcoming WHERE as_of_date_local=?", (as_of,)
    ).fetchall()

    # Build V5 structure: portfolio date vs price date (use EOD of portfolio date for portfolio_data_as_of_utc)
    prices_utc = r.get("prices_as_of_utc")
    portfolio_utc = None
    if as_of:
        try:
            local_tz = ZoneInfo(settings.local_tz)
            d = date.fromisoformat(as_of[:10])
            eod_local = datetime.combine(d, datetime.max.time().replace(microsecond=0), tzinfo=local_tz)
            portfolio_utc = eod_local.astimezone(timezone.utc).isoformat()
        except Exception:
            portfolio_utc = as_of + "T00:00:00+00:00" if as_of else None
    timestamps = {
        "portfolio_data_as_of_local": as_of,
        "portfolio_data_as_of_utc": portfolio_utc or prices_utc or (as_of + "T00:00:00Z" if as_of else None),
        "price_data_as_of_utc": prices_utc,
    }
    holdings_list = [_row_to_holding_legacy(h) for h in holdings_rows]
    # Use assembled holdings count so validation (totals.holdings_count == len(holdings)) passes
    holdings_count = len(holdings_list)
    portfolio = {
        "totals": {
            "market_value": r.get("market_value"),
            "cost_basis": r.get("cost_basis"),
            "net_liquidation_value": r.get("net_liquidation_value"),
            "unrealized_pnl": r.get("unrealized_pnl"),
            "unrealized_pct": r.get("unrealized_pct"),
            "margin_loan_balance": r.get("margin_loan_balance"),
            "margin_to_portfolio_pct": r.get("ltv_pct"),
            "holdings_count": holdings_count,
            "positions_profitable": r.get("positions_profitable"),
            "positions_losing": r.get("positions_losing"),
        },
        "income": {
            "projected_monthly_income": r.get("projected_monthly_income"),
            "forward_12m_total": r.get("forward_12m_total"),
            "portfolio_current_yield_pct": r.get("portfolio_yield_pct"),
            "portfolio_yield_on_cost_pct": r.get("portfolio_yield_on_cost_pct"),
            "income_growth": _json_load(r.get("income_growth_json")),
        },
        "performance": {
            "twr_1m_pct": r.get("twr_1m_pct"),
            "twr_3m_pct": r.get("twr_3m_pct"),
            "twr_6m_pct": r.get("twr_6m_pct"),
            "twr_12m_pct": r.get("twr_12m_pct"),
            "vs_benchmark": {
                "symbol": r.get("vs_benchmark_symbol"),
                "benchmark_twr_1y_pct": r.get("vs_benchmark_twr_1y_pct"),
                "excess_1y_pct": r.get("vs_benchmark_excess_1y_pct"),
                "corr_1y": r.get("vs_benchmark_corr_1y"),
            } if r.get("vs_benchmark_symbol") else None,
        },
        "risk": {
            "volatility": {"vol_30d_pct": r.get("vol_30d_pct"), "vol_90d_pct": r.get("vol_90d_pct")},
            "ratios": {
                "sharpe_1y": r.get("sharpe_1y"), "sortino_1y": r.get("sortino_1y"),
                "sortino_6m": r.get("sortino_6m"), "sortino_3m": r.get("sortino_3m"), "sortino_1m": r.get("sortino_1m"),
                "sortino_sharpe_ratio": r.get("sortino_sharpe_ratio"), "sortino_sharpe_divergence": r.get("sortino_sharpe_divergence"),
                "calmar_1y": r.get("calmar_1y"), "omega_ratio_1y": r.get("omega_ratio_1y"),
                "ulcer_index_1y": r.get("ulcer_index_1y"), "pain_adjusted_return": r.get("pain_adjusted_return"),
                "information_ratio_1y": r.get("information_ratio_1y"), "tracking_error_1y_pct": r.get("tracking_error_1y_pct"),
            },
            "drawdown": {
                "max_drawdown_1y_pct": r.get("max_drawdown_1y_pct"),
                "drawdown_duration_1y_days": r.get("drawdown_duration_1y_days"),
                "currently_in_drawdown": bool(r.get("currently_in_drawdown")) if r.get("currently_in_drawdown") is not None else None,
                "current_drawdown_depth_pct": r.get("drawdown_depth_pct"),
                "current_drawdown_duration_days": r.get("drawdown_duration_days"),
                "peak_value": r.get("drawdown_peak_value"),
                "peak_date": r.get("drawdown_peak_date"),
            },
            "var": {
                "var_90_1d_pct": r.get("var_90_1d_pct"), "var_95_1d_pct": r.get("var_95_1d_pct"),
                "var_99_1d_pct": r.get("var_99_1d_pct"), "var_95_1w_pct": r.get("var_95_1w_pct"), "var_95_1m_pct": r.get("var_95_1m_pct"),
                "cvar_90_1d_pct": r.get("cvar_90_1d_pct"), "cvar_95_1d_pct": r.get("cvar_95_1d_pct"),
                "cvar_99_1d_pct": r.get("cvar_99_1d_pct"), "cvar_95_1w_pct": r.get("cvar_95_1w_pct"), "cvar_95_1m_pct": r.get("cvar_95_1m_pct"),
            },
            "tail_risk": {
                "tail_risk_category": r.get("tail_risk_category"),
                "cvar_to_income_ratio": r.get("cvar_to_income_ratio"),
            },
            "beta_portfolio": r.get("beta_portfolio"),
            "portfolio_risk_quality": r.get("portfolio_risk_quality"),
            "income_stability_score": r.get("income_stability_score"),
        },
        "allocation": {
            "concentration": {
                "top3_weight_pct": r.get("top3_weight_pct"),
                "top5_weight_pct": r.get("top5_weight_pct"),
                "herfindahl_index": r.get("herfindahl_index"),
                "category": r.get("concentration_category"),
            },
        },
        "attribution": _attribution_by_window(attr_rows),
    }

    holdings = holdings_list
    goals = {
        "baseline": {
            "target_monthly_income": r.get("goal_target_monthly_income"),
            "required_portfolio_value": r.get("goal_required_portfolio_value"),
            "progress_pct": r.get("goal_progress_pct"),
            "months_to_goal": r.get("goal_months_to_goal"),
            "current_projected_monthly": r.get("goal_current_projected_monthly"),
            "estimated_goal_date": r.get("goal_estimated_goal_date"),
        },
        "net_of_interest": {
            "progress_pct": r.get("goal_net_progress_pct"),
            "months_to_goal": r.get("goal_net_months_to_goal"),
            "current_projected_monthly_net": r.get("goal_net_current_projected_monthly"),
        },
        "tiers": [],
        "current_state": {
            "portfolio_value": r.get("goal_tiers_portfolio_value"),
            "projected_monthly_income": r.get("goal_tiers_projected_monthly"),
            "portfolio_yield_pct": r.get("goal_tiers_yield_pct"),
            "target_monthly": r.get("goal_target_monthly_income"),
        },
        "pace": {
            "current_pace": {
                "months_ahead_behind": r.get("goal_pace_months_ahead_behind"),
                "pace_category": r.get("goal_pace_category"),
                "on_track": bool(r.get("goal_pace_on_track")) if r.get("goal_pace_on_track") is not None else None,
                "revised_goal_date": r.get("goal_pace_revised_goal_date"),
                "pct_of_tier_pace": r.get("goal_pace_pct_of_tier"),
            },
            "likely_tier": {
                "tier": r.get("goal_likely_tier"),
                "name": r.get("goal_likely_tier_name"),
                "confidence": r.get("goal_likely_tier_confidence"),
                "reason": r.get("goal_likely_tier_reason"),
            },
            "windows": _json_load(r.get("goal_pace_windows_json")),
            "baseline_projection": _json_load(r.get("goal_pace_baseline_json")),
        },
    }
    # Derive pace nulls from windows or detection when flat row has them
    _derive_pace_nulls(goals["pace"], r, conn)
    # Backfill baseline goals from portfolio when flat row has nulls (e.g. pre-persist run)
    if goals["baseline"].get("progress_pct") is None:
        mv = r.get("market_value")
        income = r.get("projected_monthly_income")
        target = getattr(settings, "goal_target_monthly", None)
        if target is None and isinstance(income, (int, float)) and income > 0:
            target = float(income) * 1.2
        as_of_dt = _parse_date(as_of)
        if mv is not None and income is not None and target and as_of_dt:
            computed = _build_goal_progress(float(income), float(mv), float(target), as_of_dt)
            if computed:
                goals["baseline"]["target_monthly_income"] = computed.get("target_monthly")
                goals["baseline"]["required_portfolio_value"] = computed.get("required_portfolio_value_at_goal")
                goals["baseline"]["progress_pct"] = computed.get("progress_pct")
                goals["baseline"]["months_to_goal"] = computed.get("months_to_goal")
                goals["baseline"]["current_projected_monthly"] = computed.get("current_projected_monthly")
                goals["baseline"]["estimated_goal_date"] = computed.get("estimated_goal_date")

    # current_state.target_monthly for Telegram/API (e.g. "Target Monthly: $X")
    if goals["current_state"].get("target_monthly") is None:
        goals["current_state"]["target_monthly"] = (
            goals["baseline"].get("target_monthly_income")
            or goals["baseline"].get("target_monthly")
            or r.get("goal_target_monthly_income")
            or getattr(settings, "goal_target_monthly", None)
        )

    # Build tiers and backfill nulls from baseline so "goal data not available" doesn't appear
    baseline_target = goals["baseline"].get("target_monthly_income") or goals["baseline"].get("target_monthly") or r.get("goal_target_monthly_income")
    baseline_progress = goals["baseline"].get("progress_pct") or r.get("goal_progress_pct")
    target_ltv = getattr(settings, "goal_target_ltv_pct", 30.0)
    for t in tiers_rows:
        row = dict(t)
        if row.get("target_monthly") is None and baseline_target is not None:
            row["target_monthly"] = baseline_target
        if row.get("progress_pct") is None and baseline_progress is not None:
            row["progress_pct"] = baseline_progress
        # Replace None or the lazy default "medium" with derived confidence (high for likely tier, low for others)
        confidence = row.get("confidence")
        if confidence is None or confidence == "medium":
            likely_tier = r.get("goal_likely_tier")
            if row.get("tier") == likely_tier:
                row["confidence"] = r.get("goal_likely_tier_confidence") or "high"
            else:
                row["confidence"] = "low"
        if row.get("assumption_target_ltv_pct") is None:
            row["assumption_target_ltv_pct"] = target_ltv
        goals["tiers"].append(row)

    # Backfill net_of_interest.months_to_goal when flat row has null
    if goals["net_of_interest"].get("months_to_goal") is None:
        mv = r.get("market_value")
        income = r.get("projected_monthly_income")
        loan = r.get("margin_loan_balance") or 0
        target = goals["baseline"].get("target_monthly_income") or goals["baseline"].get("target_monthly") or r.get("goal_target_monthly_income")
        if mv is not None and income is not None and target and float(target) > 0:
            try:
                gpn = _build_goal_progress_net(float(income), float(mv), float(loan), float(target))
                if gpn:
                    goals["net_of_interest"]["progress_pct"] = goals["net_of_interest"].get("progress_pct") or gpn.get("progress_pct")
                    goals["net_of_interest"]["current_projected_monthly_net"] = goals["net_of_interest"].get("current_projected_monthly_net") or gpn.get("current_projected_monthly_net")
                    addl = gpn.get("additional_investment_needed_now")
                    net_income = gpn.get("current_projected_monthly_net") or (float(income) - float(loan) * getattr(settings, "margin_apr_current", 0.0415) / 12.0)
                    contrib = getattr(settings, "goal_monthly_contribution", 0) or 0
                    inflow = contrib + (net_income if isinstance(net_income, (int, float)) else 0)
                    if addl is not None and inflow > 0:
                        goals["net_of_interest"]["months_to_goal"] = max(0, int(math.ceil(addl / inflow)))
                    elif addl is not None and addl <= 0:
                        goals["net_of_interest"]["months_to_goal"] = 0
            except (TypeError, ValueError):
                pass

    rate_shock = {}
    for rr in rate_rows:
        rr = dict(rr)
        # If margin_impact_pct is null in DB, derive as expense-as-%-of-income so we don't show placeholder
        mc = rr.get("new_monthly_cost")
        inc = r.get("projected_monthly_income")
        impact = rr.get("margin_impact_pct")
        if impact is None and isinstance(mc, (int, float)) and isinstance(inc, (int, float)) and inc:
            impact = round(float(mc) / float(inc) * 100.0, 2)
        rate_shock[rr["scenario"]] = {
            "new_rate_pct": rr.get("new_rate_pct"),
            "new_monthly_cost": mc,
            "income_coverage_ratio": rr.get("income_coverage"),
            "margin_impact_pct": impact,
        }
    # When no rate scenarios in DB, derive from margin_loan_balance so we don't show "margin data not available"
    if not rate_shock and isinstance(r.get("margin_loan_balance"), (int, float)) and r.get("margin_loan_balance"):
        loan = float(r["margin_loan_balance"])
        income = (r.get("projected_monthly_income") or 0) or 1.0
        apr = getattr(settings, "margin_apr_current", 0.0415)
        current_pct = round(apr * 100, 2)
        current_monthly = round(loan * (current_pct / 100.0) / 12.0, 2)
        for shock in (1.0, 2.0, 3.0):
            new_pct = current_pct + shock
            monthly_cost = round(loan * (new_pct / 100.0) / 12.0, 2)
            cov = round(income / monthly_cost, 2) if monthly_cost else None
            # margin_impact_pct = expense as % of income (so client sees impact of rate shock)
            impact_pct = round(monthly_cost / income * 100.0, 2) if income else None
            rate_shock[f"rate_plus_{int(shock * 100)}bp"] = {
                "new_rate_pct": new_pct,
                "new_monthly_cost": monthly_cost,
                "income_coverage_ratio": cov,
                "margin_impact_pct": impact_pct,
            }
    # Backfill margin.current from margin_loan_balance when flat row has null (persist used different keys)
    loan = r.get("margin_loan_balance")
    monthly_interest = r.get("monthly_interest_current")
    annual_interest = r.get("annual_interest_current")
    if (monthly_interest is None or annual_interest is None) and isinstance(loan, (int, float)) and loan:
        apr = getattr(settings, "margin_apr_current", 0.0415)
        monthly_interest = round(loan * apr / 12.0, 2)
        annual_interest = round(monthly_interest * 12.0, 2)
    income = r.get("projected_monthly_income")
    income_coverage = r.get("margin_income_coverage")
    if income_coverage is None and isinstance(monthly_interest, (int, float)) and monthly_interest and isinstance(income, (int, float)) and income:
        income_coverage = round(income / monthly_interest, 2)
    interest_to_income_pct = r.get("margin_interest_to_income_pct")
    if interest_to_income_pct is None and isinstance(monthly_interest, (int, float)) and isinstance(income, (int, float)) and income:
        interest_to_income_pct = round(monthly_interest / income * 100.0, 2)
    margin = {
        "current": {
            "monthly_interest": monthly_interest,
            "annual_interest": annual_interest,
            "income_interest_coverage": income_coverage,
            "interest_to_income_pct": interest_to_income_pct,
        },
        "stress": {
            "margin_call_risk": {
                "buffer_to_margin_call_pct": r.get("buffer_to_margin_call_pct"),
                "dollar_decline_to_call": r.get("dollar_decline_to_call"),
                "days_at_current_volatility": r.get("days_at_current_volatility"),
                "buffer_status": r.get("margin_call_buffer_status"),
            },
            "rate_shock_scenarios": rate_shock if rate_shock else None,
        },
        "guidance": _json_load(r.get("margin_guidance_json")) or {"recommended_mode": r.get("margin_guidance_selected_mode")},
        "history_90d": _json_load(r.get("margin_history_90d_json")),
    }

    div_realized = {}
    # Always provide mtd so enrichment can fill from DB when flat has null/empty
    mtd_total = r.get("dividends_realized_mtd")
    if mtd_total is not None:
        by_sym = _json_load(r.get("dividends_by_symbol_json"))
        div_realized["mtd"] = {"total_dividends": mtd_total, "by_symbol": by_sym if by_sym is not None else {}}
    else:
        div_realized["mtd"] = None
    div_realized["30d"] = {"total_dividends": r.get("dividends_realized_30d")} if r.get("dividends_realized_30d") is not None else None
    div_realized["ytd"] = {"total_dividends": r.get("dividends_realized_ytd")} if r.get("dividends_realized_ytd") is not None else None
    div_realized["qtd"] = {"total_dividends": r.get("dividends_realized_qtd")} if r.get("dividends_realized_qtd") is not None else None
    _enrich_dividends_from_db(conn, as_of, holdings_list, div_realized)
    _enrich_holding_dividends_from_db(conn, as_of, holdings_list)
    dividends = {
        "realized": div_realized,
        "projected_vs_received": _json_load(r.get("dividends_projected_vs_received_json")),
        "upcoming_this_month": {
            "events": [
                {"symbol": u["symbol"], "ex_date_est": u["ex_date_est"], "pay_date_est": u.get("pay_date_est"), "amount_est": u.get("amount_est")}
                for u in (dict(x) for x in upc_rows)
            ],
        },
    }

    macro = {
        "snapshot": {
            "vix": r.get("macro_vix"),
            "ten_year_yield": r.get("macro_ten_year_yield"),
            "two_year_yield": r.get("macro_two_year_yield"),
            "hy_spread_bps": r.get("macro_hy_spread_bps"),
            "yield_spread_10y_2y": r.get("macro_yield_spread_10y_2y"),
            "macro_stress_score": r.get("macro_stress_score"),
            "cpi_yoy": r.get("macro_cpi_yoy"),
        },
        "as_of_date": r.get("macro_data_as_of_date"),
    }

    missing_paths = _json_load(r.get("coverage_missing_paths_json"))
    meta = {
        "schema_version": r.get("schema_version"),
        "data_quality": {
            "derived_pct": r.get("coverage_derived_pct"),
            "pulled_pct": r.get("coverage_pulled_pct"),
            "missing_pct": r.get("coverage_missing_pct"),
            "filled_pct": r.get("coverage_filled_pct"),
            "missing_paths": missing_paths if missing_paths is not None else [],
        },
        "health_status": r.get("health_status"),
        "snapshot_created_at": r.get("created_at_utc"),
    }

    out = {
        "timestamps": timestamps,
        "portfolio": portfolio,
        "holdings": holdings,
        "goals": goals,
        "margin": margin,
        "dividends": dividends,
        "macro": macro,
        "meta": meta,
        "as_of_date_local": as_of,
        "created_at_utc": r.get("created_at_utc"),
    }
    if slim:
        out = slim_snapshot(out)
    return out


def _json_load(s: str | None) -> Any:
    if not s:
        return None
    try:
        return json.loads(s)
    except (TypeError, json.JSONDecodeError):
        return None


def _attribution_by_window(attr_rows: list) -> dict:
    out = {}
    for row in attr_rows:
        row = dict(row)
        w = row["window"]
        sym = row["symbol"]
        if w not in out:
            out[w] = {}
        data = {
            "contribution_pct": row.get("contribution_pct"),
            "weight_avg_pct": row.get("weight_avg_pct"),
            "return_pct": row.get("return_pct"),
        }
        if sym == "_portfolio":
            out[w]["window"] = data
        else:
            out[w][sym] = data
    # Ensure each window has a "window" aggregate; derive from per-symbol if missing
    for w in ("1m", "3m", "6m", "12m"):
        if w not in out:
            out[w] = {}
        if "window" not in out[w]:
            per_sym = [r for s, r in out[w].items() if s != "window" and isinstance(r, dict)]
            contrib_sum = sum((r.get("contribution_pct") or 0) for r in per_sym) if per_sym else None
            out[w]["window"] = {
                "contribution_pct": contrib_sum,
                "weight_avg_pct": 100.0 if contrib_sum is not None else None,
                "return_pct": contrib_sum,
            }
        # API/period expect top-level total_return_pct (same as portfolio contribution)
        window_data = out[w].get("window") or {}
        if window_data.get("contribution_pct") is not None and out[w].get("total_return_pct") is None:
            out[w]["total_return_pct"] = window_data["contribution_pct"]
    return out


def _row_to_holding_legacy(row: sqlite3.Row) -> dict:
    """Convert a daily_holdings row to V5 holding shape (cost/valuation/income/analytics)."""
    r = dict(row)
    return {
        "symbol": r.get("symbol"),
        "shares": r.get("shares"),
        "trades_count": r.get("trades_count"),
        "cost": {"cost_basis": r.get("cost_basis"), "avg_cost_per_share": r.get("avg_cost_per_share")},
        "valuation": {
            "last_price": r.get("last_price"),
            "market_value": r.get("market_value"),
            "unrealized_pnl": r.get("unrealized_pnl"),
            "unrealized_pct": r.get("unrealized_pct"),
            "portfolio_weight_pct": r.get("weight_pct"),
        },
        "income": {
            "forward_12m_dividend": r.get("forward_12m_dividend"),
            "projected_monthly_dividend": r.get("projected_monthly_dividend"),
            "projected_annual_dividend": r.get("projected_annual_dividend"),
            "current_yield_pct": r.get("current_yield_pct"),
            "yield_on_cost_pct": r.get("yield_on_cost_pct"),
            "dividends_30d": r.get("dividends_30d"),
            "dividends_qtd": r.get("dividends_qtd"),
            "dividends_ytd": r.get("dividends_ytd"),
        },
        "analytics": {
            "distribution": {
                "trailing_12m_yield_pct": r.get("trailing_12m_yield_pct"),
                "forward_yield_pct": r.get("forward_yield_pct"),
                "distribution_frequency": r.get("distribution_frequency"),
                "next_ex_date_est": r.get("next_ex_date_est"),
                "last_ex_date": r.get("last_ex_date"),
                "trailing_12m_div_per_share": r.get("trailing_12m_div_per_share"),
                "forward_12m_div_per_share": r.get("forward_12m_div_per_share"),
            },
            "risk": {
                "vol_30d_pct": r.get("vol_30d_pct"), "vol_90d_pct": r.get("vol_90d_pct"),
                "beta_3y": r.get("beta_3y"),
                "max_drawdown_1y_pct": r.get("max_drawdown_1y_pct"),
                "drawdown_duration_1y_days": r.get("drawdown_duration_1y_days"),
                "downside_dev_1y_pct": r.get("downside_dev_1y_pct"),
                "sortino_1y": r.get("sortino_1y"), "sortino_6m": r.get("sortino_6m"),
                "sortino_3m": r.get("sortino_3m"), "sortino_1m": r.get("sortino_1m"),
                "sharpe_1y": r.get("sharpe_1y"), "calmar_1y": r.get("calmar_1y"),
                "risk_quality_score": r.get("risk_quality_score"),
                "risk_quality_category": r.get("risk_quality_category"),
                "volatility_profile": r.get("volatility_profile"),
            },
            "performance": {
                "twr_1m_pct": r.get("twr_1m_pct"), "twr_3m_pct": r.get("twr_3m_pct"),
                "twr_6m_pct": r.get("twr_6m_pct"), "twr_12m_pct": r.get("twr_12m_pct"),
                "corr_1y": r.get("corr_1y"),
            },
        },
        "reliability": {
            "consistency_score": r.get("reliability_consistency_score"),
            "trend_6m": r.get("reliability_trend_6m"),
            "missed_payments_12m": r.get("reliability_missed_payments_12m"),
        },
    }


def _period_coverage_summary(conn: sqlite3.Connection, start_date: str, end_date: str) -> dict:
    """Compute average derived_pct and pulled_pct from daily snapshots in the period range."""
    rows = conn.execute(
        "SELECT coverage_derived_pct, coverage_pulled_pct FROM daily_portfolio WHERE as_of_date_local BETWEEN ? AND ?",
        (start_date, end_date),
    ).fetchall()
    derived = [float(r[0]) for r in rows if r[0] is not None]
    pulled = [float(r[1]) for r in rows if r[1] is not None]
    return {
        "derived_pct": round(sum(derived) / len(derived), 2) if derived else None,
        "pulled_pct": round(sum(pulled) / len(pulled), 2) if pulled else None,
    }


def _period_benchmark(r: dict) -> dict:
    """Build benchmark block for period snapshot; include both twr_1y object and twr_1y_pct_* for compatibility."""
    s, e, d = r.get("benchmark_twr_1y_start"), r.get("benchmark_twr_1y_end"), r.get("benchmark_twr_1y_delta")
    out = {
        "symbol": r.get("benchmark_symbol"),
        "period_return_pct": r.get("benchmark_period_return_pct"),
        "twr_1y": {"start": s, "end": e, "delta": d},
        "twr_1y_pct_start": s,
        "twr_1y_pct_end": e,
        "twr_1y_pct_delta": d,
    }
    return out


def assemble_period_snapshot(
    conn: sqlite3.Connection,
    period_type: str,
    period_end_date: str,
    period_start_date: str | None = None,
    rolling: bool = False,
) -> dict | None:
    """Reconstruct period snapshot JSON from period_summary + child tables.

    Output shape matches build_period_snapshot minus full daily snapshot embedding
    in intervals (no holdings arrays, no full goal_tiers per interval).
    """
    conn.row_factory = sqlite3.Row
    if not period_start_date:
        row = conn.execute(
            "SELECT period_start_date FROM period_summary WHERE period_type=? AND period_end_date=? AND is_rolling=? LIMIT 1",
            (period_type, period_end_date, 1 if rolling else 0),
        ).fetchone()
        if not row:
            return None
        period_start_date = row["period_start_date"]
    row = conn.execute(
        "SELECT * FROM period_summary WHERE period_type=? AND period_start_date=? AND period_end_date=? AND is_rolling=?",
        (period_type, period_start_date, period_end_date, 1 if rolling else 0),
    ).fetchone()
    if not row:
        return None
    r = dict(row)

    # ── Child tables ────────────────────────────────────────────────────
    interval_rows = conn.execute(
        """SELECT * FROM period_intervals
           WHERE period_type=? AND period_start_date=? AND period_end_date=?
           ORDER BY interval_start""",
        (period_type, period_start_date, period_end_date),
    ).fetchall()

    risk_stat_rows = conn.execute(
        """SELECT * FROM period_risk_stats
           WHERE period_type=? AND period_start_date=? AND period_end_date=?""",
        (period_type, period_start_date, period_end_date),
    ).fetchall()

    change_rows = conn.execute(
        """SELECT * FROM period_holding_changes
           WHERE period_type=? AND period_start_date=? AND period_end_date=?""",
        (period_type, period_start_date, period_end_date),
    ).fetchall()

    holdings_by_interval = {}
    try:
        pih_rows = conn.execute(
            """SELECT * FROM period_interval_holdings
               WHERE period_type=? AND period_start_date=? AND period_end_date=?
               ORDER BY interval_label, weight_pct DESC""",
            (period_type, period_start_date, period_end_date),
        ).fetchall()
        for row in pih_rows:
            row = dict(row)
            label = row.get("interval_label") or ""
            holdings_by_interval.setdefault(label, []).append({
                "symbol": row.get("symbol"),
                "weight_pct": row.get("weight_pct"),
                "market_value": row.get("market_value"),
                "pnl_pct": row.get("pnl_pct"),
                "pnl_dollar": row.get("pnl_dollar"),
                "projected_monthly_dividend": row.get("projected_monthly_dividend"),
                "current_yield_pct": row.get("current_yield_pct"),
                "sharpe_1y": row.get("sharpe_1y"),
                "sortino_1y": row.get("sortino_1y"),
                "risk_quality_category": row.get("risk_quality_category"),
            })
    except sqlite3.OperationalError:
        pass  # table may not exist before 003

    attribution_by_interval = {}
    try:
        pia_rows = conn.execute(
            """SELECT * FROM period_interval_attribution
               WHERE period_type=? AND period_start_date=? AND period_end_date=?""",
            (period_type, period_start_date, period_end_date),
        ).fetchall()
        for row in pia_rows:
            row = dict(row)
            label = row.get("interval_label") or ""
            window = row.get("window") or ""
            if label not in attribution_by_interval:
                attribution_by_interval[label] = {}
            attribution_by_interval[label][window] = {
                "total_return_pct": row.get("total_return_pct"),
                "income_contribution_pct": row.get("income_contribution_pct"),
                "price_contribution_pct": row.get("price_contribution_pct"),
                "top_contributors": _json_load(row.get("top_json")) or [],
                "bottom_contributors": _json_load(row.get("bottom_json")) or [],
            }
    except sqlite3.OperationalError:
        pass  # table may not exist before 003

    # ── Map snapshot_type from period_type ───────────────────────────────
    _type_map = {"WEEK": "weekly", "MONTH": "monthly", "QUARTER": "quarterly", "YEAR": "yearly"}
    snapshot_type = _type_map.get(period_type, period_type.lower())

    # ── Risk stats (min/avg/max) ─────────────────────────────────────────
    risk_stats = {}
    for rs in risk_stat_rows:
        rs = dict(rs)
        risk_stats[rs["metric"]] = {"avg": rs.get("avg_val"), "min": rs.get("min_val"), "max": rs.get("max_val")}

    # ── Risk start/end/delta ─────────────────────────────────────────────
    _risk_keys = [
        ("vol_30d_pct", "vol_30d"), ("vol_90d_pct", "vol_90d"),
        ("sharpe_1y", "sharpe_1y"), ("sortino_1y", "sortino_1y"),
        ("sortino_6m", "sortino_6m"), ("sortino_3m", "sortino_3m"), ("sortino_1m", "sortino_1m"),
        ("calmar_1y", "calmar_1y"), ("max_drawdown_1y_pct", "max_dd_1y"),
    ]
    risk_start = {}
    risk_end = {}
    risk_delta = {}
    for metric_key, col_prefix in _risk_keys:
        s = r.get(f"{col_prefix}_start")
        e = r.get(f"{col_prefix}_end")
        d = r.get(f"{col_prefix}_delta")
        if s is not None or e is not None:
            risk_start[metric_key] = s
            risk_end[metric_key] = e
            risk_delta[metric_key] = d
    risk_start["portfolio_risk_quality"] = r.get("portfolio_risk_quality_start")
    risk_end["portfolio_risk_quality"] = r.get("portfolio_risk_quality_end")
    risk_block = {"start": risk_start, "end": risk_end, "delta": risk_delta}
    if risk_stats:
        risk_block["stats"] = risk_stats

    def _fill_interval_period_summary_from_daily(
        conn: sqlite3.Connection, interval_end: str | None, ps: dict
    ) -> None:
        """Fill nulls in period_summary from daily snapshot for interval_end (true source)."""
        if not interval_end:
            return
        daily = assemble_daily_snapshot(conn, as_of_date=interval_end)
        if not daily:
            return
        portfolio = daily.get("portfolio") or {}
        income = portfolio.get("income") or {}
        risk = portfolio.get("risk") or {}
        volatility = risk.get("volatility") or {}
        ratios = risk.get("ratios") or {}
        drawdown = risk.get("drawdown") or {}
        var = risk.get("var") or {}
        goals = daily.get("goals") or {}
        baseline = goals.get("baseline") or {}
        net = goals.get("net_of_interest") or {}
        margin = daily.get("margin") or {}
        margin_current = margin.get("current") or {}
        mc = (margin.get("stress") or {}).get("margin_call_risk") or {}
        # Income end
        inc_end = ps.get("income") or {}
        inc_end = inc_end.get("end") or {}
        if inc_end.get("forward_12m_total") is None and income.get("forward_12m_total") is not None:
            ps.setdefault("income", {}).setdefault("end", {})["forward_12m_total"] = income["forward_12m_total"]
        if inc_end.get("portfolio_current_yield_pct") is None and income.get("portfolio_current_yield_pct") is not None:
            ps.setdefault("income", {}).setdefault("end", {})["portfolio_current_yield_pct"] = income["portfolio_current_yield_pct"]
        if inc_end.get("portfolio_yield_on_cost_pct") is None and income.get("portfolio_yield_on_cost_pct") is not None:
            ps.setdefault("income", {}).setdefault("end", {})["portfolio_yield_on_cost_pct"] = income["portfolio_yield_on_cost_pct"]
        # Risk end
        risk_end_dict = (ps.get("risk") or {}).get("end") or {}
        for key, val in [
            ("vol_30d_pct", volatility.get("vol_30d_pct")),
            ("vol_90d_pct", volatility.get("vol_90d_pct")),
            ("calmar_1y", ratios.get("calmar_1y")),
            ("max_drawdown_1y_pct", drawdown.get("max_drawdown_1y_pct")),
            ("var_90_1d_pct", var.get("var_90_1d_pct")),
            ("var_95_1d_pct", var.get("var_95_1d_pct")),
            ("cvar_90_1d_pct", var.get("cvar_90_1d_pct")),
            ("omega_ratio_1y", ratios.get("omega_ratio_1y")),
            ("ulcer_index_1y", ratios.get("ulcer_index_1y")),
            ("income_stability_score", risk.get("income_stability_score")),
            ("beta_portfolio", risk.get("beta_portfolio")),
            ("portfolio_risk_quality", risk.get("portfolio_risk_quality")),
        ]:
            if risk_end_dict.get(key) is None and val is not None:
                ps.setdefault("risk", {}).setdefault("end", {})[key] = val
        # Performance (twr_6m from daily)
        perf = ps.get("performance") or {}
        if perf.get("twr_6m_pct") is None:
            daily_twr = (portfolio.get("performance") or {}).get("twr_6m_pct")
            if daily_twr is not None:
                ps.setdefault("performance", {})["twr_6m_pct"] = daily_twr
        # Margin
        margin_ps = ps.get("margin") or {}
        if margin_ps.get("annual_interest_expense") is None:
            ann = margin_current.get("annual_interest") or margin_current.get("annual_interest_expense")
            if ann is not None:
                ps.setdefault("margin", {})["annual_interest_expense"] = ann
        if margin_ps.get("margin_call_buffer_pct") is None and mc.get("buffer_to_margin_call_pct") is not None:
            ps.setdefault("margin", {})["margin_call_buffer_pct"] = mc["buffer_to_margin_call_pct"]
        # Goal progress
        gp_end = (ps.get("goal_progress") or {}).get("end") or {}
        for key, val in [
            ("progress_pct", baseline.get("progress_pct")),
            ("months_to_goal", baseline.get("months_to_goal")),
            ("current_projected_monthly", baseline.get("current_projected_monthly")),
        ]:
            if gp_end.get(key) is None and val is not None:
                ps.setdefault("goal_progress", {}).setdefault("end", {})[key] = val
        gpn_end = (ps.get("goal_progress_net") or {}).get("end") or {}
        if gpn_end.get("progress_pct") is None and net.get("progress_pct") is not None:
            ps.setdefault("goal_progress_net", {}).setdefault("end", {})["progress_pct"] = net["progress_pct"]

    # ── Intervals ────────────────────────────────────────────────────────
    intervals = []
    for iv in interval_rows:
        iv = dict(iv)
        label = iv.get("interval_label") or ""
        attr_windows = attribution_by_interval.get(label) or {}
        interval_holdings = holdings_by_interval.get(label) or []
        ps = {
            "totals": {
                "end": {
                    "total_market_value": iv.get("mv"),
                    "cost_basis": iv.get("cost_basis"),
                    "net_liquidation_value": iv.get("nlv"),
                    "unrealized_pct": iv.get("unrealized_pct"),
                    "unrealized_pnl": iv.get("unrealized_pnl"),
                    "margin_loan_balance": iv.get("margin_loan"),
                    "margin_to_portfolio_pct": iv.get("ltv_pct"),
                },
            },
            "performance": {
                "period": {
                    "pnl_dollar_period": iv.get("pnl_period"),
                    "pnl_pct_period": iv.get("pnl_pct_period"),
                },
                "twr_1m_pct": iv.get("twr_1m_pct"),
                "twr_3m_pct": iv.get("twr_3m_pct"),
                "twr_6m_pct": iv.get("twr_6m_pct"),
                "twr_12m_pct": iv.get("twr_12m_pct"),
            },
            "risk": {
                "end": {
                    "vol_30d_pct": iv.get("vol_30d_pct"),
                    "vol_90d_pct": iv.get("vol_90d_pct"),
                    "sharpe_1y": iv.get("sharpe_1y"),
                    "sortino_1y": iv.get("sortino_1y"),
                    "sortino_6m": iv.get("sortino_6m"),
                    "sortino_3m": iv.get("sortino_3m"),
                    "sortino_1m": iv.get("sortino_1m"),
                    "calmar_1y": iv.get("calmar_1y"),
                    "max_drawdown_1y_pct": iv.get("max_drawdown_1y_pct"),
                    "var_90_1d_pct": iv.get("var_90_1d_pct"),
                    "var_95_1d_pct": iv.get("var_95_1d_pct"),
                    "cvar_90_1d_pct": iv.get("cvar_90_1d_pct"),
                    "omega_ratio_1y": iv.get("omega_ratio_1y"),
                    "ulcer_index_1y": iv.get("ulcer_index_1y"),
                    "income_stability_score": iv.get("income_stability_score"),
                    "beta_portfolio": iv.get("beta_portfolio"),
                    "portfolio_risk_quality": iv.get("portfolio_risk_quality"),
                },
            },
            "income": {
                "end": {
                    "projected_monthly_income": iv.get("monthly_income"),
                    "forward_12m_total": iv.get("forward_12m_total"),
                    "portfolio_current_yield_pct": iv.get("yield_pct"),
                    "portfolio_yield_on_cost_pct": iv.get("yield_on_cost_pct"),
                },
            },
            "margin": {
                "annual_interest_expense": iv.get("annual_interest_expense"),
                "margin_call_buffer_pct": iv.get("margin_call_buffer_pct"),
            },
            "goal_progress": {
                "end": {
                    "progress_pct": iv.get("goal_progress_pct"),
                    "months_to_goal": iv.get("goal_months_to_goal"),
                    "current_projected_monthly": iv.get("goal_projected_monthly"),
                },
            },
            "goal_progress_net": {
                "end": {
                    "progress_pct": iv.get("goal_net_progress_pct"),
                },
            },
        }
        _fill_interval_period_summary_from_daily(conn, iv.get("interval_end"), ps)
        intervals.append({
            "period": {
                "label": label,
                "start_date": iv.get("interval_start"),
                "end_date": iv.get("interval_end"),
            },
            "end_date": iv.get("interval_end"),
            "holdings": interval_holdings,
            "return_attribution_1m": attr_windows.get("1m"),
            "return_attribution_3m": attr_windows.get("3m"),
            "return_attribution_6m": attr_windows.get("6m"),
            "return_attribution_12m": attr_windows.get("12m"),
            "period_summary": ps,
        })

    # ── Portfolio changes ────────────────────────────────────────────────
    added = []
    removed = []
    top_gainers = []
    top_losers = []
    weight_increases = []
    weight_decreases = []
    for ch in change_rows:
        ch = dict(ch)
        entry = {
            "symbol": ch.get("symbol"),
            "weight_start_pct": ch.get("weight_start_pct"),
            "weight_end_pct": ch.get("weight_end_pct"),
            "weight_delta_pct": ch.get("weight_delta_pct"),
            "pnl_pct_period": ch.get("pnl_pct_period"),
            "pnl_dollar_period": ch.get("pnl_dollar_period"),
        }
        ct = ch.get("change_type")
        if ct == "added":
            added.append(entry)
        elif ct == "removed":
            removed.append(entry)
        elif ct == "top_gainer":
            top_gainers.append(entry)
        elif ct == "top_loser":
            top_losers.append(entry)
        elif ct == "weight_increase":
            weight_increases.append(entry)
        elif ct == "weight_decrease":
            weight_decreases.append(entry)
    top_gainers = sorted(top_gainers, key=lambda x: (x.get("pnl_dollar_period") or 0), reverse=True)[:5]
    top_losers = sorted(top_losers, key=lambda x: (x.get("pnl_dollar_period") or 0))[:5]
    weight_increases = sorted(weight_increases, key=lambda x: (x.get("weight_delta_pct") or 0), reverse=True)
    weight_decreases = sorted(weight_decreases, key=lambda x: (x.get("weight_delta_pct") or 0))
    portfolio_changes = {
        "holdings_added": added,
        "holdings_removed": removed,
        "top_gainers": top_gainers,
        "top_losers": top_losers,
        "weight_increases": weight_increases,
        "weight_decreases": weight_decreases,
    }

    # ── Goal tiers and goal_pace from daily at end-of-period (fallback to latest on or before) ──
    goal_tiers_block = None  # { tiers: [...], current_state: {...} } or None
    goal_pace_from_daily = {}
    try:
        # Prefer exact period_end_date; else latest daily on or before it
        dp_row = conn.execute(
            "SELECT * FROM daily_portfolio WHERE as_of_date_local=?",
            (period_end_date,),
        ).fetchone()
        if not dp_row:
            dp_row = conn.execute(
                "SELECT * FROM daily_portfolio WHERE as_of_date_local <= ? ORDER BY as_of_date_local DESC LIMIT 1",
                (period_end_date,),
            ).fetchone()
        daily_date = dp_row["as_of_date_local"] if dp_row else None
        if dp_row:
            dp_row = dict(dp_row)
            goal_pace_from_daily = {
                "current_pace": {
                    "months_ahead_behind": dp_row.get("goal_pace_months_ahead_behind"),
                    "pace_category": dp_row.get("goal_pace_category"),
                    "on_track": bool(dp_row.get("goal_pace_on_track")) if dp_row.get("goal_pace_on_track") is not None else None,
                    "revised_goal_date": dp_row.get("goal_pace_revised_goal_date"),
                    "pct_of_tier_pace": dp_row.get("goal_pace_pct_of_tier"),
                },
                "likely_tier": {
                    "tier": dp_row.get("goal_likely_tier"),
                    "name": dp_row.get("goal_likely_tier_name"),
                    "confidence": dp_row.get("goal_likely_tier_confidence"),
                    "reason": dp_row.get("goal_likely_tier_reason"),
                },
                "windows": _json_load(dp_row.get("goal_pace_windows_json")),
                "baseline_projection": _json_load(dp_row.get("goal_pace_baseline_json")),
            }
        if daily_date:
            tier_rows = conn.execute(
                "SELECT * FROM daily_goal_tiers WHERE as_of_date_local=? ORDER BY tier",
                (daily_date,),
            ).fetchall()
            if tier_rows:
                tiers = []
                for tr in tier_rows:
                    row = dict(tr)
                    # Shape like API contract: nested assumptions, omit as_of_date_local
                    tier = {
                        "tier": row.get("tier"),
                        "name": row.get("name"),
                        "description": row.get("description"),
                        "target_monthly": row.get("target_monthly"),
                        "required_portfolio_value": row.get("required_portfolio_value"),
                        "final_portfolio_value": row.get("final_portfolio_value"),
                        "progress_pct": row.get("progress_pct"),
                        "months_to_goal": row.get("months_to_goal"),
                        "estimated_goal_date": row.get("estimated_goal_date"),
                        "confidence": row.get("confidence"),
                        "assumptions": {
                            "monthly_contribution": row.get("assumption_monthly_contribution"),
                            "drip_enabled": bool(row.get("assumption_drip_enabled")) if row.get("assumption_drip_enabled") is not None else None,
                            "annual_appreciation_pct": row.get("assumption_annual_appreciation_pct"),
                            "ltv_maintained": bool(row.get("assumption_ltv_maintained")) if row.get("assumption_ltv_maintained") is not None else None,
                            "target_ltv_pct": row.get("assumption_target_ltv_pct")
                            if row.get("assumption_target_ltv_pct") is not None
                            else (30.0 if row.get("assumption_ltv_maintained") else None),
                        },
                    }
                    tiers.append(tier)
                goal_tiers_block = {
                    "tiers": tiers,
                    "current_state": {
                        "portfolio_value": dp_row.get("goal_tiers_portfolio_value") if dp_row else None,
                        "projected_monthly_income": dp_row.get("goal_tiers_projected_monthly") if dp_row else None,
                        "portfolio_yield_pct": dp_row.get("goal_tiers_yield_pct") if dp_row else None,
                    } if dp_row else None,
                }
    except sqlite3.OperationalError:
        pass

    # ── Assemble (matches build_period_snapshot shape) ───────────────────
    return {
        "summary_id": f"{snapshot_type}_{period_end_date}",
        "snapshot_type": snapshot_type,
        "snapshot_mode": r.get("snapshot_mode") or ("to_date" if rolling else "final"),
        "as_of": period_end_date,
        "period": {
            "label": r.get("period_label"),
            "start_date": period_start_date,
            "end_date": period_end_date,
            "expected_days": r.get("expected_days"),
            "observed_days": r.get("observed_days"),
            "coverage_pct": r.get("coverage_pct"),
            "is_complete": bool(r.get("is_complete")) if r.get("is_complete") is not None else None,
        },
        "benchmark": _period_benchmark(r),
        "period_summary": {
            "goal_tiers": goal_tiers_block,
            "goal_pace": (
                goal_pace_from_daily
                if (goal_pace_from_daily.get("current_pace") or goal_pace_from_daily.get("likely_tier"))
                else {"current_pace": {"pace_category": r.get("goal_pace_category_end"), "months_ahead_behind": r.get("goal_pace_months_end")}}
            ),
            "totals": {
                "start": {
                    "total_market_value": r.get("mv_start"),
                    "net_liquidation_value": r.get("nlv_start"),
                    "cost_basis": r.get("cost_basis_start"),
                    "unrealized_pct": r.get("unrealized_pct_start"),
                    "unrealized_pnl": r.get("unrealized_pnl_start"),
                    "margin_loan_balance": r.get("margin_balance_start"),
                    "margin_to_portfolio_pct": r.get("ltv_pct_start"),
                },
                "end": {
                    "total_market_value": r.get("mv_end"),
                    "net_liquidation_value": r.get("nlv_end"),
                    "cost_basis": r.get("cost_basis_end"),
                    "unrealized_pct": r.get("unrealized_pct_end"),
                    "unrealized_pnl": r.get("unrealized_pnl_end"),
                    "margin_loan_balance": r.get("margin_balance_end"),
                    "margin_to_portfolio_pct": r.get("ltv_pct_end"),
                },
                "delta": {
                    "total_market_value": r.get("mv_delta"),
                    "net_liquidation_value": r.get("nlv_delta"),
                    "cost_basis": r.get("cost_basis_delta"),
                    "unrealized_pct": r.get("unrealized_pct_delta"),
                    "unrealized_pnl": r.get("unrealized_pnl_delta"),
                    "margin_loan_balance": r.get("margin_balance_delta"),
                    "margin_to_portfolio_pct": r.get("ltv_pct_delta"),
                },
            },
            "income": {
                "start": {
                    "projected_monthly_income": r.get("monthly_income_start"),
                    "forward_12m_total": r.get("forward_12m_start"),
                    "portfolio_current_yield_pct": r.get("yield_start"),
                    "portfolio_yield_on_cost_pct": r.get("yield_on_cost_start"),
                },
                "end": {
                    "projected_monthly_income": r.get("monthly_income_end"),
                    "forward_12m_total": r.get("forward_12m_end"),
                    "portfolio_current_yield_pct": r.get("yield_end"),
                    "portfolio_yield_on_cost_pct": r.get("yield_on_cost_end"),
                },
                "delta": {
                    "projected_monthly_income": r.get("monthly_income_delta"),
                    "forward_12m_total": r.get("forward_12m_delta"),
                    "portfolio_current_yield_pct": r.get("yield_delta"),
                    "portfolio_yield_on_cost_pct": r.get("yield_on_cost_delta"),
                },
            },
            "performance": {
                "period": {
                    "twr_period_pct": r.get("twr_period_pct"),
                    "pnl_dollar_period": r.get("pnl_dollar_period"),
                    "pnl_pct_period": r.get("pnl_pct_period"),
                },
                "twr_windows": {
                    "twr_1m_pct_start": r.get("twr_1m_start"), "twr_1m_pct_end": r.get("twr_1m_end"), "twr_1m_pct_delta": r.get("twr_1m_delta"),
                    "twr_3m_pct_start": r.get("twr_3m_start"), "twr_3m_pct_end": r.get("twr_3m_end"), "twr_3m_pct_delta": r.get("twr_3m_delta"),
                    "twr_6m_pct_start": r.get("twr_6m_start"), "twr_6m_pct_end": r.get("twr_6m_end"), "twr_6m_pct_delta": r.get("twr_6m_delta"),
                    "twr_12m_pct_start": r.get("twr_12m_start"), "twr_12m_pct_end": r.get("twr_12m_end"), "twr_12m_pct_delta": r.get("twr_12m_delta"),
                },
            },
            "risk": risk_block,
            "goal_progress": {
                "start": {
                    "progress_pct": r.get("goal_progress_pct_start"),
                    "current_projected_monthly": r.get("goal_monthly_start"),
                    "months_to_goal": r.get("goal_months_to_goal_start"),
                },
                "end": {
                    "progress_pct": r.get("goal_progress_pct_end"),
                    "current_projected_monthly": r.get("goal_monthly_end"),
                    "months_to_goal": r.get("goal_months_to_goal_end"),
                },
                "delta": {
                    "progress_pct": r.get("goal_progress_pct_delta"),
                    "current_projected_monthly": r.get("goal_monthly_delta"),
                    "months_to_goal": r.get("goal_months_to_goal_delta"),
                },
            },
            "goal_progress_net": {
                "start": {
                    "progress_pct": r.get("goal_net_progress_pct_start"),
                    "current_projected_monthly_net": r.get("goal_net_monthly_start"),
                },
                "end": {
                    "progress_pct": r.get("goal_net_progress_pct_end"),
                    "current_projected_monthly_net": r.get("goal_net_monthly_end"),
                },
                "delta": {
                    "progress_pct": r.get("goal_net_progress_pct_delta"),
                    "current_projected_monthly_net": r.get("goal_net_monthly_delta"),
                },
            },
            "composition": {
                "start": {"holding_count": r.get("holding_count_start"), "concentration_top5_pct": r.get("concentration_top5_start")},
                "end": {"holding_count": r.get("holding_count_end"), "concentration_top5_pct": r.get("concentration_top5_end")},
                "delta": {"holding_count": r.get("holding_count_delta"), "concentration_top5_pct": r.get("concentration_top5_delta")},
            },
            "macro": {
                "start": {"ten_year_yield": r.get("macro_10y_start"), "two_year_yield": r.get("macro_2y_start"), "vix": r.get("macro_vix_start"), "cpi_yoy": r.get("macro_cpi_start")},
                "end": {"ten_year_yield": r.get("macro_10y_end"), "two_year_yield": r.get("macro_2y_end"), "vix": r.get("macro_vix_end"), "cpi_yoy": r.get("macro_cpi_end")},
                "avg": {"ten_year_yield": r.get("macro_10y_avg"), "two_year_yield": r.get("macro_2y_avg"), "vix": r.get("macro_vix_avg"), "cpi_yoy": r.get("macro_cpi_avg")},
                "delta": {"ten_year_yield": r.get("macro_10y_delta"), "two_year_yield": r.get("macro_2y_delta"), "vix": r.get("macro_vix_delta"), "cpi_yoy": r.get("macro_cpi_delta")},
            },
        },
        "portfolio_changes": portfolio_changes,
        "intervals": intervals,
        "summary": _period_coverage_summary(conn, period_start_date, period_end_date),
    }


SIGNIFICANT_MISSING_PCT = 1.0


def slim_snapshot(snapshot: dict, missing_pct_threshold: float = SIGNIFICANT_MISSING_PCT) -> dict:
    if not isinstance(snapshot, dict):
        return snapshot
    missing_pct = _missing_pct(snapshot)
    keep_notes = missing_pct is not None and missing_pct >= missing_pct_threshold
    return _strip_noise(snapshot, keep_notes=keep_notes)


def _missing_pct(snapshot: dict) -> float | None:
    # V4: top-level coverage key
    coverage = snapshot.get("coverage")
    if isinstance(coverage, dict):
        val = coverage.get("missing_pct")
        if isinstance(val, (int, float)):
            return float(val)
    # V5: coverage moved to meta.data_quality
    dq = (snapshot.get("meta") or {}).get("data_quality")
    if isinstance(dq, dict):
        val = dq.get("missing_pct")
        if isinstance(val, (int, float)):
            return float(val)
    return None


def _strip_noise(obj, keep_notes: bool, parent_key: str | None = None):
    if isinstance(obj, dict):
        out = {}
        for key, value in obj.items():
            key_lower = str(key).lower()
            if "provenance" in key_lower:
                continue
            if key_lower == "notes" and not keep_notes:
                continue
            if parent_key == "meta" and key_lower in ("cache", "cache_control"):
                continue
            out[key] = _strip_noise(value, keep_notes=keep_notes, parent_key=str(key))
        return out
    if isinstance(obj, list):
        return [_strip_noise(item, keep_notes=keep_notes, parent_key=parent_key) for item in obj]
    return obj
