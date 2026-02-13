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
    _dividend_reliability_metrics,
    _dividend_window,
    _frequency_from_ex_dates,
    _frequency_from_recent_ex_dates,
    _build_pay_history,
    _build_position_index,
    _load_first_acquired_dates,
    _load_dividend_transactions,
    _load_provider_dividends,
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
            tot = round(sum(e["amount"] for e in events if e["date"] >= cut_30d and isinstance(e.get("amount"), (int, float))), 2)
            inc["dividends_30d"] = tot if tot != 0 else 0.0
        if inc.get("dividends_qtd") is None:
            tot = round(sum(e["amount"] for e in events if e["date"] >= cut_qtd and isinstance(e.get("amount"), (int, float))), 2)
            inc["dividends_qtd"] = tot if tot != 0 else 0.0
        if inc.get("dividends_ytd") is None:
            tot = round(sum(e["amount"] for e in events if e["date"] >= cut_ytd and isinstance(e.get("amount"), (int, float))), 2)
            inc["dividends_ytd"] = tot if tot != 0 else 0.0


def _enrich_holding_reliability_from_db(
    conn: sqlite3.Connection,
    as_of_str: str,
    holdings_list: list[dict],
) -> None:
    """Backfill holdings[].reliability from source dividend history when flat values are stale."""
    as_of_dt = _parse_date(as_of_str)
    if not as_of_dt or not holdings_list:
        return

    symbols_to_fill: set[str] = set()
    for h in holdings_list:
        sym = h.get("symbol")
        if not sym:
            continue
        rel = h.get("reliability") if isinstance(h.get("reliability"), dict) else {}
        score = rel.get("consistency_score")
        trend = rel.get("trend_6m")
        if trend is None or not isinstance(score, (int, float)) or score <= 0:
            symbols_to_fill.add(str(sym).upper())
    if not symbols_to_fill:
        return

    div_tx = _load_dividend_transactions(conn)
    provider_divs = _load_provider_dividends(conn)
    if not provider_divs:
        return
    pay_history = _build_pay_history(div_tx)
    first_acquired = _load_first_acquired_dates(conn)
    position_index = _build_position_index(conn, symbols_to_fill)

    for h in holdings_list:
        sym_raw = h.get("symbol")
        if not sym_raw:
            continue
        sym = str(sym_raw).upper()
        if sym not in symbols_to_fill:
            continue
        provider_events = provider_divs.get(sym, [])
        ex_dates = sorted(
            ev.get("ex_date")
            for ev in provider_events
            if ev.get("ex_date") and ev.get("ex_date") <= as_of_dt
        )
        expected_frequency = (
            _frequency_from_recent_ex_dates(ex_dates, recent=6)
            or _frequency_from_ex_dates(ex_dates)
        )
        try:
            rel = _dividend_reliability_metrics(
                sym,
                div_tx,
                provider_divs,
                pay_history,
                expected_frequency,
                as_of_dt,
                first_acquired.get(sym),
                position_index,
            )
        except Exception:
            continue

        rel_out = h.get("reliability")
        if not isinstance(rel_out, dict):
            rel_out = {}
            h["reliability"] = rel_out
        rel_out["consistency_score"] = rel.get("consistency_score")
        rel_out["trend_6m"] = rel.get("trend_6m") or rel.get("dividend_trend_6m")
        rel_out["missed_payments_12m"] = rel.get("missed_payments_12m")


def _daily_reliability_score_guide() -> dict:
    return {
        "version": "1.0",
        "consistency_score": {
            "range": [0.0, 1.0],
            "method": (
                "Weighted blend of payout-amount stability, payment hit-rate, "
                "and payment-timing consistency, with adjustments for 6m trend and dividend cuts."
            ),
            "bands": [
                {"label": "excellent", "min": 0.85, "max": 1.0, "meaning": "very stable payouts and reliable payment behavior"},
                {"label": "good", "min": 0.7, "max": 0.849, "meaning": "mostly stable with minor fluctuation"},
                {"label": "watch", "min": 0.5, "max": 0.699, "meaning": "noticeable variability or weaker payment reliability"},
                {"label": "weak", "min": 0.0, "max": 0.499, "meaning": "high fluctuation, missed payments, or repeated cuts"},
            ],
        },
        "trend_6m": {
            "values": ["growing", "stable", "declining", "insufficient_history"],
            "notes": "Calculated from trailing 6-month per-share dividend event trend.",
        },
        "missed_payments_12m": {
            "definition": "Expected provider payments due in last 12 months that were not observed as received.",
            "better_is": "lower",
        },
    }


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
    _enrich_holding_reliability_from_db(conn, as_of, holdings_list)
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
    out["reliability_score_guide"] = _daily_reliability_score_guide()
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
            # Type safety: only sum numeric contribution_pct values
            contrib_sum = sum((r.get("contribution_pct") or 0) for r in per_sym if isinstance(r.get("contribution_pct"), (int, float))) if per_sym else None
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

    # ── Activity section (NEW) ───────────────────────────────────────────
    activity_block = None
    try:
        activity_row = conn.execute(
            """SELECT * FROM period_activity
               WHERE period_type=? AND period_start_date=? AND period_end_date=?""",
            (period_type, period_start_date, period_end_date),
        ).fetchone()

        if activity_row:
            activity_row = dict(activity_row)

            # Contributions
            contribution_rows = conn.execute(
                """SELECT contribution_date, amount FROM period_contributions
                   WHERE period_type=? AND period_start_date=? AND period_end_date=?
                   ORDER BY contribution_date""",
                (period_type, period_start_date, period_end_date),
            ).fetchall()

            # Withdrawals
            withdrawal_rows = conn.execute(
                """SELECT withdrawal_date, amount FROM period_withdrawals
                   WHERE period_type=? AND period_start_date=? AND period_end_date=?
                   ORDER BY withdrawal_date""",
                (period_type, period_start_date, period_end_date),
            ).fetchall()

            # Dividends
            dividend_rows = conn.execute(
                """SELECT symbol, ex_date, pay_date, amount FROM period_dividend_events
                   WHERE period_type=? AND period_start_date=? AND period_end_date=?
                   ORDER BY pay_date""",
                (period_type, period_start_date, period_end_date),
            ).fetchall()

            # Trades
            trade_rows = conn.execute(
                """SELECT symbol, buy_count, sell_count FROM period_trades
                   WHERE period_type=? AND period_start_date=? AND period_end_date=?""",
                (period_type, period_start_date, period_end_date),
            ).fetchall()

            # Position lists
            position_rows = conn.execute(
                """SELECT list_type, symbol FROM period_position_lists
                   WHERE period_type=? AND period_start_date=? AND period_end_date=?""",
                (period_type, period_start_date, period_end_date),
            ).fetchall()

            # Build by_symbol dicts
            dividends_by_symbol = {}
            for row in dividend_rows:
                symbol = row[0]
                amount = row[3]
                dividends_by_symbol[symbol] = dividends_by_symbol.get(symbol, 0) + amount

            trades_by_symbol = {}
            for row in trade_rows:
                symbol = row[0]
                trades_by_symbol[symbol] = {
                    "buy_count": row[1] or 0,
                    "sell_count": row[2] or 0
                }

            positions_added = []
            positions_removed = []
            positions_increased = []
            positions_decreased = []
            for list_type, symbol in position_rows:
                if not symbol:
                    continue
                if list_type == "added":
                    positions_added.append(symbol)
                elif list_type == "removed":
                    positions_removed.append(symbol)
                elif list_type == "increased":
                    positions_increased.append(symbol)
                elif list_type == "decreased":
                    positions_decreased.append(symbol)

            activity_block = {
                "contributions": {
                    "total": activity_row.get("contributions_total") or 0,
                    "count": activity_row.get("contributions_count") or 0,
                    "dates": [row[0] for row in contribution_rows]
                },
                "withdrawals": {
                    "total": activity_row.get("withdrawals_total") or 0,
                    "count": activity_row.get("withdrawals_count") or 0,
                    "dates": [row[0] for row in withdrawal_rows]
                },
                "dividends": {
                    "total_received": activity_row.get("dividends_total_received") or 0,
                    "count": activity_row.get("dividends_count") or 0,
                    "by_symbol": dividends_by_symbol,
                    "events": [
                        {
                            "symbol": row[0],
                            "ex_date": row[1],
                            "pay_date": row[2],
                            "amount": row[3]
                        }
                        for row in dividend_rows
                    ]
                },
                "interest": {
                    "total_paid": activity_row.get("interest_total_paid") or 0,
                    "avg_daily_balance": activity_row.get("interest_avg_daily_balance") or 0,
                    "avg_rate_pct": activity_row.get("interest_avg_rate_pct") or 0,
                    "annualized": activity_row.get("interest_annualized") or 0
                },
                "trades": {
                    "total_count": activity_row.get("trades_total_count") or 0,
                    "buy_count": activity_row.get("trades_buy_count") or 0,
                    "sell_count": activity_row.get("trades_sell_count") or 0,
                    "by_symbol": trades_by_symbol
                },
                "positions": {
                    "added": sorted(positions_added),
                    "removed": sorted(positions_removed),
                    "symbols_increased": sorted(positions_increased),
                    "symbols_decreased": sorted(positions_decreased)
                },
                "margin": {
                    "borrowed": activity_row.get("margin_borrowed") or 0,
                    "repaid": activity_row.get("margin_repaid") or 0,
                    "net_change": activity_row.get("margin_net_change") or 0
                }
            }
    except sqlite3.OperationalError:
        pass

    # ── Macro period stats (NEW) ─────────────────────────────────────────
    macro_period_stats = {}
    try:
        macro_stat_rows = conn.execute(
            """SELECT metric, avg_val, min_val, max_val, std_val, min_date, max_date FROM period_macro_stats
               WHERE period_type=? AND period_start_date=? AND period_end_date=?""",
            (period_type, period_start_date, period_end_date),
        ).fetchall()

        for row in macro_stat_rows:
            metric = row[0]
            macro_period_stats[metric] = {
                "avg": row[1],
                "min": row[2],
                "max": row[3],
                "std": row[4],
                "min_date": row[5],
                "max_date": row[6]
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
                "period_stats": macro_period_stats if macro_period_stats else None,
            },
            "period_stats": {
                "market_value": {
                    "avg": r.get("mv_avg"),
                    "min": r.get("mv_min"),
                    "max": r.get("mv_max"),
                    "std": r.get("mv_std"),
                    "min_date": r.get("mv_min_date"),
                    "max_date": r.get("mv_max_date"),
                },
                "net_liquidation_value": {
                    "avg": r.get("nlv_avg"),
                    "min": r.get("nlv_min"),
                    "max": r.get("nlv_max"),
                    "std": r.get("nlv_std"),
                    "min_date": r.get("nlv_min_date"),
                    "max_date": r.get("nlv_max_date"),
                },
                "margin_to_portfolio_pct": {
                    "avg": r.get("margin_to_portfolio_pct_avg"),
                    "min": r.get("margin_to_portfolio_pct_min"),
                    "max": r.get("margin_to_portfolio_pct_max"),
                    "std": r.get("margin_to_portfolio_pct_std"),
                },
                "projected_monthly_income": {
                    "avg": r.get("projected_monthly_avg"),
                    "min": r.get("projected_monthly_min"),
                    "max": r.get("projected_monthly_max"),
                },
                "yield_pct": {
                    "avg": r.get("yield_pct_avg"),
                    "min": r.get("yield_pct_min"),
                    "max": r.get("yield_pct_max"),
                },
            },
            "period_drawdown": {
                "period_max_drawdown_pct": r.get("period_max_drawdown_pct"),
                "period_max_drawdown_date": r.get("period_max_drawdown_date"),
                "period_recovery_date": r.get("period_recovery_date"),
                "period_days_in_drawdown": r.get("period_days_in_drawdown"),
                "period_drawdown_count": r.get("period_drawdown_count"),
            },
            "period_var": {
                "avg_var_95_1d_pct": r.get("avg_var_95_1d_pct"),
                "days_exceeding_var_95": r.get("days_exceeding_var_95"),
                "worst_day_return_pct": r.get("worst_day_return_pct"),
                "worst_day_date": r.get("worst_day_date"),
                "best_day_return_pct": r.get("best_day_return_pct"),
                "best_day_date": r.get("best_day_date"),
            },
            "margin_safety": {
                "min_buffer_to_call_pct": r.get("min_buffer_to_call_pct"),
                "min_buffer_date": r.get("min_buffer_date"),
                "days_below_50pct_buffer": r.get("days_below_50pct_buffer"),
                "days_below_40pct_buffer": r.get("days_below_40pct_buffer"),
                "margin_call_events": r.get("margin_call_events"),
            },
        },
        "activity": activity_block,
        "portfolio_changes": portfolio_changes,
        "intervals": intervals,
        "summary": _period_coverage_summary(conn, period_start_date, period_end_date),
    }


def _to_float(val: Any) -> float | None:
    if isinstance(val, bool):
        return None
    if isinstance(val, (int, float)):
        return float(val)
    return None


def _avg(values: list[float], digits: int = 3) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), digits)


def _pct_change(delta: Any, base: Any, digits: int = 3) -> float | None:
    d = _to_float(delta)
    b = _to_float(base)
    if d is None or b is None or abs(b) < 1e-12:
        return None
    return round((d / b) * 100.0, digits)


def _coalesce(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _series_stats(values: list[float], digits: int = 3) -> dict[str, float | None]:
    if not values:
        return {"avg": None, "min": None, "max": None, "std": None}
    if len(values) == 1:
        return {
            "avg": round(values[0], digits),
            "min": round(values[0], digits),
            "max": round(values[0], digits),
            "std": None,
        }
    mean_val = sum(values) / len(values)
    variance = sum((v - mean_val) ** 2 for v in values) / len(values)
    return {
        "avg": round(mean_val, digits),
        "min": round(min(values), digits),
        "max": round(max(values), digits),
        "std": round(math.sqrt(variance), digits),
    }


def _twr_window_stats(daily_rows: list[dict], key: str) -> dict[str, float | None]:
    values = [_to_float(row.get(key)) for row in daily_rows]
    values = [v for v in values if v is not None]
    if not values:
        return {"avg": None, "min": None, "max": None, "end": None}
    stats = _series_stats(values, digits=3)
    return {
        "avg": stats["avg"],
        "min": stats["min"],
        "max": stats["max"],
        "end": round(values[-1], 3),
    }


def _local_date_to_utc_iso(date_text: str | None) -> str | None:
    dt = _parse_date(date_text)
    if not dt:
        return None
    local_tz = ZoneInfo(settings.local_tz)
    local_dt = datetime(dt.year, dt.month, dt.day, 0, 0, 0, tzinfo=local_tz)
    return local_dt.astimezone(timezone.utc).isoformat()


def _add_months(base_date: str | None, months: Any) -> str | None:
    dt = _parse_date(base_date)
    m = _to_float(months)
    if not dt or m is None:
        return None
    total_months = dt.year * 12 + (dt.month - 1) + int(round(m))
    year = total_months // 12
    month = total_months % 12 + 1
    # Keep day bounded to month length.
    if month == 2:
        leap = year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
        max_day = 29 if leap else 28
    elif month in (4, 6, 9, 11):
        max_day = 30
    else:
        max_day = 31
    day = min(dt.day, max_day)
    return date(year, month, day).isoformat()


def _calendar_period(snapshot_type: str, start_dt: date) -> str:
    if snapshot_type == "weekly":
        iso = start_dt.isocalendar()
        return f"{iso.year}-W{iso.week:02d}"
    if snapshot_type == "monthly":
        return f"{start_dt.year}-{start_dt.month:02d}"
    if snapshot_type == "quarterly":
        q = (start_dt.month - 1) // 3 + 1
        return f"{start_dt.year}-Q{q}"
    return f"{start_dt.year}"


def assemble_period_snapshot_target(
    conn: sqlite3.Connection,
    period_type: str,
    period_end_date: str,
    period_start_date: str | None = None,
    rolling: bool = False,
) -> dict | None:
    """Assemble period response in target schema shape from flat DB tables."""
    conn.row_factory = sqlite3.Row

    if not period_start_date:
        start_row = conn.execute(
            "SELECT period_start_date FROM period_summary WHERE period_type=? AND period_end_date=? AND is_rolling=? LIMIT 1",
            (period_type, period_end_date, 1 if rolling else 0),
        ).fetchone()
        if not start_row:
            return None
        period_start_date = start_row["period_start_date"]

    ps_row = conn.execute(
        """SELECT * FROM period_summary
           WHERE period_type=? AND period_start_date=? AND period_end_date=? AND is_rolling=?""",
        (period_type, period_start_date, period_end_date, 1 if rolling else 0),
    ).fetchone()
    if not ps_row:
        return None
    r = dict(ps_row)

    start_dt = _parse_date(period_start_date)
    end_dt = _parse_date(period_end_date)
    if not start_dt or not end_dt:
        return None

    _type_map = {"WEEK": "weekly", "MONTH": "monthly", "QUARTER": "quarterly", "YEAR": "yearly"}
    snapshot_type = _type_map.get(period_type, period_type.lower())

    daily_rows = [
        dict(row)
        for row in conn.execute(
            """SELECT * FROM daily_portfolio
               WHERE as_of_date_local BETWEEN ? AND ?
               ORDER BY as_of_date_local""",
            (period_start_date, period_end_date),
        ).fetchall()
    ]
    observed_dates = [row.get("as_of_date_local") for row in daily_rows if row.get("as_of_date_local")]
    observed_set = set(observed_dates)
    start_daily = next((row for row in daily_rows if row.get("as_of_date_local") == period_start_date), None)
    end_daily = next((row for row in reversed(daily_rows) if row.get("as_of_date_local") == period_end_date), None)
    if not start_daily and daily_rows:
        start_daily = daily_rows[0]
    if not end_daily and daily_rows:
        end_daily = daily_rows[-1]

    pace_months_start_fallback = _to_float(start_daily.get("goal_pace_months_ahead_behind")) if start_daily else None
    pace_months_end_fallback = _to_float(end_daily.get("goal_pace_months_ahead_behind")) if end_daily else None
    pace_months_delta_fallback = (
        round(pace_months_end_fallback - pace_months_start_fallback, 2)
        if isinstance(pace_months_start_fallback, (int, float))
        and isinstance(pace_months_end_fallback, (int, float))
        else None
    )
    pace_tier_start_fallback = _to_float(start_daily.get("goal_pace_pct_of_tier")) if start_daily else None
    pace_tier_end_fallback = _to_float(end_daily.get("goal_pace_pct_of_tier")) if end_daily else None

    days_in_period = (end_dt - start_dt).days + 1
    snapshots_expected = r.get("expected_days") or days_in_period
    snapshots_count = r.get("observed_days") or len(daily_rows)
    coverage_pct = r.get("coverage_pct")
    if coverage_pct is None and snapshots_expected:
        coverage_pct = round((snapshots_count / snapshots_expected) * 100.0, 3)

    missing_dates = []
    cur_day = start_dt
    while cur_day <= end_dt:
        day_text = cur_day.isoformat()
        if day_text not in observed_set:
            missing_dates.append(day_text)
        cur_day += timedelta(days=1)

    def _series_pairs(column: str) -> list[tuple[str, float]]:
        pairs: list[tuple[str, float]] = []
        for row in daily_rows:
            day = row.get("as_of_date_local")
            val = _to_float(row.get(column))
            if day and val is not None:
                pairs.append((day, val))
        return pairs

    def _series_stats_from_pairs(pairs: list[tuple[str, float]], digits: int = 3) -> dict[str, Any]:
        values = [val for _, val in pairs]
        if not values:
            return {"avg": None, "min": None, "max": None, "std": None, "min_date": None, "max_date": None}
        stats = _series_stats(values, digits=digits)
        min_val = stats["min"]
        max_val = stats["max"]
        min_date = next((day for day, val in pairs if val == min_val), None)
        max_date = next((day for day, val in pairs if val == max_val), None)
        return {
            "avg": stats["avg"],
            "min": stats["min"],
            "max": stats["max"],
            "std": stats["std"],
            "min_date": min_date,
            "max_date": max_date,
        }

    nlv_pairs = _series_pairs("net_liquidation_value")
    ltv_pairs = _series_pairs("ltv_pct")
    margin_balance_pairs = _series_pairs("margin_loan_balance")
    projected_monthly_pairs = _series_pairs("projected_monthly_income")
    yield_pairs = _series_pairs("portfolio_yield_pct")
    vol_30d_pairs = _series_pairs("vol_30d_pct")
    vol_90d_pairs = _series_pairs("vol_90d_pct")
    sharpe_pairs = _series_pairs("sharpe_1y")
    sortino_pairs = _series_pairs("sortino_1y")
    calmar_pairs = _series_pairs("calmar_1y")
    buffer_pairs = _series_pairs("buffer_to_margin_call_pct")

    nlv_fallback = _series_stats_from_pairs(nlv_pairs)
    ltv_fallback = _series_stats_from_pairs(ltv_pairs)
    margin_balance_fallback = _series_stats_from_pairs(margin_balance_pairs)
    projected_monthly_fallback = _series_stats_from_pairs(projected_monthly_pairs)
    yield_fallback = _series_stats_from_pairs(yield_pairs)
    vol_30d_fallback = _series_stats_from_pairs(vol_30d_pairs)
    vol_90d_fallback = _series_stats_from_pairs(vol_90d_pairs)
    sharpe_fallback = _series_stats_from_pairs(sharpe_pairs)
    sortino_fallback = _series_stats_from_pairs(sortino_pairs)
    calmar_fallback = _series_stats_from_pairs(calmar_pairs)
    buffer_fallback = _series_stats_from_pairs(buffer_pairs)
    buffer_days_below_50 = sum(1 for _, val in buffer_pairs if val < 50) if buffer_pairs else None
    buffer_days_below_40 = sum(1 for _, val in buffer_pairs if val < 40) if buffer_pairs else None
    buffer_call_events = sum(1 for _, val in buffer_pairs if val < 0) if buffer_pairs else None

    # Activity and child tables
    activity_row = conn.execute(
        """SELECT * FROM period_activity
           WHERE period_type=? AND period_start_date=? AND period_end_date=?""",
        (period_type, period_start_date, period_end_date),
    ).fetchone()
    activity = dict(activity_row) if activity_row else {}

    contribution_rows = conn.execute(
        """SELECT contribution_date, amount, account_id
           FROM period_contributions
           WHERE period_type=? AND period_start_date=? AND period_end_date=?
           ORDER BY contribution_date""",
        (period_type, period_start_date, period_end_date),
    ).fetchall()
    withdrawal_rows = conn.execute(
        """SELECT withdrawal_date, amount, account_id
           FROM period_withdrawals
           WHERE period_type=? AND period_start_date=? AND period_end_date=?
           ORDER BY withdrawal_date""",
        (period_type, period_start_date, period_end_date),
    ).fetchall()
    dividend_rows = conn.execute(
        """SELECT symbol, ex_date, pay_date, amount
           FROM period_dividend_events
           WHERE period_type=? AND period_start_date=? AND period_end_date=?
           ORDER BY ex_date, symbol""",
        (period_type, period_start_date, period_end_date),
    ).fetchall()
    trade_rows = conn.execute(
        """SELECT symbol, buy_count, sell_count
           FROM period_trades
           WHERE period_type=? AND period_start_date=? AND period_end_date=?
           ORDER BY symbol""",
        (period_type, period_start_date, period_end_date),
    ).fetchall()
    position_rows = conn.execute(
        """SELECT list_type, symbol
           FROM period_position_lists
           WHERE period_type=? AND period_start_date=? AND period_end_date=?
           ORDER BY list_type, symbol""",
        (period_type, period_start_date, period_end_date),
    ).fetchall()
    macro_stat_rows = conn.execute(
        """SELECT metric, avg_val, min_val, max_val, std_val
           FROM period_macro_stats
           WHERE period_type=? AND period_start_date=? AND period_end_date=?""",
        (period_type, period_start_date, period_end_date),
    ).fetchall()

    contribution_dates = sorted({row[0] for row in contribution_rows if row[0]})
    withdrawal_dates = sorted({row[0] for row in withdrawal_rows if row[0]})

    dividends_by_symbol: dict[str, float] = {}
    dividend_count_by_symbol: dict[str, int] = {}
    dividend_events = []
    for sym, ex_date, pay_date, amount in dividend_rows:
        if not sym:
            continue
        amt = _to_float(amount) or 0.0
        dividends_by_symbol[sym] = round(dividends_by_symbol.get(sym, 0.0) + amt, 2)
        dividend_count_by_symbol[sym] = dividend_count_by_symbol.get(sym, 0) + 1
        dividend_events.append(
            {
                "symbol": sym,
                "ex_date": ex_date,
                "pay_date": pay_date,
                "amount": amt,
            }
        )

    trades_by_symbol: dict[str, dict[str, int]] = {}
    for sym, buy_count, sell_count in trade_rows:
        if not sym:
            continue
        trades_by_symbol[sym] = {
            "buy_count": int(buy_count or 0),
            "sell_count": int(sell_count or 0),
        }

    positions = {
        "added": [],
        "removed": [],
        "symbols_increased": [],
        "symbols_decreased": [],
    }
    for list_type, symbol in position_rows:
        if not symbol:
            continue
        if list_type == "added":
            positions["added"].append(symbol)
        elif list_type == "removed":
            positions["removed"].append(symbol)
        elif list_type == "increased":
            positions["symbols_increased"].append(symbol)
        elif list_type == "decreased":
            positions["symbols_decreased"].append(symbol)
    for key in positions:
        positions[key] = sorted(set(positions[key]))

    macro_stats: dict[str, dict[str, float | None]] = {}
    for metric, avg_val, min_val, max_val, std_val in macro_stat_rows:
        macro_stats[metric] = {
            "avg": _to_float(avg_val),
            "min": _to_float(min_val),
            "max": _to_float(max_val),
            "std": _to_float(std_val),
        }

    # Holdings summary from daily holdings start/end plus period history.
    start_holdings_date = start_daily.get("as_of_date_local") if start_daily else period_start_date
    end_holdings_date = end_daily.get("as_of_date_local") if end_daily else period_end_date
    start_holdings_rows = conn.execute(
        "SELECT * FROM daily_holdings WHERE as_of_date_local=?",
        (start_holdings_date,),
    ).fetchall()
    end_holdings_rows = conn.execute(
        "SELECT * FROM daily_holdings WHERE as_of_date_local=?",
        (end_holdings_date,),
    ).fetchall()
    start_holdings = {row["symbol"]: dict(row) for row in start_holdings_rows if row["symbol"]}
    end_holdings = {row["symbol"]: dict(row) for row in end_holdings_rows if row["symbol"]}
    symbols = sorted(set(start_holdings.keys()) | set(end_holdings.keys()))

    holdings_history: dict[str, list[dict]] = {sym: [] for sym in symbols}
    if symbols:
        placeholders = ",".join("?" for _ in symbols)
        rows = conn.execute(
            f"""SELECT as_of_date_local, symbol, market_value, weight_pct, vol_30d_pct
                FROM daily_holdings
                WHERE as_of_date_local BETWEEN ? AND ?
                  AND symbol IN ({placeholders})
                ORDER BY symbol, as_of_date_local""",
            (period_start_date, period_end_date, *symbols),
        ).fetchall()
        for row in rows:
            row_dict = dict(row)
            holdings_history.setdefault(row_dict["symbol"], []).append(row_dict)

    mv_start = _to_float(r.get("mv_start"))
    holdings_summary = []
    for sym in symbols:
        s = start_holdings.get(sym) or {}
        e = end_holdings.get(sym) or {}
        hist = holdings_history.get(sym) or []

        start_mv = _to_float(s.get("market_value"))
        end_mv = _to_float(e.get("market_value"))
        mv_delta = None if start_mv is None or end_mv is None else round(end_mv - start_mv, 2)
        start_shares = _to_float(s.get("shares"))
        end_shares = _to_float(e.get("shares"))
        shares_delta = None if start_shares is None or end_shares is None else round(end_shares - start_shares, 4)

        weight_vals = [_to_float(row.get("weight_pct")) for row in hist]
        weight_vals = [v for v in weight_vals if v is not None]
        vol_vals = [_to_float(row.get("vol_30d_pct")) for row in hist]
        vol_vals = [v for v in vol_vals if v is not None]

        day_returns = []
        for idx in range(1, len(hist)):
            prev_mv = _to_float(hist[idx - 1].get("market_value"))
            curr_mv = _to_float(hist[idx].get("market_value"))
            if prev_mv is None or curr_mv is None or abs(prev_mv) < 1e-12:
                continue
            day_returns.append(
                (
                    round(((curr_mv - prev_mv) / prev_mv) * 100.0, 3),
                    hist[idx].get("as_of_date_local"),
                )
            )

        worst_day = min(day_returns, default=(None, None), key=lambda x: x[0] if x[0] is not None else float("inf"))
        best_day = max(day_returns, default=(None, None), key=lambda x: x[0] if x[0] is not None else float("-inf"))

        dividends_received = round(dividends_by_symbol.get(sym, 0.0), 2)
        period_return_pct = None
        if start_mv is not None and end_mv is not None and abs(start_mv) > 1e-12:
            period_return_pct = round(((end_mv - start_mv + dividends_received) / start_mv) * 100.0, 3)

        contribution_pct = None
        if mv_start is not None and mv_delta is not None and abs(mv_start) > 1e-12:
            contribution_pct = round((mv_delta / mv_start) * 100.0, 3)

        holdings_summary.append(
            {
                "symbol": sym,
                "values": {
                    "start_shares": start_shares,
                    "end_shares": end_shares,
                    "shares_delta": shares_delta,
                    "shares_delta_pct": _pct_change(shares_delta, start_shares, digits=3),
                    "start_market_value": start_mv,
                    "end_market_value": end_mv,
                    "market_value_delta": mv_delta,
                    "market_value_delta_pct": _pct_change(mv_delta, start_mv, digits=3),
                    "start_weight_pct": _to_float(s.get("weight_pct")),
                    "end_weight_pct": _to_float(e.get("weight_pct")),
                    "avg_weight_pct": _avg(weight_vals, digits=3),
                },
                "performance": {
                    "period_return_pct": period_return_pct,
                    "contribution_to_portfolio_pct": contribution_pct,
                    "start_twr_12m_pct": _to_float(s.get("twr_12m_pct")),
                    "end_twr_12m_pct": _to_float(e.get("twr_12m_pct")),
                },
                "income": {
                    "dividends_received": dividends_received,
                    "dividend_events_count": int(dividend_count_by_symbol.get(sym, 0)),
                    "start_yield_pct": _to_float(s.get("current_yield_pct")),
                    "end_yield_pct": _to_float(e.get("current_yield_pct")),
                    "start_projected_monthly": _to_float(s.get("projected_monthly_dividend")),
                    "end_projected_monthly": _to_float(e.get("projected_monthly_dividend")),
                },
                "risk": {
                    "avg_vol_30d_pct": _avg(vol_vals, digits=3),
                    "period_max_drawdown_pct": _to_float(e.get("max_drawdown_1y_pct")),
                    "worst_day_pct": worst_day[0],
                    "worst_day_date": worst_day[1],
                    "best_day_pct": best_day[0],
                    "best_day_date": best_day[1],
                },
            }
        )
    holdings_summary.sort(key=lambda item: _to_float((item.get("values") or {}).get("end_market_value")) or 0.0, reverse=True)

    twr_1m_stats = _twr_window_stats(daily_rows, "twr_1m_pct")
    twr_3m_stats = _twr_window_stats(daily_rows, "twr_3m_pct")
    twr_12m_stats = _twr_window_stats(daily_rows, "twr_12m_pct")

    avg_top3 = _to_float(r.get("concentration_top3_avg"))
    avg_top5 = _to_float(r.get("concentration_top5_avg"))
    avg_herf = _to_float(r.get("concentration_herfindahl_avg"))
    if daily_rows:
        top3_values = [_to_float(row.get("top3_weight_pct")) for row in daily_rows]
        top5_values = [_to_float(row.get("top5_weight_pct")) for row in daily_rows]
        herf_values = [_to_float(row.get("herfindahl_index")) for row in daily_rows]
        if avg_top3 is None:
            avg_top3 = _avg([v for v in top3_values if v is not None], digits=3)
        if avg_top5 is None:
            avg_top5 = _avg([v for v in top5_values if v is not None], digits=3)
        if avg_herf is None:
            avg_herf = _avg([v for v in herf_values if v is not None], digits=3)

    benchmark_corr = _to_float(r.get("benchmark_correlation_1y_avg"))
    if benchmark_corr is None and daily_rows:
        corr_vals = [_to_float(row.get("vs_benchmark_corr_1y")) for row in daily_rows]
        benchmark_corr = _avg([v for v in corr_vals if v is not None], digits=3)

    start_daily_excess = _to_float(start_daily.get("vs_benchmark_excess_1y_pct")) if start_daily else None
    end_daily_excess = _to_float(end_daily.get("vs_benchmark_excess_1y_pct")) if end_daily else None

    start_hy_spread = _to_float(start_daily.get("macro_hy_spread_bps")) if start_daily else None
    end_hy_spread = _to_float(end_daily.get("macro_hy_spread_bps")) if end_daily else None
    start_macro_stress = _to_float(start_daily.get("macro_stress_score")) if start_daily else None
    end_macro_stress = _to_float(end_daily.get("macro_stress_score")) if end_daily else None

    avg_rate_pct_fallback = _coalesce(_to_float(activity.get("interest_avg_rate_pct")), _to_float(r.get("margin_apr_avg")))
    avg_daily_balance_fallback = _coalesce(
        _to_float(activity.get("interest_avg_daily_balance")),
        _to_float(r.get("margin_balance_avg")),
        margin_balance_fallback.get("avg"),
    )
    annualized_interest_fallback = None
    if avg_daily_balance_fallback is not None and avg_rate_pct_fallback is not None:
        annualized_interest_fallback = round(avg_daily_balance_fallback * (avg_rate_pct_fallback / 100.0), 3)

    margin_interest_total_paid = _to_float(activity.get("interest_total_paid"))
    if margin_interest_total_paid is None and annualized_interest_fallback is not None and days_in_period > 0:
        margin_interest_total_paid = round(annualized_interest_fallback * (days_in_period / 365.0), 3)
    avg_daily_cost = None
    if margin_interest_total_paid is not None and days_in_period > 0:
        avg_daily_cost = round(margin_interest_total_paid / days_in_period, 3)

    contributions_total = abs(_to_float(activity.get("contributions_total")) or 0.0)
    withdrawals_total = abs(_to_float(activity.get("withdrawals_total")) or 0.0)

    yield_start = _to_float(r.get("yield_start"))
    yield_end = _to_float(r.get("yield_end"))
    monthly_start = _to_float(r.get("monthly_income_start"))
    monthly_end = _to_float(r.get("monthly_income_end"))
    mv_delta = _to_float(r.get("mv_delta"))
    cb_delta = _to_float(r.get("cost_basis_delta"))
    nlv_delta = _to_float(r.get("nlv_delta"))
    margin_delta = _to_float(r.get("margin_balance_delta"))

    return {
        "meta": {
            "schema_version": "5.0",
            "summary_type": "period",
            "period": {
                "type": snapshot_type,
                "start_date_local": period_start_date,
                "end_date_local": period_end_date,
                "calendar_period": _calendar_period(snapshot_type, start_dt),
                "days_in_period": days_in_period,
                "snapshots_count": snapshots_count,
                "snapshots_expected": snapshots_expected,
                "coverage_pct": coverage_pct,
                "missing_dates": missing_dates,
            },
            "created_at_utc": r.get("created_at_utc"),
        },
        "timestamps": {
            "period_start_local": period_start_date,
            "period_end_local": period_end_date,
            "period_start_utc": _local_date_to_utc_iso(period_start_date),
            "period_end_utc": _local_date_to_utc_iso(period_end_date),
        },
        "portfolio": {
            "values": {
                "start": {
                    "market_value": _to_float(r.get("mv_start")),
                    "cost_basis": _to_float(r.get("cost_basis_start")),
                    "net_liquidation_value": _to_float(r.get("nlv_start")),
                    "unrealized_pnl": _to_float(r.get("unrealized_pnl_start")),
                    "unrealized_pct": _to_float(r.get("unrealized_pct_start")),
                    "margin_loan_balance": _to_float(r.get("margin_balance_start")),
                    "margin_to_portfolio_pct": _to_float(r.get("ltv_pct_start")),
                    "holdings_count": r.get("holding_count_start"),
                },
                "end": {
                    "market_value": _to_float(r.get("mv_end")),
                    "cost_basis": _to_float(r.get("cost_basis_end")),
                    "net_liquidation_value": _to_float(r.get("nlv_end")),
                    "unrealized_pnl": _to_float(r.get("unrealized_pnl_end")),
                    "unrealized_pct": _to_float(r.get("unrealized_pct_end")),
                    "margin_loan_balance": _to_float(r.get("margin_balance_end")),
                    "margin_to_portfolio_pct": _to_float(r.get("ltv_pct_end")),
                    "holdings_count": r.get("holding_count_end"),
                },
                "delta": {
                    "market_value": mv_delta,
                    "market_value_pct": _pct_change(mv_delta, r.get("mv_start"), digits=3),
                    "cost_basis": cb_delta,
                    "cost_basis_pct": _pct_change(cb_delta, r.get("cost_basis_start"), digits=3),
                    "net_liquidation_value": nlv_delta,
                    "net_liquidation_value_pct": _pct_change(nlv_delta, r.get("nlv_start"), digits=3),
                    "unrealized_pnl": _to_float(r.get("unrealized_pnl_delta")),
                    "margin_loan_balance": margin_delta,
                    "margin_loan_balance_pct": _pct_change(margin_delta, r.get("margin_balance_start"), digits=3),
                },
                "period_stats": {
                    "market_value": {
                        "avg": _to_float(r.get("mv_avg")),
                        "min": _to_float(r.get("mv_min")),
                        "max": _to_float(r.get("mv_max")),
                        "std": _to_float(r.get("mv_std")),
                        "min_date": r.get("mv_min_date"),
                        "max_date": r.get("mv_max_date"),
                    },
                    "net_liquidation_value": {
                        "avg": _coalesce(_to_float(r.get("nlv_avg")), nlv_fallback.get("avg")),
                        "min": _coalesce(_to_float(r.get("nlv_min")), nlv_fallback.get("min")),
                        "max": _coalesce(_to_float(r.get("nlv_max")), nlv_fallback.get("max")),
                        "std": _coalesce(_to_float(r.get("nlv_std")), nlv_fallback.get("std")),
                        "min_date": _coalesce(r.get("nlv_min_date"), nlv_fallback.get("min_date")),
                        "max_date": _coalesce(r.get("nlv_max_date"), nlv_fallback.get("max_date")),
                    },
                    "margin_to_portfolio_pct": {
                        "avg": _coalesce(_to_float(r.get("margin_to_portfolio_pct_avg")), ltv_fallback.get("avg")),
                        "min": _coalesce(_to_float(r.get("margin_to_portfolio_pct_min")), ltv_fallback.get("min")),
                        "max": _coalesce(_to_float(r.get("margin_to_portfolio_pct_max")), ltv_fallback.get("max")),
                        "std": _coalesce(_to_float(r.get("margin_to_portfolio_pct_std")), ltv_fallback.get("std")),
                    },
                },
            },
            "income": {
                "start_projected_monthly": monthly_start,
                "end_projected_monthly": monthly_end,
                "delta_projected_monthly": _to_float(r.get("monthly_income_delta")),
                "delta_projected_monthly_pct": _pct_change(r.get("monthly_income_delta"), monthly_start, digits=3),
                "start_yield_pct": yield_start,
                "end_yield_pct": yield_end,
                "delta_yield_pct": _to_float(r.get("yield_delta")),
                "period_stats": {
                    "projected_monthly": {
                        "avg": _coalesce(_to_float(r.get("projected_monthly_avg")), projected_monthly_fallback.get("avg")),
                        "min": _coalesce(_to_float(r.get("projected_monthly_min")), projected_monthly_fallback.get("min")),
                        "max": _coalesce(_to_float(r.get("projected_monthly_max")), projected_monthly_fallback.get("max")),
                        "std": _coalesce(_to_float(r.get("projected_monthly_std")), projected_monthly_fallback.get("std")),
                    },
                    "yield_pct": {
                        "avg": _coalesce(_to_float(r.get("yield_pct_avg")), yield_fallback.get("avg")),
                        "min": _coalesce(_to_float(r.get("yield_pct_min")), yield_fallback.get("min")),
                        "max": _coalesce(_to_float(r.get("yield_pct_max")), yield_fallback.get("max")),
                        "std": _coalesce(_to_float(r.get("yield_pct_std")), yield_fallback.get("std")),
                    },
                },
            },
            "performance": {
                "period_return_pct": _to_float(r.get("pnl_pct_period")),
                "twr_period_pct": _to_float(r.get("twr_period_pct")),
                "period_stats": {
                    "twr_1m_pct": twr_1m_stats,
                    "twr_3m_pct": twr_3m_stats,
                    "twr_12m_pct": twr_12m_stats,
                },
                "vs_benchmark": {
                    "benchmark_symbol": r.get("benchmark_symbol"),
                    "start_benchmark_twr_1y_pct": _to_float(r.get("benchmark_twr_1y_start")),
                    "end_benchmark_twr_1y_pct": _to_float(r.get("benchmark_twr_1y_end")),
                    "start_excess_1y_pct": start_daily_excess,
                    "end_excess_1y_pct": end_daily_excess,
                    "avg_correlation_1y": benchmark_corr,
                },
            },
            "risk": {
                "volatility": {
                    "start_vol_30d_pct": _to_float(r.get("vol_30d_start")),
                    "end_vol_30d_pct": _to_float(r.get("vol_30d_end")),
                    "avg_vol_30d_pct": _coalesce(_to_float(r.get("vol_30d_avg")), vol_30d_fallback.get("avg")),
                    "start_vol_90d_pct": _to_float(r.get("vol_90d_start")),
                    "end_vol_90d_pct": _to_float(r.get("vol_90d_end")),
                    "avg_vol_90d_pct": _coalesce(_to_float(r.get("vol_90d_avg")), vol_90d_fallback.get("avg")),
                },
                "ratios": {
                    "start_sharpe_1y": _to_float(r.get("sharpe_1y_start")),
                    "end_sharpe_1y": _to_float(r.get("sharpe_1y_end")),
                    "avg_sharpe_1y": _coalesce(_to_float(r.get("sharpe_1y_avg")), sharpe_fallback.get("avg")),
                    "start_sortino_1y": _to_float(r.get("sortino_1y_start")),
                    "end_sortino_1y": _to_float(r.get("sortino_1y_end")),
                    "avg_sortino_1y": _coalesce(_to_float(r.get("sortino_1y_avg")), sortino_fallback.get("avg")),
                    "start_calmar_1y": _to_float(r.get("calmar_1y_start")),
                    "end_calmar_1y": _to_float(r.get("calmar_1y_end")),
                    "avg_calmar_1y": _coalesce(_to_float(r.get("calmar_1y_avg")), calmar_fallback.get("avg")),
                },
                "drawdown": {
                    "period_max_drawdown_pct": _to_float(r.get("period_max_drawdown_pct")),
                    "period_max_drawdown_date": r.get("period_max_drawdown_date"),
                    "period_recovery_date": r.get("period_recovery_date"),
                    "period_days_in_drawdown": r.get("period_days_in_drawdown"),
                    "period_drawdown_count": r.get("period_drawdown_count"),
                    "start_current_drawdown_pct": _to_float(start_daily.get("drawdown_depth_pct")) if start_daily else None,
                    "end_current_drawdown_pct": _to_float(end_daily.get("drawdown_depth_pct")) if end_daily else None,
                    "start_max_drawdown_1y_pct": _to_float(r.get("max_dd_1y_start")),
                    "end_max_drawdown_1y_pct": _to_float(r.get("max_dd_1y_end")),
                },
                "var": {
                    "start_var_95_1d_pct": _to_float(start_daily.get("var_95_1d_pct")) if start_daily else None,
                    "end_var_95_1d_pct": _to_float(end_daily.get("var_95_1d_pct")) if end_daily else None,
                    "avg_var_95_1d_pct": _to_float(r.get("var_95_1d_avg")),
                    "start_cvar_95_1d_pct": _to_float(start_daily.get("cvar_95_1d_pct")) if start_daily else None,
                    "end_cvar_95_1d_pct": _to_float(end_daily.get("cvar_95_1d_pct")) if end_daily else None,
                    "avg_cvar_95_1d_pct": _to_float(r.get("cvar_95_1d_avg")),
                    "days_exceeding_var_95": r.get("days_exceeding_var_95"),
                    "worst_day_return_pct": _to_float(r.get("worst_day_return_pct")),
                    "worst_day_date": r.get("worst_day_date"),
                    "best_day_return_pct": _to_float(r.get("best_day_return_pct")),
                    "best_day_date": r.get("best_day_date"),
                },
            },
            "allocation": {
                "start_concentration": {
                    "top3_weight_pct": _to_float(r.get("concentration_top3_start")),
                    "top5_weight_pct": _to_float(r.get("concentration_top5_start")),
                    "herfindahl_index": _to_float(r.get("concentration_herfindahl_start")),
                },
                "end_concentration": {
                    "top3_weight_pct": _to_float(r.get("concentration_top3_end")),
                    "top5_weight_pct": _to_float(r.get("concentration_top5_end")),
                    "herfindahl_index": _to_float(r.get("concentration_herfindahl_end")),
                },
                "avg_concentration": {
                    "top3_weight_pct": avg_top3,
                    "top5_weight_pct": avg_top5,
                    "herfindahl_index": avg_herf,
                },
            },
        },
        "activity": {
            "contributions": {
                "total": contributions_total,
                "count": int(activity.get("contributions_count") or 0),
                "dates": contribution_dates,
            },
            "withdrawals": {
                "total": withdrawals_total,
                "count": int(activity.get("withdrawals_count") or 0),
                "dates": withdrawal_dates,
            },
            "dividends": {
                "total_received": _to_float(activity.get("dividends_total_received")) or 0.0,
                "count": int(activity.get("dividends_count") or 0),
                "by_symbol": dividends_by_symbol,
                "events": dividend_events,
            },
            "interest": {
                "total_paid": _to_float(activity.get("interest_total_paid")) or 0.0,
                "avg_daily_balance": avg_daily_balance_fallback,
                "avg_rate_pct": avg_rate_pct_fallback,
                "annualized": _coalesce(_to_float(activity.get("interest_annualized")), annualized_interest_fallback),
            },
            "trades": {
                "total_count": int(activity.get("trades_total_count") or 0),
                "buy_count": int(activity.get("trades_buy_count") or 0),
                "sell_count": int(activity.get("trades_sell_count") or 0),
                "by_symbol": trades_by_symbol,
            },
            "positions": positions,
            "margin": {
                "borrowed": _to_float(activity.get("margin_borrowed")) or 0.0,
                "repaid": _to_float(activity.get("margin_repaid")) or 0.0,
                "net_change": _to_float(activity.get("margin_net_change")) or 0.0,
            },
        },
        "holdings_summary": holdings_summary,
        "goals": {
            "start": {
                "portfolio_value": _to_float(r.get("mv_start")),
                "projected_monthly_income": monthly_start,
                "progress_pct": _to_float(r.get("goal_progress_pct_start")),
                "months_to_goal": _to_float(r.get("goal_months_to_goal_start")),
                "estimated_goal_date": _add_months(period_start_date, r.get("goal_months_to_goal_start")),
            },
            "end": {
                "portfolio_value": _to_float(r.get("mv_end")),
                "projected_monthly_income": monthly_end,
                "progress_pct": _to_float(r.get("goal_progress_pct_end")),
                "months_to_goal": _to_float(r.get("goal_months_to_goal_end")),
                "estimated_goal_date": _add_months(period_end_date, r.get("goal_months_to_goal_end")),
            },
            "delta": {
                "portfolio_value": mv_delta,
                "projected_monthly_income": _to_float(r.get("monthly_income_delta")),
                "progress_pct": _to_float(r.get("goal_progress_pct_delta")),
                "months_to_goal": _to_float(r.get("goal_months_to_goal_delta")),
            },
            "pace": {
                "start_months_ahead_behind": _coalesce(
                    pace_months_start_fallback,
                    _to_float(r.get("goal_pace_months_start")),
                ),
                "end_months_ahead_behind": _coalesce(
                    pace_months_end_fallback,
                    _to_float(r.get("goal_pace_months_end")),
                ),
                "delta_months_ahead_behind": _coalesce(
                    _to_float(r.get("goal_pace_months_delta")),
                    pace_months_delta_fallback,
                ),
                "start_tier_pace_pct": _coalesce(
                    pace_tier_start_fallback,
                    _to_float(r.get("goal_pace_tier_pace_pct_start")),
                ),
                "end_tier_pace_pct": _coalesce(
                    pace_tier_end_fallback,
                    _to_float(r.get("goal_pace_tier_pace_pct_end")),
                ),
            },
        },
        "margin": {
            "balance": {
                "start": _to_float(r.get("margin_balance_start")),
                "end": _to_float(r.get("margin_balance_end")),
                "delta": margin_delta,
                "delta_pct": _pct_change(margin_delta, r.get("margin_balance_start"), digits=3),
                "avg": _coalesce(_to_float(r.get("margin_balance_avg")), margin_balance_fallback.get("avg")),
                "min": _coalesce(_to_float(r.get("margin_balance_min")), margin_balance_fallback.get("min")),
                "max": _coalesce(_to_float(r.get("margin_balance_max")), margin_balance_fallback.get("max")),
                "std": _coalesce(_to_float(r.get("margin_balance_std")), margin_balance_fallback.get("std")),
            },
            "ltv": {
                "start_pct": _to_float(r.get("ltv_pct_start")),
                "end_pct": _to_float(r.get("ltv_pct_end")),
                "avg_pct": _coalesce(_to_float(r.get("ltv_pct_avg")), ltv_fallback.get("avg")),
                "min_pct": _coalesce(_to_float(r.get("ltv_pct_min")), ltv_fallback.get("min")),
                "max_pct": _coalesce(_to_float(r.get("ltv_pct_max")), ltv_fallback.get("max")),
                "std_pct": _coalesce(_to_float(r.get("ltv_pct_std")), ltv_fallback.get("std")),
            },
            "interest": {
                "total_paid": margin_interest_total_paid,
                "avg_daily_cost": avg_daily_cost,
                "start_apr_pct": _coalesce(_to_float(r.get("margin_apr_start")), avg_rate_pct_fallback),
                "end_apr_pct": _coalesce(_to_float(r.get("margin_apr_end")), avg_rate_pct_fallback),
                "avg_apr_pct": _coalesce(_to_float(r.get("margin_apr_avg")), avg_rate_pct_fallback),
            },
            "safety": {
                "min_buffer_to_call_pct": _coalesce(_to_float(r.get("margin_min_buffer_to_call_pct")), buffer_fallback.get("min")),
                "min_buffer_date": _coalesce(r.get("margin_min_buffer_date"), buffer_fallback.get("min_date")),
                "days_below_50pct_buffer": _coalesce(
                    r.get("margin_days_below_50pct_buffer"),
                    buffer_days_below_50,
                ),
                "days_below_40pct_buffer": _coalesce(
                    r.get("margin_days_below_40pct_buffer"),
                    buffer_days_below_40,
                ),
                "margin_call_events": _coalesce(
                    r.get("margin_call_events"),
                    buffer_call_events,
                ),
            },
        },
        "macro": {
            "start": {
                "vix": _to_float(r.get("macro_vix_start")),
                "ten_year_yield": _to_float(r.get("macro_10y_start")),
                "two_year_yield": _to_float(r.get("macro_2y_start")),
                "hy_spread_bps": start_hy_spread,
                "yield_spread_10y_2y": _to_float(start_daily.get("macro_yield_spread_10y_2y")) if start_daily else None,
                "macro_stress_score": start_macro_stress,
                "cpi_yoy": _to_float(r.get("macro_cpi_start")),
            },
            "end": {
                "vix": _to_float(r.get("macro_vix_end")),
                "ten_year_yield": _to_float(r.get("macro_10y_end")),
                "two_year_yield": _to_float(r.get("macro_2y_end")),
                "hy_spread_bps": end_hy_spread,
                "yield_spread_10y_2y": _to_float(end_daily.get("macro_yield_spread_10y_2y")) if end_daily else None,
                "macro_stress_score": end_macro_stress,
                "cpi_yoy": _to_float(r.get("macro_cpi_end")),
            },
            "period_stats": {
                "vix": macro_stats.get("vix")
                or {
                    "avg": _to_float(r.get("macro_vix_avg")),
                    "min": None,
                    "max": None,
                    "std": None,
                },
                "ten_year_yield": macro_stats.get("ten_year_yield")
                or {
                    "avg": _to_float(r.get("macro_10y_avg")),
                    "min": None,
                    "max": None,
                    "std": None,
                },
                "hy_spread_bps": macro_stats.get("hy_spread_bps")
                or {"avg": _avg([v for v in [_to_float(start_hy_spread), _to_float(end_hy_spread)] if v is not None], digits=3), "min": None, "max": None, "std": None},
                "macro_stress_score": macro_stats.get("macro_stress_score")
                or {"avg": _avg([v for v in [_to_float(start_macro_stress), _to_float(end_macro_stress)] if v is not None], digits=3), "min": None, "max": None, "std": None},
            },
        },
    }

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
