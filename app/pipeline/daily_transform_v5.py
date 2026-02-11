"""V5 schema transformer for daily snapshots (renamed from snapshot_v5).

Transforms v4 daily snapshot dict into v5 schema:
- Holdings restructured into sub-objects (cost, valuation, income, analytics, reliability, performance[])
- 'ultimate' renamed to 'analytics' with distribution/risk/performance sub-groups
- contribution_analysis_* flattened to performance[] array
- portfolio_rollups → portfolio with nested totals/income/performance/risk/allocation/attribution
- Goals nested under single parent
- margin_guidance + margin_stress consolidated into 'margin' with sign fix
- dividends + dividends_upcoming merged
- Timestamps consolidated
- New: summary (dashboard quick-view), alerts[] (6 types), portfolio.allocation

Full snapshots keep all provenance; slim_snapshot() strips it as before.
"""
from __future__ import annotations

from datetime import date, timedelta


def transform_to_v5(daily: dict) -> dict:
    """Transform a v4 daily snapshot to v5 schema. No-op if already v5."""
    meta = daily.get("meta") or {}
    if meta.get("schema_version") == "5.0":
        return daily

    as_of_str = daily.get("as_of", "")[:10]
    totals = daily.get("totals") or {}
    income_data = daily.get("income") or {}
    rollups = daily.get("portfolio_rollups") or {}
    perf = rollups.get("performance") or {}
    risk = rollups.get("risk") or {}

    # ── Timestamps ───────────────────────────────────────────────────────
    price_as_of_utc = daily.get("prices_as_of_utc")
    price_as_of_date = daily.get("prices_as_of")
    # Derive price_data_as_of_utc from date when UTC timestamp is missing
    # (common for historical rebuilds that have no intraday quotes)
    if not price_as_of_utc and price_as_of_date:
        price_as_of_utc = f"{price_as_of_date}T20:00:00+00:00"

    timestamps = {
        "snapshot_created_utc": meta.get("snapshot_created_at"),
        "portfolio_data_as_of_utc": daily.get("as_of_utc"),
        "portfolio_data_as_of_local": daily.get("as_of_date_local"),
        "price_data_as_of_utc": price_as_of_utc,
        "price_data_as_of_date": price_as_of_date,
        "macro_data_as_of_date": (
            (daily.get("macro") or {}).get("snapshot") or {}
        ).get("date"),
        "last_transaction_sync_utc": meta.get("last_transaction_sync_at"),
    }

    # ── Account ──────────────────────────────────────────────────────────
    account = {
        "plaid_account_id": daily.get("plaid_account_id"),
        "account_type": "margin",
        "base_currency": "USD",
    }

    # ── Holdings ─────────────────────────────────────────────────────────
    v4_holdings = daily.get("holdings") or []
    v5_holdings = [_transform_holding(h, as_of_str) for h in v4_holdings]

    # ── Summary ──────────────────────────────────────────────────────────
    summary = _build_summary(daily)

    # ── Alerts ───────────────────────────────────────────────────────────
    alerts = _build_alerts(daily)

    # ── Portfolio ────────────────────────────────────────────────────────
    allocation = _build_allocation(v4_holdings)

    portfolio_risk = {
        "volatility": {
            "vol_30d_pct": risk.get("vol_30d_pct"),
            "vol_90d_pct": risk.get("vol_90d_pct"),
            "downside_dev_1y_pct": risk.get("downside_dev_1y_pct"),
        },
        "ratios": _pick(risk, [
            "sharpe_1y", "sortino_1y", "sortino_6m", "sortino_3m", "sortino_1m",
            "sortino_sharpe_ratio", "sortino_sharpe_divergence",
            "calmar_1y", "omega_ratio_1y", "ulcer_index_1y", "ulcer_index_category",
            "pain_adjusted_return",
        ]),
        "drawdown": {
            "max_drawdown_1y_pct": risk.get("max_drawdown_1y_pct"),
            "drawdown_duration_1y_days": risk.get("drawdown_duration_1y_days"),
        },
        "var": _pick(risk, [
            "var_90_1d_pct", "var_95_1d_pct", "var_99_1d_pct",
            "var_95_1w_pct", "var_95_1m_pct",
            "cvar_90_1d_pct", "cvar_95_1d_pct", "cvar_99_1d_pct",
            "cvar_95_1w_pct", "cvar_95_1m_pct",
        ]),
        "tail_risk": rollups.get("tail_risk"),
        "beta_portfolio": risk.get("beta_portfolio"),
        "portfolio_risk_quality": risk.get("portfolio_risk_quality"),
        "income_stability_score": risk.get("income_stability_score"),
    }
    # Merge drawdown status + recovery if present
    if risk.get("drawdown_status"):
        portfolio_risk["drawdown"].update(risk["drawdown_status"])
    if risk.get("recovery_metrics"):
        portfolio_risk["drawdown"]["recovery"] = risk["recovery_metrics"]
    # Sortino/Sharpe ratio
    sortino = risk.get("sortino_1y")
    sharpe = risk.get("sharpe_1y")
    if isinstance(sortino, (int, float)) and isinstance(sharpe, (int, float)) and sharpe != 0:
        portfolio_risk["ratios"]["sortino_sharpe_ratio"] = round(sortino / sharpe, 3)

    # Attribution from return_attribution_* keys
    attribution = {}
    for key in ["return_attribution_1m", "return_attribution_3m",
                "return_attribution_6m", "return_attribution_12m"]:
        label = key.replace("return_attribution_", "")
        if rollups.get(key):
            attribution[label] = rollups[key]

    # Profitable / losing positions
    profitable = sum(
        1 for h in v4_holdings
        if isinstance(h.get("unrealized_pnl"), (int, float)) and h["unrealized_pnl"] > 0
    )
    losing = sum(
        1 for h in v4_holdings
        if isinstance(h.get("unrealized_pnl"), (int, float)) and h["unrealized_pnl"] < 0
    )

    portfolio = {
        "totals": {
            **totals,
            "holdings_count": daily.get("count") or len(v5_holdings),
            "positions_profitable": profitable,
            "positions_losing": losing,
        },
        "income": {
            **income_data,
            "income_stability": rollups.get("income_stability"),
            "income_growth": rollups.get("income_growth"),
        },
        "performance": {
            **perf,
            "benchmark": rollups.get("benchmark"),
            "vs_benchmark": rollups.get("vs_benchmark"),
        },
        "risk": portfolio_risk,
        "allocation": allocation,
        "attribution": attribution,
    }

    # ── Dividends ────────────────────────────────────────────────────────
    div_v4 = daily.get("dividends") or {}
    dividends = {
        "upcoming_this_month": daily.get("dividends_upcoming"),
        "projected_vs_received": div_v4.get("projected_vs_received"),
        "realized": {
            "mtd": div_v4.get("realized_mtd"),
            "30d": (div_v4.get("windows") or {}).get("30d"),
            "ytd": (div_v4.get("windows") or {}).get("ytd"),
            "qtd": (div_v4.get("windows") or {}).get("qtd"),
        },
    }

    # ── Margin ───────────────────────────────────────────────────────────
    margin = _build_margin(daily)

    # ── Goals ────────────────────────────────────────────────────────────
    goals = _build_goals(daily, income_data)

    # ── Meta ─────────────────────────────────────────────────────────────
    new_meta = {
        **meta,
        "schema_version": "5.0",
        "migration_notes": [
            "V4→V5: Renamed 'ultimate' to 'analytics' with sub-grouping",
            "V4→V5: Flattened contribution_analysis_* to performance[] array",
            "V4→V5: Nested all goal objects under 'goals'",
            "V4→V5: Consolidated margin_guidance + margin_stress into 'margin'",
            "V4→V5: Added summary, alerts, allocation sections",
            "V4→V5: Consolidated timestamps",
            "V4→V5: Fixed margin call sign → buffer_to_margin_call_pct",
            "V4→V5: Restructured holdings into sub-objects",
        ],
        "data_quality": daily.get("coverage"),
    }

    # ── Provenance ───────────────────────────────────────────────────────
    provenance = {
        "holdings": daily.get("holdings_provenance"),
        "totals": daily.get("totals_provenance"),
        "income": daily.get("income_provenance"),
        "portfolio": daily.get("portfolio_rollups_provenance"),
        "dividends": daily.get("dividends_provenance"),
        "dividends_upcoming": daily.get("dividends_upcoming_provenance"),
        "margin_guidance": daily.get("margin_guidance_provenance"),
        "margin_stress": daily.get("margin_stress_provenance"),
        "goals": daily.get("goal_progress_provenance"),
    }

    return {
        "meta": new_meta,
        "timestamps": timestamps,
        "summary": summary,
        "alerts": alerts,
        "account": account,
        "holdings": v5_holdings,
        "portfolio": portfolio,
        "dividends": dividends,
        "margin": margin,
        "goals": goals,
        "macro": daily.get("macro"),
        "provenance": provenance,
    }


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _pick(d: dict, keys: list[str]) -> dict:
    """Extract subset of keys from dict."""
    return {k: d.get(k) for k in keys}


# ─── Holding transform ──────────────────────────────────────────────────────


def _transform_holding(h: dict, as_of_str: str) -> dict:
    """Restructure a flat v4 holding into v5 sub-grouped format."""
    ult = h.get("ultimate") or {}

    analytics = {
        "distribution": {
            "trailing_12m_div_per_share": ult.get("trailing_12m_div_ps"),
            "trailing_12m_yield_pct": ult.get("trailing_12m_yield_pct"),
            "forward_yield_pct": ult.get("forward_yield_pct"),
            "forward_yield_method": ult.get("forward_yield_method"),
            "distribution_frequency": ult.get("distribution_frequency"),
            "next_ex_date_est": ult.get("next_ex_date_est"),
            "last_ex_date": ult.get("last_ex_date"),
            "forward_12m_div_per_share": ult.get("forward_12m_div_ps"),
            "derived_fields": ult.get("derived_fields"),
        },
        "risk": _pick(ult, [
            "vol_30d_pct", "vol_90d_pct", "beta_3y",
            "max_drawdown_1y_pct", "drawdown_duration_1y_days",
            "downside_dev_1y_pct",
            "sortino_1y", "sortino_6m", "sortino_3m", "sortino_1m",
            "sharpe_1y", "calmar_1y",
            "risk_quality_score", "risk_quality_category", "volatility_profile",
            "var_90_1d_pct", "var_95_1d_pct", "var_99_1d_pct",
            "var_95_1w_pct", "var_95_1m_pct",
            "cvar_90_1d_pct", "cvar_95_1d_pct", "cvar_99_1d_pct",
            "cvar_95_1w_pct", "cvar_95_1m_pct",
        ]),
        "performance": _pick(ult, [
            "twr_1m_pct", "twr_3m_pct", "twr_6m_pct", "twr_12m_pct", "corr_1y",
        ]),
    }

    # Flatten contribution_analysis_* → performance[]
    performance = []
    period_days = {"1m": 30, "3m": 90, "6m": 180, "12m": 365}
    try:
        end_dt = date.fromisoformat(as_of_str)
    except (ValueError, TypeError):
        end_dt = None

    for label, days in period_days.items():
        ca = h.get(f"contribution_analysis_{label}")
        if not ca:
            continue
        entry = {"period": label}
        if end_dt:
            start_dt = end_dt - timedelta(days=days)
            entry["window"] = {
                "start": start_dt.isoformat(),
                "end": end_dt.isoformat(),
                "days": days,
            }
        entry.update(ca)
        performance.append(entry)

    out = {
        "symbol": h.get("symbol"),
        "shares": h.get("shares"),
        "trades_count": h.get("trades"),
        "cost": {
            "cost_basis": h.get("cost_basis"),
            "avg_cost_per_share": h.get("avg_cost"),
        },
        "valuation": {
            "last_price": h.get("last_price"),
            "market_value": h.get("market_value"),
            "unrealized_pnl": h.get("unrealized_pnl"),
            "unrealized_pct": h.get("unrealized_pct"),
            "portfolio_weight_pct": h.get("weight_pct"),
        },
        "income": {
            "forward_12m_dividend": h.get("forward_12m_dividend"),
            "forward_method": h.get("forward_method"),
            "projected_monthly_dividend": h.get("projected_monthly_dividend"),
            "current_yield_pct": h.get("current_yield_pct"),
            "yield_on_cost_pct": h.get("yield_on_cost_pct"),
            "last_ex_date": h.get("last_ex_date"),
            "dividends_30d": h.get("dividends_30d"),
            "dividends_qtd": h.get("dividends_qtd"),
            "dividends_ytd": h.get("dividends_ytd"),
        },
        "analytics": analytics,
        "reliability": h.get("dividend_reliability"),
        "performance": performance,
    }

    # Carry provenance (renamed keys)
    if h.get("ultimate_provenance"):
        out["analytics_provenance"] = h["ultimate_provenance"]
    if h.get("dividend_reliability_provenance"):
        out["reliability_provenance"] = h["dividend_reliability_provenance"]

    return out


# ─── Summary ────────────────────────────────────────────────────────────────


def _build_summary(daily: dict) -> dict:
    """Build dashboard quick-view summary from existing data."""
    totals = daily.get("totals") or {}
    income = daily.get("income") or {}
    rollups = daily.get("portfolio_rollups") or {}
    perf = rollups.get("performance") or {}
    risk = rollups.get("risk") or {}
    gp = daily.get("goal_progress") or {}
    holdings = daily.get("holdings") or []

    ltv = totals.get("margin_to_portfolio_pct") or 0
    rq = risk.get("portfolio_risk_quality", "unknown")
    dd = risk.get("drawdown_status") or {}
    dd_depth = abs(dd.get("current_drawdown_depth_pct") or 0)

    if ltv > 40 or rq == "poor":
        health = "critical"
    elif ltv > 35 or (dd.get("currently_in_drawdown") and dd_depth > 10):
        health = "caution"
    elif ltv > 30 or (dd.get("currently_in_drawdown") and dd_depth > 5):
        health = "fair"
    else:
        health = "good"

    return {
        "net_liquidation_value": totals.get("net_liquidation_value"),
        "total_market_value": totals.get("market_value"),
        "cost_basis": totals.get("cost_basis"),
        "margin_loan_balance": totals.get("margin_loan_balance"),
        "ltv_pct": totals.get("margin_to_portfolio_pct"),
        "unrealized_pnl": totals.get("unrealized_pnl"),
        "unrealized_pct": totals.get("unrealized_pct"),
        "monthly_income": income.get("projected_monthly_income"),
        "portfolio_yield_pct": income.get("portfolio_current_yield_pct"),
        "twr_1m_pct": perf.get("twr_1m_pct"),
        "twr_3m_pct": perf.get("twr_3m_pct"),
        "twr_6m_pct": perf.get("twr_6m_pct"),
        "twr_12m_pct": perf.get("twr_12m_pct"),
        "months_to_goal": gp.get("months_to_goal"),
        "goal_progress_pct": gp.get("progress_pct"),
        "holdings_count": len(holdings),
        "health_status": health,
        "risk_category": rq,
    }


# ─── Alerts engine ──────────────────────────────────────────────────────────


def _build_alerts(daily: dict) -> list[dict]:
    """Generate actionable alerts from snapshot data.

    Six alert types:
    1. margin   — LTV exceeds target (>30%)
    2. dividend — upcoming ex-dates within 14 days
    3. performance — holdings down >15% with declining dividends
    4. goal    — pace vs baseline (ahead/behind)
    5. risk    — currently in drawdown
    6. rate    — scheduled interest rate change
    """
    alerts: list[dict] = []
    n = 0
    now_str = (daily.get("meta") or {}).get("snapshot_created_at") or daily.get("as_of_utc")
    as_of_str = daily.get("as_of", "")[:10]

    totals = daily.get("totals") or {}
    rollups = daily.get("portfolio_rollups") or {}
    risk = rollups.get("risk") or {}
    mg = daily.get("margin_guidance") or {}
    gp = daily.get("goal_pace") or {}

    # ── 1. Margin ────────────────────────────────────────────────────────
    ltv = totals.get("margin_to_portfolio_pct")
    if isinstance(ltv, (int, float)) and ltv > 30:
        n += 1
        severity = "critical" if ltv > 40 else "warning"
        repay = None
        for mode in (mg.get("modes") or []):
            if mode.get("mode") == "aggressive" and mode.get("action") == "repay":
                repay = mode.get("amount")
        alerts.append({
            "id": f"alert_{n:03d}",
            "type": "margin",
            "severity": severity,
            "message": f"LTV at {ltv}% exceeds target of 30.0%",
            "action_required": (
                f"Consider repaying ${repay:,.0f} to reach target LTV"
                if repay else "Consider reducing margin balance"
            ),
            "triggered_at": now_str,
        })

    # ── 2. Dividend — upcoming ex-dates within 14 days ───────────────────
    try:
        as_of_dt = date.fromisoformat(as_of_str)
    except (ValueError, TypeError):
        as_of_dt = None

    if as_of_dt:
        for h in (daily.get("holdings") or []):
            ult = h.get("ultimate") or {}
            nex = ult.get("next_ex_date_est")
            if not nex:
                continue
            try:
                ex_dt = date.fromisoformat(nex[:10])
                days_until = (ex_dt - as_of_dt).days
            except (ValueError, TypeError):
                continue
            if 0 < days_until <= 14:
                n += 1
                sym = h.get("symbol")
                fwd = h.get("forward_12m_dividend") or 0
                monthly_est = round(fwd / 12, 2) if fwd else None
                alerts.append({
                    "id": f"alert_{n:03d}",
                    "type": "dividend",
                    "severity": "info",
                    "message": f"{sym} ex-date in {days_until} days ({nex[:10]})",
                    "details": {
                        "symbol": sym,
                        "ex_date": nex[:10],
                        "expected_amount": monthly_est,
                        "shares": h.get("shares"),
                    },
                    "triggered_at": now_str,
                })

    # ── 3. Performance — underperformers with negative momentum ──────────
    for h in (daily.get("holdings") or []):
        up = h.get("unrealized_pct")
        if not isinstance(up, (int, float)) or up >= -15:
            continue

        rel = h.get("dividend_reliability") or {}
        trend = rel.get("dividend_trend_6m")
        missed = rel.get("missed_payments_12m")
        ult = h.get("ultimate") or {}
        twr_3m = ult.get("twr_3m_pct")

        # Trigger if declining dividends, missed payments, or severe 3m loss
        has_issue = (
            trend == "declining"
            or (isinstance(missed, str) and "no dividend" in missed.lower())
            or (isinstance(twr_3m, (int, float)) and twr_3m < -20)
        )
        if not has_issue:
            continue

        n += 1
        sym = h.get("symbol")
        msg = f"{sym} down {abs(up):.0f}% from cost basis"
        if trend == "declining":
            msg += " with declining dividends"

        details: dict = {"symbol": sym, "unrealized_pct": up}
        if isinstance(twr_3m, (int, float)):
            details["twr_3m_pct"] = twr_3m
        if isinstance(trend, str):
            details["dividend_trend"] = trend
        if isinstance(missed, str):
            details["missed_payments"] = missed

        alerts.append({
            "id": f"alert_{n:03d}",
            "type": "performance",
            "severity": "critical",
            "message": msg,
            "action_required": f"Review position — {abs(up):.0f}% decline with negative momentum",
            "details": details,
            "triggered_at": now_str,
        })

    # ── 4. Goal pace ─────────────────────────────────────────────────────
    cp = gp.get("current_pace") or {}
    ahead = cp.get("months_ahead_behind")
    if isinstance(ahead, (int, float)):
        n += 1
        if ahead >= 0:
            alerts.append({
                "id": f"alert_{n:03d}",
                "type": "goal",
                "severity": "success",
                "message": f"{abs(ahead):.1f} months ahead of baseline goal pace",
                "details": {
                    "months_ahead": ahead,
                    "on_track": cp.get("on_track"),
                    "revised_goal_date": cp.get("revised_goal_date"),
                    "pace_category": cp.get("pace_category"),
                },
                "triggered_at": now_str,
            })
        else:
            alerts.append({
                "id": f"alert_{n:03d}",
                "type": "goal",
                "severity": "warning",
                "message": f"{abs(ahead):.1f} months behind baseline goal pace",
                "action_required": "Review contribution strategy or consider rebalancing",
                "details": {
                    "months_behind": abs(ahead),
                    "on_track": cp.get("on_track"),
                    "pace_category": cp.get("pace_category"),
                },
                "triggered_at": now_str,
            })

    # ── 5. Drawdown / risk ───────────────────────────────────────────────
    dd = risk.get("drawdown_status") or {}
    if dd.get("currently_in_drawdown"):
        n += 1
        depth = dd.get("current_drawdown_depth_pct", 0)
        duration = dd.get("current_drawdown_duration_days", 0)
        rec = risk.get("recovery_metrics") or {}
        est_recovery = rec.get("estimated_recovery_days")
        # Derive estimated recovery when not available from recovery_metrics
        if est_recovery is None and isinstance(depth, (int, float)) and abs(depth) > 0:
            vol_30d = risk.get("vol_30d_pct")
            if isinstance(vol_30d, (int, float)) and vol_30d > 0:
                # Estimate: days to recover = depth / (daily vol * expected drift)
                daily_vol = vol_30d / (252 ** 0.5)
                est_recovery = max(1, round(abs(depth) / max(daily_vol * 0.5, 0.01)))
            else:
                # Rough heuristic: ~5 trading days per 1% drawdown
                est_recovery = max(1, round(abs(depth) * 5))

        severity = "warning" if isinstance(depth, (int, float)) and abs(depth) > 10 else "info"
        alert_entry = {
            "id": f"alert_{n:03d}",
            "type": "risk",
            "severity": severity,
            "message": f"Currently in drawdown: {depth}% from peak {duration} days ago",
            "details": {
                "drawdown_depth_pct": depth,
                "drawdown_duration_days": duration,
                "estimated_recovery_days": est_recovery,
                "peak_value": dd.get("peak_value"),
                "peak_date": dd.get("peak_date"),
            },
            "triggered_at": now_str,
        }
        if severity == "warning":
            alert_entry["action_required"] = f"Portfolio down {abs(depth):.1f}% — review positions for rebalancing"
        alerts.append(alert_entry)

    # ── 6. Rate change ───────────────────────────────────────────────────
    rates = mg.get("rates") or {}
    fr = rates.get("apr_future_pct")
    cr = rates.get("apr_current_pct")
    fd = rates.get("apr_future_date")
    if fr and cr and fr != cr:
        n += 1
        loan = totals.get("margin_loan_balance") or 0
        cm = round(loan * (cr / 100) / 12, 2)
        fm = round(loan * (fr / 100) / 12, 2)
        alerts.append({
            "id": f"alert_{n:03d}",
            "type": "rate",
            "severity": "warning",
            "message": f"Interest rate change scheduled {fd} ({cr}% \u2192 {fr}%)",
            "action_required": f"Monthly interest will change from ${cm:,.2f} to ${fm:,.2f}",
            "details": {
                "current_rate_pct": cr,
                "future_rate_pct": fr,
                "effective_date": fd,
                "monthly_increase": round(fm - cm, 2),
            },
            "triggered_at": now_str,
        })

    return alerts


# ─── Allocation ──────────────────────────────────────────────────────────────


def _build_allocation(holdings: list[dict]) -> dict:
    """Build allocation breakdown from v4 holdings."""
    sorted_h = sorted(holdings, key=lambda h: h.get("weight_pct") or 0, reverse=True)

    by_position: list[dict] = []
    weights: list[float] = []
    freq_groups: dict[str, dict] = {}
    risk_groups: dict[str, dict] = {}

    for h in sorted_h:
        sym = h.get("symbol")
        w = h.get("weight_pct") or 0
        mv = h.get("market_value") or 0
        by_position.append({"symbol": sym, "weight_pct": w, "market_value": mv})
        weights.append(w)

        ult = h.get("ultimate") or {}
        freq = ult.get("distribution_frequency") or "unknown"
        freq_groups.setdefault(freq, {"weight_pct": 0.0, "count": 0})
        freq_groups[freq]["weight_pct"] += w
        freq_groups[freq]["count"] += 1

        rcat = ult.get("risk_quality_category") or "unknown"
        risk_groups.setdefault(rcat, {"weight_pct": 0.0, "count": 0})
        risk_groups[rcat]["weight_pct"] += w
        risk_groups[rcat]["count"] += 1

    # Round group weights
    for groups in [freq_groups, risk_groups]:
        for g in groups.values():
            g["weight_pct"] = round(g["weight_pct"], 3)

    # Concentration metrics
    sw = sorted(weights, reverse=True)
    top3 = sum(sw[:3]) if len(sw) >= 3 else sum(sw)
    top5 = sum(sw[:5]) if len(sw) >= 5 else sum(sw)
    hhi = sum((w / 100) ** 2 for w in weights) if weights else 0

    if hhi < 0.10:
        cat = "diversified"
    elif hhi < 0.20:
        cat = "moderate"
    elif hhi < 0.30:
        cat = "concentrated"
    else:
        cat = "highly_concentrated"

    return {
        "by_position": by_position,
        "concentration": {
            "top_3_weight_pct": round(top3, 3),
            "top_5_weight_pct": round(top5, 3),
            "herfindahl_index": round(hhi, 3),
            "concentration_category": cat,
        },
        "by_distribution_frequency": freq_groups,
        "risk_profile_mix": risk_groups,
    }


# ─── Margin consolidation ───────────────────────────────────────────────────


def _build_margin(daily: dict) -> dict:
    """Consolidate margin_guidance + margin_stress into single section."""
    ms = daily.get("margin_stress") or {}
    mg = daily.get("margin_guidance") or {}
    scenarios = ms.get("stress_scenarios") or {}
    mc = scenarios.get("margin_call_distance") or {}
    rate_shock = scenarios.get("interest_rate_shock") or {}

    # Flatten rate shock dict → array
    rate_arr = []
    for key in sorted(rate_shock.keys()):
        label = key.replace("rate_plus_", "+").replace("bp", "bp")
        rate_arr.append({"scenario": label, **rate_shock[key]})

    # Sign fix: negative decline → positive buffer
    decline = mc.get("portfolio_decline_to_call_pct")
    buffer = abs(decline) if isinstance(decline, (int, float)) else decline

    rates = mg.get("rates") or {}
    totals = daily.get("totals") or {}
    income_data = daily.get("income") or {}
    loan = totals.get("margin_loan_balance") or 0
    cr_pct = rates.get("apr_current_pct")
    fr_pct = rates.get("apr_future_pct")

    # Build scheduled_rate_change with derived expense + coverage
    sched_rate = {
        "effective_date": rates.get("apr_future_date"),
        "current_rate_pct": cr_pct,
        "future_rate_pct": fr_pct,
    }
    if isinstance(fr_pct, (int, float)) and loan > 0:
        future_monthly = round(loan * (fr_pct / 100) / 12, 2)
        sched_rate["future_monthly_expense"] = future_monthly
        monthly_income = income_data.get("projected_monthly_income")
        if isinstance(monthly_income, (int, float)) and monthly_income > 0:
            sched_rate["income_coverage_ratio"] = round(monthly_income / future_monthly, 2)
    if isinstance(cr_pct, (int, float)) and loan > 0:
        sched_rate["current_monthly_expense"] = round(loan * (cr_pct / 100) / 12, 2)

    return {
        "current": ms.get("current"),
        "stress": {
            "margin_call_risk": {
                "buffer_to_margin_call_pct": buffer,
                "dollar_decline_to_call": mc.get("dollar_decline_to_call"),
                "days_at_current_volatility": mc.get("days_at_current_volatility"),
                "buffer_status": mc.get("buffer_status"),
            },
            "rate_shock_scenarios": rate_arr,
            "scheduled_rate_change": sched_rate,
        },
        "efficiency": ms.get("efficiency"),
        "guidance": {
            "recommended_mode": mg.get("selected_mode"),
            "modes": mg.get("modes"),
            "rates": rates,
        },
        "history_90d": ms.get("historical_trends_90d"),
    }


# ─── Goals nesting ───────────────────────────────────────────────────────────


def _build_goals(daily: dict, income_data: dict) -> dict:
    """Nest all goal objects under single parent."""
    gp = daily.get("goal_progress") or {}
    gn = daily.get("goal_progress_net") or {}
    gt = daily.get("goal_tiers") or {}
    yield_pct = income_data.get("portfolio_current_yield_pct") or 0

    return {
        "target": {
            "target_monthly_income": gp.get("target_monthly"),
            "required_portfolio_value": gp.get("required_portfolio_value_at_goal"),
            "assumptions": f"{yield_pct}% yield maintained",
        },
        "baseline": gp,
        "net_of_interest": gn,
        "optimistic": daily.get("goal_progress_optimistic"),
        "tiers": gt.get("tiers") or [],
        "tiers_provenance": gt.get("provenance"),
        "current_state": gt.get("current_state"),
        "pace": daily.get("goal_pace"),
    }
