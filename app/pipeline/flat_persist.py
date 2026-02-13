"""Persist daily and period snapshots to flat tables (replacing JSON blobs)."""
from __future__ import annotations

import json
import sqlite3

from .diff_daily import (
    _totals,
    _income,
    _perf,
    _risk_flat,
    _rollups,
    _goal_progress,
    _goal_progress_net,
    _goal_pace,
    _margin_stress,
    _dividends,
    _dividends_upcoming,
    _coverage,
    _holdings_flat,
)
from ..alerts.evaluator import _holding_ultimate


def _flatten_holding_for_persist(h: dict) -> dict:
    """Extract flat keys from V5 nested holding (cost, valuation, income, analytics, reliability).
    Supports both V5 (nested) and V4 (flat) holding shapes.
    """
    if not h:
        return {}
    # If already flat (V4), return as-is
    if h.get("cost_basis") is not None or h.get("market_value") is not None:
        return dict(h)
    out = dict(h)
    cost = h.get("cost") or {}
    val = h.get("valuation") or {}
    inc = h.get("income") or {}
    ana = h.get("analytics") or {}
    rel = h.get("reliability") or h.get("dividend_reliability") or {}
    if isinstance(rel, dict):
        out.update(rel)
    out["cost_basis"] = cost.get("cost_basis") or h.get("cost_basis")
    out["avg_cost_per_share"] = cost.get("avg_cost_per_share") or h.get("avg_cost") or h.get("avg_cost_per_share")
    out["last_price"] = val.get("last_price") or h.get("last_price")
    out["market_value"] = val.get("market_value") or h.get("market_value")
    out["unrealized_pnl"] = val.get("unrealized_pnl") or h.get("unrealized_pnl")
    out["unrealized_pct"] = val.get("unrealized_pct") or h.get("unrealized_pct")
    out["weight_pct"] = val.get("portfolio_weight_pct") or val.get("weight_pct") or h.get("weight_pct")
    out["forward_12m_dividend"] = inc.get("forward_12m_dividend") or h.get("forward_12m_dividend")
    out["projected_monthly_dividend"] = inc.get("projected_monthly_dividend") or h.get("projected_monthly_dividend")
    out["projected_annual_dividend"] = (
        inc.get("projected_annual_dividend") or h.get("projected_annual_dividend")
        or inc.get("forward_12m_dividend") or h.get("forward_12m_dividend")
    )
    out["current_yield_pct"] = inc.get("current_yield_pct") or h.get("current_yield_pct")
    out["yield_on_cost_pct"] = inc.get("yield_on_cost_pct") or h.get("yield_on_cost_pct")
    out["dividends_30d"] = inc.get("dividends_30d") or h.get("dividends_30d")
    out["dividends_qtd"] = inc.get("dividends_qtd") or h.get("dividends_qtd")
    out["dividends_ytd"] = inc.get("dividends_ytd") or h.get("dividends_ytd")
    dist = ana.get("distribution") or {}
    risk = ana.get("risk") or {}
    perf = ana.get("performance") or {}
    out.update(dist)
    out.update(risk)
    out.update(perf)
    if "portfolio_weight_pct" in out and "weight_pct" not in out:
        out["weight_pct"] = out["portfolio_weight_pct"]
    return out


def _now_utc_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _goal_tiers(snap: dict) -> dict:
    """Goal tiers (tiers[], current_state) from V5 goals or legacy goal_tiers."""
    v4 = snap.get("goal_tiers")
    if v4:
        return v4
    goals = snap.get("goals") or {}
    return {"tiers": goals.get("tiers") or [], "current_state": goals.get("current_state")}


def _margin_guidance(snap: dict) -> dict:
    """Margin guidance from V5 margin.guidance or legacy margin_guidance."""
    v4 = snap.get("margin_guidance")
    if v4:
        return v4
    g = (snap.get("margin") or {}).get("guidance") or {}
    return {"selected_mode": g.get("recommended_mode"), "modes": g.get("modes"), "rates": g.get("rates")}


def _macro_snapshot(snap: dict) -> dict:
    return (snap.get("macro") or {}).get("snapshot") or {}


def _as_of(snap: dict) -> str:
    ts = snap.get("timestamps") or {}
    return ts.get("portfolio_data_as_of_local") or snap.get("as_of_date_local") or snap.get("as_of") or ""


def _write_daily_flat(conn: sqlite3.Connection, daily: dict, run_id: str) -> None:
    """Extract values from daily snapshot dict and write to daily_portfolio + child tables."""
    cur = conn.cursor()

    totals = _totals(daily)
    income = _income(daily)
    perf = _perf(daily)
    risk = _risk_flat(daily)
    rollups = _rollups(daily)
    gp = _goal_progress(daily)
    gpn = _goal_progress_net(daily)
    gt = _goal_tiers(daily)
    pace = _goal_pace(daily)
    ms = _margin_stress(daily)
    mg = _margin_guidance(daily)
    divs = _dividends(daily)
    div_up = _dividends_upcoming(daily)
    cov = _coverage(daily)
    macro_snap = _macro_snapshot(daily)
    as_of = _as_of(daily) or ""

    istab = rollups.get("income_stability") or {}
    if isinstance(istab, (int, float)):
        istab = {"stability_score": istab} if not isinstance(istab, dict) else {}
    tail = rollups.get("tail_risk") or {}
    vs_bench = (perf.get("vs_benchmark") or rollups.get("vs_benchmark") or {})
    stress = ms.get("stress_scenarios") or {}
    mc_dist = stress.get("margin_call_distance") or {}
    margin_stress_full = (daily.get("margin") or {}).get("stress") or {}
    margin_current = ms.get("current") or {}
    alloc = (daily.get("portfolio") or {}).get("allocation") or {}
    conc = alloc.get("concentration") or {}
    pvr = (daily.get("dividends") or {}).get("projected_vs_received") or divs.get("projected_vs_received") or {}
    rmtd = divs.get("realized_mtd") or (divs.get("realized") or {}).get("mtd") or {}
    if not rmtd and isinstance(divs.get("realized"), dict):
        rmtd = divs["realized"].get("mtd") or {}
    windows = divs.get("windows") or {}
    events = (div_up.get("events") or []) if isinstance(div_up, dict) else []
    cs = gt.get("current_state") or {}
    likely = pace.get("likely_tier") or {}
    cur_pace = pace.get("current_pace") or {}
    timestamps = daily.get("timestamps") or {}
    meta = daily.get("meta") or {}
    macro_obj = daily.get("macro") or {}

    created = (
        daily.get("created_at_utc")
        or meta.get("snapshot_created_at")
        or _now_utc_iso()
    )
    prices_as_of = (
        timestamps.get("price_data_as_of_utc")
        or daily.get("prices_as_of_utc")
        or daily.get("prices_as_of")
    )
    # Type safety: only sum numeric dividend amounts (can be 'N/A' string from bad data)
    div_upcoming_total = sum(
        (e.get("amount_est") if isinstance(e.get("amount_est"), (int, float))
         else e.get("estimated_amount") if isinstance(e.get("estimated_amount"), (int, float))
         else 0)
        for e in events
    )
    margin_history = ms.get("historical_trends_90d") or ms.get("history_90d")

    cur.execute(
        """
        INSERT OR REPLACE INTO daily_portfolio (
            as_of_date_local, built_from_run_id, created_at_utc,
            market_value, cost_basis, net_liquidation_value,
            unrealized_pnl, unrealized_pct, margin_loan_balance, ltv_pct,
            holdings_count, positions_profitable, positions_losing,
            projected_monthly_income, forward_12m_total,
            portfolio_yield_pct, portfolio_yield_on_cost_pct,
            income_stability_score, income_trend_6m, income_volatility_30d_pct,
            dividend_cut_count_12m, income_growth_json,
            twr_1m_pct, twr_3m_pct, twr_6m_pct, twr_12m_pct,
            vs_benchmark_symbol, vs_benchmark_twr_1y_pct,
            vs_benchmark_excess_1y_pct, vs_benchmark_corr_1y,
            vol_30d_pct, vol_90d_pct,
            sharpe_1y, sortino_1y, sortino_6m, sortino_3m, sortino_1m,
            sortino_sharpe_ratio, sortino_sharpe_divergence, calmar_1y,
            omega_ratio_1y, ulcer_index_1y, pain_adjusted_return,
            beta_portfolio, information_ratio_1y, tracking_error_1y_pct,
            portfolio_risk_quality,
            var_90_1d_pct, var_95_1d_pct, var_99_1d_pct, var_95_1w_pct, var_95_1m_pct,
            cvar_90_1d_pct, cvar_95_1d_pct, cvar_99_1d_pct, cvar_95_1w_pct, cvar_95_1m_pct,
            tail_risk_category, cvar_to_income_ratio,
            max_drawdown_1y_pct, drawdown_duration_1y_days,
            currently_in_drawdown, drawdown_depth_pct, drawdown_duration_days,
            drawdown_peak_value, drawdown_peak_date,
            dividends_realized_mtd, dividends_realized_30d,
            dividends_realized_ytd, dividends_realized_qtd,
            dividends_projected_monthly, dividends_received_pct,
            dividends_upcoming_count, dividends_upcoming_total_est,
            dividends_by_symbol_json, dividends_projected_vs_received_json,
            top3_weight_pct, top5_weight_pct, herfindahl_index, concentration_category,
            monthly_interest_current, annual_interest_current,
            margin_income_coverage, margin_interest_to_income_pct,
            buffer_to_margin_call_pct, dollar_decline_to_call,
            days_at_current_volatility, margin_call_buffer_status,
            margin_guidance_selected_mode, margin_guidance_json, margin_history_90d_json,
            goal_target_monthly_income, goal_required_portfolio_value,
            goal_progress_pct, goal_months_to_goal,
            goal_current_projected_monthly, goal_estimated_goal_date,
            goal_net_progress_pct, goal_net_months_to_goal,
            goal_net_current_projected_monthly,
            goal_tiers_portfolio_value, goal_tiers_projected_monthly, goal_tiers_yield_pct,
            goal_pace_months_ahead_behind, goal_pace_category,
            goal_pace_on_track, goal_pace_revised_goal_date, goal_pace_pct_of_tier,
            goal_likely_tier, goal_likely_tier_name,
            goal_likely_tier_confidence, goal_likely_tier_reason,
            goal_pace_windows_json, goal_pace_baseline_json,
            macro_vix, macro_ten_year_yield, macro_two_year_yield,
            macro_hy_spread_bps, macro_yield_spread_10y_2y, macro_stress_score,
            macro_cpi_yoy, macro_data_as_of_date,
            coverage_derived_pct, coverage_pulled_pct,
            coverage_missing_pct, coverage_filled_pct, coverage_missing_paths_json,
            prices_as_of_utc, schema_version, health_status
        ) VALUES (
            """ + ",".join(["?"] * 130) + """
        )
        """,
        (
            as_of,
            run_id,
            created,
            totals.get("market_value"),
            totals.get("cost_basis"),
            totals.get("net_liquidation_value"),
            totals.get("unrealized_pnl"),
            totals.get("unrealized_pct"),
            totals.get("margin_loan_balance"),
            totals.get("margin_to_portfolio_pct") or totals.get("ltv_pct"),
            totals.get("holdings_count"),
            totals.get("positions_profitable"),
            totals.get("positions_losing"),
            income.get("projected_monthly_income"),
            income.get("forward_12m_total"),
            income.get("portfolio_current_yield_pct") or income.get("portfolio_yield_pct"),
            income.get("portfolio_yield_on_cost_pct"),
            istab.get("stability_score") if isinstance(istab, dict) else None,
            istab.get("income_trend_6m") if isinstance(istab, dict) else None,
            istab.get("income_volatility_30d_pct") if isinstance(istab, dict) else None,
            istab.get("dividend_cut_count_12m") if isinstance(istab, dict) else None,
            json.dumps(income.get("income_growth")) if income.get("income_growth") else None,
            perf.get("twr_1m_pct"),
            perf.get("twr_3m_pct"),
            perf.get("twr_6m_pct"),
            perf.get("twr_12m_pct"),
            vs_bench.get("symbol") or vs_bench.get("benchmark"),
            vs_bench.get("benchmark_twr_1y_pct"),
            vs_bench.get("excess_1y_pct") or vs_bench.get("excess_return_1y_pct"),
            vs_bench.get("corr_1y") or vs_bench.get("correlation_to_benchmark"),
            risk.get("vol_30d_pct"),
            risk.get("vol_90d_pct"),
            risk.get("sharpe_1y"),
            risk.get("sortino_1y"),
            risk.get("sortino_6m"),
            risk.get("sortino_3m"),
            risk.get("sortino_1m"),
            risk.get("sortino_sharpe_ratio"),
            risk.get("sortino_sharpe_divergence"),
            risk.get("calmar_1y"),
            risk.get("omega_ratio_1y"),
            risk.get("ulcer_index_1y"),
            risk.get("pain_adjusted_return"),
            risk.get("beta_portfolio"),
            risk.get("information_ratio_1y"),
            risk.get("tracking_error_1y_pct"),
            risk.get("portfolio_risk_quality"),
            risk.get("var_90_1d_pct"),
            risk.get("var_95_1d_pct"),
            risk.get("var_99_1d_pct"),
            risk.get("var_95_1w_pct"),
            risk.get("var_95_1m_pct"),
            risk.get("cvar_90_1d_pct"),
            risk.get("cvar_95_1d_pct"),
            risk.get("cvar_99_1d_pct"),
            risk.get("cvar_95_1w_pct"),
            risk.get("cvar_95_1m_pct"),
            tail.get("tail_risk_category"),
            tail.get("cvar_to_income_ratio"),
            risk.get("max_drawdown_1y_pct"),
            risk.get("drawdown_duration_1y_days"),
            1 if risk.get("currently_in_drawdown") else (0 if risk.get("currently_in_drawdown") is not None else None),
            risk.get("current_drawdown_depth_pct") or risk.get("drawdown_depth_pct"),
            risk.get("current_drawdown_duration_days") or risk.get("drawdown_duration_days"),
            risk.get("peak_value") or risk.get("drawdown_peak_value"),
            risk.get("peak_date") or risk.get("drawdown_peak_date"),
            rmtd.get("total_dividends"),
            windows.get("30d", {}).get("total_dividends") if isinstance(windows.get("30d"), dict) else None,
            windows.get("ytd", {}).get("total_dividends") if isinstance(windows.get("ytd"), dict) else None,
            windows.get("qtd", {}).get("total_dividends") if isinstance(windows.get("qtd"), dict) else None,
            pvr.get("projected"),
            pvr.get("pct_of_projection"),
            len(events),
            div_upcoming_total,
            json.dumps(rmtd.get("by_symbol")) if rmtd.get("by_symbol") else None,
            json.dumps(pvr) if pvr else None,
            conc.get("top3_weight_pct") or conc.get("top_3_weight_pct"),
            conc.get("top5_weight_pct") or conc.get("top_5_weight_pct"),
            conc.get("herfindahl_index"),
            conc.get("category") or conc.get("concentration_category"),
            margin_current.get("monthly_interest") or margin_current.get("monthly_interest_expense"),
            margin_current.get("annual_interest") or margin_current.get("annual_interest_expense"),
            margin_current.get("income_interest_coverage") or margin_current.get("income_coverage"),
            margin_current.get("interest_to_income_pct") or (ms.get("efficiency") or {}).get("interest_expense_as_pct_of_income"),
            mc_dist.get("portfolio_decline_to_call_pct") or mc_dist.get("buffer_to_margin_call_pct"),
            mc_dist.get("dollar_decline_to_call"),
            mc_dist.get("days_at_current_volatility"),
            mc_dist.get("buffer_status"),
            mg.get("selected_mode") or mg.get("recommended_mode"),
            json.dumps(mg) if mg else None,
            json.dumps(margin_history) if margin_history else None,
            gp.get("target_monthly") or gp.get("target_monthly_income"),
            gp.get("required_portfolio_value") or gp.get("required_portfolio_value_at_goal"),
            gp.get("progress_pct"),
            gp.get("months_to_goal"),
            gp.get("current_projected_monthly"),
            gp.get("estimated_goal_date"),
            gpn.get("progress_pct"),
            gpn.get("months_to_goal"),
            gpn.get("current_projected_monthly_net") or gpn.get("current_projected_monthly"),
            cs.get("portfolio_value"),
            cs.get("projected_monthly_income"),
            cs.get("portfolio_yield_pct"),
            cur_pace.get("months_ahead_behind"),
            cur_pace.get("pace_category"),
            1 if cur_pace.get("on_track") else (0 if cur_pace.get("on_track") is not None else None),
            cur_pace.get("revised_goal_date"),
            cur_pace.get("pct_of_tier_pace"),
            likely.get("tier"),
            likely.get("name"),
            likely.get("confidence"),
            likely.get("reason"),
            json.dumps(pace.get("windows")) if pace.get("windows") else None,
            json.dumps(pace.get("baseline_projection")) if pace.get("baseline_projection") else None,
            macro_snap.get("vix"),
            macro_snap.get("ten_year_yield"),
            macro_snap.get("two_year_yield"),
            macro_snap.get("hy_spread_bps"),
            macro_snap.get("yield_spread_10y_2y"),
            macro_snap.get("macro_stress_score"),
            macro_snap.get("cpi_yoy"),
            macro_obj.get("as_of_date") or timestamps.get("macro_data_as_of_date"),
            cov.get("derived_pct"),
            cov.get("pulled_pct"),
            cov.get("missing_pct"),
            cov.get("filled_pct"),
            json.dumps(cov.get("missing_paths")) if cov.get("missing_paths") else None,
            prices_as_of,
            meta.get("schema_version") or "v4",
            meta.get("health_status"),
        ),
    )

    # Child tables: delete-then-insert for this date
    cur.execute("DELETE FROM daily_holdings WHERE as_of_date_local=?", (as_of,))
    cur.execute("DELETE FROM daily_goal_tiers WHERE as_of_date_local=?", (as_of,))
    cur.execute("DELETE FROM daily_margin_rate_scenarios WHERE as_of_date_local=?", (as_of,))
    cur.execute("DELETE FROM daily_return_attribution WHERE as_of_date_local=?", (as_of,))
    cur.execute("DELETE FROM daily_dividends_upcoming WHERE as_of_date_local=?", (as_of,))

    # Holdings
    holdings = _holdings_flat(daily)
    for h in holdings:
        flat = _flatten_holding_for_persist(h)
        # Use _holding_ultimate for analytics when flat lacks them (v4-derived data)
        use_ult = _holding_ultimate(h) if not all(k in flat for k in ("vol_30d_pct", "sortino_1y")) else flat
        cur.execute(
            """
            INSERT INTO daily_holdings (
                as_of_date_local, symbol, shares, trades_count,
                cost_basis, avg_cost_per_share,
                last_price, market_value, unrealized_pnl, unrealized_pct, weight_pct,
                forward_12m_dividend, projected_monthly_dividend, projected_annual_dividend,
                current_yield_pct, yield_on_cost_pct,
                dividends_30d, dividends_qtd, dividends_ytd,
                trailing_12m_yield_pct, forward_yield_pct, distribution_frequency,
                next_ex_date_est, last_ex_date,
                trailing_12m_div_per_share, forward_12m_div_per_share,
                vol_30d_pct, vol_90d_pct, beta_3y,
                max_drawdown_1y_pct, drawdown_duration_1y_days, downside_dev_1y_pct,
                sortino_1y, sortino_6m, sortino_3m, sortino_1m,
                sharpe_1y, calmar_1y,
                risk_quality_score, risk_quality_category, volatility_profile,
                var_90_1d_pct, var_95_1d_pct, var_99_1d_pct, var_95_1w_pct, var_95_1m_pct,
                cvar_90_1d_pct, cvar_95_1d_pct, cvar_99_1d_pct, cvar_95_1w_pct, cvar_95_1m_pct,
                twr_1m_pct, twr_3m_pct, twr_6m_pct, twr_12m_pct, corr_1y,
                reliability_consistency_score, reliability_trend_6m, reliability_missed_payments_12m
            ) VALUES (
                """ + ",".join(["?"] * 59) + """
            )
        """,
            (
                as_of,
                flat.get("symbol"),
                flat.get("shares"),
                flat.get("trades_count") or flat.get("trades"),
                flat.get("cost_basis"),
                flat.get("avg_cost") or flat.get("avg_cost_per_share"),
                flat.get("last_price"),
                flat.get("market_value"),
                flat.get("unrealized_pnl"),
                flat.get("unrealized_pct"),
                flat.get("weight_pct"),
                flat.get("forward_12m_dividend"),
                flat.get("projected_monthly_dividend"),
                flat.get("projected_annual_dividend"),
                flat.get("current_yield_pct"),
                flat.get("yield_on_cost_pct"),
                flat.get("dividends_30d"),
                flat.get("dividends_qtd"),
                flat.get("dividends_ytd"),
                use_ult.get("trailing_12m_yield_pct"),
                use_ult.get("forward_yield_pct"),
                use_ult.get("distribution_frequency"),
                use_ult.get("next_ex_date_est"),
                use_ult.get("last_ex_date"),
                use_ult.get("trailing_12m_div_per_share"),
                use_ult.get("forward_12m_div_per_share"),
                use_ult.get("vol_30d_pct"),
                use_ult.get("vol_90d_pct"),
                use_ult.get("beta_3y"),
                use_ult.get("max_drawdown_1y_pct"),
                use_ult.get("drawdown_duration_1y_days"),
                use_ult.get("downside_dev_1y_pct"),
                use_ult.get("sortino_1y"),
                use_ult.get("sortino_6m"),
                use_ult.get("sortino_3m"),
                use_ult.get("sortino_1m"),
                use_ult.get("sharpe_1y"),
                use_ult.get("calmar_1y"),
                use_ult.get("risk_quality_score"),
                use_ult.get("risk_quality_category"),
                use_ult.get("volatility_profile"),
                use_ult.get("var_90_1d_pct"),
                use_ult.get("var_95_1d_pct"),
                use_ult.get("var_99_1d_pct"),
                use_ult.get("var_95_1w_pct"),
                use_ult.get("var_95_1m_pct"),
                use_ult.get("cvar_90_1d_pct"),
                use_ult.get("cvar_95_1d_pct"),
                use_ult.get("cvar_99_1d_pct"),
                use_ult.get("cvar_95_1w_pct"),
                use_ult.get("cvar_95_1m_pct"),
                use_ult.get("twr_1m_pct"),
                use_ult.get("twr_3m_pct"),
                use_ult.get("twr_6m_pct"),
                use_ult.get("twr_12m_pct"),
                use_ult.get("corr_1y"),
                flat.get("consistency_score"),
                flat.get("trend_6m") or flat.get("dividend_trend_6m"),
                flat.get("missed_payments_12m"),
            ),
        )

    # Goal tiers (snapshot tiers often lack target_monthly/progress_pct/confidence; fill from baseline)
    tiers = gt.get("tiers") or []
    gp = _goal_progress(daily)
    baseline_target = gp.get("target_monthly") or gp.get("target_monthly_income")
    baseline_progress = gp.get("progress_pct")
    likely_tier_num = likely.get("tier")
    likely_confidence = likely.get("confidence") or "high"
    for t in tiers:
        assumptions = t.get("assumptions") or {}
        # Never persist the lazy default "medium"; derive from likely_tier (high for match, low for others)
        t_conf = t.get("confidence")
        if t_conf is None or t_conf == "medium":
            t_conf = likely_confidence if t.get("tier") == likely_tier_num else "low"
        cur.execute(
            """
            INSERT INTO daily_goal_tiers (
                as_of_date_local, tier, name, description,
                target_monthly, required_portfolio_value, final_portfolio_value,
                progress_pct, months_to_goal, estimated_goal_date, confidence,
                assumption_monthly_contribution, assumption_drip_enabled,
                assumption_annual_appreciation_pct, assumption_ltv_maintained,
                assumption_target_ltv_pct
            ) VALUES (?,?,?,?, ?,?,?, ?,?,?,?, ?,?,?,?,?)
        """,
            (
                as_of,
                t.get("tier"),
                t.get("name"),
                t.get("description"),
                t.get("target_monthly") if t.get("target_monthly") is not None else baseline_target,
                t.get("required_portfolio_value"),
                t.get("final_portfolio_value"),
                t.get("progress_pct") if t.get("progress_pct") is not None else baseline_progress,
                t.get("months_to_goal"),
                t.get("estimated_goal_date"),
                t_conf,
                assumptions.get("monthly_contribution"),
                1 if assumptions.get("drip_enabled") else (0 if assumptions.get("drip_enabled") is not None else None),
                assumptions.get("annual_appreciation_pct"),
                1 if assumptions.get("ltv_maintained") else (0 if assumptions.get("ltv_maintained") is not None else None),
                assumptions.get("target_ltv_pct"),
            ),
        )

    # Margin rate scenarios: V5 has rate_shock_scenarios (array), v4 has interest_rate_shock (dict)
    rate_shock = margin_stress_full.get("interest_rate_shock") or stress.get("interest_rate_shock") or {}
    if not rate_shock and isinstance(margin_stress_full.get("rate_shock_scenarios"), list):
        for item in margin_stress_full["rate_shock_scenarios"]:
            if isinstance(item, dict) and item.get("scenario"):
                rate_shock[item["scenario"]] = item
    if isinstance(rate_shock, dict):
        for scenario_key, scenario_data in rate_shock.items():
            if isinstance(scenario_data, dict):
                cur.execute(
                    """
                    INSERT INTO daily_margin_rate_scenarios (
                        as_of_date_local, scenario, new_rate_pct, new_monthly_cost,
                        income_coverage, margin_impact_pct
                    ) VALUES (?,?,?,?,?,?)
                """,
                    (
                        as_of,
                        scenario_key,
                        scenario_data.get("new_rate_pct"),
                        scenario_data.get("new_monthly_cost") or scenario_data.get("monthly_expense"),
                        scenario_data.get("income_coverage_ratio") or scenario_data.get("income_coverage"),
                        scenario_data.get("margin_impact_pct"),
                    ),
                )

    # Return attribution: per-symbol from holdings (contribution_analysis_*), portfolio from summary
    holdings_list = _holdings_flat(daily)
    for window_key in ("1m", "3m", "6m", "12m"):
        # Per-symbol from holdings
        for h in holdings_list:
            sym = h.get("symbol")
            if not sym:
                continue
            ca = h.get(f"contribution_analysis_{window_key}")
            if not isinstance(ca, dict):
                continue
            contrib = ca.get("return_contribution_pct") or ca.get("contribution_pct")
            weight = ca.get("position_weight_pct") or ca.get("weight_avg_pct")
            ret = ca.get("roi_on_cost_pct") or ca.get("return_pct")
            cur.execute(
                """
                INSERT INTO daily_return_attribution (
                    as_of_date_local, window, symbol,
                    contribution_pct, weight_avg_pct, return_pct
                ) VALUES (?,?,?,?,?,?)
            """,
                (as_of, window_key, sym, contrib, weight, ret),
            )
        # Portfolio-level aggregate from summary
        summary = rollups.get(f"return_attribution_{window_key}") or {}
        if isinstance(summary, dict) and "total_return_pct" in summary:
            total_ret = summary.get("total_return_pct")
            cur.execute(
                """
                INSERT INTO daily_return_attribution (
                    as_of_date_local, window, symbol,
                    contribution_pct, weight_avg_pct, return_pct
                ) VALUES (?,?,?,?,?,?)
            """,
                (as_of, window_key, "_portfolio", total_ret, 100.0, total_ret),
            )

    # Upcoming dividends
    for e in events:
        ex_date = e.get("ex_date_est") or e.get("ex_date") or ""
        if ex_date:
            cur.execute(
                """
                INSERT OR IGNORE INTO daily_dividends_upcoming (
                    as_of_date_local, symbol, ex_date_est, pay_date_est, amount_est
                ) VALUES (?,?,?,?,?)
            """,
                (
                    as_of,
                    e.get("symbol"),
                    ex_date,
                    e.get("pay_date_est") or e.get("pay_date"),
                    e.get("amount_est") or e.get("estimated_amount"),
                ),
            )

    conn.commit()


_SNAPSHOT_TYPE_TO_DB = {
    "weekly": "WEEK",
    "monthly": "MONTH",
    "quarterly": "QUARTER",
    "yearly": "YEAR",
}


def _write_period_flat(conn: sqlite3.Connection, snapshot: dict, run_id: str) -> None:
    """Extract values from period snapshot dict and write to period_summary + child tables."""
    from datetime import datetime, timezone

    cur = conn.cursor()
    period = snapshot.get("period") or {}
    ps = snapshot.get("period_summary") or {}
    bench = snapshot.get("benchmark") or {}
    mode = snapshot.get("snapshot_mode") or "final"
    snapshot_type = snapshot.get("snapshot_type") or ""
    is_rolling = 1 if mode == "to_date" else 0

    period_type = _SNAPSHOT_TYPE_TO_DB.get(snapshot_type, snapshot_type.upper())
    period_start = period.get("start_date") or ""
    period_end = period.get("end_date") or snapshot.get("as_of") or ""

    # Totals
    t_s = (ps.get("totals") or {}).get("start") or {}
    t_e = (ps.get("totals") or {}).get("end") or {}
    t_d = (ps.get("totals") or {}).get("delta") or {}

    # Income
    i_s = (ps.get("income") or {}).get("start") or {}
    i_e = (ps.get("income") or {}).get("end") or {}
    i_d = (ps.get("income") or {}).get("delta") or {}

    # Performance
    perf_period = (ps.get("performance") or {}).get("period") or {}
    twr_w = (ps.get("performance") or {}).get("twr_windows") or {}

    # Risk
    r_s = (ps.get("risk") or {}).get("start") or {}
    r_e = (ps.get("risk") or {}).get("end") or {}
    r_d = (ps.get("risk") or {}).get("delta") or {}
    risk_stats_dict = (ps.get("risk") or {}).get("stats") or {}

    # Goals
    gp_s = (ps.get("goal_progress") or {}).get("start") or {}
    gp_e = (ps.get("goal_progress") or {}).get("end") or {}
    gp_d = (ps.get("goal_progress") or {}).get("delta") or {}
    gpn_s = (ps.get("goal_progress_net") or {}).get("start") or {}
    gpn_e = (ps.get("goal_progress_net") or {}).get("end") or {}
    gpn_d = (ps.get("goal_progress_net") or {}).get("delta") or {}
    goal_pace = ps.get("goal_pace") or {}
    pace_cur = goal_pace.get("current_pace") or {}
    pace_category_vals = goal_pace.get("pace_category") or {}
    pace_months_vals = goal_pace.get("months_ahead_behind") or {}
    pace_tier_vals = goal_pace.get("tier_pace_pct") or {}

    pace_category_start = pace_category_vals.get("start")
    pace_category_end = pace_category_vals.get("end")
    if pace_category_start is None:
        pace_category_start = pace_cur.get("pace_category")
    if pace_category_end is None:
        pace_category_end = pace_cur.get("pace_category")

    pace_months_start = pace_months_vals.get("start")
    pace_months_end = pace_months_vals.get("end")
    pace_months_delta = pace_months_vals.get("delta")
    if pace_months_start is None:
        pace_months_start = pace_cur.get("months_ahead_behind")
    if pace_months_end is None:
        pace_months_end = pace_cur.get("months_ahead_behind")
    if pace_months_delta is None and isinstance(pace_months_start, (int, float)) and isinstance(pace_months_end, (int, float)):
        pace_months_delta = round(pace_months_end - pace_months_start, 2)

    # Composition
    comp_s = (ps.get("composition") or {}).get("start") or {}
    comp_e = (ps.get("composition") or {}).get("end") or {}
    comp_d = (ps.get("composition") or {}).get("delta") or {}

    # Macro
    mac_s = (ps.get("macro") or {}).get("start") or {}
    mac_e = (ps.get("macro") or {}).get("end") or {}
    mac_a = (ps.get("macro") or {}).get("avg") or {}
    mac_d = (ps.get("macro") or {}).get("delta") or {}

    # Period stats (new)
    period_stats = ps.get("period_stats") or {}
    mv_stats_dict = period_stats.get("market_value") or {}
    nlv_stats_dict = period_stats.get("net_liquidation_value") or {}
    income_stats_dict = period_stats.get("projected_monthly_income") or {}
    yield_stats_dict = period_stats.get("yield_pct") or {}
    ltv_stats_dict = period_stats.get("margin_to_portfolio_pct") or {}
    margin_balance_stats_dict = period_stats.get("margin_balance") or {}
    var_95_1d_avg = period_stats.get("var_95_1d_avg")
    var_90_1d_avg = period_stats.get("var_90_1d_avg")
    var_99_1d_avg = period_stats.get("var_99_1d_avg")
    var_95_1w_avg = period_stats.get("var_95_1w_avg")
    var_95_1m_avg = period_stats.get("var_95_1m_avg")
    cvar_95_1d_avg = period_stats.get("cvar_95_1d_avg")
    cvar_90_1d_avg = period_stats.get("cvar_90_1d_avg")
    cvar_99_1d_avg = period_stats.get("cvar_99_1d_avg")
    cvar_95_1w_avg = period_stats.get("cvar_95_1w_avg")
    cvar_95_1m_avg = period_stats.get("cvar_95_1m_avg")
    margin_interest_dict = period_stats.get("margin_interest") or {}

    # Period drawdown (new)
    period_dd = ps.get("period_drawdown") or {}

    # Var breach (new)
    var_breach = ps.get("var_breach") or {}

    # Margin safety (new)
    margin_safety = ps.get("margin_safety") or {}

    # Macro period stats (new)
    macro_period_stats = ps.get("macro_period_stats") or {}

    now_utc = datetime.now(timezone.utc).isoformat()

    # Delete existing row + children before insert (idempotent upsert)
    for tbl in (
        "period_risk_stats",
        "period_interval_holdings",
        "period_interval_attribution",
        "period_intervals",
        "period_holding_changes",
        "period_macro_stats",
        "period_activity",
        "period_contributions",
        "period_withdrawals",
        "period_dividend_events",
        "period_trades",
        "period_margin_detail",
    ):
        cur.execute(
            f"DELETE FROM {tbl} WHERE period_type=? AND period_start_date=? AND period_end_date=?",
            (period_type, period_start, period_end),
        )
    cur.execute(
        "DELETE FROM period_summary WHERE period_type=? AND period_start_date=? AND period_end_date=? AND is_rolling=?",
        (period_type, period_start, period_end, is_rolling),
    )

    # Build tuple first for debugging
    values_tuple = (
            period_type, period_start, period_end, period.get("label"), is_rolling, mode,
            period.get("expected_days"), period.get("observed_days"), period.get("coverage_pct"),
            1 if period.get("is_complete") else 0,
            # Totals
            t_s.get("total_market_value"), t_e.get("total_market_value"), t_d.get("total_market_value"),
            t_s.get("net_liquidation_value"), t_e.get("net_liquidation_value"), t_d.get("net_liquidation_value"),
            t_s.get("cost_basis"), t_e.get("cost_basis"), t_d.get("cost_basis"),
            t_s.get("unrealized_pct"), t_e.get("unrealized_pct"), t_d.get("unrealized_pct"),
            t_s.get("unrealized_pnl"), t_e.get("unrealized_pnl"), t_d.get("unrealized_pnl"),
            t_s.get("margin_loan_balance"), t_e.get("margin_loan_balance"), t_d.get("margin_loan_balance"),
            margin_balance_stats_dict.get("avg"), margin_balance_stats_dict.get("min"), margin_balance_stats_dict.get("max"),
            margin_balance_stats_dict.get("std"), margin_balance_stats_dict.get("min_date"), margin_balance_stats_dict.get("max_date"),
            t_s.get("margin_to_portfolio_pct"), t_e.get("margin_to_portfolio_pct"), t_d.get("margin_to_portfolio_pct"),
            ltv_stats_dict.get("avg"), ltv_stats_dict.get("min"), ltv_stats_dict.get("max"),
            ltv_stats_dict.get("std"), ltv_stats_dict.get("min_date"), ltv_stats_dict.get("max_date"),
            # Performance
            perf_period.get("twr_period_pct"), perf_period.get("pnl_dollar_period"), perf_period.get("pnl_pct_period"),
            twr_w.get("twr_1m_pct_start"), twr_w.get("twr_1m_pct_end"), twr_w.get("twr_1m_pct_delta"),
            twr_w.get("twr_3m_pct_start"), twr_w.get("twr_3m_pct_end"), twr_w.get("twr_3m_pct_delta"),
            twr_w.get("twr_6m_pct_start"), twr_w.get("twr_6m_pct_end"), twr_w.get("twr_6m_pct_delta"),
            twr_w.get("twr_12m_pct_start"), twr_w.get("twr_12m_pct_end"), twr_w.get("twr_12m_pct_delta"),
            # Income
            i_s.get("projected_monthly_income"), i_e.get("projected_monthly_income"), i_d.get("projected_monthly_income"),
            i_s.get("forward_12m_total"), i_e.get("forward_12m_total"), i_d.get("forward_12m_total"),
            i_s.get("portfolio_current_yield_pct"), i_e.get("portfolio_current_yield_pct"), i_d.get("portfolio_current_yield_pct"),
            i_s.get("portfolio_yield_on_cost_pct"), i_e.get("portfolio_yield_on_cost_pct"), i_d.get("portfolio_yield_on_cost_pct"),
            ps.get("income", {}).get("dividends_received_period"),  # dividends_received_period
            # Period stats: market value
            mv_stats_dict.get("avg"), mv_stats_dict.get("min"), mv_stats_dict.get("max"), mv_stats_dict.get("std"),
            mv_stats_dict.get("min_date"), mv_stats_dict.get("max_date"),
            # Period stats: net liquidation value
            nlv_stats_dict.get("avg"), nlv_stats_dict.get("min"), nlv_stats_dict.get("max"), nlv_stats_dict.get("std"),
            nlv_stats_dict.get("min_date"), nlv_stats_dict.get("max_date"),
            # Period stats: income
            income_stats_dict.get("avg"), income_stats_dict.get("min"), income_stats_dict.get("max"), income_stats_dict.get("std"),
            # Period stats: yield
            yield_stats_dict.get("avg"), yield_stats_dict.get("min"), yield_stats_dict.get("max"), yield_stats_dict.get("std"),
            # Period stats: margin to portfolio
            ltv_stats_dict.get("avg"), ltv_stats_dict.get("min"), ltv_stats_dict.get("max"), ltv_stats_dict.get("std"),
            # Period stats: VaR
            var_95_1d_avg, var_90_1d_avg, var_99_1d_avg, var_95_1w_avg, var_95_1m_avg,
            cvar_95_1d_avg, cvar_90_1d_avg, cvar_99_1d_avg, cvar_95_1w_avg, cvar_95_1m_avg,
            # Period drawdown
            period_dd.get("period_max_drawdown_pct"), period_dd.get("period_max_drawdown_date"),
            period_dd.get("period_recovery_date"), period_dd.get("period_days_in_drawdown"),
            period_dd.get("period_drawdown_count"),
            # Var breach
            var_breach.get("days_exceeding_var_95"),
            var_breach.get("worst_day_return_pct"), var_breach.get("worst_day_date"),
            var_breach.get("best_day_return_pct"), var_breach.get("best_day_date"),
            # Margin safety metrics
            margin_safety.get("min_buffer_to_call_pct"), margin_safety.get("min_buffer_date"),
            margin_safety.get("days_below_50pct_buffer"), margin_safety.get("days_below_40pct_buffer"),
            margin_safety.get("margin_call_events"),
            # Margin interest
            margin_interest_dict.get("start_apr_pct"), margin_interest_dict.get("end_apr_pct"), margin_interest_dict.get("avg_apr_pct"),
            # Risk
            r_s.get("vol_30d_pct"), r_e.get("vol_30d_pct"), r_d.get("vol_30d_pct"),
            r_s.get("vol_90d_pct"), r_e.get("vol_90d_pct"), r_d.get("vol_90d_pct"),
            r_s.get("sharpe_1y"), r_e.get("sharpe_1y"), r_d.get("sharpe_1y"),
            r_s.get("sortino_1y"), r_e.get("sortino_1y"), r_d.get("sortino_1y"),
            r_s.get("sortino_6m"), r_e.get("sortino_6m"), r_d.get("sortino_6m"),
            r_s.get("sortino_3m"), r_e.get("sortino_3m"), r_d.get("sortino_3m"),
            r_s.get("sortino_1m"), r_e.get("sortino_1m"), r_d.get("sortino_1m"),
            r_s.get("calmar_1y"), r_e.get("calmar_1y"), r_d.get("calmar_1y"),
            r_s.get("max_drawdown_1y_pct"), r_e.get("max_drawdown_1y_pct"), r_d.get("max_drawdown_1y_pct"),
            r_s.get("portfolio_risk_quality"), r_e.get("portfolio_risk_quality"),
            # Goals
            gp_s.get("progress_pct"), gp_e.get("progress_pct"), gp_d.get("progress_pct"),
            gp_s.get("current_projected_monthly"), gp_e.get("current_projected_monthly"), gp_d.get("current_projected_monthly"),
            gp_s.get("months_to_goal"), gp_e.get("months_to_goal"), gp_d.get("months_to_goal"),
            gpn_s.get("progress_pct"), gpn_e.get("progress_pct"), gpn_d.get("progress_pct"),
            gpn_s.get("current_projected_monthly_net"), gpn_e.get("current_projected_monthly_net"), gpn_d.get("current_projected_monthly_net"),
            # Goal pace
            pace_category_start, pace_category_end,
            pace_months_start, pace_months_end, pace_months_delta,
            pace_tier_vals.get("start"), pace_tier_vals.get("end"), pace_tier_vals.get("delta"),
            # Composition
            comp_s.get("holding_count"), comp_e.get("holding_count"), comp_d.get("holding_count"),
            comp_s.get("concentration_top3_pct"), comp_e.get("concentration_top3_pct"), comp_d.get("concentration_top3_pct"),
            comp_s.get("concentration_top5_pct"), comp_e.get("concentration_top5_pct"), comp_d.get("concentration_top5_pct"),
            comp_s.get("concentration_herfindahl"), comp_e.get("concentration_herfindahl"), comp_d.get("concentration_herfindahl"),
            # Macro
            mac_s.get("ten_year_yield"), mac_e.get("ten_year_yield"), mac_a.get("ten_year_yield"), mac_d.get("ten_year_yield"),
            mac_s.get("two_year_yield"), mac_e.get("two_year_yield"), mac_a.get("two_year_yield"), mac_d.get("two_year_yield"),
            mac_s.get("vix"), mac_e.get("vix"), mac_a.get("vix"), mac_d.get("vix"),
            mac_s.get("cpi_yoy"), mac_e.get("cpi_yoy"), mac_a.get("cpi_yoy"), mac_d.get("cpi_yoy"),
            # Benchmark
            bench.get("symbol"), bench.get("period_return_pct"),
            bench.get("twr_1y_pct_start"), bench.get("twr_1y_pct_end"), bench.get("twr_1y_pct_delta"),
            # Meta
            run_id, now_utc,
    )

    cur.execute(
            """
            INSERT INTO period_summary (
                period_type, period_start_date, period_end_date, period_label, is_rolling, snapshot_mode,
                expected_days, observed_days, coverage_pct, is_complete,
                mv_start, mv_end, mv_delta,
                nlv_start, nlv_end, nlv_delta,
                cost_basis_start, cost_basis_end, cost_basis_delta,
                unrealized_pct_start, unrealized_pct_end, unrealized_pct_delta,
                unrealized_pnl_start, unrealized_pnl_end, unrealized_pnl_delta,
                margin_balance_start, margin_balance_end, margin_balance_delta,
                margin_balance_avg, margin_balance_min, margin_balance_max, margin_balance_std, margin_balance_min_date, margin_balance_max_date,
                ltv_pct_start, ltv_pct_end, ltv_pct_delta,
                ltv_pct_avg, ltv_pct_min, ltv_pct_max, ltv_pct_std, ltv_pct_min_date, ltv_pct_max_date,
                twr_period_pct, pnl_dollar_period, pnl_pct_period,
                twr_1m_start, twr_1m_end, twr_1m_delta,
                twr_3m_start, twr_3m_end, twr_3m_delta,
                twr_6m_start, twr_6m_end, twr_6m_delta,
                twr_12m_start, twr_12m_end, twr_12m_delta,
                monthly_income_start, monthly_income_end, monthly_income_delta,
                forward_12m_start, forward_12m_end, forward_12m_delta,
                yield_start, yield_end, yield_delta,
                yield_on_cost_start, yield_on_cost_end, yield_on_cost_delta,
                dividends_received_period,
                mv_avg, mv_min, mv_max, mv_std, mv_min_date, mv_max_date,
                nlv_avg, nlv_min, nlv_max, nlv_std, nlv_min_date, nlv_max_date,
                projected_monthly_avg, projected_monthly_min, projected_monthly_max, projected_monthly_std,
                yield_pct_avg, yield_pct_min, yield_pct_max, yield_pct_std,
                margin_to_portfolio_pct_avg, margin_to_portfolio_pct_min, margin_to_portfolio_pct_max, margin_to_portfolio_pct_std,
                var_95_1d_avg, var_90_1d_avg, var_99_1d_avg, var_95_1w_avg, var_95_1m_avg,
                cvar_95_1d_avg, cvar_90_1d_avg, cvar_99_1d_avg, cvar_95_1w_avg, cvar_95_1m_avg,
                period_max_drawdown_pct, period_max_drawdown_date, period_recovery_date, period_days_in_drawdown, period_drawdown_count,
                days_exceeding_var_95, worst_day_return_pct, worst_day_date, best_day_return_pct, best_day_date,
                margin_min_buffer_to_call_pct, margin_min_buffer_date, margin_days_below_50pct_buffer, margin_days_below_40pct_buffer, margin_call_events,
                margin_apr_start, margin_apr_end, margin_apr_avg,
                vol_30d_start, vol_30d_end, vol_30d_delta,
                vol_90d_start, vol_90d_end, vol_90d_delta,
                sharpe_1y_start, sharpe_1y_end, sharpe_1y_delta,
                sortino_1y_start, sortino_1y_end, sortino_1y_delta,
                sortino_6m_start, sortino_6m_end, sortino_6m_delta,
                sortino_3m_start, sortino_3m_end, sortino_3m_delta,
                sortino_1m_start, sortino_1m_end, sortino_1m_delta,
                calmar_1y_start, calmar_1y_end, calmar_1y_delta,
                max_dd_1y_start, max_dd_1y_end, max_dd_1y_delta,
                portfolio_risk_quality_start, portfolio_risk_quality_end,
                goal_progress_pct_start, goal_progress_pct_end, goal_progress_pct_delta,
                goal_monthly_start, goal_monthly_end, goal_monthly_delta,
                goal_months_to_goal_start, goal_months_to_goal_end, goal_months_to_goal_delta,
                goal_net_progress_pct_start, goal_net_progress_pct_end, goal_net_progress_pct_delta,
                goal_net_monthly_start, goal_net_monthly_end, goal_net_monthly_delta,
                goal_pace_category_start, goal_pace_category_end,
                goal_pace_months_start, goal_pace_months_end, goal_pace_months_delta,
                goal_pace_tier_pace_pct_start, goal_pace_tier_pace_pct_end, goal_pace_tier_pace_pct_delta,
                holding_count_start, holding_count_end, holding_count_delta,
                concentration_top3_start, concentration_top3_end, concentration_top3_delta,
                concentration_top5_start, concentration_top5_end, concentration_top5_delta,
                concentration_herfindahl_start, concentration_herfindahl_end, concentration_herfindahl_delta,
                macro_10y_start, macro_10y_end, macro_10y_avg, macro_10y_delta,
                macro_2y_start, macro_2y_end, macro_2y_avg, macro_2y_delta,
                macro_vix_start, macro_vix_end, macro_vix_avg, macro_vix_delta,
                macro_cpi_start, macro_cpi_end, macro_cpi_avg, macro_cpi_delta,
                benchmark_symbol, benchmark_period_return_pct,
                benchmark_twr_1y_start, benchmark_twr_1y_end, benchmark_twr_1y_delta,
                built_from_run_id, created_at_utc
            ) VALUES (
                ?,?,?,?,?,?,?,?,?,?,
                ?,?,?,?,?,?,?,?,?,?,
                ?,?,?,?,?,?,?,?,?,?,
                ?,?,?,?,?,?,?,?,?,?,
                ?,?,?,?,?,?,?,?,?,?,
                ?,?,?,?,?,?,?,?,?,?,
                ?,?,?,?,?,?,?,?,?,?,
                ?,?,?,?,?,?,?,?,?,?,
                ?,?,?,?,?,?,?,?,?,?,
                ?,?,?,?,?,?,?,?,?,?,
                ?,?,?,?,?,?,?,?,?,?,
                ?,?,?,?,?,?,?,?,?,?,
                ?,?,?,?,?,?,?,?,?,?,
                ?,?,?,?,?,?,?,?,?,?,
                ?,?,?,?,?,?,?,?,?,?,
                ?,?,?,?,?,?,?,?,?,?,
                ?,?,?,?,?,?,?,?,?,?,
                ?,?,?,?,?,?,?,?,?,?,
                ?,?,?,?,?,?,?,?,?,?,
                ?,?,?,?,?,?,?,?,?,?,
                ?,?,?,?,?,?,?,?,?,?
            )
            """,
            values_tuple,
    )

    # Persist risk period averages that are stored in period_summary but not part of
    # the flat insert column list to keep backward-compatible SQL manageable.
    cur.execute(
        """UPDATE period_summary
           SET vol_30d_avg=?, vol_90d_avg=?, sharpe_1y_avg=?, sortino_1y_avg=?, calmar_1y_avg=?
           WHERE period_type=? AND period_start_date=? AND period_end_date=? AND is_rolling=?""",
        (
            (risk_stats_dict.get("vol_30d_pct") or {}).get("avg"),
            (risk_stats_dict.get("vol_90d_pct") or {}).get("avg"),
            (risk_stats_dict.get("sharpe_1y") or {}).get("avg"),
            (risk_stats_dict.get("sortino_1y") or {}).get("avg"),
            (risk_stats_dict.get("calmar_1y") or {}).get("avg"),
            period_type,
            period_start,
            period_end,
            is_rolling,
        ),
    )

    # Child: period_risk_stats (min/avg/max)
    risk_stats = risk_stats_dict
    for metric, vals in risk_stats.items():
        if isinstance(vals, dict):
            cur.execute(
                """INSERT INTO period_risk_stats (period_type, period_start_date, period_end_date, metric, avg_val, min_val, max_val)
                   VALUES (?,?,?,?,?,?,?)""",
                (period_type, period_start, period_end, metric, vals.get("avg"), vals.get("min"), vals.get("max")),
            )

    # Child: period_macro_stats (avg/min/max/std with dates)
    for metric, vals in macro_period_stats.items():
        if isinstance(vals, dict):
            cur.execute(
                """INSERT INTO period_macro_stats (period_type, period_start_date, period_end_date, metric, avg_val, min_val, max_val, std_val, min_date, max_date)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (period_type, period_start, period_end, metric,
                 vals.get("avg"), vals.get("min"), vals.get("max"), vals.get("std"),
                 vals.get("min_date"), vals.get("max_date")),
            )

    # Child: period_intervals — intervals are flat dicts with interval_label, start_date,
    # end_date, totals, performance, risk, income at top level (not nested period_summary)
    intervals = snapshot.get("intervals") or []
    iv_margin_stress = None
    iv_gp = None
    iv_gpn = None
    for iv in intervals:
        if not isinstance(iv, dict):
            continue
        label = iv.get("interval_label") or iv.get("label") or ""
        if not label:
            continue
        iv_totals = iv.get("totals") or {}
        iv_perf = iv.get("performance") or {}
        iv_risk = iv.get("risk") or {}
        iv_income = iv.get("income") or {}
        iv_margin = iv.get("margin") or {}
        iv_margin_stress = iv.get("margin_stress") or {}
        iv_gp = iv.get("goal_progress") or {}
        iv_gpn = iv.get("goal_progress_net") or {}
        mc_dist = (iv_margin_stress.get("stress_scenarios") or {}).get("margin_call_distance") or {}
        margin_current = iv_margin_stress.get("current") or {}
        cur.execute(
            """INSERT OR REPLACE INTO period_intervals (
                period_type, period_start_date, period_end_date, interval_label,
                interval_start, interval_end,
                mv, cost_basis, nlv, unrealized_pct, unrealized_pnl,
                pnl_period, pnl_pct_period,
                twr_1m_pct, twr_3m_pct, twr_12m_pct,
                sharpe_1y, sortino_1y, sortino_6m, sortino_3m, sortino_1m,
                portfolio_risk_quality, monthly_income, margin_loan, ltv_pct,
                forward_12m_total, yield_pct, yield_on_cost_pct,
                twr_6m_pct,
                vol_30d_pct, vol_90d_pct, calmar_1y, max_drawdown_1y_pct,
                var_90_1d_pct, var_95_1d_pct, cvar_90_1d_pct,
                omega_ratio_1y, ulcer_index_1y, income_stability_score, beta_portfolio,
                annual_interest_expense, margin_call_buffer_pct,
                goal_progress_pct, goal_months_to_goal, goal_projected_monthly, goal_net_progress_pct
            ) VALUES (?,?,?,?, ?,?, ?,?,?,?,?, ?,?, ?,?,?, ?,?,?,?,?, ?,?,?,?
                ,?,?,?, ?,?,?,?, ?,?,?, ?,?,?,?, ?,?, ?,?,?,?,?)""",
            (
                period_type, period_start, period_end, label,
                iv.get("start_date") or "",
                iv.get("end_date") or "",
                iv_totals.get("total_market_value") or iv_totals.get("market_value"),
                iv_totals.get("cost_basis"),
                iv_totals.get("net_liquidation_value"),
                iv_totals.get("unrealized_pct"),
                iv_totals.get("unrealized_pnl"),
                iv_perf.get("pnl_dollar_period"),
                iv_perf.get("pnl_pct_period"),
                iv_perf.get("twr_1m_pct"),
                iv_perf.get("twr_3m_pct"),
                iv_perf.get("twr_12m_pct"),
                iv_risk.get("sharpe_1y"),
                iv_risk.get("sortino_1y"),
                iv_risk.get("sortino_6m"),
                iv_risk.get("sortino_3m"),
                iv_risk.get("sortino_1m"),
                iv_risk.get("portfolio_risk_quality"),
                iv_income.get("projected_monthly_income"),
                iv_margin.get("margin_loan_balance") or iv_totals.get("margin_loan_balance"),
                iv_margin.get("margin_to_portfolio_pct") or iv_margin.get("ltv_pct") or iv_totals.get("margin_to_portfolio_pct"),
                iv_income.get("forward_12m_total"),
                iv_income.get("portfolio_current_yield_pct") or iv_income.get("portfolio_yield_pct"),
                iv_income.get("portfolio_yield_on_cost_pct"),
                iv_perf.get("twr_6m_pct"),
                iv_risk.get("vol_30d_pct"),
                iv_risk.get("vol_90d_pct"),
                iv_risk.get("calmar_1y"),
                iv_risk.get("max_drawdown_1y_pct"),
                iv_risk.get("var_90_1d_pct"),
                iv_risk.get("var_95_1d_pct"),
                iv_risk.get("cvar_90_1d_pct"),
                iv_risk.get("omega_ratio_1y"),
                iv_risk.get("ulcer_index_1y"),
                iv_risk.get("income_stability_score"),
                iv_risk.get("beta_portfolio"),
                margin_current.get("annual_interest") or margin_current.get("annual_interest_expense"),
                mc_dist.get("buffer_to_margin_call_pct") or mc_dist.get("portfolio_decline_to_call_pct"),
                iv_gp.get("progress_pct"),
                iv_gp.get("months_to_goal"),
                iv_gp.get("current_projected_monthly"),
                iv_gpn.get("progress_pct"),
            ),
        )
        # period_interval_holdings
        for h in iv.get("holdings") or []:
            sym = h.get("symbol")
            if not sym:
                continue
            cur.execute(
                """INSERT INTO period_interval_holdings (
                    period_type, period_start_date, period_end_date, interval_label, symbol,
                    weight_pct, market_value, pnl_pct, pnl_dollar,
                    projected_monthly_dividend, current_yield_pct,
                    sharpe_1y, sortino_1y, risk_quality_category
                ) VALUES (?,?,?,?,?, ?,?,?,?, ?,?, ?,?,?)""",
                (
                    period_type, period_start, period_end, label, sym,
                    h.get("weight_pct"),
                    h.get("market_value"),
                    h.get("pnl_pct"),
                    h.get("pnl_dollar"),
                    h.get("projected_monthly_dividend"),
                    h.get("current_yield_pct"),
                    h.get("sharpe_1y"),
                    h.get("sortino_1y"),
                    h.get("risk_quality_category"),
                ),
            )
        # period_interval_attribution (per window)
        for window_key in ("1m", "3m", "6m", "12m"):
            attr = iv.get(f"return_attribution_{window_key}") or {}
            if not isinstance(attr, dict):
                continue
            top_list = attr.get("top_contributors") or []
            bottom_list = attr.get("bottom_contributors") or []
            top_json = json.dumps([{"symbol": c.get("symbol"), "contribution_pct": c.get("contribution_pct")} for c in top_list[:3]]) if top_list else None
            bottom_json = json.dumps([{"symbol": c.get("symbol"), "contribution_pct": c.get("contribution_pct")} for c in bottom_list[:3]]) if bottom_list else None
            cur.execute(
                """INSERT INTO period_interval_attribution (
                    period_type, period_start_date, period_end_date, interval_label, window,
                    total_return_pct, income_contribution_pct, price_contribution_pct,
                    top_json, bottom_json
                ) VALUES (?,?,?,?,?, ?,?,?, ?,?)""",
                (
                    period_type, period_start, period_end, label, window_key,
                    attr.get("total_return_pct"),
                    attr.get("income_contribution_pct"),
                    attr.get("price_contribution_pct"),
                    top_json,
                    bottom_json,
                ),
            )

    # Child: period_holding_changes — added, removed, top_gainers, top_losers, weight_increases, weight_decreases
    changes = snapshot.get("portfolio_changes") or {}
    _change_map = {
        "added": changes.get("holdings_added") or changes.get("added") or [],
        "removed": changes.get("holdings_removed") or changes.get("removed") or [],
        "top_gainer": changes.get("top_gainers") or [],
        "top_loser": changes.get("top_losers") or [],
        "weight_increase": changes.get("weight_increases") or [],
        "weight_decrease": changes.get("weight_decreases") or [],
    }
    for change_type, items in _change_map.items():
        for h in items:
            sym = h.get("symbol")
            if not sym:
                continue
            cur.execute(
                """INSERT OR REPLACE INTO period_holding_changes (
                    period_type, period_start_date, period_end_date, symbol, change_type,
                    weight_start_pct, weight_end_pct, weight_delta_pct, avg_weight_pct,
                    pnl_pct_period, pnl_dollar_period, contribution_to_portfolio_pct,
                    start_twr_12m_pct, end_twr_12m_pct,
                    start_shares, end_shares, shares_delta, shares_delta_pct,
                    start_market_value, end_market_value, market_value_delta, market_value_delta_pct,
                    dividends_received, dividend_events_count,
                    start_yield_pct, end_yield_pct,
                    start_projected_monthly, end_projected_monthly,
                    avg_vol_30d_pct, period_max_drawdown_pct,
                    worst_day_pct, worst_day_date, best_day_pct, best_day_date
                ) VALUES (?,?,?,?,?, ?,?,?,?, ?,?,?, ?,?, ?,?,?,?, ?,?,?,?, ?,?, ?,?, ?,?, ?,?, ?,?,?,?)""",
                (
                    period_type, period_start, period_end, sym, change_type,
                    h.get("weight_start_pct") or h.get("weight_pct_start"),
                    h.get("weight_end_pct") or h.get("weight_pct_end"),
                    h.get("weight_delta_pct") or h.get("weight_pct_delta"),
                    h.get("avg_weight_pct"),
                    h.get("pnl_pct_period") or h.get("pnl_pct"),
                    h.get("pnl_dollar_period") or h.get("pnl_dollar"),
                    h.get("contribution_to_portfolio_pct"),
                    h.get("start_twr_12m_pct"),
                    h.get("end_twr_12m_pct"),
                    h.get("start_shares"),
                    h.get("end_shares"),
                    h.get("shares_delta"),
                    h.get("shares_delta_pct"),
                    h.get("start_market_value"),
                    h.get("end_market_value"),
                    h.get("market_value_delta"),
                    h.get("market_value_delta_pct"),
                    h.get("dividends_received"),
                    h.get("dividend_events_count"),
                    h.get("start_yield_pct"),
                    h.get("end_yield_pct"),
                    h.get("start_projected_monthly"),
                    h.get("end_projected_monthly"),
                    h.get("avg_vol_30d_pct"),
                    h.get("period_max_drawdown_pct"),
                    h.get("worst_day_pct"),
                    h.get("worst_day_date"),
                    h.get("best_day_pct"),
                    h.get("best_day_date"),
                ),
            )

    # Child: Activity tables
    activity = ps.get("activity") or {}

    # Delete existing activity data for this period first
    for tbl in ("period_activity", "period_contributions", "period_withdrawals",
                "period_dividend_events", "period_trades", "period_margin_detail", "period_position_lists"):
        cur.execute(
            f"DELETE FROM {tbl} WHERE period_type=? AND period_start_date=? AND period_end_date=?",
            (period_type, period_start, period_end),
        )

    # period_activity - Main activity summary
    contributions = activity.get("contributions") or {}
    withdrawals = activity.get("withdrawals") or {}
    dividends = activity.get("dividends") or {}
    interest = activity.get("interest") or {}
    trades = activity.get("trades") or {}
    margin = activity.get("margin") or {}
    positions = activity.get("positions") or {}

    cur.execute(
        """INSERT INTO period_activity (
            period_type, period_start_date, period_end_date,
            contributions_total, contributions_count,
            withdrawals_total, withdrawals_count,
            dividends_total_received, dividends_count,
            interest_total_paid, interest_avg_daily_balance, interest_avg_rate_pct, interest_annualized,
            trades_total_count, trades_buy_count, trades_sell_count,
            margin_borrowed, margin_repaid, margin_net_change,
            positions_added_count, positions_removed_count, positions_increased_count, positions_decreased_count
        ) VALUES (?,?,?, ?,?, ?,?, ?,?, ?,?,?,?, ?,?,?, ?,?,?, ?,?,?,?)""",
        (
            period_type, period_start, period_end,
            contributions.get("total"), contributions.get("count"),
            withdrawals.get("total"), withdrawals.get("count"),
            dividends.get("total_received"), dividends.get("count"),
            interest.get("total_paid"), interest.get("avg_daily_balance"), interest.get("avg_rate_pct"), interest.get("annualized"),
            trades.get("total_count"), trades.get("buy_count"), trades.get("sell_count"),
            margin.get("borrowed"), margin.get("repaid"), margin.get("net_change"),
            len(positions.get("added") or []),
            len(positions.get("removed") or []),
            len(positions.get("symbols_increased") or []),
            len(positions.get("symbols_decreased") or []),
        ),
    )

    # period_position_lists - Store position symbols in normalized table
    for symbol in positions.get("added") or []:
        cur.execute(
            """INSERT INTO period_position_lists (
                period_type, period_start_date, period_end_date, list_type, symbol
            ) VALUES (?,?,?,?,?)""",
            (period_type, period_start, period_end, "added", symbol),
        )
    for symbol in positions.get("removed") or []:
        cur.execute(
            """INSERT INTO period_position_lists (
                period_type, period_start_date, period_end_date, list_type, symbol
            ) VALUES (?,?,?,?,?)""",
            (period_type, period_start, period_end, "removed", symbol),
        )
    for symbol in positions.get("symbols_increased") or []:
        cur.execute(
            """INSERT INTO period_position_lists (
                period_type, period_start_date, period_end_date, list_type, symbol
            ) VALUES (?,?,?,?,?)""",
            (period_type, period_start, period_end, "increased", symbol),
        )
    for symbol in positions.get("symbols_decreased") or []:
        cur.execute(
            """INSERT INTO period_position_lists (
                period_type, period_start_date, period_end_date, list_type, symbol
            ) VALUES (?,?,?,?,?)""",
            (period_type, period_start, period_end, "decreased", symbol),
        )

    # period_contributions - Aggregate by date and account
    contrib_agg = {}
    for contrib in contributions.get("details") or []:
        key = (contrib.get("date"), contrib.get("account_id"))
        if key not in contrib_agg:
            contrib_agg[key] = 0
        contrib_agg[key] += contrib.get("amount") or 0

    for (date, account_id), total_amount in contrib_agg.items():
        cur.execute(
            """INSERT INTO period_contributions (
                period_type, period_start_date, period_end_date,
                contribution_date, amount, account_id
            ) VALUES (?,?,?,?,?,?)""",
            (period_type, period_start, period_end, date, total_amount, account_id),
        )

    # period_withdrawals - Aggregate by date and account
    withdrawal_agg = {}
    for withdrawal in withdrawals.get("details") or []:
        key = (withdrawal.get("date"), withdrawal.get("account_id"))
        if key not in withdrawal_agg:
            withdrawal_agg[key] = 0
        withdrawal_agg[key] += withdrawal.get("amount") or 0

    for (date, account_id), total_amount in withdrawal_agg.items():
        cur.execute(
            """INSERT INTO period_withdrawals (
                period_type, period_start_date, period_end_date,
                withdrawal_date, amount, account_id
            ) VALUES (?,?,?,?,?,?)""",
            (period_type, period_start, period_end, date, total_amount, account_id),
        )

    # period_dividend_events - Dividend events
    for div_event in dividends.get("events") or []:
        if not div_event.get("symbol"):
            continue
        cur.execute(
            """INSERT INTO period_dividend_events (
                period_type, period_start_date, period_end_date,
                symbol, ex_date, pay_date, amount
            ) VALUES (?,?,?, ?,?,?,?)""",
            (
                period_type, period_start, period_end,
                div_event.get("symbol"),
                div_event.get("ex_date"),
                div_event.get("pay_date"),
                div_event.get("amount"),
            ),
        )

    # period_trades - Trades by symbol
    for symbol, trade_counts in (trades.get("by_symbol") or {}).items():
        if not symbol:
            continue
        cur.execute(
            """INSERT INTO period_trades (
                period_type, period_start_date, period_end_date,
                symbol, buy_count, sell_count
            ) VALUES (?,?,?, ?,?,?)""",
            (
                period_type, period_start, period_end,
                symbol,
                trade_counts.get("buy_count") or 0,
                trade_counts.get("sell_count") or 0,
            ),
        )

    # period_margin_detail - Margin activity
    if margin.get("borrowed") is not None or margin.get("repaid") is not None:
        cur.execute(
            """INSERT INTO period_margin_detail (
                period_type, period_start_date, period_end_date,
                borrowed, repaid, net_change
            ) VALUES (?,?,?, ?,?,?)""",
            (
                period_type, period_start, period_end,
                margin.get("borrowed"),
                margin.get("repaid"),
                margin.get("net_change"),
            ),
        )
