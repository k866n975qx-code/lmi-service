"""AI-powered portfolio insights using Anthropic Claude API."""
from __future__ import annotations

import structlog

from ..pipeline.diff_daily import (
    _totals,
    _income,
    _goal_tiers,
    _goal_pace,
    _rollups,
    _margin_stress,
    _holdings_flat,
)
from ..alerts.evaluator import _holding_ultimate

log = structlog.get_logger()


def _as_of(snap: dict) -> str:
    """As-of date from V5 or legacy snapshot."""
    if not snap:
        return "unknown"
    ts = snap.get("timestamps") or {}
    return (
        snap.get("as_of_date_local")
        or snap.get("as_of")
        or ts.get("portfolio_data_as_of_local")
        or "unknown"
    )

SYSTEM_PROMPT = """\
You are a dividend portfolio analyst assistant. The user tracks a dividend-income \
portfolio with a monthly income goal. You receive a structured snapshot of their \
portfolio and produce concise, actionable insights.

Rules:
- Be direct. No fluff, no disclaimers, no "I'm an AI" caveats.
- Focus on what matters: pace toward goal, income sustainability, risk flags, \
  and 1-2 concrete action items.
- Use plain language a retail investor understands.
- Keep total response under 1500 characters (Telegram limit).
- Format output as Telegram HTML: use <b>bold</b> for headers, plain text for body.
- Do NOT use markdown. Only Telegram-safe HTML tags: <b>, <i>, <u>, <code>, <a>.
- Use line breaks (newlines) to separate sections, not <br> tags.
"""

USER_PROMPT_TEMPLATE = """\
Here is my portfolio snapshot as of {as_of}:

PORTFOLIO TOTALS
- Net Liquidation Value: ${nlv:,.2f}
- Market Value: ${market_value:,.2f}
- Cost Basis: ${cost_basis:,.2f}
- Unrealized P&L: ${unrealized_pnl:,.2f} ({unrealized_pct:+.1f}%)
- Margin Loan: ${margin_loan:,.2f} (LTV: {ltv_pct:.1f}%)

INCOME
- Projected Monthly: ${monthly_income:,.2f}
- Forward 12M Total: ${forward_12m:,.2f}
- Current Yield: {current_yield:.2f}%
- Yield on Cost: {yoc:.2f}%

GOAL PROGRESS
- Target Monthly: ${target_monthly:,.2f}
- Progress: {goal_pct:.1f}%
- Months to Goal: {months_to_goal}
- Strategy: {likely_tier}
- Pace: {pace_category} ({pace_pct:.0f}% of tier pace, {months_ahead_behind:+.1f}m)

RISK
- 30d Volatility: {vol_30d:.1f}%
- Sharpe (1Y): {sharpe}
- Sortino (1Y): {sortino}
- Max Drawdown (1Y): {max_dd:.1f}%
- Risk Quality: {risk_quality}

MARGIN STRESS
- Decline to Margin Call: {decline_to_call:.0f}%
- Income Coverage Ratio: {income_coverage:.1f}x

MACRO
- VIX: {vix}
- 10Y Yield: {ten_year}%
- Yield Spread (10Y-2Y): {spread}bps
- Macro Stress Score: {macro_stress}

TOP HOLDINGS (by weight)
{top_holdings}

Give me a brief portfolio insight covering:
1. How am I tracking toward my income goal?
2. Any risk or income sustainability concerns?
3. One or two actionable suggestions.
"""


def _safe_get(d: dict, *keys, default=0):
    """Safely traverse nested dict keys."""
    val = d
    for k in keys:
        if not isinstance(val, dict):
            return default
        val = val.get(k, default)
    return val if val is not None else default


def _extract_context(snap: dict) -> dict:
    """Extract key metrics from snapshot for the AI prompt."""
    totals = _totals(snap)
    income = _income(snap)
    goal_tiers_data = _goal_tiers(snap)
    goal_cs = goal_tiers_data.get("current_state") or {}
    pace = _goal_pace(snap)
    rollups = _rollups(snap)
    risk = rollups.get("risk") or {}
    macro_data = snap.get("macro") or {}
    macro_snap = macro_data.get("snapshot") or {}
    stress = _margin_stress(snap)
    stress_scenarios = stress.get("stress_scenarios") or {}

    current_pace = pace.get("current_pace") or {}
    likely_tier = pace.get("likely_tier") or {}
    tiers = goal_tiers_data.get("tiers") or []
    # Use likely tier's months_to_goal if available
    likely_tier_data = next((t for t in tiers if t.get("tier") == likely_tier.get("tier")), {})
    months_to_goal = likely_tier_data.get("months_to_goal")

    # Top holdings by weight (use flattened holding for weight/yield/dividend)
    holdings = _holdings_flat(snap)
    sorted_h = sorted(
        holdings,
        key=lambda h: (_holding_ultimate(h).get("weight_pct") or 0),
        reverse=True,
    )
    top_lines = []
    for h in sorted_h[:10]:
        u = _holding_ultimate(h)
        sym = h.get("symbol", "?")
        w = u.get("weight_pct", 0) or 0
        yld = u.get("current_yield_pct", 0) or 0
        mo_div = u.get("projected_monthly_dividend", 0) or 0
        top_lines.append(f"  {sym}: {w:.1f}% weight, {yld:.1f}% yield, ${mo_div:.2f}/mo")

    return {
        "as_of": _as_of(snap),
        "nlv": _safe_get(totals, "net_liquidation_value"),
        "market_value": _safe_get(totals, "market_value"),
        "cost_basis": _safe_get(totals, "cost_basis"),
        "unrealized_pnl": _safe_get(totals, "unrealized_pnl"),
        "unrealized_pct": _safe_get(totals, "unrealized_pct"),
        "margin_loan": _safe_get(totals, "margin_loan_balance"),
        "ltv_pct": _safe_get(totals, "margin_to_portfolio_pct"),
        "monthly_income": _safe_get(income, "projected_monthly_income"),
        "forward_12m": _safe_get(income, "forward_12m_total"),
        "current_yield": _safe_get(income, "portfolio_current_yield_pct"),
        "yoc": _safe_get(income, "portfolio_yield_on_cost_pct"),
        "target_monthly": _safe_get(goal_cs, "target_monthly"),
        "goal_pct": round(goal_cs.get("projected_monthly_income", 0) / goal_cs.get("target_monthly", 1) * 100, 1) if goal_cs.get("target_monthly") else 0,
        "months_to_goal": str(months_to_goal) if months_to_goal is not None else "N/A",
        "likely_tier": f"Tier {likely_tier.get('tier', '—')} - {likely_tier.get('name', '—')} ({likely_tier.get('confidence', '—')})" if likely_tier.get("tier") else "N/A",
        "pace_category": current_pace.get("pace_category", "unknown"),
        "pace_pct": _safe_get(current_pace, "pct_of_tier_pace", default=0),
        "months_ahead_behind": _safe_get(current_pace, "months_ahead_behind", default=0),
        "vol_30d": _safe_get(risk, "vol_30d_pct"),
        "sharpe": f"{_safe_get(risk, 'sharpe_1y'):.2f}" if isinstance(_safe_get(risk, "sharpe_1y"), (int, float)) else "N/A",
        "sortino": f"{_safe_get(risk, 'sortino_1y'):.2f}" if isinstance(_safe_get(risk, "sortino_1y"), (int, float)) else "N/A",
        "max_dd": _safe_get(risk, "max_drawdown_1y_pct"),
        "risk_quality": risk.get("portfolio_risk_quality", "N/A"),
        "decline_to_call": _safe_get(stress_scenarios, "margin_call_distance", "portfolio_decline_to_call_pct", default=0),
        "income_coverage": _safe_get(stress_scenarios, "interest_rate_shock", "+50bps", "income_coverage_ratio", default=0),
        "vix": f"{macro_snap.get('vix', 'N/A')}",
        "ten_year": f"{macro_snap.get('ten_year_yield', 'N/A')}",
        "spread": f"{macro_snap.get('yield_spread_10y_2y', 'N/A')}",
        "macro_stress": f"{macro_snap.get('macro_stress_score', 'N/A')}",
        "top_holdings": "\n".join(top_lines) if top_lines else "  No holdings data",
    }


def generate_insight(snap: dict, api_key: str, model: str = "claude-sonnet-4-20250514") -> str | None:
    """Send snapshot context to Claude and return HTML insight text.

    Args:
        snap: Full daily snapshot dict.
        api_key: Anthropic API key.
        model: Model to use. Defaults to Sonnet 4 for cost/speed.
               Use "claude-opus-4-20250514" for deeper analysis.

    Returns:
        HTML-formatted insight string, or None on error.
    """
    try:
        import anthropic
    except ImportError:
        log.warning("ai_insights_missing_sdk", msg="anthropic package not installed")
        return None

    context = _extract_context(snap)
    user_prompt = USER_PROMPT_TEMPLATE.format(**context)

    try:
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model=model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return message.content[0].text
    except anthropic.BadRequestError as e:
        log.error("ai_insights_bad_request", error=str(e))
        return None
    except anthropic.AuthenticationError as e:
        log.error("ai_insights_auth_error", error=str(e))
        return None
    except Exception:
        log.exception("ai_insights_error")
        return None


# ===== PERIOD INSIGHTS =====

PERIOD_SYSTEM_PROMPT = """\
You are a dividend portfolio analyst reviewing period performance. The user tracks \
a dividend-income portfolio with a monthly income goal. You receive a structured \
period summary showing performance, activity, and risk metrics over the period.

Rules:
- Be direct. No fluff, no disclaimers, no "I'm an AI" caveats.
- Focus on: period performance vs goals, activity summary, risk events, and 1-2 \
  actionable suggestions for next period.
- Use plain language a retail investor understands.
- Keep total response under 1500 characters (Telegram limit).
- Format output as Telegram HTML: use <b>bold</b> for headers, plain text for body.
- Do NOT use markdown. Only Telegram-safe HTML tags: <b>, <i>, <u>, <code>, <a>.
- Use line breaks (newlines) to separate sections, not <br> tags.
"""

PERIOD_USER_PROMPT_TEMPLATE = """\
Here is my portfolio {period_type} period summary ({period_start} to {period_end}):

PERIOD PERFORMANCE
- Portfolio Return: {period_return_pct:+.2f}%
- Starting Value: ${mv_start:,.2f}
- Ending Value: ${mv_end:,.2f}
- Value Change: ${mv_delta:+,.2f}
- Income Generated: ${monthly_income_start:.2f}/mo → ${monthly_income_end:.2f}/mo ({monthly_income_delta:+.2f})

ACTIVITY THIS PERIOD
- Contributions: ${contributions_total:,.2f} ({contributions_count} deposits)
- Withdrawals: ${withdrawals_total:,.2f} ({withdrawals_count} withdrawals)
- Dividends Received: ${dividends_total:,.2f} ({dividends_count} events)
- Trades: {trades_total} ({trades_buy} buys, {trades_sell} sells)
- Positions Added: {positions_added}
- Positions Removed: {positions_removed}

GOAL PROGRESS
- Monthly Income: ${monthly_income_end:,.2f} (target: ${target_monthly:,.2f})
- Progress: {goal_progress_pct:.1f}%
- Months to Goal: {months_to_goal_start} → {months_to_goal_end}

RISK EVENTS
- Max Drawdown: {period_max_drawdown_pct:.2f}% on {period_max_drawdown_date}
- Days Exceeding VaR 95: {days_exceeding_var_95}
- Worst Day: {worst_day_return_pct:.2f}% on {worst_day_date}
- Best Day: {best_day_return_pct:+.2f}% on {best_day_date}
- Volatility: {vol_30d_start:.1f}% → {vol_30d_end:.1f}%

PERIOD STATS
- Avg Portfolio Value: ${mv_avg:,.2f}
- Avg Monthly Income: ${projected_monthly_avg:,.2f}
- Avg Yield: {yield_pct_avg:.2f}%

Give me a brief period review covering:
1. How did I perform vs my goals this period?
2. Any notable activity, risks, or concerns?
3. One or two suggestions for the next period.
"""


def _extract_period_context(period_snap: dict) -> dict:
    """Extract key metrics from period snapshot for AI prompt (target schema first)."""
    def _num(value):
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            return float(value)
        return None

    def _first_num(*values, default=0.0):
        for value in values:
            num = _num(value)
            if num is not None:
                return num
        return default

    def _first(*values, default=None):
        for value in values:
            if value is not None:
                return value
        return default

    period = period_snap.get("period") or {}
    meta = period_snap.get("meta") or {}
    meta_period = meta.get("period") or {}
    timestamps = period_snap.get("timestamps") or {}

    # Target V5 blocks
    portfolio = period_snap.get("portfolio") or {}
    values = portfolio.get("values") or {}
    income_v5 = portfolio.get("income") or {}
    performance_v5 = portfolio.get("performance") or {}
    risk_v5 = portfolio.get("risk") or {}
    drawdown_v5 = risk_v5.get("drawdown") or {}
    var_v5 = risk_v5.get("var") or {}
    vol_v5 = risk_v5.get("volatility") or {}
    activity_v5 = period_snap.get("activity") or {}
    goals_v5 = period_snap.get("goals") or {}

    # Legacy fallback blocks
    ps = period_snap.get("period_summary") or {}
    totals_legacy = ps.get("totals") or {}
    t_s = totals_legacy.get("start") or {}
    t_e = totals_legacy.get("end") or {}
    t_d = totals_legacy.get("delta") or {}
    perf_legacy = ps.get("performance") or {}
    perf_period_legacy = perf_legacy.get("period") or {}
    income_legacy = ps.get("income") or {}
    i_s = income_legacy.get("start") or {}
    i_e = income_legacy.get("end") or {}
    i_d = income_legacy.get("delta") or {}
    activity_legacy = ps.get("activity") or {}
    goal_prog_legacy = ps.get("goal_progress") or {}
    gp_s = goal_prog_legacy.get("start") or {}
    gp_e = goal_prog_legacy.get("end") or {}
    risk_legacy = ps.get("risk") or {}
    r_s = risk_legacy.get("start") or {}
    r_e = risk_legacy.get("end") or {}
    period_dd_legacy = ps.get("period_drawdown") or {}
    var_breach_legacy = ps.get("var_breach") or {}
    period_stats_legacy = ps.get("period_stats") or {}
    mv_stats_legacy = period_stats_legacy.get("market_value") or {}
    income_stats_legacy = period_stats_legacy.get("projected_monthly_income") or {}
    yield_stats_legacy = period_stats_legacy.get("yield_pct") or {}

    activity = activity_v5 or activity_legacy
    contributions = activity.get("contributions") or {}
    withdrawals = activity.get("withdrawals") or {}
    dividends = activity.get("dividends") or {}
    trades = activity.get("trades") or {}
    positions = activity.get("positions") or {}

    period_type = _first(
        meta_period.get("type"),
        period.get("label"),
        period.get("type"),
        "period",
    )
    period_start = _first(
        meta_period.get("start_date_local"),
        timestamps.get("period_start_local"),
        period.get("start_date"),
        "",
    )
    period_end = _first(
        meta_period.get("end_date_local"),
        timestamps.get("period_end_local"),
        period.get("end_date"),
        "",
    )

    mv_start = _first_num(
        _safe_get(values, "start", "market_value", default=None),
        t_s.get("total_market_value"),
        default=0.0,
    )
    mv_end = _first_num(
        _safe_get(values, "end", "market_value", default=None),
        t_e.get("total_market_value"),
        default=0.0,
    )
    mv_delta = _first_num(
        _safe_get(values, "delta", "market_value", default=None),
        t_d.get("total_market_value"),
        default=0.0,
    )
    mv_avg = _first_num(
        _safe_get(values, "period_stats", "market_value", "avg", default=None),
        mv_stats_legacy.get("avg"),
        default=0.0,
    )

    monthly_income_start = _first_num(
        income_v5.get("start_projected_monthly"),
        i_s.get("projected_monthly_income"),
        default=0.0,
    )
    monthly_income_end = _first_num(
        income_v5.get("end_projected_monthly"),
        i_e.get("projected_monthly_income"),
        default=0.0,
    )
    monthly_income_delta = _first_num(
        income_v5.get("delta_projected_monthly"),
        i_d.get("projected_monthly_income"),
        default=0.0,
    )
    projected_monthly_avg = _first_num(
        _safe_get(income_v5, "period_stats", "projected_monthly", "avg", default=None),
        income_stats_legacy.get("avg"),
        default=0.0,
    )
    yield_pct_avg = _first_num(
        _safe_get(income_v5, "period_stats", "yield_pct", "avg", default=None),
        yield_stats_legacy.get("avg"),
        default=0.0,
    )

    goals_start = goals_v5.get("start") or {}
    goals_end = goals_v5.get("end") or {}
    target_monthly = _first_num(
        goals_end.get("target_monthly"),
        goals_start.get("target_monthly"),
        default=0.0,
    )
    goal_progress_pct = _first_num(
        goals_end.get("progress_pct"),
        gp_e.get("progress_pct"),
        default=0.0,
    )
    if target_monthly <= 0 and monthly_income_end > 0 and goal_progress_pct > 0:
        target_monthly = monthly_income_end / (goal_progress_pct / 100.0)
    if target_monthly <= 0:
        target_monthly = monthly_income_end

    months_to_goal_start = _first(
        goals_start.get("months_to_goal"),
        gp_s.get("months_to_goal"),
        "N/A",
    )
    months_to_goal_end = _first(
        goals_end.get("months_to_goal"),
        gp_e.get("months_to_goal"),
        "N/A",
    )

    return {
        "period_type": str(period_type).replace("_", " ").title(),
        "period_start": period_start,
        "period_end": period_end,
        "period_return_pct": _first_num(
            performance_v5.get("twr_period_pct"),
            performance_v5.get("period_return_pct"),
            perf_period_legacy.get("twr_period_pct"),
            default=0.0,
        ),
        "mv_start": mv_start,
        "mv_end": mv_end,
        "mv_delta": mv_delta,
        "mv_avg": mv_avg,
        "monthly_income_start": monthly_income_start,
        "monthly_income_end": monthly_income_end,
        "monthly_income_delta": monthly_income_delta,
        "projected_monthly_avg": projected_monthly_avg,
        "yield_pct_avg": yield_pct_avg,
        "contributions_total": _first_num(contributions.get("total"), default=0.0),
        "contributions_count": int(contributions.get("count") or 0),
        "withdrawals_total": _first_num(withdrawals.get("total"), default=0.0),
        "withdrawals_count": int(withdrawals.get("count") or 0),
        "dividends_total": abs(_first_num(dividends.get("total_received"), default=0.0)),
        "dividends_count": int(dividends.get("count") or 0),
        "trades_total": int(trades.get("total_count") or 0),
        "trades_buy": int(trades.get("buy_count") or 0),
        "trades_sell": int(trades.get("sell_count") or 0),
        "positions_added": len(positions.get("added") or []),
        "positions_removed": len(positions.get("removed") or []),
        "goal_progress_pct": goal_progress_pct,
        "months_to_goal_start": months_to_goal_start,
        "months_to_goal_end": months_to_goal_end,
        "target_monthly": target_monthly,
        "period_max_drawdown_pct": _first_num(
            drawdown_v5.get("period_max_drawdown_pct"),
            period_dd_legacy.get("period_max_drawdown_pct"),
            default=0.0,
        ),
        "period_max_drawdown_date": _first(
            drawdown_v5.get("period_max_drawdown_date"),
            period_dd_legacy.get("period_max_drawdown_date"),
            "N/A",
        ),
        "days_exceeding_var_95": int(_first_num(
            var_v5.get("days_exceeding_var_95"),
            var_breach_legacy.get("days_exceeding_var_95"),
            default=0.0,
        )),
        "worst_day_return_pct": _first_num(
            var_v5.get("worst_day_return_pct"),
            var_breach_legacy.get("worst_day_return_pct"),
            default=0.0,
        ),
        "worst_day_date": _first(
            var_v5.get("worst_day_date"),
            var_breach_legacy.get("worst_day_date"),
            "N/A",
        ),
        "best_day_return_pct": _first_num(
            var_v5.get("best_day_return_pct"),
            var_breach_legacy.get("best_day_return_pct"),
            default=0.0,
        ),
        "best_day_date": _first(
            var_v5.get("best_day_date"),
            var_breach_legacy.get("best_day_date"),
            "N/A",
        ),
        "vol_30d_start": _first_num(
            vol_v5.get("start_vol_30d_pct"),
            r_s.get("vol_30d_pct"),
            default=0.0,
        ),
        "vol_30d_end": _first_num(
            vol_v5.get("end_vol_30d_pct"),
            r_e.get("vol_30d_pct"),
            default=0.0,
        ),
    }


def generate_period_insight(
    period_snap: dict,
    api_key: str,
    model: str = "claude-sonnet-4-20250514"
) -> str | None:
    """Send period snapshot context to Claude and return HTML insight text.

    Args:
        period_snap: Full period snapshot dict (from build_period_snapshot).
        api_key: Anthropic API key.
        model: Model to use. Defaults to Sonnet 4 for cost/speed.
               Use "claude-opus-4-20250514" for deeper analysis.

    Returns:
        HTML-formatted insight string, or None on error.
    """
    try:
        import anthropic
    except ImportError:
        log.warning("ai_insights_missing_sdk", msg="anthropic package not installed")
        return None

    context = _extract_period_context(period_snap)
    user_prompt = PERIOD_USER_PROMPT_TEMPLATE.format(**context)

    try:
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model=model,
            max_tokens=1024,
            system=PERIOD_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return message.content[0].text
    except anthropic.BadRequestError as e:
        log.error("ai_insights_period_bad_request", error=str(e))
        return None
    except anthropic.AuthenticationError as e:
        log.error("ai_insights_period_auth_error", error=str(e))
        return None
    except Exception:
        log.exception("ai_insights_period_error")
        return None
