"""AI-powered portfolio insights using Anthropic Claude API."""
from __future__ import annotations

import structlog
from ..pipeline import snap_compat as sc

log = structlog.get_logger()

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
    totals = sc.get_totals(snap)
    income = sc.get_income(snap)
    goal_tiers = sc.get_goal_tiers(snap)
    goal_cs = goal_tiers.get("current_state") or {}
    pace = sc.get_goal_pace(snap)
    rollups = sc.get_rollups(snap)
    risk = rollups.get("risk") or {}
    macro_data = snap.get("macro") or {}
    macro_snap = macro_data.get("snapshot") or {}
    stress = sc.get_margin_stress(snap)
    stress_scenarios = stress.get("stress_scenarios") or {}

    current_pace = pace.get("current_pace") or {}
    likely_tier = pace.get("likely_tier") or {}
    tiers = goal_tiers.get("tiers") or []
    # Use likely tier's months_to_goal if available
    likely_tier_data = next((t for t in tiers if t.get("tier") == likely_tier.get("tier")), {})
    months_to_goal = likely_tier_data.get("months_to_goal")

    # Top holdings by weight
    holdings = sc.get_holdings_flat(snap)
    sorted_h = sorted(holdings, key=lambda h: h.get("weight_pct", 0), reverse=True)
    top_lines = []
    for h in sorted_h[:10]:
        sym = h.get("symbol", "?")
        w = h.get("weight_pct", 0)
        yld = h.get("current_yield_pct", 0)
        mo_div = h.get("projected_monthly_dividend", 0)
        top_lines.append(f"  {sym}: {w:.1f}% weight, {yld:.1f}% yield, ${mo_div:.2f}/mo")

    return {
        "as_of": sc.get_as_of(snap) or "unknown",
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
