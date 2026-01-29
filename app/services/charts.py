from __future__ import annotations

import io
import json
import sqlite3
from datetime import date, timedelta

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import numpy as np

import structlog

log = structlog.get_logger()


def _load_snapshots(conn: sqlite3.Connection, days: int = 90) -> list[tuple[date, dict]]:
    """Load recent daily snapshots ordered by date ascending."""
    cur = conn.cursor()
    rows = cur.execute(
        "SELECT as_of_date_local, payload_json FROM snapshot_daily_current "
        "WHERE as_of_date_local >= date(?, '-' || ? || ' days') ORDER BY as_of_date_local ASC",
        (date.today().isoformat(), str(days)),
    ).fetchall()
    out = []
    for row in rows:
        try:
            out.append((date.fromisoformat(row[0]), json.loads(row[1])))
        except (json.JSONDecodeError, ValueError):
            continue
    return out


def _fig_to_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor="#1e1e2e")
    buf.seek(0)
    plt.close(fig)
    return buf.read()


def _apply_dark_theme(ax):
    """Apply a dark theme to axes."""
    ax.set_facecolor("#1e1e2e")
    ax.tick_params(colors="#cdd6f4")
    ax.xaxis.label.set_color("#cdd6f4")
    ax.yaxis.label.set_color("#cdd6f4")
    ax.title.set_color("#cdd6f4")
    for spine in ax.spines.values():
        spine.set_color("#45475a")
    ax.grid(True, alpha=0.2, color="#45475a")


def generate_pace_chart(conn: sqlite3.Connection, days: int = 90) -> bytes | None:
    """Generate goal pace tracking chart: actual vs expected portfolio value."""
    snapshots = _load_snapshots(conn, days)
    if len(snapshots) < 2:
        return None

    dates = []
    actual_mv = []
    expected_mv = []

    for d, snap in snapshots:
        totals = snap.get("totals") or {}
        mv = totals.get("market_value")
        if not isinstance(mv, (int, float)):
            continue
        dates.append(d)
        actual_mv.append(mv)

        goal_pace = snap.get("goal_pace") or {}
        ytd = (goal_pace.get("windows") or {}).get("ytd") or {}
        exp = (ytd.get("expected") or {}).get("portfolio_value")
        expected_mv.append(exp if isinstance(exp, (int, float)) else mv)

    if len(dates) < 2:
        return None

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#1e1e2e")
    _apply_dark_theme(ax)

    ax.plot(dates, actual_mv, label="Actual", linewidth=2.2, color="#a6e3a1")
    ax.plot(dates, expected_mv, label="Expected (Tier Pace)", linewidth=2, linestyle="--", color="#89b4fa")

    # Fill ahead/behind
    ax.fill_between(
        dates, actual_mv, expected_mv,
        where=[a >= e for a, e in zip(actual_mv, expected_mv)],
        alpha=0.15, color="#a6e3a1", label="Ahead",
    )
    ax.fill_between(
        dates, actual_mv, expected_mv,
        where=[a < e for a, e in zip(actual_mv, expected_mv)],
        alpha=0.15, color="#f38ba8", label="Behind",
    )

    ax.set_title(f"Goal Pace Tracking — Last {days} Days", fontsize=14, fontweight="bold")
    ax.set_ylabel("Portfolio Value ($)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.legend(facecolor="#313244", edgecolor="#45475a", labelcolor="#cdd6f4")

    return _fig_to_bytes(fig)


def generate_income_chart(conn: sqlite3.Connection, days: int = 90) -> bytes | None:
    """Generate monthly dividend income over time."""
    snapshots = _load_snapshots(conn, days)
    if len(snapshots) < 2:
        return None

    dates = []
    monthly_income = []

    for d, snap in snapshots:
        inc = snap.get("income") or {}
        proj = inc.get("projected_monthly_income")
        if isinstance(proj, (int, float)):
            dates.append(d)
            monthly_income.append(proj)

    if len(dates) < 2:
        return None

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#1e1e2e")
    _apply_dark_theme(ax)

    ax.fill_between(dates, monthly_income, alpha=0.3, color="#f9e2af")
    ax.plot(dates, monthly_income, linewidth=2.2, color="#f9e2af", label="Projected Monthly")

    # Target line
    goal = (snapshots[-1][1].get("goal_progress") or {}).get("target_monthly")
    if isinstance(goal, (int, float)):
        ax.axhline(y=goal, color="#f38ba8", linestyle="--", linewidth=1.5, label=f"Target ${goal:,.0f}/mo")

    ax.set_title(f"Projected Monthly Income — Last {days} Days", fontsize=14, fontweight="bold")
    ax.set_ylabel("Monthly Income ($)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.legend(facecolor="#313244", edgecolor="#45475a", labelcolor="#cdd6f4")

    return _fig_to_bytes(fig)


def generate_performance_chart(conn: sqlite3.Connection, days: int = 90) -> bytes | None:
    """Generate portfolio value (NLV) over time."""
    snapshots = _load_snapshots(conn, days)
    if len(snapshots) < 2:
        return None

    dates = []
    nlv_values = []

    for d, snap in snapshots:
        totals = snap.get("totals") or {}
        nlv = totals.get("net_liquidation_value")
        if isinstance(nlv, (int, float)):
            dates.append(d)
            nlv_values.append(nlv)

    if len(dates) < 2:
        return None

    # Calculate color based on trend
    color = "#a6e3a1" if nlv_values[-1] >= nlv_values[0] else "#f38ba8"

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#1e1e2e")
    _apply_dark_theme(ax)

    ax.fill_between(dates, nlv_values, alpha=0.15, color=color)
    ax.plot(dates, nlv_values, linewidth=2.2, color=color)

    # Start/end markers
    ax.scatter([dates[0]], [nlv_values[0]], color="#89b4fa", zorder=5, s=40)
    ax.scatter([dates[-1]], [nlv_values[-1]], color=color, zorder=5, s=40)

    delta = nlv_values[-1] - nlv_values[0]
    delta_pct = (nlv_values[-1] / nlv_values[0] - 1.0) * 100.0 if nlv_values[0] else 0
    sign = "+" if delta >= 0 else ""
    ax.set_title(
        f"Net Liquidation Value — Last {days} Days ({sign}${delta:,.0f}, {sign}{delta_pct:.1f}%)",
        fontsize=14, fontweight="bold",
    )
    ax.set_ylabel("Net Value ($)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

    return _fig_to_bytes(fig)


def generate_attribution_chart(snap: dict) -> bytes | None:
    """Generate pie chart of income by top positions."""
    holdings = snap.get("holdings") or []
    income_by_sym = {}
    for h in holdings:
        sym = h.get("symbol")
        annual = h.get("forward_12m_dividend") or h.get("projected_annual_dividend") or 0
        if sym and isinstance(annual, (int, float)) and annual > 0:
            income_by_sym[sym] = annual

    if not income_by_sym:
        return None

    sorted_items = sorted(income_by_sym.items(), key=lambda x: x[1], reverse=True)
    top = sorted_items[:10]
    other_total = sum(v for _, v in sorted_items[10:])

    labels = [s for s, _ in top]
    values = [v for _, v in top]
    if other_total > 0:
        labels.append("Other")
        values.append(other_total)

    # Catppuccin Mocha palette
    colors = [
        "#f38ba8", "#fab387", "#f9e2af", "#a6e3a1", "#94e2d5",
        "#89dceb", "#74c7ec", "#89b4fa", "#b4befe", "#cba6f7",
        "#f5c2e7",
    ]

    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor("#1e1e2e")
    ax.set_facecolor("#1e1e2e")

    wedges, texts, autotexts = ax.pie(
        values, labels=labels, autopct="%1.1f%%",
        colors=colors[:len(values)], startangle=90,
        textprops={"color": "#cdd6f4"},
    )
    for at in autotexts:
        at.set_fontsize(9)
        at.set_fontweight("bold")

    total = sum(values)
    ax.set_title(
        f"Income Attribution — ${total:,.0f}/yr",
        fontsize=14, fontweight="bold", color="#cdd6f4",
    )

    return _fig_to_bytes(fig)


def generate_yield_chart(conn: sqlite3.Connection, days: int = 90) -> bytes | None:
    """Generate portfolio yield % and yield-on-cost % over time."""
    snapshots = _load_snapshots(conn, days)
    if len(snapshots) < 2:
        return None

    dates = []
    current_yield = []
    yoc = []

    for d, snap in snapshots:
        inc = snap.get("income") or {}
        cy = inc.get("portfolio_current_yield_pct")
        yc = inc.get("portfolio_yield_on_cost_pct")
        if isinstance(cy, (int, float)):
            dates.append(d)
            current_yield.append(cy)
            yoc.append(yc if isinstance(yc, (int, float)) else cy)

    if len(dates) < 2:
        return None

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#1e1e2e")
    _apply_dark_theme(ax)

    ax.plot(dates, current_yield, linewidth=2.2, color="#89b4fa", label="Current Yield")
    ax.plot(dates, yoc, linewidth=2, linestyle="--", color="#cba6f7", label="Yield on Cost")
    ax.fill_between(dates, current_yield, alpha=0.15, color="#89b4fa")

    # Annotate latest values
    if current_yield:
        ax.annotate(
            f"{current_yield[-1]:.2f}%", xy=(dates[-1], current_yield[-1]),
            xytext=(10, 5), textcoords="offset points", color="#89b4fa",
            fontsize=10, fontweight="bold",
        )
    if yoc and yoc[-1] != current_yield[-1]:
        ax.annotate(
            f"{yoc[-1]:.2f}%", xy=(dates[-1], yoc[-1]),
            xytext=(10, -12), textcoords="offset points", color="#cba6f7",
            fontsize=10, fontweight="bold",
        )

    ax.set_title(f"Portfolio Yield — Last {days} Days", fontsize=14, fontweight="bold")
    ax.set_ylabel("Yield (%)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.1f}%"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.legend(facecolor="#313244", edgecolor="#45475a", labelcolor="#cdd6f4")

    return _fig_to_bytes(fig)


def generate_risk_chart(conn: sqlite3.Connection, days: int = 90) -> bytes | None:
    """Generate risk dashboard: 30d volatility area + Sharpe/Sortino lines."""
    snapshots = _load_snapshots(conn, days)
    if len(snapshots) < 2:
        return None

    dates = []
    vol_30d = []
    sharpe = []
    sortino = []

    for d, snap in snapshots:
        rollups = snap.get("portfolio_rollups") or {}
        risk = rollups.get("risk") or {}
        v = risk.get("vol_30d_pct")
        if isinstance(v, (int, float)):
            dates.append(d)
            vol_30d.append(v)
            sh = risk.get("sharpe_1y")
            so = risk.get("sortino_1y")
            sharpe.append(sh if isinstance(sh, (int, float)) else None)
            sortino.append(so if isinstance(so, (int, float)) else None)

    if len(dates) < 2:
        return None

    fig, ax1 = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#1e1e2e")
    _apply_dark_theme(ax1)

    # Volatility on left axis
    ax1.fill_between(dates, vol_30d, alpha=0.25, color="#f38ba8")
    ax1.plot(dates, vol_30d, linewidth=2.2, color="#f38ba8", label="30d Volatility")
    ax1.set_ylabel("Volatility (%)", color="#f38ba8")
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.1f}%"))
    ax1.tick_params(axis="y", labelcolor="#f38ba8")

    # Sharpe/Sortino on right axis
    ax2 = ax1.twinx()
    _apply_dark_theme(ax2)
    ax2.set_facecolor("none")

    sharpe_clean = [(d, v) for d, v in zip(dates, sharpe) if v is not None]
    sortino_clean = [(d, v) for d, v in zip(dates, sortino) if v is not None]

    if sharpe_clean:
        sd, sv = zip(*sharpe_clean)
        ax2.plot(sd, sv, linewidth=1.8, color="#a6e3a1", label="Sharpe (1Y)", linestyle="-.")
    if sortino_clean:
        sd, sv = zip(*sortino_clean)
        ax2.plot(sd, sv, linewidth=1.8, color="#89b4fa", label="Sortino (1Y)", linestyle="--")

    ax2.set_ylabel("Ratio", color="#cdd6f4")
    ax2.tick_params(axis="y", labelcolor="#cdd6f4")
    ax2.axhline(y=0, color="#45475a", linewidth=0.8, alpha=0.5)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               facecolor="#313244", edgecolor="#45475a", labelcolor="#cdd6f4", loc="upper left")

    ax1.set_title(f"Risk Dashboard — Last {days} Days", fontsize=14, fontweight="bold")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

    return _fig_to_bytes(fig)


def generate_drawdown_chart(conn: sqlite3.Connection, days: int = 90) -> bytes | None:
    """Generate underwater chart showing drawdown from peak."""
    snapshots = _load_snapshots(conn, days)
    if len(snapshots) < 2:
        return None

    dates = []
    nlv_values = []

    for d, snap in snapshots:
        totals = snap.get("totals") or {}
        nlv = totals.get("net_liquidation_value")
        if isinstance(nlv, (int, float)):
            dates.append(d)
            nlv_values.append(nlv)

    if len(dates) < 2:
        return None

    # Compute drawdown series
    peak = nlv_values[0]
    drawdowns = []
    for v in nlv_values:
        peak = max(peak, v)
        dd_pct = ((v - peak) / peak) * 100.0 if peak > 0 else 0.0
        drawdowns.append(dd_pct)

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#1e1e2e")
    _apply_dark_theme(ax)

    ax.fill_between(dates, drawdowns, 0, alpha=0.35, color="#f38ba8")
    ax.plot(dates, drawdowns, linewidth=2, color="#f38ba8")
    ax.axhline(y=0, color="#45475a", linewidth=1)

    # Mark max drawdown point
    min_dd = min(drawdowns)
    min_idx = drawdowns.index(min_dd)
    ax.scatter([dates[min_idx]], [min_dd], color="#fab387", zorder=5, s=60)
    ax.annotate(
        f"{min_dd:.1f}%", xy=(dates[min_idx], min_dd),
        xytext=(10, -15), textcoords="offset points",
        color="#fab387", fontsize=11, fontweight="bold",
    )

    # Current drawdown annotation
    curr_dd = drawdowns[-1]
    if curr_dd < -0.5:
        ax.annotate(
            f"Now: {curr_dd:.1f}%", xy=(dates[-1], curr_dd),
            xytext=(-60, -15), textcoords="offset points",
            color="#f9e2af", fontsize=10, fontweight="bold",
        )

    ax.set_title(f"Drawdown from Peak — Last {days} Days (Max: {min_dd:.1f}%)",
                 fontsize=14, fontweight="bold")
    ax.set_ylabel("Drawdown (%)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.1f}%"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

    return _fig_to_bytes(fig)


def generate_allocation_chart(snap: dict) -> bytes | None:
    """Generate horizontal bar chart of position weights."""
    holdings = snap.get("holdings") or []
    positions = []
    for h in holdings:
        sym = h.get("symbol")
        weight = h.get("weight_pct")
        if sym and isinstance(weight, (int, float)) and weight > 0:
            positions.append((sym, weight))

    if not positions:
        return None

    positions.sort(key=lambda x: x[1], reverse=True)

    # Top 15 + Other
    top = positions[:15]
    other_weight = sum(w for _, w in positions[15:])

    syms = [s for s, _ in top]
    weights = [w for _, w in top]
    if other_weight > 0.1:
        syms.append("Other")
        weights.append(other_weight)

    syms.reverse()
    weights.reverse()

    # Catppuccin palette cycling
    palette = [
        "#f38ba8", "#fab387", "#f9e2af", "#a6e3a1", "#94e2d5",
        "#89dceb", "#74c7ec", "#89b4fa", "#b4befe", "#cba6f7",
        "#f5c2e7", "#f38ba8", "#fab387", "#f9e2af", "#a6e3a1", "#94e2d5",
    ]
    bar_colors = [palette[i % len(palette)] for i in range(len(syms))]

    fig_height = max(5, len(syms) * 0.38)
    fig, ax = plt.subplots(figsize=(10, fig_height))
    fig.patch.set_facecolor("#1e1e2e")
    _apply_dark_theme(ax)

    bars = ax.barh(syms, weights, color=bar_colors, edgecolor="#45475a", linewidth=0.5)

    # Value labels on bars
    for bar, w in zip(bars, weights):
        ax.text(
            bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
            f"{w:.1f}%", va="center", color="#cdd6f4", fontsize=9,
        )

    ax.set_xlabel("Weight (%)")
    ax.set_title("Portfolio Allocation", fontsize=14, fontweight="bold")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.0f}%"))

    return _fig_to_bytes(fig)


def generate_margin_chart(conn: sqlite3.Connection, days: int = 90) -> bytes | None:
    """Generate margin utilization chart: LTV % and margin balance over time."""
    snapshots = _load_snapshots(conn, days)
    if len(snapshots) < 2:
        return None

    dates = []
    ltv_pct = []
    margin_bal = []

    for d, snap in snapshots:
        totals = snap.get("totals") or {}
        ltv = totals.get("margin_to_portfolio_pct")
        bal = totals.get("margin_loan_balance")
        if isinstance(ltv, (int, float)):
            dates.append(d)
            ltv_pct.append(ltv)
            margin_bal.append(bal if isinstance(bal, (int, float)) else 0)

    if len(dates) < 2:
        return None

    fig, ax1 = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#1e1e2e")
    _apply_dark_theme(ax1)

    # LTV % on left axis with color zones
    ax1.fill_between(dates, ltv_pct, alpha=0.2, color="#f9e2af")
    ax1.plot(dates, ltv_pct, linewidth=2.2, color="#f9e2af", label="LTV %")

    # Warning/critical threshold lines
    ax1.axhline(y=30, color="#fab387", linewidth=1, linestyle=":", alpha=0.7, label="Warning (30%)")
    ax1.axhline(y=40, color="#f38ba8", linewidth=1, linestyle=":", alpha=0.7, label="Critical (40%)")

    ax1.set_ylabel("LTV (%)", color="#f9e2af")
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.0f}%"))
    ax1.tick_params(axis="y", labelcolor="#f9e2af")
    ax1.set_ylim(bottom=0)

    # Margin balance on right axis
    ax2 = ax1.twinx()
    _apply_dark_theme(ax2)
    ax2.set_facecolor("none")

    ax2.plot(dates, margin_bal, linewidth=1.8, color="#74c7ec", linestyle="--", label="Margin Balance")
    ax2.set_ylabel("Balance ($)", color="#74c7ec")
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))
    ax2.tick_params(axis="y", labelcolor="#74c7ec")

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               facecolor="#313244", edgecolor="#45475a", labelcolor="#cdd6f4", loc="upper left")

    ax1.set_title(f"Margin Utilization — Last {days} Days", fontsize=14, fontweight="bold")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

    return _fig_to_bytes(fig)


def generate_dividend_calendar_chart(snap: dict) -> bytes | None:
    """Generate bar chart of upcoming dividend payments by date."""
    upcoming = snap.get("dividends_upcoming") or {}
    events = upcoming.get("events") or []

    if not events:
        return None

    # Aggregate amounts by pay date
    by_date: dict[date, float] = {}
    sym_by_date: dict[date, list[str]] = {}
    for ev in events:
        pay_str = ev.get("pay_date_est") or ev.get("ex_date_est")
        amt = ev.get("amount_est", 0)
        sym = ev.get("symbol", "?")
        if not pay_str or not isinstance(amt, (int, float)):
            continue
        try:
            pd = date.fromisoformat(pay_str)
        except ValueError:
            continue
        by_date[pd] = by_date.get(pd, 0) + amt
        sym_by_date.setdefault(pd, []).append(sym)

    if not by_date:
        return None

    sorted_dates = sorted(by_date.keys())
    amounts = [by_date[d] for d in sorted_dates]

    # Catppuccin palette cycling per bar
    palette = ["#a6e3a1", "#89b4fa", "#f9e2af", "#cba6f7", "#94e2d5",
               "#fab387", "#f38ba8", "#74c7ec", "#b4befe", "#f5c2e7"]
    bar_colors = [palette[i % len(palette)] for i in range(len(sorted_dates))]

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#1e1e2e")
    _apply_dark_theme(ax)

    bars = ax.bar(sorted_dates, amounts, color=bar_colors, edgecolor="#45475a",
                  linewidth=0.5, width=0.8)

    # Labels on each bar showing symbols
    for bar, d, amt in zip(bars, sorted_dates, amounts):
        syms = sym_by_date.get(d, [])
        label = ", ".join(syms[:3])
        if len(syms) > 3:
            label += f" +{len(syms)-3}"
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + max(amounts) * 0.02,
            label, ha="center", va="bottom", color="#cdd6f4", fontsize=7, rotation=45,
        )

    total = sum(amounts)
    ax.set_title(f"Upcoming Dividends — ${total:,.2f} Total", fontsize=14, fontweight="bold")
    ax.set_ylabel("Amount ($)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.2f}"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    fig.autofmt_xdate(rotation=45)

    return _fig_to_bytes(fig)
