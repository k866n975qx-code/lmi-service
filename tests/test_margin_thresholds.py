from app.pipeline.daily_transform_v5 import _build_alerts
from app.pipeline.snapshots import _build_margin_guidance
from app.pipeline import snapshot_views


def _daily_payload(ltv_pct: float) -> dict:
    return {
        "meta": {"snapshot_created_at": "2026-01-15T00:00:00Z"},
        "as_of": "2026-01-15",
        "totals": {
            "margin_to_portfolio_pct": ltv_pct,
            "margin_loan_balance": 10000.0,
        },
        "portfolio_rollups": {"risk": {"drawdown_status": {}, "recovery_metrics": {}}},
        "margin_guidance": {"modes": [], "rates": {}},
        "goal_pace": {},
        "holdings": [],
    }


def _margin_alerts(ltv_pct: float) -> list[dict]:
    alerts = _build_alerts(_daily_payload(ltv_pct))
    return [a for a in alerts if a.get("type") == "margin"]


def test_margin_alert_threshold_is_strictly_above_35pct():
    assert _margin_alerts(35.0) == []


def test_margin_alert_warns_above_35pct():
    alerts = _margin_alerts(35.1)
    assert len(alerts) == 1
    assert alerts[0]["severity"] == "warning"
    assert "target of 35.0%" in alerts[0]["message"]


def test_margin_alert_critical_starts_above_40pct():
    alerts = _margin_alerts(40.1)
    assert len(alerts) == 1
    assert alerts[0]["severity"] == "critical"


def test_margin_guidance_aggressive_mode_max_margin_pct_is_35():
    guidance = _build_margin_guidance(
        total_market_value=100000.0,
        margin_loan_balance=20000.0,
        projected_monthly_income=1000.0,
    )
    aggressive = next(mode for mode in guidance["modes"] if mode["mode"] == "aggressive")
    assert aggressive["constraints"]["max_margin_pct"] == 35.0


def test_snapshot_fallback_target_ltv_uses_settings_value(monkeypatch):
    monkeypatch.setattr(snapshot_views.settings, "goal_target_ltv_pct", 35.0)
    assert snapshot_views._resolve_target_ltv_pct(None, True) == 35.0
    assert snapshot_views._resolve_target_ltv_pct(None, 1) == 35.0
    assert snapshot_views._resolve_target_ltv_pct(None, 0) is None
    assert snapshot_views._resolve_target_ltv_pct(33.0, True) == 33.0
