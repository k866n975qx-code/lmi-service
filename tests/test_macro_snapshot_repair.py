import sqlite3
from datetime import date

from app.pipeline import orchestrator as orchestrator_mod
from app.pipeline import snapshots as snapshots_mod
from app.providers.fred_adapter import FredAdapter


def _make_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute(
        """
        CREATE TABLE daily_portfolio (
            as_of_date_local TEXT PRIMARY KEY,
            market_value REAL,
            prices_as_of_utc TEXT,
            macro_vix REAL,
            macro_ten_year_yield REAL,
            macro_two_year_yield REAL,
            macro_hy_spread_bps REAL,
            macro_yield_spread_10y_2y REAL,
            macro_stress_score REAL,
            macro_cpi_yoy REAL,
            macro_data_as_of_date TEXT
        )
        """
    )
    return conn


def test_stabilize_macro_snapshot_uses_latest_complete_row():
    conn = _make_conn()
    conn.execute(
        """
        INSERT INTO daily_portfolio (
            as_of_date_local,
            market_value,
            prices_as_of_utc,
            macro_vix,
            macro_ten_year_yield,
            macro_two_year_yield,
            macro_hy_spread_bps,
            macro_yield_spread_10y_2y,
            macro_stress_score,
            macro_cpi_yoy,
            macro_data_as_of_date
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "2026-03-10",
            100.0,
            "2026-03-10T13:00:02+00:00",
            25.5,
            4.12,
            3.56,
            319.0,
            0.56,
            -0.006,
            2.83,
            "2026-03-09",
        ),
    )
    current_macro = {
        "snapshot": {
            "date": "2026-02-01",
            "vix": None,
            "ten_year_yield": None,
            "two_year_yield": None,
            "hy_spread_bps": None,
            "yield_spread_10y_2y": None,
            "macro_stress_score": None,
            "cpi_yoy": 2.66,
        }
    }

    stabilized = snapshots_mod._stabilize_macro_snapshot(conn, current_macro, date(2026, 3, 11))

    assert stabilized["snapshot"]["vix"] == 25.5
    assert stabilized["snapshot"]["ten_year_yield"] == 4.12
    assert stabilized["snapshot"]["macro_stress_score"] == -0.006
    assert stabilized["snapshot"]["cpi_yoy"] == 2.83
    assert stabilized["snapshot"]["date"] == "2026-03-09"


def test_stabilize_macro_snapshot_keeps_complete_current_values():
    conn = _make_conn()
    current_macro = {
        "snapshot": {
            "date": "2026-03-11",
            "vix": 27.0,
            "ten_year_yield": 4.2,
            "two_year_yield": 3.6,
            "hy_spread_bps": 325.0,
            "yield_spread_10y_2y": 0.6,
            "macro_stress_score": 0.1,
            "cpi_yoy": 2.7,
        }
    }

    stabilized = snapshots_mod._stabilize_macro_snapshot(conn, current_macro, date(2026, 3, 11))

    assert stabilized is current_macro


def test_daily_persist_state_rewrites_when_macro_changes_only():
    existing_flat = (
        100.0,
        "2026-03-11T13:00:02+00:00",
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )
    daily = {
        "portfolio": {"totals": {"market_value": 100.0}},
        "timestamps": {"price_data_as_of_utc": "2026-03-11T13:00:02+00:00"},
        "macro": {
            "snapshot": {
                "vix": 25.5,
                "ten_year_yield": 4.12,
                "two_year_yield": 3.56,
                "hy_spread_bps": 319.0,
                "macro_stress_score": -0.006,
                "cpi_yoy": 2.83,
                "date": "2026-03-09",
            }
        },
    }

    state = orchestrator_mod._daily_persist_state(existing_flat, daily, new_tx_count=0)

    assert state["market_value_changed"] is False
    assert state["prices_as_of_changed"] is False
    assert state["macro_changed"] is True
    assert state["should_persist_daily"] is True


def test_daily_persist_state_skips_when_nothing_changed():
    existing_flat = (
        100.0,
        "2026-03-11T13:00:02+00:00",
        25.5,
        4.12,
        3.56,
        319.0,
        -0.006,
        2.83,
        "2026-03-09",
    )
    daily = {
        "portfolio": {"totals": {"market_value": 100.0}},
        "timestamps": {"price_data_as_of_utc": "2026-03-11T13:00:02+00:00"},
        "macro": {
            "snapshot": {
                "vix": 25.5,
                "ten_year_yield": 4.12,
                "two_year_yield": 3.56,
                "hy_spread_bps": 319.0,
                "macro_stress_score": -0.006,
                "cpi_yoy": 2.83,
                "date": "2026-03-09",
            }
        },
    }

    state = orchestrator_mod._daily_persist_state(existing_flat, daily, new_tx_count=0)

    assert state["macro_changed"] is False
    assert state["should_persist_daily"] is False


def test_fred_adapter_uses_json_api_when_key_present_and_pyfredapi_missing(monkeypatch):
    payload = {
        "observations": [
            {"date": "2026-03-10", "value": "4.15"},
            {"date": "2026-03-11", "value": "4.20"},
        ]
    }
    calls: list[str] = []

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            import json

            return json.dumps(payload).encode("utf-8")

    def fake_urlopen(req, timeout=None):
        calls.append(req.full_url)
        return _Resp()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    adapter = FredAdapter(api_key="secret", timeout_seconds=5.0)
    adapter.fred = None
    df = adapter.series("DGS10")

    assert df is not None
    assert calls and calls[0].startswith("https://api.stlouisfed.org/fred/series/observations?")
    assert df["value"].tolist() == [4.15, 4.2]


def test_fred_adapter_falls_back_to_csv_when_api_unavailable(monkeypatch):
    calls: list[str] = []

    class _Resp:
        def __init__(self, body: str):
            self._body = body

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return self._body.encode("utf-8")

    def fake_urlopen(req, timeout=None):
        url = req.full_url
        calls.append(url)
        if "api.stlouisfed.org" in url:
            raise TimeoutError("api timeout")
        return _Resp("DATE,DGS10\n2026-03-10,4.15\n2026-03-11,4.20\n")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    adapter = FredAdapter(api_key="secret", timeout_seconds=5.0)
    adapter.fred = None
    df = adapter.series("DGS10")

    assert df is not None
    assert len(calls) == 2
    assert "api.stlouisfed.org" in calls[0]
    assert "fredgraph.csv" in calls[1]
    assert df["value"].tolist() == [4.15, 4.2]
