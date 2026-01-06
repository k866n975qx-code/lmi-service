import unittest

from app.pipeline.diff_daily import build_daily_diff


def _snapshot(market_value, margin_loan=100.0, twr_12m=12.0, vol_30d=5.0, vol_90d=10.0, sharpe=0.6, calmar=0.7):
    return {
        "totals": {
            "market_value": market_value,
            "net_liquidation_value": market_value - margin_loan,
            "cost_basis": market_value - 10.0,
            "unrealized_pnl": 10.0,
            "unrealized_pct": 1.0,
            "margin_loan_balance": margin_loan,
            "margin_to_portfolio_pct": round((margin_loan / market_value) * 100, 3) if market_value else None,
        },
        "income": {
            "projected_monthly_income": 10.0,
            "forward_12m_total": 120.0,
            "portfolio_current_yield_pct": 10.0,
            "portfolio_yield_on_cost_pct": 10.0,
        },
        "portfolio_rollups": {
            "performance": {"twr_1m_pct": 1.0, "twr_3m_pct": 2.0, "twr_6m_pct": 3.0, "twr_12m_pct": twr_12m},
            "risk": {
                "vol_30d_pct": vol_30d,
                "vol_90d_pct": vol_90d,
                "sharpe_1y": sharpe,
                "calmar_1y": calmar,
                "max_drawdown_1y_pct": -10.0,
            },
        },
        "dividends_upcoming": {"projected": 5.0},
        "coverage": {"derived_pct": 100.0, "missing_pct": 0.0},
        "holdings": [
            {
                "symbol": "AAA",
                "shares": 10.0,
                "market_value": market_value,
                "weight_pct": 100.0,
                "last_price": market_value / 10.0,
                "unrealized_pnl": 10.0,
            }
        ],
    }


class PeriodDiffTests(unittest.TestCase):
    def test_weekly_rolling_diff(self):
        left = _snapshot(1000.0)
        right = _snapshot(1100.0)
        diff = build_daily_diff(
            left,
            right,
            "2026-01-01",
            "2026-01-08",
            period_type="weekly",
            calendar_aligned=False,
            period_start="2026-01-01",
            period_end="2026-01-08",
            dividends_period_totals=(25.0, 30.0),
        )
        self.assertEqual(diff["comparison"]["period_type"], "weekly")
        self.assertFalse(diff["comparison"]["calendar_aligned"])
        self.assertIn("per_day_return_pct", diff["range_metrics"])
        self.assertNotIn("annualized_return_pct", diff["range_metrics"])
        self.assertIn("realized_period_total", diff["dividends"])
        self.assertIn("realized_mtd_total", diff["dividends"])
        self.assertIsNone(diff["dividends"]["realized_mtd_total"]["left"])

    def test_monthly_calendar_aligned_diff(self):
        left = _snapshot(1000.0)
        right = _snapshot(1100.0)
        diff = build_daily_diff(
            left,
            right,
            "2026-02-01",
            "2026-02-28",
            period_type="monthly",
            calendar_aligned=True,
            period_start="2026-02-01",
            period_end="2026-02-28",
            dividends_period_totals=(40.0, 55.0),
            dividends_mtd_totals=(40.0, 55.0),
        )
        self.assertEqual(diff["comparison"]["period_type"], "monthly")
        self.assertTrue(diff["comparison"]["calendar_aligned"])
        self.assertIn("annualized_return_pct", diff["range_metrics"])
        self.assertIsNone(diff["range_metrics"]["per_day_return_pct"])
        self.assertEqual(diff["dividends"]["realized_mtd_total"]["left"], 40.0)

    def test_quarterly_calendar_aligned_diff(self):
        left = _snapshot(1000.0)
        right = _snapshot(1100.0)
        diff = build_daily_diff(
            left,
            right,
            "2026-01-01",
            "2026-03-31",
            period_type="quarterly",
            calendar_aligned=True,
            period_start="2026-01-01",
            period_end="2026-03-31",
            dividends_period_totals=(75.0, 90.0),
        )
        self.assertEqual(diff["comparison"]["period_type"], "quarterly")
        self.assertTrue(diff["comparison"]["calendar_aligned"])
        self.assertIn("annualized_return_pct", diff["range_metrics"])
        self.assertIsNone(diff["range_metrics"]["per_day_return_pct"])
        self.assertIsNone(diff["dividends"]["realized_mtd_total"]["left"])

    def test_yearly_calendar_aligned_diff(self):
        left = _snapshot(1000.0, twr_12m=11.5, sharpe=0.55, calmar=0.65)
        right = _snapshot(1100.0, twr_12m=12.5, sharpe=0.66, calmar=0.77)
        diff = build_daily_diff(
            left,
            right,
            "2025-01-01",
            "2025-12-31",
            period_type="yearly",
            calendar_aligned=True,
            period_start="2025-01-01",
            period_end="2025-12-31",
            dividends_period_totals=(120.0, 140.0),
        )
        self.assertEqual(diff["comparison"]["period_type"], "yearly")
        self.assertTrue(diff["comparison"]["calendar_aligned"])
        self.assertIn("annualized_return_pct", diff["range_metrics"])
        self.assertIsNone(diff["range_metrics"]["per_day_return_pct"])
        highlights = diff["summary"]["highlights"]
        self.assertTrue(any(line.startswith("TWR 12m") for line in highlights))


if __name__ == "__main__":
    unittest.main()
