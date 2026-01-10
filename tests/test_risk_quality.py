import unittest

from app.pipeline.snapshots import _risk_quality_category


class RiskQualityCategoryTests(unittest.TestCase):
    def test_excellent_high_sortino_high_ratio(self):
        self.assertEqual(_risk_quality_category(2.0, 1.5), "excellent")

    def test_good_decent_sortino_good_ratio(self):
        self.assertEqual(_risk_quality_category(0.7, 0.6), "good")

    def test_acceptable_high_ratio_low_sortino(self):
        self.assertEqual(_risk_quality_category(0.4, 0.3), "acceptable")

    def test_concerning_low_both(self):
        self.assertEqual(_risk_quality_category(0.3, 0.4), "concerning")

    def test_concerning_negative(self):
        self.assertEqual(_risk_quality_category(-0.2, -0.3), "concerning")

    def test_concerning_sharpe_zero(self):
        self.assertEqual(_risk_quality_category(0.5, 0.0), "concerning")

    def test_none_when_missing(self):
        self.assertIsNone(_risk_quality_category(None, 0.5))
        self.assertIsNone(_risk_quality_category(0.5, None))


if __name__ == "__main__":
    unittest.main()
