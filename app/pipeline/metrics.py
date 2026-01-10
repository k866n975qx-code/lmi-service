from __future__ import annotations

import numpy as np
import pandas as pd

ANNUAL_PERIODS = 252


def _as_datetime_index(series: pd.Series) -> pd.Series:
    if series is None:
        return series
    if not isinstance(series.index, pd.DatetimeIndex):
        series = series.copy()
        series.index = pd.to_datetime(series.index)
    return series.sort_index()


def _slice_window(series: pd.Series, window_days: int | None = None, window_points: int | None = None) -> pd.Series:
    if series is None or series.empty:
        return series
    if window_points:
        return series.tail(window_points)
    if window_days:
        series = _as_datetime_index(series)
        end = series.index.max()
        start = end - pd.Timedelta(days=window_days)
        return series.loc[series.index >= start]
    return series


def time_weighted_returns(values: pd.Series, cashflows: pd.Series | None = None) -> pd.Series:
    if values is None or values.empty:
        return pd.Series(dtype=float)
    v = _as_datetime_index(values).dropna()
    if v.size < 2:
        return pd.Series(dtype=float)
    if cashflows is None:
        cashflows = pd.Series(0.0, index=v.index)
    else:
        cashflows = _as_datetime_index(cashflows).reindex(v.index).fillna(0.0)
    prev = v.shift(1)
    returns = (v - prev - cashflows) / prev
    return returns.dropna()


def twr(values: pd.Series, cashflows: pd.Series | None = None, window_days: int | None = None, window_points: int | None = None) -> float | None:
    rets = time_weighted_returns(values, cashflows)
    rets = _slice_window(rets, window_days=window_days, window_points=window_points)
    if rets is None or rets.empty:
        return None
    return float((1.0 + rets).prod() - 1.0)


def annualized_volatility(returns: pd.Series, window_days: int | None = None, window_points: int | None = None, periods: int = ANNUAL_PERIODS) -> float | None:
    rets = _slice_window(returns, window_days=window_days, window_points=window_points)
    if rets is None or rets.empty:
        return None
    return float(rets.std(ddof=0) * np.sqrt(periods))


def downside_deviation(returns: pd.Series, window_days: int | None = None, window_points: int | None = None, periods: int = ANNUAL_PERIODS) -> float | None:
    rets = _slice_window(returns, window_days=window_days, window_points=window_points)
    if rets is None or rets.empty:
        return None
    downside = rets[rets < 0]
    if downside.empty:
        return 0.0
    return float(downside.std(ddof=0) * np.sqrt(periods))


def sharpe_ratio(returns: pd.Series, rf_annual: float = 0.0, window_days: int | None = None, window_points: int | None = None, periods: int = ANNUAL_PERIODS) -> float | None:
    rets = _slice_window(returns, window_days=window_days, window_points=window_points)
    if rets is None or rets.empty:
        return None
    vol = rets.std(ddof=0)
    if vol == 0:
        return None
    ann_return = rets.mean() * periods
    return float((ann_return - rf_annual) / (vol * np.sqrt(periods)))


def sortino_ratio(returns: pd.Series, rf_annual: float = 0.0, window_days: int | None = None, window_points: int | None = None, periods: int = ANNUAL_PERIODS) -> float | None:
    rets = _slice_window(returns, window_days=window_days, window_points=window_points)
    if rets is None or rets.empty:
        return None
    dd = downside_deviation(rets, periods=periods)
    if dd in (None, 0.0):
        return None
    ann_return = rets.mean() * periods
    return float((ann_return - rf_annual) / dd)


def var_cvar(returns: pd.Series, alpha: float = 0.05, window_days: int | None = None, window_points: int | None = None) -> tuple[float | None, float | None]:
    rets = _slice_window(returns, window_days=window_days, window_points=window_points)
    if rets is None or rets.empty:
        return None, None
    var = float(np.quantile(rets, alpha))
    tail = rets[rets <= var]
    cvar = float(tail.mean()) if not tail.empty else None
    return var, cvar


def max_drawdown(values: pd.Series, window_days: int | None = None) -> tuple[float | None, int | None]:
    v = _slice_window(_as_datetime_index(values).dropna(), window_days=window_days)
    if v is None or v.empty:
        return None, None
    peak = v.cummax()
    dd = v / peak - 1.0
    max_dd = float(dd.min())
    # longest drawdown duration in data points
    duration = 0
    current = 0
    for val in dd:
        if val < 0:
            current += 1
        else:
            duration = max(duration, current)
            current = 0
    duration = max(duration, current)
    return max_dd, duration


def beta_and_corr(portfolio_returns: pd.Series, benchmark_returns: pd.Series, window_days: int | None = None) -> tuple[float | None, float | None]:
    if portfolio_returns is None or benchmark_returns is None:
        return None, None
    pr = _slice_window(portfolio_returns, window_days=window_days)
    br = _slice_window(benchmark_returns, window_days=window_days)
    df = pd.concat([pr, br], axis=1).dropna()
    if df.empty or df.shape[0] < 2:
        return None, None
    cov = df.iloc[:, 0].cov(df.iloc[:, 1])
    var = df.iloc[:, 1].var()
    beta = float(cov / var) if var not in (0.0, None) else None
    corr = float(df.iloc[:, 0].corr(df.iloc[:, 1]))
    return beta, corr


def portfolio_performance(values: pd.Series, cashflows: pd.Series | None = None) -> dict:
    return {
        "twr_1m_pct": _pct(twr(values, cashflows, window_days=30)),
        "twr_3m_pct": _pct(twr(values, cashflows, window_days=90)),
        "twr_6m_pct": _pct(twr(values, cashflows, window_days=180)),
        "twr_12m_pct": _pct(twr(values, cashflows, window_days=365)),
    }


def portfolio_risk(values: pd.Series, benchmark_values: pd.Series | None = None, rf_annual: float = 0.0) -> dict:
    returns = time_weighted_returns(values)
    beta, corr = (None, None)
    if benchmark_values is not None:
        bench_returns = time_weighted_returns(benchmark_values)
        beta, corr = beta_and_corr(returns, bench_returns, window_days=365)
    max_dd, dd_dur = max_drawdown(values, window_days=365)
    var_95, cvar_95 = var_cvar(returns, alpha=0.05, window_days=365)
    downside_1y = downside_deviation(returns, window_days=365)
    sharpe_1y = sharpe_ratio(returns, rf_annual=rf_annual, window_days=365)
    sortino_1y = sortino_ratio(returns, rf_annual=rf_annual, window_days=365)
    sortino_6m = sortino_ratio(returns, rf_annual=rf_annual, window_days=180)
    sortino_3m = sortino_ratio(returns, rf_annual=rf_annual, window_days=90)
    sortino_1m = sortino_ratio(returns, rf_annual=rf_annual, window_days=30)
    sortino_sharpe_ratio = _safe_divide(sortino_1y, sharpe_1y)
    sortino_sharpe_divergence = None
    if sortino_1y is not None and sharpe_1y is not None:
        sortino_sharpe_divergence = sortino_1y - sharpe_1y
    return {
        "vol_30d_pct": _pct(annualized_volatility(returns, window_days=30)),
        "vol_90d_pct": _pct(annualized_volatility(returns, window_days=90)),
        "downside_dev_1y_pct": _pct(downside_1y),
        "sharpe_1y": _round(sharpe_1y),
        "sortino_1y": _round(sortino_1y),
        "sortino_6m": _round(sortino_6m),
        "sortino_3m": _round(sortino_3m),
        "sortino_1m": _round(sortino_1m),
        "sortino_sharpe_ratio": _round(sortino_sharpe_ratio),
        "sortino_sharpe_divergence": _round(sortino_sharpe_divergence),
        "calmar_1y": _round(_safe_divide(twr(values, window_days=365), abs(max_dd) if max_dd is not None else None)),
        "max_drawdown_1y_pct": _pct(max_dd),
        "drawdown_duration_1y_days": dd_dur,
        "var_95_1d_pct": _pct(var_95),
        "cvar_95_1d_pct": _pct(cvar_95),
        "beta_portfolio": _round(beta),
        "corr_1y": _round(corr),
        "income_stability_score": None,
    }


def _pct(val: float | None) -> float | None:
    return None if val is None else round(val * 100, 3)


def _round(val: float | None) -> float | None:
    return None if val is None else round(val, 3)


def _safe_divide(a: float | None, b: float | None) -> float | None:
    if a is None or b in (None, 0.0):
        return None
    return float(a / b)
