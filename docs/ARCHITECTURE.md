# Architecture Notes

- **Pipeline order:** LM → Holdings (LM only) → Providers (needed symbols) → Derived metrics → Daily snapshot → Periodic snapshots.
- **Provider chain:** OpenBB(yfinance) → yahooquery → Stooq; FRED for macro. Record `{provider, endpoint, params}` in `facts_source_daily`.
- **Benchmarks:** primary `^GSPC`, secondary `SPY`.
- **Facts vs Derived:** Only source-backed fields go into `facts_source_daily`. Derived fields computed during snapshot build.
