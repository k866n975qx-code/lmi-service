from __future__ import annotations

SIGNIFICANT_MISSING_PCT = 1.0


def slim_snapshot(snapshot: dict, missing_pct_threshold: float = SIGNIFICANT_MISSING_PCT) -> dict:
    if not isinstance(snapshot, dict):
        return snapshot
    missing_pct = _missing_pct(snapshot)
    keep_notes = missing_pct is not None and missing_pct >= missing_pct_threshold
    return _strip_noise(snapshot, keep_notes=keep_notes)


def _missing_pct(snapshot: dict) -> float | None:
    coverage = snapshot.get("coverage")
    if isinstance(coverage, dict):
        val = coverage.get("missing_pct")
        if isinstance(val, (int, float)):
            return float(val)
    return None


def _strip_noise(obj, keep_notes: bool, parent_key: str | None = None):
    if isinstance(obj, dict):
        out = {}
        for key, value in obj.items():
            key_lower = str(key).lower()
            if "provenance" in key_lower:
                continue
            if key_lower == "notes" and not keep_notes:
                continue
            if parent_key == "meta" and key_lower in ("cache", "cache_control"):
                continue
            out[key] = _strip_noise(value, keep_notes=keep_notes, parent_key=str(key))
        return out
    if isinstance(obj, list):
        return [_strip_noise(item, keep_notes=keep_notes, parent_key=parent_key) for item in obj]
    return obj
