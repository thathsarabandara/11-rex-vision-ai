from datetime import datetime, timezone


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def utcnow() -> datetime:
    return datetime.now(timezone.utc)
