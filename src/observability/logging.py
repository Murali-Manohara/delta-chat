"""
Structured JSON logging with a correlation id.

Every log line is one JSON object on one line (so `jq`, `grep`, or any log
aggregator can consume it without a custom parser). No free-text logs
anywhere in src/ -- if you need to log something, call `get_logger(...)`
and pass structured fields, not an f-string.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from typing import Any


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": round(time.time(), 3),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage() if record.args else record.msg,
        }
        for key in ("request_id", "stage", "pid", "extra"):
            val = getattr(record, key, None)
            if val is not None:
                payload[key] = val
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def get_logger(name: str = "delta_chat") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def log_event(logger: logging.Logger, level: str, msg: str, **fields):
    extra = {k: v for k, v in fields.items() if k in ("request_id", "stage", "pid")}
    extra["extra"] = {k: v for k, v in fields.items() if k not in extra}
    getattr(logger, level)(msg, extra=extra)
