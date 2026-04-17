"""
utils/logger.py
──────────────
Centralised logger using loguru.
All modules import `logger` from here.
"""

import sys
from pathlib import Path
from loguru import logger

# Remove default handler
logger.remove()

# ── Console handler (coloured) ───────────────────────────────
logger.add(
    sys.stdout,
    colorize=True,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
           "<level>{level: <8}</level> | "
           "<cyan>{name}</cyan>:<cyan>{line}</cyan> — <level>{message}</level>",
    level="INFO",
)

# ── File handler (full debug log) ────────────────────────────
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

logger.add(
    LOG_DIR / "run.log",
    rotation="10 MB",
    retention="14 days",
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name}:{line} — {message}",
)

__all__ = ["logger"]
