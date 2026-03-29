import os
import logging
from datetime import datetime

from rich.console import Console
from rich.logging import RichHandler

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
_log_path = os.path.join(LOG_DIR, f"run_{datetime.now():%Y%m%d_%H%M%S}.log")
_log_file = open(_log_path, "w", encoding="utf-8")

_term = Console(stderr=True)
_file = Console(file=_log_file, no_color=True, width=160)

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        RichHandler(console=_term, rich_tracebacks=True, markup=False),
        RichHandler(console=_file, rich_tracebacks=True, markup=False),
    ],
)
log = logging.getLogger("vm_agent")


def printout(*args, **kwargs):
    """Print rich renderables (Panel, Table, Rule, etc.) to both outputs."""
    _term.print(*args, **kwargs)
    _file.print(*args, **kwargs)
