"""Furiosa Warboy Vision Models"""

from src.warboy.cfg import (
    MODEL_LIST,
    TASKS,
    get_demo_params_from_cfg,
    get_model_params_from_cfg,
)

from .viewer import spawn_web_viewer
from .warboy_api import WARBOY_APP

__version__ = "0.10.2.dev0"
__all__ = [
    "__version__",
    "spawn_web_viewer",
    "get_demo_params_from_cfg",
    "get_model_params_from_cfg",
    "MODEL_LIST",
    "TASKS",
    "WARBOY_APP",
]
