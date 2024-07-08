"""Furiosa Warboy Vision Models"""

from .models import WARBOY_YOLO
from .warboy_api import WARBOY_APP
from warboy.cfg import (
    MODEL_LIST,
    TASKS,
    get_demo_params_from_cfg,
    get_model_params_from_cfg,
)
from .viewer import spawn_web_viewer

__version__ = "0.10.2.dev0"
__all__ = [
    "__version__",
    "WARBOY_YOLO",
    "WARBOY_APP",
    "spawn_web_viewer",
    "get_demo_params_from_cfg",
    "get_model_params_from_cfg" "MODEL_LIST",
    "TASKS",
]
