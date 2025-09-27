import importlib
import pkgutil
from typing import Dict
from .base import BaseModelRunner

MODEL_REGISTRY: Dict[str, BaseModelRunner] = {}

def _auto_register():
    # import every submodule under models.* (except private/base/dunder)
    package = __name__
    for m in pkgutil.iter_modules(__path__):
        name = m.name
        if name.startswith("_") or name in {"base"}:
            continue
        module = importlib.import_module(f"{package}.{name}")
        # convention: module must define MODEL_NAME: str and RUNNER: BaseModelRunner
        if hasattr(module, "MODEL_NAME") and hasattr(module, "RUNNER"):
            runner = getattr(module, "RUNNER")
            if isinstance(runner, BaseModelRunner):
                MODEL_REGISTRY[getattr(module, "MODEL_NAME")] = runner

_auto_register()
