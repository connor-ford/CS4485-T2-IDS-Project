import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

class BaseModelRunner(ABC):
    """minimal runner interface."""

    @abstractmethod
    def validate(self, inputs: Dict[str, Any]) -> None:
        """raise ValueError if inputs are missing or malformed."""
        ...

    @abstractmethod
    def _predict_impl(self, inputs: Dict[str, Any]) -> Any:
        """return raw prediction (scalar, label, or dict)."""
        ...

    def predict(self, inputs: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        """wrapper that adds simple timing meta."""
        self.validate(inputs)
        t0 = time.perf_counter()
        out = self._predict_impl(inputs)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        return out, {"inference_ms": round(dt_ms, 3)}
