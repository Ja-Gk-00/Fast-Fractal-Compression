from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Logger:
    emit: Callable[[str], None]

    def info(self, msg: str) -> None:
        self.emit(msg)
