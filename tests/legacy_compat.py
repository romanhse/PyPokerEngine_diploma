"""Tiny compatibility helpers for the historical Python-2-era test suite."""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar

import pytest

F = TypeVar("F", bound=Callable[..., Any])


def raises(exception_type: type[BaseException]) -> Callable[[F], F]:
    """Drop-in replacement for the only ``nose.tools`` decorator still used."""

    def decorator(function: F) -> F:
        @wraps(function)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            with pytest.raises(exception_type):
                return function(*args, **kwargs)

        return wrapped  # type: ignore[return-value]

    return decorator
