from dataclasses import dataclass
from typing import Callable


@dataclass
class Interval:
    left: float
    right: float


@dataclass
class LocalMinimum:
    argument: float
    function_value: float


ScalarFunction = Callable[[float], float]


def scipy_implementation(
        function_to_minimize: ScalarFunction,
        interval: Interval,
        tolerance: float = 10 ** -8,
) -> LocalMinimum:
    return LocalMinimum(argument=0.1, function_value=0.1)


def bisection_implementation(
        function_to_minimize: ScalarFunction,
        interval: Interval,
        tolerance: float = 10 ** -8,
) -> LocalMinimum:
    return LocalMinimum(argument=0.1, function_value=0.1)
