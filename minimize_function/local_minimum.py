from dataclasses import dataclass
from typing import Callable
import math


@dataclass
class Interval:
    left: float
    right: float


@dataclass
class LocalMinimum:
    argument: float
    function_value: float


@dataclass
class ScoreIteration:
    iterationNumber: int
    iterationFunction: int


ScalarFunction = Callable[[float], float]


def scipy_implementation(
    function_to_minimize: ScalarFunction,
    interval: Interval,
    tolerance: float = 10**-8,
) -> LocalMinimum:
    first, third = interval.left, interval.right
    delta = abs((first + third) / 2)
    second = first + delta

    nums = first, second, third
    S1, S2, S3 = list(map(function_to_minimize, nums))
    values = S1, S2, S3
    while True:
        S_k = min(values)
        k = values.index(S_k) + 1

        if delta < tolerance:
            argument = nums[values.index(S_k)]
            function_value = S_k

            decimal_places = -int(math.log10(tolerance)) - 1

            argument = round(argument, decimal_places)
            function_value = round(function_value, decimal_places)

            return LocalMinimum(argument=argument, function_value=function_value)

        delta /= 2

        first, second, third, S1, S2, S3 = update_values(
            k, function_to_minimize, nums, values, delta
        )
        values = S1, S2, S3
        nums = first, second, third


def update_values(k, function_to_minimize, func_args, func_values, delta, iterationFunction):
    first, second, third = func_args
    S1, S2, S3 = func_values

    if k == 1:
        third = second
        S3 = S2
        second = first + delta
        S2 = function_to_minimize(second)
        iterationFunction += 1

    elif k == 3:
        first = second
        S1 = S2
        second = third - delta
        S2 = function_to_minimize(second)
        iterationFunction += 1

    elif k == 2:
        y1 = function_to_minimize(second - delta)
        y2 = function_to_minimize(second + delta)
        iterationFunction += 2

        if y1 < S2:
            return update_values(
                1, function_to_minimize, func_args, func_values, delta, iterationFunction
            )
        elif y2 < S2:
            return update_values(
                3, function_to_minimize, func_args, func_values, delta, iterationFunction
            )
        else:
            first, third = second - delta, second + delta
            S1, S3 = y1, y2

    return first, second, third, S1, S2, S3, iterationFunction


def bisection_implementation(
    function_to_minimize: ScalarFunction,
    interval: Interval,
    tolerance: float = 10**-8,
) -> LocalMinimum:
    iterationNumber, iterationFunction = 0, 0

    first, third = interval.left, interval.right
    delta = abs((first + third) / 2)
    second = first + delta

    nums = first, second, third
    S1, S2, S3 = list(map(function_to_minimize, nums))
    iterationFunction += 3
    values = S1, S2, S3
    S_k, k = 0, 0

    while delta > tolerance:
        S_k = min(values)
        k = values.index(S_k) + 1

        delta /= 2

        first, second, third, S1, S2, S3, iterationFunction = update_values(
            k, function_to_minimize, nums, values, delta, iterationFunction
        )
        values = S1, S2, S3
        nums = first, second, third

        iterationNumber += 1

    argument = nums[values.index(S_k)]
    function_value = S_k

    decimal_places = -int(math.log10(tolerance)) - 1
    argument = round(argument, decimal_places)
    function_value = round(function_value, decimal_places)

    return LocalMinimum(argument=argument, function_value=function_value), ScoreIteration(iterationNumber=iterationNumber, iterationFunction=iterationFunction)

def main():

    print(
        bisection_implementation(
            function_to_minimize=lambda x: math.sin(x)
            + math.cos(math.sqrt(2 * x))
            + math.sin(math.sqrt(3 * x)),
            interval=Interval(left=12, right=50),
        )
    )


if __name__ == "__main__":
    main()
