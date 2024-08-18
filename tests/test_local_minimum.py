import unittest
import math
from parameterized import parameterized
from minimize_function.local_minimum import (
    scipy_implementation,
    bisection_implementation,
    LocalMinimum,
    Interval,
    ScalarFunction,
)
from scipy import optimize
import numpy as np

def _get_test_parameters():
    return [
        (
            lambda x: (x - 2) * x * (x + 2) ** 2,
            Interval(left=0.0, right=2.0),
            LocalMinimum(argument=1.2807764, function_value=-9.9149496),
        ),
        (
            lambda x: (x - 2) * x * (x + 2) ** 2,
            Interval(left=-5, right=-1.0),
            LocalMinimum(argument=-2.0, function_value=0.0),
        ),
        (
            lambda x: math.sin(x)
            + math.cos(math.sqrt(2 * x))
            + math.sin(math.sqrt(3 * x)),
            Interval(left=2.0, right=6.0),
            LocalMinimum(argument=4.9983516, function_value=-2.6266173),
        ),
        (
            lambda x: x * x,
            Interval(left=-(10**5), right=10**6),
            LocalMinimum(argument=0.0, function_value=0.0),
        ),
        (
            lambda x: x * x,
            Interval(left=-(10**-12), right=10**-12),
            LocalMinimum(argument=0.0, function_value=0.0),
        ),
        (
            lambda x: x * x,
            Interval(left=11, right=11),
            LocalMinimum(argument=11, function_value=121),
        ),
        (
            lambda x: x,
            Interval(left=0.0, right=5),
            LocalMinimum(argument=0.0, function_value=0.0),
        ),
        (
            lambda x: -x,
            Interval(left=-3.0, right=2.0),
            LocalMinimum(argument=2.0, function_value=-2.0),
        ),
    ]


class LocalMinimumTestCase(unittest.TestCase):
    @parameterized.expand(_get_test_parameters())
    def test_one(
        self,
        function_to_minimize: ScalarFunction,
        interval: Interval,
        expected_result: LocalMinimum,
    ):
        tolerance = 10**-8
        for algorithm in scipy_implementation, bisection_implementation:
            with self.subTest(msg=algorithm.__name__):
                result: LocalMinimum = algorithm(
                    function_to_minimize, interval, tolerance
                )
                self.assertAlmostEqual(
                    expected_result.argument, result.argument, delta=10**-6
                )
                self.assertAlmostEqual(
                    expected_result.function_value, result.function_value, delta=10**-6
                )

    def test_two(self):
        interval = Interval(left=12, right=50)
        tolerance = 10**-15
        bisection_implementation_result = optimize.differential_evolution(
            lambda x: np.sin(x) + np.cos(np.sqrt(2 * x)) + np.sin(np.sqrt(3 * x)),
            bounds=[(interval.left, interval.right)],
            tol=tolerance,
        )

        argument = bisection_implementation_result.x[0]
        function_value = bisection_implementation_result.fun

        # Множество локальных минимумов
        self.assertAlmostEqual(42.3977447, argument)
        self.assertAlmostEqual(-2.9369797, function_value)
