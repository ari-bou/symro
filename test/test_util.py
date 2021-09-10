import os
from typing import Callable, Iterable, List, Tuple, Union
import warnings

SCRIPT_DIR_PATH = os.path.join(os.getcwd(), "scripts")


def run_tests(tests: Iterable[Tuple[str, Callable[[], Union[bool, List[bool]]]]]):

    total_count = 0
    true_count = 0

    def run_test(header: str, test: Callable[[], Union[bool, List[bool]]]):

        print("\n" + header)
        print('-' * 50)

        result = test()
        if not isinstance(result, list):
            result = [result]

        nonlocal total_count
        nonlocal true_count
        total_count_t = 0
        true_count_t = 0

        for r in result:
            total_count_t += 1
            true_count_t += 1 if r else 0

        total_count += total_count_t
        true_count += true_count_t

        print("{0}/{1} test{2} conducted successfully".format(true_count_t,
                                                              total_count_t,
                                                              "s" if total_count_t != 1 else ''))

    for h, t in tests:
        run_test(h, t)

    return true_count, total_count


def run_test_groups(test_groups: Iterable[Callable[[], Tuple[int, int]]]):

    true_count = 0
    total_count = 0

    for test_group in test_groups:
        true_count_g, total_count_g = test_group()
        true_count += true_count_g
        total_count += total_count_g

    return true_count, total_count


def check_str_result(actual_result, expected_result) -> bool:
    if str(actual_result) == str(expected_result):
        print("Correct result: {0}".format(actual_result))
        return True
    else:
        warnings.warn("Incorrect result: {0} \nExpected result: {1}".format(actual_result, expected_result))
        return False


def check_num_result(actual_result: Union[int, float],
                     expected_result: Union[int, float],
                     tol: float = 0.01) -> bool:
    if abs(actual_result - expected_result) <= tol:
        print("Correct result: {0}".format(actual_result))
        return True
    else:
        warnings.warn("Incorrect result: {0} \nExpected result: {1}".format(actual_result, expected_result))
        return False
