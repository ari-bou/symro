import symro
from symro.core.parsing.amplparser import AMPLParser
from symro.test.test_util import *


def run_ampl_parser_test_group():
    tests = [("Parse expression literals", ampl_parser_core_logic_test)]
    return run_tests(tests)


def ampl_parser_core_logic_test():

    problem = symro.read_ampl(script_literal="",
                              working_dir_path=SCRIPT_DIR_PATH)
    ampl_parser = AMPLParser(problem)

    results = []

    # logical expression
    node = ampl_parser.parse_logical_expression("1 == 1 and 2 + 3 > 1")
    results.append(check_str_result(node, "1 == 1 && 2 + 3 > 1"))

    # 2D set expression
    node = ampl_parser.parse_set_expression("{(1, 'A'), (2, 'B')}")
    results.append(check_str_result(node, "{(1,'A'),(2,'B')}"))

    # conditional arithmetic expression without trailing else clause
    literal = "if 1 > 2 then 1 + 2 else if 1 = 2 then 3 + 4 else if 1 < 2 then 5 + 6"
    node = ampl_parser.parse_arithmetic_expression(literal)
    results.append(check_str_result(node,
                                    ("(sum {1..1: (1 > 2)} (1 + 2))"
                                     + " + (sum {1..1: (! (1 > 2)) && (1 == 2)} (3 + 4))"
                                     + " + (sum {1..1: (! (1 > 2)) && (! (1 == 2)) && (1 < 2)} (5 + 6))")))

    # conditional arithmetic expression with trailing else clause
    literal = "if 1 > 2 then 1 + 2 else if 1 = 2 then 3 + 4 else 5 + 6"
    node = ampl_parser.parse_arithmetic_expression(literal)
    results.append(check_str_result(node,
                                    ("(sum {1..1: (1 > 2)} (1 + 2))"
                                     + " + (sum {1..1: (! (1 > 2)) && (1 == 2)} (3 + 4))"
                                     + " + (sum {1..1: (! (1 > 2)) && (! (1 == 2))} (5 + 6))")))

    return results
