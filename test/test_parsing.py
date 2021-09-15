import symro
from symro.core.parsing.amplparser import AMPLParser
from symro.test.test_util import *


def run_ampl_parser_test_group():
    tests = [("Parse logical expression literal", ampl_parser_logical_expression_test),
             ("Parse 2D enumerated set declaration", ampl_parser_2d_enumerated_set_declaration_test)]
    return run_tests(tests)


def ampl_parser_logical_expression_test():

    problem = symro.read_ampl(script_literal="",
                              working_dir_path=SCRIPT_DIR_PATH)
    ampl_parser = AMPLParser(problem)

    results = []

    node = ampl_parser.parse_logical_expression("1 == 1 and 2 + 3 > 1")
    results.append(check_str_result(node, "1 == 1 and 2 + 3 > 1"))

    return results


def ampl_parser_2d_enumerated_set_declaration_test():

    problem = symro.read_ampl(script_literal="",
                              working_dir_path=SCRIPT_DIR_PATH)
    ampl_parser = AMPLParser(problem)

    results = []

    node = ampl_parser.parse_set_expression("{(1, 'A'), (2, 'B')};")
    results.append(check_str_result(node, "{(1,'A'),(2,'B')}"))

    return results
