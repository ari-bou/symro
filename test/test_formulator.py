import symro
import symro.core.mat as mat
from symro.core.prob.problem import Problem
from symro.core.parsing.amplparser import AMPLParser
import symro.core.handlers.nodebuilder as nb
import symro.core.handlers.formulator as frm
from symro.test.test_util import *


# Scripts
# ----------------------------------------------------------------------------------------------------------------------

SCRIPT = """

set I = 1..3;

var x >= 0, <= 1;
var y{I} >= 0, <= 1;

minimize OBJ: 0;

"""


# Tests
# ----------------------------------------------------------------------------------------------------------------------

def run_formulator_test_group():
    tests = [("Expand expressions", formulator_expansion_test)]
    return run_tests(tests)


def formulator_expansion_test():

    problem = symro.read_ampl(script_literal=SCRIPT,
                              working_dir_path=SCRIPT_DIR_PATH)

    ampl_parser = AMPLParser(problem)

    results = []

    # test 1: x + 1 - 2 + 3 * x + 4 / x
    node = ampl_parser.parse_arithmetic_expression("x + 1 - 2 + 3 * x + 4 / x")
    node = __standardize_expression(problem, node)
    results.append(check_str_result(node, "(x) + (1) + (-2) + (3 * x) + (4 * (1 / x))"))

    # test 2: (1 + x) * (2 + 3 * x)
    node = ampl_parser.parse_arithmetic_expression("(1 + x) * (2 + 3 * x)")
    node = __standardize_expression(problem, node)
    results.append(check_str_result(node, "(1 * 2) + (1 * 3 * x) + (x * 2) + (x * 3 * x)"))

    # test 3: (x^2 + 4 * x + 5) * (6 * x + 7) * (8 + 9 / x)
    node = ampl_parser.parse_arithmetic_expression("(x^2 + 4 * x + 5) * (6 * x + 7) * (8 + 9 / x)")
    node = __standardize_expression(problem, node)
    results.append(check_str_result(
        node,
        "((x) * (x) * 6 * x * 8) + ((x) * (x) * 6 * x * 9 * (1 / x)) + ((x) * (x) * 7 * 8)"
        " + ((x) * (x) * 7 * 9 * (1 / x)) + (4 * x * 6 * x * 8) + (4 * x * 6 * x * 9 * (1 / x)) + (4 * x * 7 * 8)"
        " + (4 * x * 7 * 9 * (1 / x)) + (5 * 6 * x * 8) + (5 * 6 * x * 9 * (1 / x)) + (5 * 7 * 8)"
        " + (5 * 7 * 9 * (1 / x))"
    ))

    # test 4: (x + sum {i in I} 2 * y[i]) * (x^2 + 3)
    node = ampl_parser.parse_arithmetic_expression("(x + sum {i in I} 2 * y[i]) * (x^2 + 3)")
    node = __standardize_expression(problem, node)
    results.append(check_str_result(
        node,
        "(x * (x) * (x)) + (x * 3) + (sum {i in I} (2 * y[i] * (x) * (x))) + (sum {i in I} (2 * y[i] * 3))"
    ))

    # test 5: (x + sum {i in I} 2 * y[i]) * (x * sum {i in I} y[i])
    node = ampl_parser.parse_arithmetic_expression("(x + sum {i in I} 2 * y[i]) * (x * sum {i in I} y[i])")
    node = __standardize_expression(problem, node)
    results.append(check_str_result(
        node,
        "(sum {i in I} (x * x * (y[i]))) + (sum {i in I, i1 in I} (2 * y[i] * x * (y[i1])))"
    ))

    return results


# Utility
# ----------------------------------------------------------------------------------------------------------------------

def __standardize_expression(problem: Problem, node: mat.ArithmeticExpressionNode):
    node = frm.reformulate_subtraction_and_unary_negation(node)
    terms = frm.expand_multiplication(problem, node)
    ref_terms = []
    for term in terms:
        if isinstance(term, mat.ArithmeticOperationNode) and term.operator == mat.MULTIPLICATION_OPERATOR:
            term = frm.combine_summation_factor_nodes(problem, term.operands)
            ref_terms.append(term)
        else:
            ref_terms.append(term)
    return nb.build_addition_node(ref_terms)
