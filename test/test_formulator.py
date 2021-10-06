import symro
import symro.src.mat as mat
from symro.src.prob.problem import Problem
from symro.src.parsing.amplparser import AMPLParser
import symro.src.handlers.nodebuilder as nb
import symro.src.handlers.formulator as frm
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

    """# test 1
    literal = "x + 1 - 2 + 3 * x + 4 / x"
    node = ampl_parser.parse_arithmetic_expression(literal)
    node = __standardize_expression(problem, node)
    results.append(check_str_result(node, "(x) + (1) + (-2) + (3 * x) + (4 * (1 / x))"))

    # test 2
    literal = "(1 + x) * (2 + 3 * x)"
    node = ampl_parser.parse_arithmetic_expression(literal)
    node = __standardize_expression(problem, node)
    results.append(check_str_result(node, "(1 * 2) + (1 * 3 * x) + (x * 2) + (x * 3 * x)"))

    # test 3:
    literal = "(x^2 + 4 * x + 5) * (6 * x + 7) * (8 + 9 / x)"
    node = ampl_parser.parse_arithmetic_expression(literal)
    node = __standardize_expression(problem, node)
    results.append(check_str_result(
        node,
        "((x) * (x) * 6 * x * 8) + ((x) * (x) * 6 * x * 9 * (1 / x)) + ((x) * (x) * 7 * 8)"
        " + ((x) * (x) * 7 * 9 * (1 / x)) + (4 * x * 6 * x * 8) + (4 * x * 6 * x * 9 * (1 / x)) + (4 * x * 7 * 8)"
        " + (4 * x * 7 * 9 * (1 / x)) + (5 * 6 * x * 8) + (5 * 6 * x * 9 * (1 / x)) + (5 * 7 * 8)"
        " + (5 * 7 * 9 * (1 / x))"
    ))

    # test 4
    literal = "(x + sum {i in I} 2 * y[i]) * (x^2 + 3)"
    node = ampl_parser.parse_arithmetic_expression(literal)
    node = __standardize_expression(problem, node)
    results.append(check_str_result(
        node,
        "(x * (x) * (x)) + (x * 3) + (sum {i in I} (2 * y[i] * (x) * (x))) + (sum {i in I} (2 * y[i] * 3))"
    ))

    # test 5
    literal = "(x + sum {i in I} 2 * y[i]) * (x * sum {i in I} y[i])"
    node = ampl_parser.parse_arithmetic_expression(literal)
    node = __standardize_expression(problem, node)
    results.append(check_str_result(
        node,
        "(sum {i in I} (x * x * (y[i]))) + (sum {i in I, i1 in I} (2 * y[i] * x * (y[i1])))"
    ))"""

    # test 6
    literal = "x * (if 1 < 2 then sum {i in I} y[i] else sum {i in I} y[i] ^ 2)"
    node = ampl_parser.parse_arithmetic_expression(literal)
    node = __standardize_expression(problem, node)
    results.append(check_str_result(
        node,
        "(sum {i in I: (1 < 2)} (x * (y[i]))) + (sum {i in I: (! (1 < 2))} (x * (y[i]) * (y[i])))"
    ))

    # test 7
    literal = "(if 1 < 2 then x else 5) * (sum {i in I} y[i] + 10)"
    node = ampl_parser.parse_arithmetic_expression(literal)
    node = __standardize_expression(problem, node)
    results.append(check_str_result(
        node,
        "(sum {i in I: (1 < 2)} ((x) * (y[i]))) + (if (1 < 2) then ((x) * 10))"
        + " + (sum {i in I: (! (1 < 2))} ((5) * (y[i]))) + (if (! (1 < 2)) then ((5) * 10))"
    ))

    return results


# Utility
# ----------------------------------------------------------------------------------------------------------------------

def __standardize_expression(problem: Problem, node: mat.ArithmeticExpressionNode):
    node = frm.reformulate_arithmetic_conditional_expressions(node)
    node = frm.reformulate_subtraction_and_unary_negation(node)
    terms = frm.expand_multiplication(problem, node)
    ref_terms = []
    for term in terms:
        if isinstance(term, mat.ArithmeticOperationNode) and term.operator == mat.MULTIPLICATION_OPERATOR:
            term = frm.combine_arithmetic_reduction_nodes(problem, term)
            ref_terms.append(term)
        else:
            ref_terms.append(term)
    return nb.build_addition_node(ref_terms)
