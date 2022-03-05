import symro
from symro.src.parsing.amplparser import AMPLParser
from symro.test.test_util import *


def test_ampl_parser_core_logic():

    problem = symro.read_ampl(script_literal="",
                              working_dir_path=SCRIPT_DIR_PATH)
    ampl_parser = AMPLParser(problem)

    # logical expression
    node = ampl_parser.parse_logical_expression("1 == 1 and 2 + 3 > 1")
    assert str(node) == "1 == 1 && 2 + 3 > 1"

    # 2D set expression
    node = ampl_parser.parse_set_expression("{(1, 'A'), (2, 'B')}")
    assert str(node) == "{(1,'A'),(2,'B')}"

    # conditional arithmetic expression without trailing else clause
    literal = "if 1 > 2 then 1 + 2 else if 1 = 2 then 3 + 4 else if 1 < 2 then 5 + 6"
    node = ampl_parser.parse_arithmetic_expression(literal)
    assert str(node) == "if 1 > 2 then 1 + 2 else if 1 == 2 then 3 + 4 else if 1 < 2 then 5 + 6"

    # conditional arithmetic expression with trailing else clause
    literal = "if 1 > 2 then 1 + 2 else if 1 = 2 then 3 + 4 else 5 + 6"
    node = ampl_parser.parse_arithmetic_expression(literal)
    assert str(node) == "if 1 > 2 then 1 + 2 else if 1 == 2 then 3 + 4 else 5 + 6"
