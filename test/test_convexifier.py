import symro
from symro.core.handlers.convexifier import Convexifier
from symro.test.test_util import *


# Scripts
# ----------------------------------------------------------------------------------------------------------------------

SCRIPT = """

var x >= 0, <= 1;
var y >= 0, <= 1;
var z >= 0, <= 1;

minimize OBJ: 0;

#CON1: x * y <= 0; 

CON2: -x * y <= 0;

"""


# Tests
# ----------------------------------------------------------------------------------------------------------------------

def run_convexifier_test_group():
    tests = [("Convexify problem", convexifier_test)]
    return run_tests(tests)


def convexifier_test():

    problem = symro.read_ampl(script_literal=SCRIPT,
                              working_dir_path=SCRIPT_DIR_PATH)

    convexifier = Convexifier()
    convex_relax = convexifier.convexify(problem)

    results = []

    # test 1: xy
    con = convex_relax.meta_cons["CON2"]
    results.append(check_str_result(
        con.get_expression().root_node,
        " x * y <= 0"
    ))

    symro.model_to_ampl(convex_relax, file_name="convex_relaxation_test.mod")

    return results
