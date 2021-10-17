import symro
from symro.src.handlers.convexifier import Convexifier
from symro.test.test_util import *


# Scripts
# ----------------------------------------------------------------------------------------------------------------------

SCRIPT = """

set I = {1, 2, 3};
set J = {4, 5, 6};
set K = {7, 8, 9};
set R = {'i', 'ii', 'iii'};
set S = {'iv', 'v', 'vi'};
set T = {'vii', 'viii', 'ix'};

var x >= 2, <= 10;
var y >= 2, <= 10;
var z >= 2, <= 10;

var a {I, R} >= 5, <= 15;
var b {J, S} >= 5, <= 15;
var c {K, T} >= 5, <= 15;

minimize OBJ: x;

#CON1: x * y = 0; 
#CON2: -x * y <= 0;
#CON3: x * y * z <= 0;
#CON4: -x * y * z <= 0;
#CON5: x / y <= 0;
#CON6: -x / y <= 0;
#CON7: x * y / z <= 0;
#CON8: -x * y / z <= 0;
#CON9: log(x) = 0;
#CON10: x^2 = 0;

#CON11{i in I}: sum {r in R, t in T, s in S} a[i,r] * sum {j in J} b[j,s] * sum {k in K} c[k,t] <= 0; 
CON12{i in I}: -(sum {r in R, s in S} a[i,r] * sum {j in J} b[j,s]) * sum {r in R} a[2,r] <= 0; 

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

    symro.model_to_ampl(convex_relax, file_name="convex_relaxation_test.mod")

    return []
