import symro.src.mat as mat
from symro.src.prob.problem import BaseProblem, Problem
from symro.src.execution.engine import Engine


def find_lower_bound(engine: Engine,
                     problem: Problem,
                     var_sym: str,
                     var_idx: mat.Element = None,
                     sp: BaseProblem = None):
    pass


def __find_bound(is_lower: bool,
                 engine: Engine,
                 problem: Problem,
                 var_sym: str,
                 var_idx: mat.Element = None,
                 sp: BaseProblem = None):
    pass
