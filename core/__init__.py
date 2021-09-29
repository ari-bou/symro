from symro.core.prob.problem import Problem
from symro.core.handlers.problembuilder import read_ampl
from symro.core.handlers.scriptbuilder import model_to_ampl
from symro.core.execution.amplengine import AMPLEngine
from symro.core.algo.gbd.gbdalgorithm import GBDAlgorithm

__version__ = "0.0.2"


def solve_problem(problem: Problem,
                  solver_name: str = None,
                  solver_options: str = None):
    engine = AMPLEngine(problem)
    engine.solve(solver_name=solver_name,
                 solver_options=solver_options)
