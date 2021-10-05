from symro.src.prob.problem import Problem
from symro.src.handlers.problembuilder import read_ampl, build_subproblem
from symro.src.handlers.scriptbuilder import model_to_ampl
from symro.src.handlers.convexifier import convexify
from symro.src.execution.amplengine import AMPLEngine
from symro.src.algo.gbd.gbdalgorithm import GBDAlgorithm


def solve_problem(problem: Problem,
                  solve_options: str = None):
    engine = AMPLEngine(problem)
    engine.solve(solve_options=solve_options)
