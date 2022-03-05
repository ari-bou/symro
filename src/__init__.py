# allow access to the following members from the stc directory
from symro.src.mat import *
from symro.src.prob.problem import Problem
from symro.src.handlers.problembuilder import read_ampl, build_subproblem
from symro.src.handlers.scriptbuilder import model_to_ampl
from symro.src.handlers.convexifier import convexify_problem, convexify_expression, Convexifier
from symro.src.execution.amplengine import AMPLEngine
from symro.src.parsing.amplparser import AMPLParser
from symro.src.writing.almdatawriter import to_alamo
from symro.src.algo.gbd.gbdalgorithm import GBDAlgorithm
from symro.src.algo.ngbd.ngbdalgorithm import NGBDAlgorithm


def solve_problem(problem: Problem,
                  solve_options: str = None):
    engine = AMPLEngine(problem)
    engine.solve(solve_options=solve_options)
