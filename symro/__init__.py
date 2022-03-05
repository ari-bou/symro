import os
import pathlib

# allow access to the following members from the symro directory
from symro.mat import *
from symro.prob.problem import Problem
from symro.handlers.problembuilder import read_ampl, build_subproblem
from symro.handlers.scriptbuilder import model_to_ampl
from symro.handlers.convexifier import (
    convexify_problem,
    convexify_expression,
    Convexifier,
)
from symro.execution.amplengine import AMPLEngine
from symro.parsing.amplparser import AMPLParser
from symro.writing.almdatawriter import to_alamo
from symro.algo.gbd.gbdalgorithm import GBDAlgorithm
from symro.algo.ngbd.ngbdalgorithm import NGBDAlgorithm


# The directory containing this file
ROOT_DIR = pathlib.Path(__file__).parent

version_file = open(os.path.join(ROOT_DIR, "VERSION"))
version = version_file.read().strip()
__version__ = version


def solve_problem(problem: Problem, solve_options: str = None):
    engine = AMPLEngine(problem)
    engine.solve(solve_options=solve_options)
