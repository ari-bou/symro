import os

from symro.core.prob.problem import Problem
import symro.core.handlers.problembuilder as __problem_builder
from symro.core.algo.gbd.gbdalgorithm import GBDAlgorithm

__version__ = "0.0.1"

DEFAULT_WORKING_DIR_PATH = os.getcwd()


def build_problem(file_path: str = None,
                  script_literal: str = None,
                  name: str = None,
                  description: str = None,
                  working_dir_path: str = None):
    if working_dir_path is None:
        working_dir_path = DEFAULT_WORKING_DIR_PATH
    return __problem_builder.build_problem_from_ampl_script(name=name,
                                                            description=description,
                                                            file_name=file_path,
                                                            script_literal=script_literal,
                                                            working_dir_path=working_dir_path)


def solve_problem(problem: Problem,
                  solver_name: str = None,
                  solver_options: str = None):
    problem.engine.solve(solver_name=solver_name,
                         solver_options=solver_options)