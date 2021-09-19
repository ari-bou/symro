import os

from symro.core.prob.problem import Problem
import symro.core.handlers.problembuilder as __problem_builder
from symro.core.execution.amplengine import AMPLEngine
from symro.core.algo.gbd.gbdalgorithm import GBDAlgorithm

__version__ = "0.0.2"

DEFAULT_WORKING_DIR_PATH = os.getcwd()


def read_ampl(file_path: str = None,
              script_literal: str = None,
              name: str = None,
              description: str = None,
              working_dir_path: str = None,
              engine: AMPLEngine = None,
              can_clean_script: bool = False):
    if working_dir_path is None:
        working_dir_path = DEFAULT_WORKING_DIR_PATH
    return __problem_builder.build_problem_from_ampl_script(name=name,
                                                            description=description,
                                                            file_name=file_path,
                                                            script_literal=script_literal,
                                                            working_dir_path=working_dir_path,
                                                            engine=engine,
                                                            can_clean_script=can_clean_script)


def solve_problem(problem: Problem,
                  solver_name: str = None,
                  solver_options: str = None):
    engine = AMPLEngine(problem)
    engine.solve(solver_name=solver_name,
                 solver_options=solver_options)
