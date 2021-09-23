from symro.core.prob.problem import Problem
from symro.core.handlers.formulator import Formulator

# See the following paper for convexification techniques
# Adjiman, Dallwig, Floudas, and Neumaier
# A global optimization method, αBB, for general twice-differentiable constrained NLPs-I. Theoretical advances


def convexify(problem: Problem,
              problem_symbol: str = None,
              description: str = None,
              working_dir_path: str = None):

    if problem_symbol is None:
        problem_symbol = problem.symbol
    problem_symbol = problem.generate_unique_symbol(base_symbol=problem_symbol)

    if description is None:
        description = "Convex relaxation of problem {0}".format(problem.symbol)

    if working_dir_path is None:
        working_dir_path = problem.working_dir_path

    convex_relaxation = Problem(symbol=problem_symbol,
                                description=description,
                                working_dir_path=working_dir_path)
    Problem.deepcopy(problem, convex_relaxation)

    formulator = Formulator(convex_relaxation)

    formulator.substitute_defined_variables()

