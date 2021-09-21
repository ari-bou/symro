from symro.core.prob.problem import Problem

# See the following paper for convexification techniques
# Adjiman, Dallwig, Floudas, and Neumaier
# A global optimization method, αBB, for general twice-differentiable constrained NLPs-I. Theoretical advances

class Convexifier:

    def __init__(self, problem: Problem):
        self.problem: Problem = problem
