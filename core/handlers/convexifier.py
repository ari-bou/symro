from symro.core.prob.problem import Problem


class Convexifier:

    def __init__(self, problem: Problem):
        self.problem: Problem = problem
