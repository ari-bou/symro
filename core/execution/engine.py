from abc import ABC, abstractmethod
from typing import Optional, Union

import symro.core.mat as mat
from symro.core.prob.problem import Problem


class Engine(ABC):

    def __init__(self, problem: Problem = None):

        # Problem
        self.problem: Problem = problem
        self._active_problem_sym: Optional[str] = None
        self._active_problem_idx: Optional[mat.Element] = None

        # Flags
        self.can_store_soln: bool = True

        # Miscellaneous
        self._solver_output: str = ""

    # Modelling
    # ------------------------------------------------------------------------------------------------------------------

    def set_active_problem(self,
                           problem_symbol: str = None,
                           problem_idx: mat.Element = None):
        self._active_problem_sym = problem_symbol
        self._active_problem_idx = problem_idx

    # Solve
    # ------------------------------------------------------------------------------------------------------------------

    @abstractmethod
    def solve(self,
              solver_name: str = None,
              solver_options: str = None):
        pass

    # Storage
    # ------------------------------------------------------------------------------------------------------------------

    def _store_var(self,
                   symbol: str,
                   idx: Optional[mat.Element],
                   value: Union[int, float],
                   lb: Union[int, float],
                   ub: Union[int, float]):

        if self.problem.state.entity_exists(symbol=symbol, idx=idx):
            var = self.problem.state.get_entity(symbol=symbol, idx=idx)
            var.value = value
            var.ub = ub
            var.lb = lb

        else:
            var = mat.Variable(symbol=symbol,
                               idx=idx,
                               value=value,
                               lb=lb,
                               ub=ub)
            self.problem.state.add_variable(var)

    def _store_obj(self,
                   symbol: str,
                   idx: Optional[mat.Element],
                   value: Union[int, float]):

        if self.problem.state.entity_exists(symbol=symbol, idx=idx):
            obj = self.problem.state.get_entity(symbol=symbol, idx=idx)
            obj.value = value

        else:
            obj = mat.Objective(symbol=symbol,
                                idx=idx,
                                value=value)
            self.problem.state.add_objective(obj)

    def _store_con(self,
                   symbol: str,
                   idx: Optional[mat.Element],
                   value: Union[int, float],
                   lb: Union[int, float],
                   ub: Union[int, float],
                   dual: Union[int, float]):

        if self.problem.state.entity_exists(symbol=symbol, idx=idx):
            con = self.problem.state.get_entity(symbol=symbol, idx=idx)
            con.value = value
            con.ub = ub
            con.lb = lb
            con.dual = dual

        else:
            con = mat.Constraint(symbol=symbol,
                                 idx=idx,
                                 value=value,
                                 lb=lb,
                                 ub=ub,
                                 dual=dual)
            self.problem.state.add_constraint(con)
