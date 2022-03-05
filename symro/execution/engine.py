from abc import ABC, abstractmethod
from typing import Dict, Optional, Union

import symro.mat as mat
from symro.prob.problem import Problem


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

    def set_active_problem(
        self, problem_symbol: str = None, problem_idx: mat.Element = None
    ):
        self._active_problem_sym = problem_symbol
        self._active_problem_idx = problem_idx

    @abstractmethod
    def fix_var(
        self,
        symbol: str,
        idx: Optional[mat.Element] = None,
        value: Union[int, float, str] = None,
    ):
        pass

    # Solve
    # ------------------------------------------------------------------------------------------------------------------

    @abstractmethod
    def solve(
        self,
        problem_symbol: str = None,
        problem_idx: mat.Element = None,
        obj_symbol: str = None,
        solve_options: Dict[str, Union[int, float, str]] = None,
    ):
        pass

    @abstractmethod
    def get_status(self) -> str:
        pass

    @abstractmethod
    def get_solver_output(self) -> str:
        pass

    # Storage
    # ------------------------------------------------------------------------------------------------------------------

    def _store_var(
        self,
        symbol: str,
        idx: Optional[mat.Element],
        value: Union[int, float],
        lb: Union[int, float],
        ub: Union[int, float],
    ):

        if self.problem.state.entity_exists(symbol=symbol, idx=idx):
            var = self.problem.state.get_variable(symbol=symbol, idx=idx)
            var.value = value
            var.ub = ub
            var.lb = lb

        else:
            self.problem.state.add_variable(
                symbol=symbol, idx=idx, value=value, lb=lb, ub=ub
            )

    def _store_obj(
        self, symbol: str, idx: Optional[mat.Element], value: Union[int, float]
    ):

        if self.problem.state.entity_exists(symbol=symbol, idx=idx):
            obj = self.problem.state.get_objective(symbol=symbol, idx=idx)
            obj.value = value

        else:
            self.problem.state.add_objective(symbol=symbol, idx=idx, value=value)

    def _store_con(
        self,
        symbol: str,
        idx: Optional[mat.Element],
        body: Union[int, float],
        lb: Union[int, float],
        ub: Union[int, float],
        dual: Union[int, float],
    ):

        if self.problem.state.entity_exists(symbol=symbol, idx=idx):
            con = self.problem.state.get_constraint(symbol=symbol, idx=idx)
            con.body = body
            con.ub = ub
            con.lb = lb
            con.dual = dual

        else:
            self.problem.state.add_constraint(
                symbol=symbol, idx=idx, body=body, lb=lb, ub=ub, dual=dual
            )

    # Accessors and Mutators
    # ------------------------------------------------------------------------------------------------------------------

    @abstractmethod
    def get_param_value(self, symbol: str, idx: Optional[mat.Element] = None):
        pass

    @abstractmethod
    def get_var_value(self, symbol: str, idx: Optional[mat.Element] = None):
        pass

    @abstractmethod
    def get_obj_value(self, symbol: str, idx: Optional[mat.Element] = None):
        pass

    @abstractmethod
    def get_con_value(self, symbol: str, idx: Optional[mat.Element] = None):
        pass

    @abstractmethod
    def get_con_dual(self, symbol: str, idx: Optional[mat.Element] = None) -> float:
        pass

    @abstractmethod
    def set_param_value(
        self, symbol: str, idx: Optional[mat.Element], value: Union[int, float, str]
    ):
        pass

    @abstractmethod
    def set_var_value(self, symbol: str, idx: Optional[mat.Element], value: float):
        pass
