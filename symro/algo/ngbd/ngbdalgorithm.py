from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from symro.prob.problem import Problem

import symro.mat as mat

from symro.execution.engine import Engine
from symro.execution.amplengine import AMPLEngine

from symro.algo.gbd import *
from symro.algo.ngbd.ngbdproblem import NGBDProblem
from symro.algo.ngbd.ngbdproblembuilder import NGBDProblemBuilder


class NGBDAlgorithm(GBDAlgorithm):
    def __init__(
        self,
        problem: Problem,
        convex_relaxation: Problem,
        complicating_vars: List[str],
        mp_symbol: str = None,
        primal_sp_symbol: str = None,
        fbl_sp_symbol: str = None,
        primal_sp_obj_symbol: str = None,
        init_lb: float = -np.inf,
        init_ub: float = np.inf,
        before_mp_solve: Callable[[GBDProblem, int], None] = None,
        before_c_sp_solve: Callable[[GBDProblem, int, str, mat.Element], None] = None,
        before_nc_sp_solve: Callable[[GBDProblem, int, str, mat.Element], None] = None,
        working_dir_path: str = None,
    ):

        self._ngbd_problem_builder: Optional[NGBDProblemBuilder] = None

        super().__init__(
            problem=convex_relaxation,
            complicating_vars=complicating_vars,
            mp_symbol=mp_symbol,
            primal_sp_symbol=primal_sp_symbol,
            fbl_sp_symbol=fbl_sp_symbol,
            primal_sp_obj_symbol=primal_sp_obj_symbol,
            init_lb=init_lb,
            init_ub=init_ub,
            before_mp_solve=before_mp_solve,
            before_sp_solve=before_c_sp_solve,
            working_dir_path=working_dir_path,
        )

        # --- Handlers ---
        self._nc_engine: Optional[Engine] = None

        # --- Algorithmic Constructs ---
        self.algorithm_name = "NGBD"
        self.ngbd_problem: NGBDProblem = self.gbd_problem

        # --- Options ---
        self.nc_sp_solve_options: Optional[Dict[str, Union[int, float, str]]] = None

        # --- Callables ---
        self.before_nc_sp_solve: Callable[
            [GBDProblem, int, str, mat.Element], None
        ] = before_nc_sp_solve

        # --- Problem Construction ---
        self._ngbd_problem_builder.build_and_initialize_original_problem(problem)

    def _build_gbd_problem_builder(self):
        self._ngbd_problem_builder = NGBDProblemBuilder()
        return self._ngbd_problem_builder

    # Run
    # ------------------------------------------------------------------------------------------------------------------

    def run(
        self,
        mp_solve_options: Dict[str, Union[int, float, str]] = None,
        c_sp_solve_options: Dict[str, Union[int, float, str]] = None,
        nc_sp_solve_options: Dict[str, Union[int, float, str]] = None,
        max_iter_count: int = 100,
        abs_opt_tol: float = None,
        rel_opt_tol: float = 0.01,
        fbl_tol: float = 0.001,
        verbosity: int = 0,
        can_catch_exceptions: bool = False,
        can_write_log: bool = True,
    ):

        self.nc_sp_solve_options = nc_sp_solve_options

        return super().run(
            mp_solve_options=mp_solve_options,
            sp_solve_options=c_sp_solve_options,
            max_iter_count=max_iter_count,
            abs_opt_tol=abs_opt_tol,
            rel_opt_tol=rel_opt_tol,
            fbl_tol=fbl_tol,
            verbosity=verbosity,
            can_catch_exceptions=can_catch_exceptions,
            can_write_log=can_write_log,
        )

    def _setup_engine(self):
        super()._setup_engine()
        self._nc_engine = AMPLEngine(self.ngbd_problem.origin_problem)

    # Algorithm
    # ------------------------------------------------------------------------------------------------------------------

    def _run_loop(
        self,
    ) -> Tuple[bool, float, Dict[str, Union[float, Dict[tuple, float]]]]:

        is_feasible = False
        ub_nc = np.inf
        y_ub = {}
        can_iterate = True

        self._reset_cut_count()

        while can_iterate:

            is_feasible_i, self.ub, y_ub = super()._run_loop()

            if not is_feasible_i:
                break

            self._log_indexed_message(
                "Validating solution of convex relaxation by solving the original problem",
                iter=self.it,
            )

            is_feasible_nc_i, ub_nc_i = self.__solve_nc_primal_problem(y_ub)

            if is_feasible_nc_i:

                is_feasible = True

                if ub_nc_i < ub_nc:
                    ub_nc = ub_nc_i

                # termination criteria
                can_iterate = self._check_outer_loop_termination_criteria(ub_nc=ub_nc)

            if can_iterate:

                # reset upper bound of convex master problem
                self.ub = self.init_ub

                # increase lower bound of the master problem in case y is not pure binary
                if not self.ngbd_problem.is_y_binary:
                    self.__update_lower_bound()

        if not is_feasible:  # no feasible solution was found
            ub_nc = None

        return is_feasible, ub_nc, y_ub

    def _check_outer_loop_termination_criteria(self, ub_nc: float) -> bool:

        if self.it >= self.max_iter_count:  # iteration limit reached
            self._log_indexed_message("iteration limit", self.it)
            return False

        if self.lb >= ub_nc:  # lower bound surpassed upper bound
            self._log_indexed_message("lower bound >= upper bound", self.it)
            return False

        epsilon_rel = self._calculate_relative_error(lb=self.lb, ub=ub_nc)
        if epsilon_rel <= self.rel_opt_tol:  # relative error within tolerance
            message = (
                f"NGBD epsilon-optimal solution with relative error = {epsilon_rel}"
            )
            self._log_indexed_message(message, self.it)
            return False

        if self.abs_opt_tol is not None:
            epsilon_abs = self._calculate_absolute_error(lb=self.lb, ub=ub_nc)
            if epsilon_abs <= self.abs_opt_tol:  # absolute error within tolerance
                message = (
                    f"NGBD epsilon-optimal solution with absolute error = {epsilon_abs}"
                )
                self._log_indexed_message(message, self.it)
                return False

        return True

    def __solve_nc_primal_problem(self, y: EntityValueCollection):

        is_feasible_i = True
        ub_nc_i = 0

        for nc_sp_container in self.ngbd_problem.nc_sp_containers:

            is_feasible_i_sp, v_sp = self.__solve_nc_primal_sp(nc_sp_container, y)

            # y is feasible for the original subproblem
            if is_feasible_i_sp:
                ub_nc_i += v_sp

            # y is not feasible for the original subproblem
            else:
                is_feasible_i = False
                ub_nc_i = None
                break

        return is_feasible_i, ub_nc_i

    def __solve_nc_primal_sp(
        self, nc_sp_container: GBDSubproblemContainer, y: EntityValueCollection
    ) -> Tuple[bool, float]:

        sp_sym = nc_sp_container.primal_sp.symbol
        sp_idx = nc_sp_container.sp_index

        self._nc_engine.set_active_problem(problem_symbol=sp_sym, problem_idx=sp_idx)

        self._log_indexed_message(
            "Fixing complicating variables", iter=self.it, sp_sym=sp_sym, sp_idx=sp_idx
        )
        self.__fix_origin_complicating_variables(
            engine=self._nc_engine, sp_container=nc_sp_container, y=y
        )

        if self.before_nc_sp_solve is not None:
            self.before_nc_sp_solve(self.gbd_problem, self.it, sp_sym, sp_idx)

        self._log_indexed_message(
            "Solving original primal subproblem",
            iter=self.it,
            sp_sym=sp_sym,
            sp_idx=sp_idx,
        )

        self._nc_engine.solve(solve_options=self.nc_sp_solve_options)
        solver_output = self._nc_engine.get_solver_output()
        self._print_solver_output(solver_output)

        if not self._interpret_solver_result(
            solver_output, sp_sym=sp_sym, sp_index=sp_idx
        ):
            return False, 0

        v_sp = self._get_sp_obj_value(
            engine=self._nc_engine,
            obj=nc_sp_container.get_primal_meta_obj(),
            sp_index=sp_idx,
        )
        self._log_indexed_message(
            "Original subproblem objective is {0}".format(v_sp),
            iter=self.it,
            sp_sym=sp_sym,
            sp_idx=sp_idx,
        )
        return True, v_sp

    def __fix_origin_complicating_variables(
        self,
        engine: Engine,
        sp_container: GBDSubproblemContainer,
        y: EntityValueCollection,
    ):
        for var_sym, idx_set in sp_container.comp_var_idx_sets.items():

            # scalar variable
            if self.gbd_problem.meta_vars[var_sym].idx_set_dim == 0:
                value: EntityValue = y[var_sym]
                engine.fix_var(symbol=var_sym, value=value)

            # indexed variable
            else:
                for var_index in idx_set:
                    value: EntityValue = y[var_sym][var_index]
                    engine.fix_var(symbol=var_sym, idx=var_index, value=value)

    def __update_lower_bound(self):
        self.lb += 1  # TODO: make this into a tunable parameter
        self._engine.set_param_value(self.gbd_problem.eta_lb.symbol, None, self.lb)
