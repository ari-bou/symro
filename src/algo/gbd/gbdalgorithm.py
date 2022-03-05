from datetime import datetime
import numpy as np
import pprint as pprint
from typing import Callable, Dict, List, Optional, Tuple, Union

import symro.src.mat as mat
from symro.src.prob.problem import Problem

from symro.src.execution.engine import Engine
from symro.src.execution.amplengine import AMPLEngine

from symro.src.parsing import outputparser

from symro.src.algo.gbd import GBDProblem, GBDSubproblemContainer, GBDProblemBuilder

import symro.src.util.util as util


EntityValue = float
IndexedEntityValue = Dict[mat.Element, EntityValue]
EntityValueCollection = Dict[str, Union[EntityValue, IndexedEntityValue]]


class GBDAlgorithm:

    # Construction
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self,
                 problem: Problem,
                 complicating_vars: List[str],
                 mp_symbol: str = None,
                 primal_sp_symbol: str = None,
                 fbl_sp_symbol: str = None,
                 primal_sp_obj_symbol: str = None,
                 init_lb: float = -np.inf,
                 init_ub: float = np.inf,
                 before_mp_solve: Callable[[GBDProblem, int], None] = None,
                 before_sp_solve: Callable[[GBDProblem, int, str, mat.Element], None] = None,
                 working_dir_path: str = None):

        # --- Problem ---
        self.gbd_problem: Optional[GBDProblem] = None

        # --- Handlers ---
        self._gbd_problem_builder: GBDProblemBuilder = self._build_gbd_problem_builder()
        self._engine: Optional[Engine] = None

        # --- Algorithmic Constructs ---

        self.algorithm_name: str = "GBD"

        self.it: int = 0

        self.init_lb: float = init_lb
        self.init_ub: float = init_ub
        self.lb: float = init_lb
        self.ub: float = init_ub

        self.log: str = ""

        # --- Options ---

        self.mp_solve_options: Optional[Dict[str, Union[int, float, str]]] = None
        self.sp_solve_options: Optional[Dict[str, Union[int, float, str]]] = None

        self.max_iter_count: int = 100
        self.abs_opt_tol: Optional[float] = None
        self.rel_opt_tol: float = 0.01
        self.fbl_tol: float = 0.001

        self.verbosity: int = 0
        self.can_catch_exceptions: bool = False

        # --- Callables ---
        self.before_mp_solve: Callable[[GBDProblem, int], None] = before_mp_solve
        self.before_sp_solve: Callable[[GBDProblem, int, str, mat.Element], None] = before_sp_solve

        # --- Problem Construction ---
        self.gbd_problem = self._gbd_problem_builder.build_and_initialize_gbd_problem(
            problem=problem,
            comp_var_defs=complicating_vars,
            mp_symbol=mp_symbol,
            primal_sp_symbol=primal_sp_symbol,
            fbl_sp_symbol=fbl_sp_symbol,
            primal_sp_obj_symbol=primal_sp_obj_symbol,
            init_lb=init_lb,
            init_ub=init_ub,
            working_dir_path=working_dir_path
        )

    def _build_gbd_problem_builder(self):
        return GBDProblemBuilder()

    # Setup
    # ------------------------------------------------------------------------------------------------------------------

    def add_decomposition_subproblem(self,
                                     sp_symbol: str,
                                     entity_defs: List[str] = None,
                                     linked_entity_defs: List[str] = None):
        self._gbd_problem_builder.build_or_retrieve_defined_primal_sp(sp_sym=sp_symbol,
                                                                      entity_defs=entity_defs,
                                                                      linked_entity_defs=linked_entity_defs)

    def add_decomposition_axes(self, idx_set_defs: List[str]):
        self._gbd_problem_builder.add_decomposition_axes(idx_set_defs)

    def setup(self):
        self._gbd_problem_builder.build_gbd_constructs()

    # Run
    # ------------------------------------------------------------------------------------------------------------------

    def run(self,
            mp_solve_options: Dict[str, Union[int, float, str]] = None,
            sp_solve_options: Dict[str, Union[int, float, str]] = None,
            max_iter_count: int = 100,
            abs_opt_tol: float = None,
            rel_opt_tol: float = 0.01,
            fbl_tol: float = 0.001,
            verbosity: int = 0,
            can_catch_exceptions: bool = False,
            can_write_log: bool = True):

        # --- Algorithm Setup --
        self.it = 0
        self.lb = self.init_lb
        self.ub = self.init_ub

        # --- Options ---

        self.mp_solve_options: Dict[str, Union[int, float, str]] = mp_solve_options
        self.sp_solve_options: Dict[str, Union[int, float, str]] = sp_solve_options

        self.max_iter_count: int = max_iter_count
        self.abs_opt_tol: float = abs_opt_tol
        self.rel_opt_tol: float = rel_opt_tol
        self.fbl_tol: float = fbl_tol

        self.verbosity: int = verbosity
        self.can_catch_exceptions: bool = can_catch_exceptions
        self.log: str = ""

        # --- Execution ---

        self._setup_engine()

        self.__log_message(f"Running {self.algorithm_name}")

        try:
            v_ub, y = self._run_algorithm()
        except Exception as e:
            v_ub = None
            y = None
            if self.can_catch_exceptions:
                self.__log_message(str(e))
            else:
                raise e

        if can_write_log:
            self.__write_log_file()

        return v_ub, y

    def _setup_engine(self):
        self.__log_message("Setting up engine")
        self._engine = AMPLEngine(self.gbd_problem)
        self._engine.can_store_soln = False

    def _run_algorithm(self) -> Tuple[float, EntityValueCollection]:

        start_time = datetime.now()
        self._reset_cut_count()
        is_feasible, v_ub, y = self._run_loop()
        sol_time = datetime.now() - start_time

        self.__print_solution(is_feasible, v_ub, y, sol_time.total_seconds())

        if is_feasible:
            self.__log_message(f"{self.algorithm_name} algorithm terminated with feasible solution")
        else:
            self.__log_message(f"{self.algorithm_name} algorithm terminated with infeasible solution")

        return v_ub, y

    # Algorithm
    # ------------------------------------------------------------------------------------------------------------------

    def _run_loop(self) -> Tuple[bool,
                                 float,
                                 Dict[str, Union[float, Dict[tuple, float]]]]:

        is_feasible = False
        ub_i = 0
        y_ub = {}
        dual_soln = {}
        can_iterate = True

        while can_iterate:

            is_it_feasible = False

            # set master problem as active problem
            self._engine.set_active_problem(self.gbd_problem.mp.symbol)

            # solve master problem
            can_iterate, lb = self.__solve_mp(self.it)

            if can_iterate:

                # update lower bound
                if lb > self.lb:
                    self.lb = lb
                    self._log_indexed_message("Updated global lower bound", self.it)

                # increment cut count
                self.__increment_cut_count()

                # store complicating variables
                self._log_indexed_message("Storing complicating variables", self.it)
                y_i = self.__store_complicating_variables()

                self.__set_is_feasible_flag(True)  # set default value of feasibility flag
                self.__set_stored_obj_value(0)  # set default value of the objective

                # solve primal subproblems
                is_it_feasible, ub_i = self.__solve_primal_problem(dual_soln=dual_soln)
                self._log_indexed_message("Primal objective is {0}".format(ub_i), self.it)

                self.__set_stored_obj_value(ub_i)  # store upper bound of current iteration

                self._log_indexed_message("Storing dual multipliers", self.it)
                self.__set_duality_multipliers(dual_soln)  # set duality multipliers

                # update global upper bound
                if ub_i < self.ub and is_it_feasible:
                    self.ub = ub_i
                    y_ub = y_i
                    is_feasible = True
                    self._log_indexed_message("Updated global upper bound", self.it)

            # output
            self.__print_iteration_output(self.it,
                                          self.lb,
                                          self.ub,
                                          is_it_feasible)

            self.it += 1  # increment iteration counter

            # termination criteria
            can_iterate = self._check_termination_criteria()

        if not is_feasible:  # no feasible solution was found
            self.ub = ub_i

        return is_feasible, self.ub, y_ub

    def _check_termination_criteria(self) -> bool:

        if self.it >= self.max_iter_count:  # iteration limit reached
            self._log_indexed_message("iteration limit", self.it)
            return False

        if self.lb >= self.ub:  # lower bound surpassed upper bound
            self._log_indexed_message("lower bound >= upper bound", self.it)
            return False

        epsilon_rel = self._calculate_relative_error(lb=self.lb, ub=self.ub)
        if epsilon_rel <= self.rel_opt_tol:  # relative error within tolerance
            message = f"GBD epsilon-optimal solution with relative error = {epsilon_rel}"
            self._log_indexed_message(message, self.it)
            return False

        if self.abs_opt_tol is not None:
            epsilon_abs = self._calculate_absolute_error(lb=self.lb, ub=self.ub)
            if epsilon_abs <= self.abs_opt_tol:  # absolute error within tolerance
                message = f"GBD epsilon-optimal solution with absolute error = {epsilon_abs}"
                self._log_indexed_message(message, self.it)
                return False

        return True

    @staticmethod
    def _calculate_absolute_error(lb: float, ub: float):
        return ub - lb

    @staticmethod
    def _calculate_relative_error(lb: float, ub: float):
        return abs(2 * (ub - lb) / (abs(ub) + abs(lb)))

    def __solve_primal_problem(self,
                               dual_soln: Dict[str, Union[float,
                                                          Dict[Tuple[Union[int, float, str, None], ...], float]]]):

        is_feasible_i = True
        ub_i = 0

        for sp_container in self.gbd_problem.sp_containers:

            if is_feasible_i:

                is_primal_sp_fbl, v_sp = self.__solve_primal_sp(sp_container=sp_container,
                                                                dual_soln=dual_soln)

                if is_primal_sp_fbl:
                    ub_i += v_sp  # update current upper bound
                else:
                    self._log_indexed_message("Primal subproblem is infeasible",
                                              iter=self.it,
                                              sp_sym=sp_container.primal_sp.symbol,
                                              sp_idx=sp_container.sp_index)
                    is_feasible_i = False
                    ub_i = 0  # reset current upper bound
                    self.__reset_dual_solution(dual_soln)

            if not is_feasible_i:
                _, v_fbl_sp = self.__solve_fbl_sp(sp_container=sp_container,
                                                  dual_soln=dual_soln)
                ub_i += v_fbl_sp  # update current upper bound

        return is_feasible_i, ub_i

    def __solve_mp(self, iter: int) -> Tuple[bool, float]:
        before_mp_solve = self.before_mp_solve
        if before_mp_solve is not None:
            before_mp_solve(self.gbd_problem, iter)

        self._log_indexed_message("Solving master problem", iter)

        self._engine.solve(solve_options=self.mp_solve_options)
        solver_output = self._engine.get_solver_output()
        self._print_solver_output(solver_output)

        status = self._engine.get_status()

        if status in ["infeasible", "failure"]:
            self._log_indexed_message("Master problem is infeasible", iter)
            return False, 0
        else:
            lb = self._engine.get_obj_value(self.gbd_problem.mp_obj_sym)
            self._log_indexed_message("Lower bound of {0}".format(lb), iter)
            return True, lb

    def __solve_primal_sp(self,
                          sp_container: GBDSubproblemContainer,
                          dual_soln: Dict[str, Union[float, Dict[Tuple[Union[int, float, str, None], ...], float]]]
                          ) -> Tuple[bool, float]:

        self._engine.set_active_problem(problem_symbol=sp_container.primal_sp.symbol,
                                        problem_idx=sp_container.sp_index)

        self._log_indexed_message("Fixing complicating variables",
                                  iter=self.it,
                                  sp_sym=sp_container.primal_sp.symbol,
                                  sp_idx=sp_container.sp_index)
        self._fix_complicating_variables(engine=self._engine, sp_container=sp_container)

        if self.before_sp_solve is not None:
            self.before_sp_solve(self.gbd_problem,
                                 self.it,
                                 sp_container.primal_sp.symbol,
                                 sp_container.sp_index)

        self._log_indexed_message("Solving primal subproblem",
                                  iter=self.it,
                                  sp_sym=sp_container.primal_sp.symbol,
                                  sp_idx=sp_container.sp_index)

        self._engine.solve(solve_options=self.sp_solve_options)
        solver_output = self._engine.get_solver_output()
        self._print_solver_output(solver_output)

        if not self._interpret_solver_result(solver_output,
                                             sp_sym=sp_container.primal_sp.symbol,
                                             sp_index=sp_container.sp_index):
            return False, 0

        v_sp = self.__store_sp_result(is_feasible=True,
                                      sp_container=sp_container,
                                      dual_soln=dual_soln)
        self._log_indexed_message("Subproblem objective is {0}".format(v_sp),
                                  iter=self.it,
                                  sp_sym=sp_container.primal_sp.symbol,
                                  sp_idx=sp_container.sp_index)
        return True, v_sp

    def __solve_fbl_sp(self,
                       sp_container: GBDSubproblemContainer,
                       dual_soln: Dict[str, Union[float, Dict[Tuple[Union[int, float, str, None], ...], float]]]
                       ) -> Tuple[bool, float]:

        self._engine.set_active_problem(problem_symbol=sp_container.fbl_sp.symbol,
                                        problem_idx=sp_container.sp_index)

        self._log_indexed_message("Fixing complicating variables",
                                  iter=self.it,
                                  sp_sym=sp_container.fbl_sp.symbol,
                                  sp_idx=sp_container.sp_index)
        self._fix_complicating_variables(engine=self._engine, sp_container=sp_container)

        if self.before_sp_solve is not None:
            self.before_sp_solve(self.gbd_problem,
                                 self.it,
                                 sp_container.fbl_sp.symbol,
                                 sp_container.sp_index)

        self._log_indexed_message("Solving feasibility subproblem",
                                  iter=self.it,
                                  sp_sym=sp_container.fbl_sp.symbol,
                                  sp_idx=sp_container.sp_index)

        self._engine.solve(solve_options=self.sp_solve_options)
        solver_output = self._engine.get_solver_output()
        self._print_solver_output(solver_output)

        if not self._interpret_solver_result(solver_output,
                                             sp_sym=sp_container.fbl_sp.symbol,
                                             sp_index=sp_container.sp_index):
            self._log_indexed_message("Feasibility subproblem is infeasible",
                                      iter=self.it,
                                      sp_sym=sp_container.fbl_sp.symbol,
                                      sp_idx=sp_container.sp_index)

        v_sp = self.__store_sp_result(is_feasible=False,
                                      sp_container=sp_container,
                                      dual_soln=dual_soln)
        self._log_indexed_message("Subproblem objective is {0}".format(v_sp),
                                  iter=self.it,
                                  sp_sym=sp_container.fbl_sp.symbol,
                                  sp_idx=sp_container.sp_index)

        return True, v_sp

    def _interpret_solver_result(self,
                                 solver_output: str,
                                 sp_sym: str,
                                 sp_index: mat.Element) -> bool:

        is_feasible = True

        status = self._engine.get_status()
        if status in ["infeasible", "failure"]:
            is_feasible = False

        # Conopt
        if self.sp_solve_options.get("solver", None) == "conopt":

            # Last iteration yielded a feasible solution
            if is_feasible:
                return True

            # Last iteration yielded an infeasible solution
            else:

                (is_feasible,
                 _,
                 solver_iter) = outputparser.parse_conopt_output(solver_output)

                # Problem is infeasible
                if not is_feasible:
                    return False

                # Problem is feasible
                else:

                    self._log_indexed_message("Resolving feasible problem declared infeasible (CONOPT)",
                                              iter=self.it,
                                              sp_sym=sp_sym,
                                              sp_idx=sp_index)

                    conopt_options = dict(self.sp_solve_options)
                    conopt_options["maxiter"] = solver_iter

                    self._engine.solve(solve_options=conopt_options)
                    solver_output = self._engine.get_solver_output()

                    (is_feasible,
                     _,
                     solver_iter) = outputparser.parse_conopt_output(solver_output)

                    if not is_feasible:
                        raise ValueError("Failed to recover last feasible solution of a feasible subproblem.")

                    self._log_indexed_message("Recovered feasible solution (CONOPT)",
                                              iter=self.it,
                                              sp_sym=sp_sym,
                                              sp_idx=sp_index)
                    return True

        # Other solver
        else:
            return is_feasible

    # Storage and Retrieval
    # ------------------------------------------------------------------------------------------------------------------

    def __store_sp_result(self,
                          is_feasible: bool,
                          sp_container: GBDSubproblemContainer,
                          dual_soln: Dict[str, Union[float, Dict[Tuple[Union[int, float, str, None], ...], float]]]
                          ) -> float:

        self.__set_is_feasible_flag(is_feasible)

        self.__retrieve_sp_dual_solution(is_feasible=is_feasible,
                                         sp_container=sp_container,
                                         dual_soln=dual_soln)

        if is_feasible:
            v_sp = self._get_sp_obj_value(
                engine=self._engine,
                obj=sp_container.get_primal_meta_obj(),
                sp_index=sp_container.sp_index
            )
        else:
            v_sp = self._get_sp_obj_value(
                engine=self._engine,
                obj=sp_container.get_fbl_meta_obj(),
                sp_index=sp_container.sp_index
            )

        return v_sp

    # Complicating Primal Solution
    # ------------------------------------------------------------------------------------------------------------------

    def __store_complicating_variables(self) -> Dict[str, Union[float, Dict[tuple, float]]]:

        def modify_value(v: float, mv: mat.MetaVariable) -> float:
            if mv.is_binary:
                if v < 0.5:
                    return 0
                else:
                    return 1
            else:
                return v

        y = {}
        cut_count = self.__get_cut_count()

        for _, comp_meta_var in self.gbd_problem.comp_meta_vars.items():

            var_sym = comp_meta_var.symbol
            storage_sym = var_sym + "_stored"

            # Scalar variable
            if comp_meta_var.idx_set_dim == 0:
                value = self._engine.get_var_value(var_sym)
                value = modify_value(value, comp_meta_var)
                self._engine.set_param_value(storage_sym, (cut_count,), value)
                y[var_sym] = value

            # Indexed variable
            else:

                var_idx_set = comp_meta_var.idx_set_node.evaluate(self.gbd_problem.state)[0]

                if var_sym in y:
                    y_var = y[var_sym]
                else:
                    y_var = {}
                    y[var_sym] = y_var

                for var_idx in var_idx_set:
                    value = self._engine.get_var_value(var_sym, var_idx)
                    value = modify_value(value, comp_meta_var)
                    self._engine.set_param_value(storage_sym, var_idx + (cut_count,), value)
                    y_var[var_idx] = value

        return y

    def _fix_complicating_variables(self,
                                    engine: Engine,
                                    sp_container: GBDSubproblemContainer):

        cut_count = self.__get_cut_count()

        for var_sym, idx_set in sp_container.comp_var_idx_sets.items():

            storage_sym = self.gbd_problem.stored_comp_decisions[var_sym].symbol

            # scalar variable
            if self.gbd_problem.meta_vars[var_sym].idx_set_dim == 0:
                value = engine.get_param_value(storage_sym, (cut_count,))
                engine.fix_var(symbol=var_sym, value=value)

            # indexed variable
            else:
                for var_index in idx_set:
                    value = engine.get_param_value(storage_sym, var_index + (cut_count,))
                    engine.fix_var(symbol=var_sym, idx=var_index, value=value)

    # Dual Solution
    # ------------------------------------------------------------------------------------------------------------------

    def __retrieve_sp_dual_solution(self,
                                    is_feasible: bool,
                                    sp_container: GBDSubproblemContainer,
                                    dual_soln: Dict[str, Union[float,
                                                               Dict[Tuple[Union[int, float, str, None], ...], float]]]
                                    ):

        def retrieve_dual_value(sym: str, con_idx: Union[tuple, list, None] = None):
            d = self._engine.get_con_dual(sym, con_idx)
            return -d

        for con_sym, idx_set in sp_container.mixed_comp_con_idx_set.items():

            dual_mult_sym = "lambda_{0}".format(con_sym)

            mod_con_sym = con_sym
            if not is_feasible:
                if con_sym + "_F" in self.gbd_problem.meta_cons:
                    mod_con_sym += "_F"

            # scalar constraint
            if self.gbd_problem.meta_cons[con_sym].idx_set_dim == 0:
                dual_soln[dual_mult_sym] = retrieve_dual_value(mod_con_sym)

            # indexed constraint
            else:
                sp_dual_soln_c = dual_soln.get(dual_mult_sym, {})
                for con_index in idx_set:
                    sp_dual_soln_c[con_index] = retrieve_dual_value(mod_con_sym, con_index)
                dual_soln[dual_mult_sym] = sp_dual_soln_c

    def __set_duality_multipliers(self, dual_mult_values: Dict[str, Union[float, Dict[tuple, float]]]):

        cut_count = self.__get_cut_count()

        for dual_id, dual_mult in self.gbd_problem.duality_multipliers.items():

            # scalar constraint
            if not isinstance(dual_mult_values[dual_mult.symbol], dict):
                self._engine.set_param_value(symbol=dual_mult.symbol,
                                             idx=(cut_count,),
                                             value=dual_mult_values[dual_mult.symbol])

            # indexed constraint
            else:
                for dual_index, value in dual_mult_values[dual_mult.symbol].items():
                    self._engine.set_param_value(symbol=dual_mult.symbol,
                                                 idx=dual_index + (cut_count,),
                                                 value=value)

    @staticmethod
    def __reset_dual_solution(dual_soln: Dict[str, Union[float,
                                                         Dict[Tuple[Union[int, float, str, None], ...], float]]]):
        for dual_sym, dual_soln_c in dual_soln.items():
            if isinstance(dual_soln_c, dict):
                for con_idx in dual_soln_c:
                    dual_soln[dual_sym][con_idx] = 0
            else:
                dual_soln[dual_sym] = 0

    # Output
    # ------------------------------------------------------------------------------------------------------------------

    def __print_iteration_output(self,
                                 iter: int,
                                 global_lb: float,
                                 global_ub: float,
                                 is_feasible: bool):
        message = "Iter: {0:^4} | LB: {1:^15} | UB: {2:^15} | Fbl: {3}".format(iter,
                                                                               int(global_lb),
                                                                               int(global_ub),
                                                                               is_feasible)
        if self.verbosity == 1:
            print(message)

        self._log_indexed_message(message=message,
                                  iter=iter)

    def _print_solver_output(self, solver_output: str):
        if self.verbosity >= 3:
            print(solver_output)

    def __print_solution(self,
                         is_feasible: bool,
                         v_ub: float,
                         y: Dict[str, Union[float, Dict[tuple, float]]],
                         runtime_seconds: float):

        if is_feasible:
            message = "\nv = {0} \ny = {1}".format(v_ub, pprint.pformat(y))
        else:
            message = "\nProblem is infeasible"

        runtime_hours = runtime_seconds / 3600
        message += "\nruntime = {0:.0f} s ({1:.1f} h)".format(runtime_seconds, runtime_hours)

        if self.verbosity == 1:
            print(message)

        self.__log_message(message)

    def _log_indexed_message(self,
                             message: str,
                             iter: int,
                             sp_sym: str = None,
                             sp_idx: mat.Element = None):

        if sp_sym is not None and sp_idx is not None:
            sp_str = "|sp {0}[{1}]".format(sp_sym, '-'.join([str(sp_i) for sp_i in sp_idx]))
        else:
            sp_str = ""

        message = "(it {0}{1}) {2}".format(iter, sp_str, message)

        self.__log_message(message)

    def __log_message(self, message: str):

        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        entry = "{0}: {1}".format(current_time, message)

        self.log += entry + '\n'

        if self.verbosity >= 2:
            print(entry)

    def __write_log_file(self):
        util.write_file(dir_path=self.gbd_problem.working_dir_path,
                        file_name="gbd.log",
                        text=self.log)

    # Getters and Setters
    # ------------------------------------------------------------------------------------------------------------------

    def __generate_entity_sp_index(self,
                                   meta_entity: mat.MetaEntity,
                                   entity_index: List[Union[int, float, str, None]] = None,
                                   sp_index: List[Union[int, float, str]] = None):

        if entity_index is None:
            entity_index = list(meta_entity.idx_set_reduced_dummy_element)  # retrieve default entity index

        sp_idx_pos = 0  # first index position of the current indexing meta-set
        for idx_meta_set in self.gbd_problem.idx_meta_sets.values():

            idx_syms = sp_index[sp_idx_pos:sp_idx_pos + idx_meta_set.reduced_dim]
            sp_idx_pos += idx_meta_set.reduced_dim  # update position of the subproblem index

            ent_idx_pos = meta_entity.get_first_reduced_dim_index_of_idx_set(idx_meta_set)
            entity_index[ent_idx_pos:ent_idx_pos + idx_meta_set.reduced_dim] = idx_syms  # update entity index

        return entity_index

    def __get_cut_count(self) -> int:
        cut_count = self._engine.get_param_value(self.gbd_problem.cut_count_sym, None)
        return int(cut_count)

    def __increment_cut_count(self):
        cut_count = self._engine.get_param_value(self.gbd_problem.cut_count_sym, None)
        self._engine.set_param_value(self.gbd_problem.cut_count_sym, None, int(cut_count + 1))

    def _reset_cut_count(self):
        self._engine.set_param_value(self.gbd_problem.cut_count_sym, None, 0)

    def __set_is_feasible_flag(self, flag: bool):
        num_flag = 1 if flag else 0
        cut_count = self.__get_cut_count()
        self._engine.set_param_value(self.gbd_problem.is_feasible_sym, (cut_count,), num_flag)

    def __set_stored_obj_value(self, value: float):
        cut_count = self.__get_cut_count()
        self._engine.set_param_value(self.gbd_problem.stored_obj_sym, (cut_count,), value)

    def __update_stored_obj_value(self, added_value: float):
        cut_count = self.__get_cut_count()
        value = self._engine.get_param_value(self.gbd_problem.stored_obj_sym, (cut_count,))
        self._engine.set_param_value(self.gbd_problem.stored_obj_sym, (cut_count,), value + added_value)

    def _get_sp_obj_value(self,
                          engine: Engine,
                          obj: mat.MetaObjective,
                          sp_index: mat.Element) -> float:
        obj_idx = self._gbd_problem_builder.generate_entity_sp_index(sp_index=sp_index, meta_entity=obj)
        return engine.get_obj_value(obj.symbol, obj_idx)
