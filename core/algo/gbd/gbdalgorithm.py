from copy import deepcopy
from datetime import datetime
import numpy as np
import pprint as pprint
from typing import Callable, Dict, List, Optional, Tuple, Union

import symro.core.mat as mat
from symro.core.prob.problem import Problem

from symro.core.execution.amplengine import AMPLEngine

from symro.core.parsing import outputparser

from symro.core.algo.gbd import GBDProblem, GBDSubproblemContainer, GBDProblemBuilder

import symro.core.util.util as util


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
        self.problem = problem
        self.gbd_problem: Optional[GBDProblem] = None

        # --- Handlers ---
        self.__gbd_problem_builder: GBDProblemBuilder = GBDProblemBuilder(problem)
        self.__engine: Optional[AMPLEngine] = None

        # --- Options ---

        self.init_lb: float = init_lb
        self.init_ub: float = init_ub

        self.mp_solver_name: Optional[str] = None
        self.mp_solver_options: Optional[str] = None
        self.sp_solver_name: Optional[str] = None
        self.sp_solver_options: Optional[str] = None

        self.max_iter_count: int = 100
        self.abs_opt_tol: Optional[float] = None
        self.rel_opt_tol: float = 0.01
        self.fbl_tol: float = 0.001

        self.lb: float = -np.inf
        self.ub: float = np.inf

        self.verbosity: int = 0
        self.can_catch_exceptions: bool = False

        # --- Callables ---
        self.before_mp_solve: Callable[[GBDProblem, int], None] = before_mp_solve
        self.before_sp_solve: Callable[[GBDProblem, int, str, mat.Element], None] = before_sp_solve

        # --- Algorithmic Constructs ---

        self.log: str = ""

        self.gbd_problem = self.__gbd_problem_builder.build_gbd_problem(problem=self.problem,
                                                                        comp_var_defs=complicating_vars,
                                                                        mp_symbol=mp_symbol,
                                                                        primal_sp_symbol=primal_sp_symbol,
                                                                        fbl_sp_symbol=fbl_sp_symbol,
                                                                        primal_sp_obj_symbol=primal_sp_obj_symbol,
                                                                        init_lb=init_lb,
                                                                        working_dir_path=working_dir_path)

    # Setup
    # ------------------------------------------------------------------------------------------------------------------

    def add_decomposition_subproblem(self,
                                     sp_symbol: str,
                                     vars: List[str] = None,
                                     obj: str = None,
                                     cons: List[str] = None):
        self.__gbd_problem_builder.build_defined_primal_sp(sp_sym=sp_symbol,
                                                           var_defs=vars,
                                                           obj_def=obj,
                                                           con_defs=cons)

    def add_decomposition_axes(self, idx_set_symbols: List[str]):
        idx_meta_sets = {sym: self.gbd_problem.meta_sets[sym] for sym in idx_set_symbols}
        self.gbd_problem.idx_meta_sets = idx_meta_sets

    def setup(self):
        self.__gbd_problem_builder.build_gbd_constructs()

    # Run
    # ------------------------------------------------------------------------------------------------------------------

    def run(self,
            mp_solver_name: str = None,
            mp_solver_options: str = None,
            sp_solver_name: str = None,
            sp_solver_options: str = None,
            max_iter_count: int = 100,
            abs_opt_tol: float = None,
            rel_opt_tol: float = 0.01,
            fbl_tol: float = 0.001,
            verbosity: int = 0,
            can_catch_exceptions: bool = False,
            can_write_log: bool = True):

        # --- Parameter Storage ---

        self.mp_solver_name: str = mp_solver_name.lower()
        self.mp_solver_options: str = mp_solver_options
        self.sp_solver_name: str = sp_solver_name.lower()
        self.sp_solver_options: str = sp_solver_options

        self.max_iter_count: int = max_iter_count
        self.abs_opt_tol: float = abs_opt_tol
        self.rel_opt_tol: float = rel_opt_tol
        self.fbl_tol: float = fbl_tol

        self.verbosity: int = verbosity
        self.can_catch_exceptions: bool = can_catch_exceptions
        self.log: str = ""

        # --- Execution ---

        self.__log_message("Running GBD")

        self.__engine = AMPLEngine(self.gbd_problem)
        self.__engine.can_store_soln = False

        self.__evaluate_script()

        try:
            v_ub, y = self.__run_algorithm()
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

    def __evaluate_script(self):
        self.__log_message("Reading model and data")
        script_literal = self.gbd_problem.compound_script.main_script.get_literal()
        self.__engine.api.eval(script_literal + '\n')

    def __run_algorithm(self):

        start_time = datetime.now()
        is_feasible, v_ub, y, dual_mult = self.__run_loop()
        sol_time = datetime.now() - start_time

        self.__print_solution(is_feasible, v_ub, y, sol_time.total_seconds())

        if is_feasible:
            self.__log_message("Algorithm terminated with feasible solution")
        else:
            self.__log_message("Algorithm terminated with infeasible solution")

        return v_ub, y

    # Algorithm
    # ------------------------------------------------------------------------------------------------------------------

    def __run_loop(self) -> Tuple[bool,
                                  float,
                                  Dict[str, Union[float, Dict[tuple, float]]],
                                  Dict[str, Dict[tuple, float]]]:

        is_feasible = False
        ub = 0
        global_lb = self.init_lb
        global_ub = self.init_ub
        y_ub = {}
        dual_soln = {}
        dual_soln_ub = {}

        iter = 0
        can_iterate = True

        self.__reset_cut_count()

        while can_iterate:

            is_it_feasible = False

            self.__set_problem(self.gbd_problem.mp.symbol)

            # solve master problem
            can_iterate, lb = self.__solve_mp(iter)

            if can_iterate:

                # Update lower bound
                if lb > global_lb:
                    global_lb = lb
                    self.__log_indexed_message("Updated global lower bound", iter)

                # Increment cut count
                self.__increment_cut_count()

                # Store complicating variables
                self.__log_indexed_message("Storing complicating variables", iter)
                y_i = self.__store_complicating_variables()

                self.__set_is_feasible_flag(True)
                self.__set_stored_obj_value(0)

                # Solve primal subproblems
                is_it_feasible, ub = self.__solve_primal_problem(iter=iter, dual_soln=dual_soln)
                self.__log_indexed_message("Objective is {0}".format(ub), iter)

                self.__set_stored_obj_value(ub)

                self.__log_indexed_message("Storing dual multipliers", iter)
                self.__assign_values_to_dual_params(dual_soln)

                # Update Global Upper Bound
                if ub < global_ub and is_it_feasible:
                    global_ub = ub
                    y_ub = y_i
                    dual_soln_ub = deepcopy(dual_soln)
                    is_feasible = True
                    self.__log_indexed_message("Updated global upper bound", iter)

            # Output
            self.__print_iteration_output(iter,
                                          global_lb,
                                          global_ub,
                                          is_it_feasible)

            # Termination Criteria

            iter += 1

            if iter >= self.max_iter_count:
                self.__log_indexed_message("Termination: iteration limit", iter)
                can_iterate = False

            elif global_lb >= global_ub:
                self.__log_indexed_message("Termination: lower bound >= upper bound", iter)
                can_iterate = False

            else:

                epsilon_rel = abs(2 * (global_ub - global_lb) / (abs(global_ub) + abs(global_lb)))
                if epsilon_rel <= self.rel_opt_tol:
                    message = "Termination: epsilon-optimal solution with relative error = {0}".format(epsilon_rel)
                    self.__log_indexed_message(message, iter)
                    can_iterate = False

                elif self.abs_opt_tol is not None:
                    epsilon_abs = global_ub - global_lb
                    if epsilon_abs <= self.abs_opt_tol:
                        message = "Termination: epsilon-optimal solution with absolute error = {0}".format(epsilon_abs)
                        self.__log_indexed_message(message, iter)
                        can_iterate = False

        if not is_feasible:
            global_ub = ub
            dual_soln_ub = deepcopy(dual_soln)

        return is_feasible, global_ub, y_ub, dual_soln_ub

    def __solve_primal_problem(self,
                               iter: int,
                               dual_soln: Dict[str, Union[float,
                                                          Dict[Tuple[Union[int, float, str, None], ...], float]]]):

        is_feasible = True
        ub = 0

        for sp_container in self.gbd_problem.sp_containers:

            if is_feasible:

                is_primal_sp_fbl, v_sp = self.__solve_primal_sp(iter=iter,
                                                                sp_container=sp_container,
                                                                dual_soln=dual_soln)

                if is_primal_sp_fbl:
                    ub += v_sp  # update current upper bound
                else:
                    self.__log_indexed_message("Primal subproblem is infeasible",
                                               iter=iter,
                                               sp_sym=sp_container.primal_sp.symbol,
                                               sp_index=sp_container.sp_index)
                    is_feasible = False
                    ub = 0  # reset current upper bound
                    self.__reset_dual_solution(dual_soln)

            if not is_feasible:
                _, v_fbl_sp = self.__solve_fbl_sp(iter=iter,
                                                  sp_container=sp_container,
                                                  dual_soln=dual_soln)
                ub += v_fbl_sp  # update current upper bound

        return is_feasible, ub

    def __solve_mp(self, iter: int) -> Tuple[bool, float]:
        before_mp_solve = self.before_mp_solve
        if before_mp_solve is not None:
            before_mp_solve(self.gbd_problem, iter)

        self.__log_indexed_message("Solving master problem", iter)

        self.__engine.solve(solver_name=self.mp_solver_name,
                            solver_options=self.mp_solver_options)
        solver_output = self.__engine.get_solver_output()
        self.__print_solver_output(solver_output)

        status = self.__engine.get_status()

        if status in ["infeasible", "failure"]:
            self.__log_indexed_message("Master problem is infeasible", iter)
            return False, 0
        else:
            lb = self.__engine.get_obj_value(self.gbd_problem.mp_obj_sym)
            self.__log_indexed_message("Lower bound of {0}".format(lb), iter)
            return True, lb

    def __solve_primal_sp(self,
                          iter: int,
                          sp_container: GBDSubproblemContainer,
                          dual_soln: Dict[str, Union[float, Dict[Tuple[Union[int, float, str, None], ...], float]]]
                          ) -> Tuple[bool, float]:

        self.__set_problem(problem_sym=sp_container.primal_sp.symbol,
                           problem_idx=sp_container.sp_index)

        self.__log_indexed_message("Fixing complicating variables",
                                   iter=iter,
                                   sp_sym=sp_container.primal_sp.symbol,
                                   sp_index=sp_container.sp_index)
        self.__fix_complicating_variables(sp_container=sp_container)

        if self.before_sp_solve is not None:
            self.before_sp_solve(self.gbd_problem,
                                 iter,
                                 sp_container.primal_sp.symbol,
                                 sp_container.sp_index)

        self.__log_indexed_message("Solving primal subproblem",
                                   iter=iter,
                                   sp_sym=sp_container.primal_sp.symbol,
                                   sp_index=sp_container.sp_index)

        self.__engine.solve(solver_name=self.sp_solver_name,
                            solver_options=self.sp_solver_options)
        solver_output = self.__engine.get_solver_output()
        self.__print_solver_output(solver_output)

        if not self.__interpret_solver_result(solver_output,
                                              iter=iter,
                                              sp_sym=sp_container.primal_sp.symbol,
                                              sp_index=sp_container.sp_index):
            return False, 0

        v_sp = self.__store_sp_result(is_feasible=True,
                                      sp_container=sp_container,
                                      dual_soln=dual_soln)
        self.__log_indexed_message("Subproblem objective is {0}".format(v_sp),
                                   iter=iter,
                                   sp_sym=sp_container.primal_sp.symbol,
                                   sp_index=sp_container.sp_index)
        return True, v_sp

    def __solve_fbl_sp(self,
                       iter: int,
                       sp_container: GBDSubproblemContainer,
                       dual_soln: Dict[str, Union[float, Dict[Tuple[Union[int, float, str, None], ...], float]]]
                       ) -> Tuple[bool, float]:

        self.__set_problem(problem_sym=sp_container.fbl_sp.symbol,
                           problem_idx=sp_container.sp_index)

        self.__log_indexed_message("Fixing complicating variables",
                                   iter=iter,
                                   sp_sym=sp_container.fbl_sp.symbol,
                                   sp_index=sp_container.sp_index)
        self.__fix_complicating_variables(sp_container=sp_container)

        if self.before_sp_solve is not None:
            self.before_sp_solve(self.gbd_problem,
                                 iter,
                                 sp_container.fbl_sp.symbol,
                                 sp_container.sp_index)

        self.__log_indexed_message("Solving feasibility subproblem",
                                   iter=iter,
                                   sp_sym=sp_container.fbl_sp.symbol,
                                   sp_index=sp_container.sp_index)

        self.__engine.solve(self.sp_solver_name, self.sp_solver_options)
        solver_output = self.__engine.get_solver_output()
        self.__print_solver_output(solver_output)

        if not self.__interpret_solver_result(solver_output,
                                              iter=iter,
                                              sp_sym=sp_container.fbl_sp.symbol,
                                              sp_index=sp_container.sp_index):
            self.__log_indexed_message("Feasibility subproblem is infeasible",
                                       iter=iter,
                                       sp_sym=sp_container.fbl_sp.symbol,
                                       sp_index=sp_container.sp_index)

        v_sp = self.__store_sp_result(is_feasible=False,
                                      sp_container=sp_container,
                                      dual_soln=dual_soln)
        self.__log_indexed_message("Subproblem objective is {0}".format(v_sp),
                                   iter=iter,
                                   sp_sym=sp_container.fbl_sp.symbol,
                                   sp_index=sp_container.sp_index)

        return True, v_sp

    def __interpret_solver_result(self,
                                  solver_output: str,
                                  iter: int,
                                  sp_sym: str,
                                  sp_index: mat.Element) -> bool:

        is_feasible = True

        status = self.__engine.get_status()
        if status in ["infeasible", "failure"]:
            is_feasible = False

        # Conopt
        if self.sp_solver_name.lower() == "conopt":

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

                    self.__log_indexed_message("Resolving feasible problem declared infeasible (CONOPT)",
                                               iter=iter,
                                               sp_sym=sp_sym,
                                               sp_index=sp_index)
                    conopt_options = self.sp_solver_options + " maxiter={0}".format(solver_iter)
                    self.__engine.solve(self.sp_solver_name, conopt_options)
                    solver_output = self.__engine.get_solver_output()

                    (is_feasible,
                     _,
                     solver_iter) = outputparser.parse_conopt_output(solver_output)

                    if not is_feasible:
                        raise ValueError("Failed to recover last feasible solution of a feasible subproblem.")

                    self.__log_indexed_message("Recovered feasible solution (CONOPT)",
                                               iter=iter,
                                               sp_sym=sp_sym,
                                               sp_index=sp_index)
                    return True

        # Other solver
        else:
            return is_feasible

    # Problem Manipulation
    # ------------------------------------------------------------------------------------------------------------------

    def __set_problem(self,
                      problem_sym: str,
                      problem_idx: Union[List[Union[int, float, str]],
                                         Tuple[Union[int, float, str], ...]] = None):
        self.__engine.set_active_problem(problem_symbol=problem_sym,
                                         problem_idx=problem_idx)

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
            v_sp = self.__get_sp_obj_value(sp_container.get_primal_meta_obj(), sp_index=sp_container.sp_index)
        else:
            v_sp = self.__get_sp_obj_value(sp_container.get_fbl_meta_obj(), sp_index=sp_container.sp_index)

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

        for _, comp_meta_var in self.gbd_problem.get_comp_meta_vars().items():

            var_sym = comp_meta_var.symbol
            storage_sym = var_sym + "_stored"

            # Scalar variable
            if comp_meta_var.get_dimension() == 0:
                value = self.__engine.get_var_value(var_sym)
                value = modify_value(value, comp_meta_var)
                self.__engine.set_param_value(storage_sym, [cut_count], value)
                y[var_sym] = value

            # Indexed variable
            else:

                var_idx_set = comp_meta_var.idx_set_node.evaluate(self.problem.state)[0]

                if var_sym in y:
                    y_var = y[var_sym]
                else:
                    y_var = {}
                    y[var_sym] = y_var

                for var_idx in var_idx_set:
                    value = self.__engine.get_var_value(var_sym, var_idx)
                    value = modify_value(value, comp_meta_var)
                    self.__engine.set_param_value(storage_sym, list(var_idx) + [cut_count], value)
                    y_var[var_idx] = value

        return y

    def __fix_complicating_variables(self, sp_container: GBDSubproblemContainer):

        cut_count = self.__get_cut_count()

        for var_sym, idx_set in sp_container.comp_var_idx_sets.items():

            storage_sym = var_sym + "_stored"

            # scalar variable
            if self.gbd_problem.meta_vars[var_sym].get_dimension() == 0:
                value = self.__engine.get_param_value(storage_sym, [cut_count])
                var = self.__engine.api.getVariable(var_sym)
                var.fix(value)

            # indexed variable
            else:

                for var_index in idx_set:
                    value = self.__engine.get_param_value(storage_sym, list(var_index) + [cut_count])
                    var = self.__engine.api.getVariable(var_sym)
                    var = var.get(var_index)
                    var.fix(value)

    # Dual Solution
    # ------------------------------------------------------------------------------------------------------------------

    def __retrieve_sp_dual_solution(self,
                                    is_feasible: bool,
                                    sp_container: GBDSubproblemContainer,
                                    dual_soln: Dict[str, Union[float,
                                                               Dict[Tuple[Union[int, float, str, None], ...], float]]]
                                    ):

        def retrieve_dual_value(sym: str, con_idx: Union[tuple, list, None] = None):
            d = self.__engine.get_con_dual(sym, con_idx)
            return -d

        for con_sym, idx_set in sp_container.mixed_comp_con_idx_set.items():

            dual_mult_sym = "lambda_{0}".format(con_sym)

            mod_con_sym = con_sym
            if not is_feasible:
                if con_sym + "_F" in self.gbd_problem.meta_cons:
                    mod_con_sym += "_F"

            # scalar constraint
            if self.gbd_problem.meta_cons[con_sym].get_dimension() == 0:
                dual_soln[dual_mult_sym] = retrieve_dual_value(mod_con_sym)

            # indexed constraint
            else:
                sp_dual_soln_c = dual_soln.get(dual_mult_sym, {})
                for con_index in idx_set:
                    sp_dual_soln_c[con_index] = retrieve_dual_value(mod_con_sym, con_index)
                dual_soln[dual_mult_sym] = sp_dual_soln_c

    def __assign_values_to_dual_params(self, dual_mult_values: Dict[str, Union[float, Dict[tuple, float]]]):

        cut_count = self.__get_cut_count()

        for dual_id, dual_mult in self.gbd_problem.duality_multipliers.items():

            # scalar constraint
            if not isinstance(dual_mult_values[dual_mult.symbol], dict):
                self.__engine.set_param_value(symbol=dual_mult.symbol,
                                              indices=[cut_count],
                                              value=dual_mult_values[dual_mult.symbol])

            # indexed constraint
            else:
                for dual_index, value in dual_mult_values[dual_mult.symbol].items():
                    self.__engine.set_param_value(symbol=dual_mult.symbol,
                                                  indices=list(dual_index) + [cut_count],
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

        self.__log_indexed_message(message=message,
                                   iter=iter)

    def __print_solver_output(self, solver_output: str):
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

    def __log_indexed_message(self,
                              message: str,
                              iter: int,
                              sp_sym: str = None,
                              sp_index: mat.Element = None):

        if sp_sym is not None and sp_index is not None:
            sp_str = "|sp {0}[{1}]".format(sp_sym, '-'.join([str(sp_i) for sp_i in sp_index]))
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
            entity_index = list(meta_entity.get_dummy_symbols())  # retrieve default entity index

        sp_idx_pos = 0  # first index position of the current indexing meta-set
        for idx_meta_set in self.gbd_problem.idx_meta_sets.values():

            idx_syms = sp_index[sp_idx_pos:sp_idx_pos + idx_meta_set.reduced_dimension]
            sp_idx_pos += idx_meta_set.reduced_dimension  # update position of the subproblem index

            ent_idx_pos = meta_entity.get_first_reduced_dim_index_of_idx_set(idx_meta_set)
            entity_index[ent_idx_pos:ent_idx_pos + idx_meta_set.reduced_dimension] = idx_syms  # update entity index

        return entity_index

    def __get_cut_count(self) -> int:
        cut_count = self.__engine.get_param_value(self.gbd_problem.cut_count_sym, None)
        return int(cut_count)

    def __increment_cut_count(self):
        cut_count = self.__engine.get_param_value(self.gbd_problem.cut_count_sym, None)
        self.__engine.set_param_value(self.gbd_problem.cut_count_sym, None, int(cut_count + 1))

    def __reset_cut_count(self):
        self.__engine.set_param_value(self.gbd_problem.cut_count_sym, None, 0)

    def __set_is_feasible_flag(self, flag: bool):
        num_flag = 1 if flag else 0
        cut_count = self.__get_cut_count()
        self.__engine.set_param_value(self.gbd_problem.is_feasible_sym, [cut_count], num_flag)

    def __set_stored_obj_value(self, value: float):
        cut_count = self.__get_cut_count()
        self.__engine.set_param_value(self.gbd_problem.stored_obj_sym, [cut_count], value)

    def __update_stored_obj_value(self, added_value: float):
        cut_count = self.__get_cut_count()
        value = self.__engine.get_param_value(self.gbd_problem.stored_obj_sym, [cut_count])
        self.__engine.set_param_value(self.gbd_problem.stored_obj_sym, [cut_count], value + added_value)

    def __get_sp_obj_value(self,
                           obj: mat.MetaObjective,
                           sp_index: mat.Element) -> float:
        obj_idx = self.__gbd_problem_builder.generate_entity_sp_index(sp_index=sp_index, meta_entity=obj)
        return self.__engine.get_obj_value(obj.symbol, obj_idx)
