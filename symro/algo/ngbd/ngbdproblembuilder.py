from typing import List, Optional

import symro.mat as mat

import symro.scripting.amplstatement as ampl_stm

from symro.prob.problem import BaseProblem, Problem

import symro.handlers.formulator as fmr
from symro.handlers.scriptbuilder import ScriptBuilder

from symro.algo.gbd import *
from symro.algo.ngbd.ngbdproblem import NGBDProblem


class NGBDProblemBuilder(GBDProblemBuilder):

    # Construction
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self):

        super().__init__()

        self.origin_problem: Optional[Problem] = None
        self.ngbd_problem: Optional[NGBDProblem] = None

    # Core
    # ------------------------------------------------------------------------------------------------------------------

    def _build_gbd_problem(
        self,
        problem: Problem,
        mp_symbol: str = None,
        primal_sp_symbol: str = None,
        fbl_sp_symbol: str = None,
        primal_sp_obj_symbol: str = None,
        working_dir_path: str = None,
    ):
        self.gbd_problem = NGBDProblem(
            problem=problem,
            mp_symbol=mp_symbol,
            default_primal_sp_symbol=primal_sp_symbol,
            default_fbl_sp_symbol=fbl_sp_symbol,
            primal_sp_obj_sym=primal_sp_obj_symbol,
            working_dir_path=working_dir_path,
        )
        self.ngbd_problem = self.gbd_problem

    def build_and_initialize_original_problem(self, problem: Problem) -> Problem:

        # build original problem
        origin_problem = Problem()
        problem.copy(problem, origin_problem)

        # generate new problem symbol
        origin_problem.symbol = problem.symbol + ".origin"

        # standardize model
        fmr.substitute_defined_variables(origin_problem)
        self.gbd_problem.origin_to_std_con_map = fmr.standardize_model(origin_problem)

        self.ngbd_problem.origin_problem = origin_problem
        self.origin_problem = origin_problem

        return origin_problem

    def build_gbd_constructs(self) -> GBDProblem:

        super().build_gbd_constructs()

        # --- Original Subproblem ---
        if len(self.ngbd_problem.origin_primal_sps) == 0:
            self._build_default_original_primal_sp()
        self._modify_primal_sp_objs(
            problem=self.origin_problem,
            subproblems=self.ngbd_problem.origin_primal_sps,
            sp_idx_meta_sets=self.ngbd_problem.idx_meta_sets,
        )
        self.__build_original_subproblem_containers()
        self.__build_and_write_ampl_script_origin()

        return self.ngbd_problem

    # Primal Subproblems
    # ------------------------------------------------------------------------------------------------------------------

    def build_or_retrieve_defined_primal_sp(
        self,
        sp_sym: str,
        entity_defs: List[str] = None,
        linked_entity_defs: List[str] = None,
    ):

        # retrieve existing subproblem
        if entity_defs is None:

            if (
                sp_sym in self.origin_problem.subproblems
            ):  # subproblem is defined in the original problem
                origin_sp = self.origin_problem.subproblems[sp_sym]
                self.gbd_problem.origin_primal_sps.append(
                    origin_sp
                )  # store original primal subproblem

            else:  # subproblem is not defined in the original problem
                raise ValueError(
                    "NGBD problem builder encountered an undefined subproblem symbol"
                )

            if (
                sp_sym in self.gbd_problem.subproblems
            ):  # subproblem is defined in the convex relaxation
                # store convexified primal subproblem
                self.gbd_problem.primal_sps.append(self.gbd_problem.subproblems[sp_sym])

            else:  # subproblem is not defined in the convex relaxation
                raise ValueError(
                    "NGBD problem builder encountered an undefined subproblem symbol"
                )

        # build new subproblem
        else:

            # build original primal subproblem

            origin_sp = BaseProblem(
                symbol=sp_sym, description="Original primal subproblem"
            )

            for entity_def in entity_defs:
                self.__add_defined_meta_entity_to_subproblem(
                    problem=self.origin_problem,
                    subproblem=origin_sp,
                    entity_def=entity_def,
                )

            self._add_linked_meta_entities_to_subproblem(
                problem=self.origin_problem,
                subproblem=origin_sp,
                linked_entity_defs=linked_entity_defs,
            )

            self.origin_problem.add_subproblem(origin_sp)
            self.gbd_problem.origin_primal_sps.append(origin_sp)

            # build convexified primal subproblem

            convex_sp = BaseProblem(
                symbol=sp_sym, description="Convexified primal subproblem"
            )

            for entity_def in entity_defs:
                self.__add_defined_meta_entity_to_subproblem(
                    problem=self.gbd_problem,
                    subproblem=convex_sp,
                    entity_def=entity_def,
                )

            self._add_linked_meta_entities_to_subproblem(
                problem=self.gbd_problem,
                subproblem=convex_sp,
                linked_entity_defs=linked_entity_defs,
            )

            self.gbd_problem.add_subproblem(convex_sp)
            self.gbd_problem.primal_sps.append(convex_sp)

    def _build_default_original_primal_sp(self):

        # retrieve meta-variables
        meta_vars = [mv for mv in self.origin_problem.model_meta_vars]

        # retrieve meta-objective

        primal_sp_obj_sym = self.gbd_problem.primal_sp_obj_sym

        if primal_sp_obj_sym is not None:
            # retrieve primal meta-objective
            primal_sp_meta_obj = self.origin_problem.meta_objs[primal_sp_obj_sym]

        # elicit a suitable primal meta-objective
        else:
            primal_sp_meta_obj = None
            idx_meta_sets = self.ngbd_problem.get_idx_meta_sets()
            for meta_obj in self.origin_problem.model_meta_objs:
                if idx_meta_sets is not None:
                    if all(
                        [
                            meta_obj.is_indexed_with(idx_meta_set)
                            for idx_meta_set in idx_meta_sets
                        ]
                    ):
                        primal_sp_meta_obj = meta_obj
                        break

        if primal_sp_meta_obj is None:
            raise ValueError(
                "NGBD problem builder could not identify a suitable objective function"
                " for the primal subproblem(s)"
            )

        # retrieve meta-constraints
        meta_cons = list(self.origin_problem.model_meta_cons)

        # build subproblem
        primal_sp = BaseProblem(
            symbol=self.gbd_problem.default_primal_sp_sym,
            description="Primal subproblem",
        )
        primal_sp.model_meta_vars = meta_vars
        primal_sp.model_meta_objs.append(primal_sp_meta_obj)
        primal_sp.model_meta_cons = meta_cons

        self.origin_problem.add_subproblem(primal_sp)
        self.ngbd_problem.origin_primal_sps.append(primal_sp)

    # Subproblem Containers
    # ------------------------------------------------------------------------------------------------------------------

    def __build_original_subproblem_containers(self):

        sp_idx_set = self._generate_sp_idx_set()

        # generate subproblem containers

        def generate_sp_container(ngbd_prob: NGBDProblem, sp_idx: mat.Element = None):
            for origin_primal_sp in ngbd_prob.origin_primal_sps:
                sp_ctn = GBDSubproblemContainer(
                    primal_sp=origin_primal_sp, fbl_sp=None, sp_index=sp_idx
                )
                ngbd_prob.nc_sp_containers.append(sp_ctn)

        if len(sp_idx_set) == 0:  # scalar subproblems
            generate_sp_container(self.ngbd_problem)

        else:  # indexed subproblems
            for sp_index in sp_idx_set:
                generate_sp_container(self.ngbd_problem, sp_index)

        # retrieve indexing sets of all complicating meta-variables partitioned by subproblem
        comp_meta_vars = list(self.ngbd_problem.comp_meta_vars.values())
        idx_sets = self._assemble_complete_entity_idx_sets_by_symbol(comp_meta_vars)
        sp_sym_idx_sets = self._partition_complete_entity_idx_sets_by_sp_sym(
            entity_type=mat.VAR_TYPE, complete_idx_sets=idx_sets
        )
        sp_idx_sets = self._partition_sp_entity_idx_sets_by_sp_index(
            sp_entity_idx_sets=sp_sym_idx_sets
        )

        # store indexing sets of all complicating meta-variables partitioned by subproblem
        for sp_idx_set_dict, nc_sp_container in zip(
            sp_idx_sets, self.ngbd_problem.nc_sp_containers
        ):
            nc_sp_container.comp_var_idx_sets = sp_idx_set_dict

    # Scripts
    # ------------------------------------------------------------------------------------------------------------------

    def __build_and_write_ampl_script_origin(self):

        self._clean_script(self.origin_problem.compound_script)
        self.__build_model_script_origin()
        self.__build_origin_problem_declarations()

        self.origin_problem.compound_script.write(
            dir_path=self.ngbd_problem.working_dir_path,
            main_file_name=self.origin_problem.symbol + ".run",
        )

    def __build_model_script_origin(self):

        script_builder = ScriptBuilder()
        mod_script = script_builder.generate_problem_model_script(
            problem=self.origin_problem, model_file_extension=".modr"
        )

        ampl_stm.add_included_script_to_compound_script(
            self.origin_problem.compound_script, mod_script, file_command="model"
        )

    def __build_origin_problem_declarations(self):

        idx_meta_sets = [ms for ms in self.ngbd_problem.idx_meta_sets.values()]
        main_script = self.origin_problem.compound_script.main_script
        script_builder = ScriptBuilder()

        for primal_sp in self.ngbd_problem.origin_primal_sps:
            primal_sp_decl = script_builder.generate_subproblem_declaration(
                problem=self.origin_problem,
                subproblem=primal_sp,
                idx_meta_sets=idx_meta_sets,
            )
            main_script.statements.append(primal_sp_decl)
