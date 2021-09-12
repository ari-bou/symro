from typing import Dict, List, Optional, Tuple

import symro.core.constants as const
import symro.core.mat as mat
from symro.core.prob.problem import Problem, BaseProblem
from symro.core.handlers.entitybuilder import EntityBuilder


DEFAULT_MP_SYMBOL = "Master"
DEFAULT_PRIMAL_SP_SYMBOL = "Primal"
DEFAULT_FBL_SP_SYMBOL = "Feasibility"

CUT_COUNT_PARAM_SYMBOL = "CUT_COUNT"
CUTS_SET_SYMBOL = "CUTS"

IS_FEASIBLE_PARAM_SYMBOL = "is_feasible"

STORED_OBJ_PARAM_SYMBOL = "obj_stored"

ETA_VAR_SYMBOL = "eta"

MASTER_OBJ_SYMBOL = "MASTER_OBJ"
PRIMAL_OBJ_SYMBOL = "PRIMAL_OBJ"
FBL_OBJ_SYMBOL = "FBL_OBJ"

OPT_CUT_CON_SYMBOL = "GBD_OPT_CUT"
FBL_CUT_CON_SYMBOL = "GBD_FBL_CUT"
CAN_INT_CUT_CON_SYMBOL = "GBD_CANON_INT_CUT"


class GBDSubproblemContainer:

    def __init__(self,
                 primal_sp: Optional[BaseProblem],
                 fbl_sp: Optional[BaseProblem],
                 sp_index: Optional[mat.IndexSetMember]):
        self.primal_sp: Optional[BaseProblem] = primal_sp
        self.fbl_sp: Optional[BaseProblem] = fbl_sp
        self.sp_index: Optional[mat.IndexSetMember] = sp_index
        self.comp_var_idx_sets: Dict[str, Optional[mat.IndexSet]] = {}
        self.mixed_comp_con_idx_set: Dict[str, Optional[mat.IndexSet]] = {}

    def get_primal_meta_obj(self) -> mat.MetaObjective:
        return self.primal_sp.model_meta_objs[0]

    def get_fbl_meta_obj(self) -> mat.MetaObjective:
        return self.fbl_sp.model_meta_objs[0]


class GBDProblem(Problem):

    def __init__(self,
                 problem: Problem,
                 mp_symbol: str,
                 default_primal_sp_symbol: str,
                 default_fbl_sp_symbol: str,
                 primal_sp_obj_sym: str,
                 init_lb: float,
                 working_dir_path: str = None):

        super(GBDProblem, self).__init__(engine=None,
                                         symbol=None,
                                         description=problem.description)
        Problem.copy(self, problem)

        # --- Name ---
        self.name = problem.symbol + ".gbd"

        # --- I/O ---
        self.working_dir_path: str = working_dir_path if working_dir_path is not None \
            else problem.engine.working_dir_path

        # --- Script ---
        self.compound_script.included_scripts.clear()

        # --- Symbols ---

        self.mp_symbol: str = mp_symbol if mp_symbol is not None else DEFAULT_MP_SYMBOL

        self.default_primal_sp_sym: str = default_primal_sp_symbol if default_primal_sp_symbol is not None \
            else DEFAULT_PRIMAL_SP_SYMBOL

        self.default_fbl_sp_sym: str = default_fbl_sp_symbol if default_fbl_sp_symbol is not None \
            else DEFAULT_FBL_SP_SYMBOL

        self.cuts_sym: str = CUTS_SET_SYMBOL

        self.cut_count_sym = CUT_COUNT_PARAM_SYMBOL
        self.is_feasible_sym = IS_FEASIBLE_PARAM_SYMBOL
        self.stored_obj_sym = STORED_OBJ_PARAM_SYMBOL

        self.eta_sym = ETA_VAR_SYMBOL

        self.mp_obj_sym: str = MASTER_OBJ_SYMBOL
        self.primal_sp_obj_sym: str = primal_sp_obj_sym
        self.default_fbl_sp_obj_sym: str = FBL_OBJ_SYMBOL

        self.opt_cut_con_sym = OPT_CUT_CON_SYMBOL
        self.fbl_cut_con_sym = FBL_CUT_CON_SYMBOL
        self.can_int_cut_con_sym = CAN_INT_CUT_CON_SYMBOL

        # --- Algorithm Meta-Entities ---

        entity_builder = EntityBuilder(problem)

        # Meta-Sets and Meta-Parameters

        self.idx_meta_sets: Dict[str, mat.MetaSet] = {}

        self.cut_count = entity_builder.build_meta_param(symbol=self.cut_count_sym,
                                                         default_value=0)
        self.add_meta_parameter(self.cut_count, is_in_model=False)

        cuts_idx_sym = "ct"
        ord_set_node = mat.OrderedSetNode(mat.NumericNode(1),
                                          mat.DeclaredEntityNode(self.cut_count_sym,
                                                                 type=const.PARAM_TYPE))
        self.cuts = entity_builder.build_meta_set(symbol=self.cuts_sym,
                                                  dimension=1,
                                                  dummy_symbols=[cuts_idx_sym],
                                                  reduced_dummy_symbols=[cuts_idx_sym],
                                                  set_node=ord_set_node)
        self.add_meta_set(self.cuts, is_in_model=False)

        self.is_feasible = entity_builder.build_meta_param(symbol=self.is_feasible_sym,
                                                           idx_meta_sets=[self.cuts],
                                                           default_value=0)
        self.add_meta_parameter(self.is_feasible, is_in_model=False)

        self.stored_obj = entity_builder.build_meta_param(symbol=self.stored_obj_sym,
                                                          idx_meta_sets=[self.cuts],
                                                          default_value=0)
        self.add_meta_parameter(self.stored_obj, is_in_model=False)

        self.stored_comp_decisions: Dict[str, mat.MetaParameter] = {}

        self.duality_multipliers: Dict[int, mat.MetaParameter] = {}

        # Meta-Variables

        self.comp_meta_vars: Dict[str, mat.MetaVariable] = {}

        self.eta = entity_builder.build_meta_var(symbol=self.eta_sym,
                                                 lower_bound=init_lb)
        self.add_meta_variable(self.eta, is_in_model=False)

        self.slack_vars: Dict[str, mat.MetaVariable] = {}

        self.aux_f_meta_var: Optional[mat.MetaVariable] = None
        self.aux_g_meta_vars: Optional[Dict[int, mat.MetaVariable]] = {}

        # Meta-Objectives
        self.master_obj: Optional[mat.MetaObjective] = None
        self.primal_sp_objs: Dict[str, mat.MetaObjective] = {}  # key: subproblem symbol; value: meta-objective
        self.fbl_sp_objs: Dict[str, mat.MetaObjective] = {}  # key: subproblem symbol; value: meta-objective

        # Meta-Constraints

        self.pure_comp_cons: Dict[str, mat.MetaConstraint] = {}
        self.mixed_comp_cons: Dict[str, mat.MetaConstraint] = {}
        self.non_comp_cons: Dict[str, mat.MetaConstraint] = {}

        self.origin_to_std_con_map: Optional[Dict[str, List[mat.MetaConstraint]]] = None
        self.std_to_sl_map: Dict[str, Tuple[List[mat.MetaVariable], mat.MetaConstraint]] = {}

        self.sl_fbl_cons: Dict[str, mat.MetaConstraint] = {}

        self.aux_f_meta_con: Optional[mat.MetaConstraint] = None
        self.aux_g_meta_cons: Optional[Dict[int, mat.MetaConstraint]] = {}

        self.gbd_cuts: Dict[str, mat.MetaConstraint] = {}

        # Problems

        self.primal_sps: List[BaseProblem] = []
        self.fbl_sps: List[BaseProblem] = []
        self.sp_containers: List[GBDSubproblemContainer] = []

        self.mp: Optional[BaseProblem] = None

    def get_idx_meta_sets(self) -> List[mat.MetaSet]:
        return [ms for ms in self.idx_meta_sets.values()]

    def get_lvl_idx_set_dim(self) -> int:
        return sum([ms.get_reduced_dimension() for ms in self.idx_meta_sets.values()])

    def get_comp_var_syms(self) -> List[str]:
        return [mv.symbol for mv in self.comp_meta_vars.values()]

    def get_comp_meta_vars(self) -> Dict[str, mat.MetaVariable]:
        return self.comp_meta_vars
