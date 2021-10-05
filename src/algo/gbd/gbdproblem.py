from typing import Dict, List, Optional, Tuple

import symro.src.constants as const
import symro.src.mat as mat
from symro.src.prob.problem import Problem, BaseProblem
import symro.src.handlers.entitybuilder as eb


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
                 sp_index: Optional[mat.Element]):
        self.primal_sp: Optional[BaseProblem] = primal_sp
        self.fbl_sp: Optional[BaseProblem] = fbl_sp
        self.sp_index: Optional[mat.Element] = sp_index
        self.comp_var_idx_sets: Dict[str, Optional[mat.IndexingSet]] = {}
        self.mixed_comp_con_idx_set: Dict[str, Optional[mat.IndexingSet]] = {}

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
                 working_dir_path: str = None):

        super(GBDProblem, self).__init__(symbol=None,
                                         description=problem.description,
                                         working_dir_path=working_dir_path)
        Problem.copy(problem, self)

        # --- Name ---
        self.symbol = problem.symbol + ".gbd"

        # --- Script ---
        self.compound_script.included_scripts.clear()

        # --- Symbols ---

        self.mp_symbol: str = mp_symbol if mp_symbol is not None else DEFAULT_MP_SYMBOL
        self.mp_symbol = self.generate_unique_symbol(self.mp_symbol)

        self.default_primal_sp_sym: str = default_primal_sp_symbol if default_primal_sp_symbol is not None \
            else DEFAULT_PRIMAL_SP_SYMBOL
        self.default_primal_sp_sym = self.generate_unique_symbol(self.default_primal_sp_sym)

        self.default_fbl_sp_sym: str = default_fbl_sp_symbol if default_fbl_sp_symbol is not None \
            else DEFAULT_FBL_SP_SYMBOL
        self.default_fbl_sp_sym = self.generate_unique_symbol(self.default_fbl_sp_sym)

        self.cuts_unb_sym: str = "ct"
        self.cuts_sym: str = self.generate_unique_symbol(CUTS_SET_SYMBOL)

        self.cut_count_sym = self.generate_unique_symbol(CUT_COUNT_PARAM_SYMBOL)
        self.is_feasible_sym = self.generate_unique_symbol(IS_FEASIBLE_PARAM_SYMBOL)
        self.stored_obj_sym = self.generate_unique_symbol(STORED_OBJ_PARAM_SYMBOL)

        self.eta_sym = self.generate_unique_symbol(ETA_VAR_SYMBOL)

        self.mp_obj_sym: str = self.generate_unique_symbol(MASTER_OBJ_SYMBOL)
        self.primal_sp_obj_sym: str = primal_sp_obj_sym
        self.default_fbl_sp_obj_sym: str = self.generate_unique_symbol(FBL_OBJ_SYMBOL)

        self.opt_cut_con_sym = self.generate_unique_symbol(OPT_CUT_CON_SYMBOL)
        self.fbl_cut_con_sym = self.generate_unique_symbol(FBL_CUT_CON_SYMBOL)
        self.can_int_cut_con_sym = self.generate_unique_symbol(CAN_INT_CUT_CON_SYMBOL)

        # --- Algorithm Meta-Entities ---

        # Meta-Sets
        self.idx_meta_sets: Dict[str, mat.MetaSet] = {}
        self.cuts: Optional[mat.MetaSet] = None

        # Meta-Parameters
        self.cut_count: Optional[mat.MetaParameter] = None
        self.is_feasible: Optional[mat.MetaParameter] = None
        self.stored_obj: Optional[mat.MetaParameter] = None
        self.stored_comp_decisions: Dict[str, mat.MetaParameter] = {}
        self.duality_multipliers: Dict[int, mat.MetaParameter] = {}

        # Meta-Variables

        self.comp_meta_vars: Dict[str, mat.MetaVariable] = {}

        self.eta: Optional[mat.MetaVariable] = None

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

    def build_mp_constructs(self, init_lb: float):

        self.cut_count = eb.build_meta_param(
            problem=self,
            symbol=self.cut_count_sym,
            default_value=0)
        self.add_meta_parameter(self.cut_count, is_in_model=False)

        self.cuts_unb_sym = self.generate_unique_symbol("ct")
        ord_set_node = mat.OrderedSetNode(start_node=mat.NumericNode(value=1),
                                          end_node=mat.DeclaredEntityNode(self.cut_count_sym,
                                                                          type=mat.PARAM_TYPE))
        self.cuts = eb.build_meta_set(
            problem=self,
            symbol=self.cuts_sym,
            dimension=1,
            dummy_symbols=[self.cuts_unb_sym],
            reduced_dummy_symbols=[self.cuts_unb_sym],
            defined_value_node=ord_set_node)
        self.add_meta_set(self.cuts, is_in_model=False)

        self.is_feasible = eb.build_meta_param(
            problem=self,
            symbol=self.is_feasible_sym,
            idx_meta_sets=[self.cuts],
            default_value=0)
        self.add_meta_parameter(self.is_feasible, is_in_model=False)

        self.stored_obj = eb.build_meta_param(
            problem=self,
            symbol=self.stored_obj_sym,
            idx_meta_sets=[self.cuts],
            default_value=0)
        self.add_meta_parameter(self.stored_obj, is_in_model=False)

        # Meta-Variables

        self.eta = eb.build_meta_var(
            problem=self,
            symbol=self.eta_sym,
            lower_bound=init_lb)
        self.add_meta_variable(self.eta, is_in_model=False)

    def get_idx_meta_sets(self) -> List[mat.MetaSet]:
        return [ms for ms in self.idx_meta_sets.values()]

    def get_comp_var_syms(self) -> List[str]:
        return [mv.get_symbol() for mv in self.comp_meta_vars.values()]
