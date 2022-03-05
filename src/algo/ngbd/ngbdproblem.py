from typing import List, Optional

from symro.src.prob.problem import Problem, BaseProblem
from symro.src.algo.gbd import GBDProblem, GBDSubproblemContainer


class NGBDProblem(GBDProblem):

    def __init__(self,
                 problem: Problem,
                 mp_symbol: str,
                 default_primal_sp_symbol: str,
                 default_fbl_sp_symbol: str,
                 primal_sp_obj_sym: str,
                 working_dir_path: str = None):

        super().__init__(
            problem=problem,
            mp_symbol=mp_symbol,
            default_primal_sp_symbol=default_primal_sp_symbol,
            default_fbl_sp_symbol=default_fbl_sp_symbol,
            primal_sp_obj_sym=primal_sp_obj_sym,
            working_dir_path=working_dir_path
        )

        # --- Problems ---
        self.origin_problem: Optional[Problem] = None
        self.origin_primal_sps: List[BaseProblem] = []
        self.nc_sp_containers: List[GBDSubproblemContainer] = []
