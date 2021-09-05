import symro
from symro.test.test_util import *


def run_gbd_test_group():
    tests = [("Run GBD on LP", gbd_lp),
             ("Run scenario-wise GBD on LP", gbd_scenario_wise_lp),
             ("Run scenario-wise GBD on convex QCP", gbd_scenario_wise_convex_qcp)]
    return run_tests(tests)


def gbd_lp() -> bool:
    problem = symro.build_problem("lp.run",
                                  working_dir_path=SCRIPT_DIR_PATH)

    output = problem.engine.solve(solver_name="cplex", solver_options="outlev=2")
    print(output)
    obj = problem.engine.get_obj_value("OBJ")

    gbd = symro.GBDAlgorithm(problem,
                             mp_symbol="Master",
                             complicating_vars=["y"],
                             primal_sp_symbol="PrimalSubproblem",
                             fbl_sp_symbol="FeasibilitySubproblem",
                             init_lb=-100000,
                             init_ub=100000)
    gbd.setup()
    gbd.run(mp_solver_name="cplex",
            sp_solver_name="cplex",
            verbosity=0)

    print(obj)

    return True


def gbd_scenario_wise_lp() -> bool:
    problem = symro.build_problem("lp_sce.run",
                                  working_dir_path=SCRIPT_DIR_PATH)

    output = problem.engine.solve(solver_name="cplex", solver_options="outlev=2")
    print(output)
    obj = problem.engine.get_obj_value("OBJ")

    gbd = symro.GBDAlgorithm(problem,
                             mp_symbol="Master",
                             complicating_vars=["y"],
                             primal_sp_symbol="PrimalSubproblem",
                             fbl_sp_symbol="FeasibilitySubproblem",
                             primal_sp_obj_symbol="OBJ_SUB",
                             init_lb=-100000,
                             init_ub=100000)
    gbd.add_decomposition_axes(idx_set_symbols=["S"])
    gbd.setup()
    gbd.run(mp_solver_name="cplex",
            sp_solver_name="cplex",
            verbosity=0)

    print(obj)

    return True


def gbd_scenario_wise_convex_qcp() -> bool:
    problem = symro.build_problem("decomp.run",
                                  working_dir_path=SCRIPT_DIR_PATH)

    output = problem.engine.solve(solver_name="cplex", solver_options="outlev=2")
    print(output)
    obj = -problem.engine.get_obj_value("OBJ")

    gbd = symro.GBDAlgorithm(problem,
                             mp_symbol="Master",
                             complicating_vars=["INLET"],
                             primal_sp_symbol="PrimalSubproblem",
                             fbl_sp_symbol="FeasibilitySubproblem",
                             primal_sp_obj_symbol="OBJ_SUB",
                             init_lb=-1000000,
                             init_ub=1000000)
    gbd.add_decomposition_axes(idx_set_symbols=["SCENARIOS"])
    gbd.setup()
    gbd.run(mp_solver_name="cplex",
            sp_solver_name="cplex",
            rel_opt_tol=0.001,
            verbosity=0)

    print(obj)

    return True