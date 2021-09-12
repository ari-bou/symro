import symro
from symro.test.test_util import *


VERBOSITY = 1


def run_gbd_test_group():
    tests = [
             ("Run GBD on an LP", gbd_lp),
             ("Run scenario-wise GBD on an LP", gbd_scenario_wise_lp),
             ("Run scenario-wise GBD on a production problem (convex QCP)", gbd_scenario_wise_convex_qp_production)]
    return run_tests(tests)


def gbd_lp() -> bool:
    problem = symro.read_ampl("lp.run",
                              working_dir_path=SCRIPT_DIR_PATH)

    problem.engine.solve(solver_name="gurobi", solver_options="outlev=1")
    v_benchmark = problem.engine.get_obj_value("OBJ")

    gbd = symro.GBDAlgorithm(problem,
                             mp_symbol="Master",
                             complicating_vars=["y"],
                             primal_sp_symbol="PrimalSubproblem",
                             fbl_sp_symbol="FeasibilitySubproblem",
                             init_lb=-100000,
                             init_ub=100000)
    gbd.setup()
    v, y = gbd.run(mp_solver_name="gurobi",
                   sp_solver_name="gurobi",
                   verbosity=VERBOSITY)

    return check_num_result(v, v_benchmark, 0.01)


def gbd_scenario_wise_lp() -> bool:
    problem = symro.read_ampl("lp_sce.run",
                              working_dir_path=SCRIPT_DIR_PATH)

    problem.engine.solve(solver_name="gurobi", solver_options="outlev=1")
    v_benchmark = problem.engine.get_obj_value("OBJ")

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
    v, y = gbd.run(mp_solver_name="gurobi",
                   sp_solver_name="gurobi",
                   verbosity=VERBOSITY)

    return check_num_result(v, v_benchmark, 0.01)


def gbd_scenario_wise_convex_qp_production() -> bool:
    problem = symro.read_ampl("convex_qp.run",
                              working_dir_path=SCRIPT_DIR_PATH)

    problem.engine.solve(solver_name="gurobi", solver_options="outlev=1")
    v_benchmark = -problem.engine.get_obj_value("OBJ")

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
    v, y = gbd.run(mp_solver_name="gurobi",
                   sp_solver_name="gurobi",
                   rel_opt_tol=0.001,
                   verbosity=VERBOSITY)

    return check_num_result(v, v_benchmark, 0.001)


def gbd_scenario_wise_convex_qp_refinery() -> bool:
    problem = symro.read_ampl("refinery.run",
                              working_dir_path=SCRIPT_DIR_PATH)

    problem.engine.solve(solver_name="gurobi", solver_options="outlev=1")
    v_benchmark = -problem.engine.get_obj_value("TOTAL_PROFIT")

    gbd = symro.GBDAlgorithm(problem,
                             mp_symbol="Master",
                             complicating_vars=["UNIT_SIZE", "TANK_SIZE", "TANK_LOCATION", "INIT_INVENTORY"],
                             primal_sp_symbol="PrimalSubproblem",
                             fbl_sp_symbol="FeasibilitySubproblem",
                             init_lb=-10000,
                             init_ub=1000000)
    gbd.add_decomposition_axes(["SCENARIOS"])
    gbd.setup()
    v, y = gbd.run(mp_solver_name="knitro",
                   sp_solver_name="cplex",
                   max_iter_count=10000,
                   verbosity=2)

    return check_num_result(v, v_benchmark, 0.01)
