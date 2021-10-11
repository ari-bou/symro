import symro
from symro.test.test_util import *
from symro.src.execution.amplengine import AMPLEngine


VERBOSITY = 1


def run_gbd_test_group():
    tests = [("Run GBD on an LP", gbd_lp),
             ("Run scenario-wise GBD on an LP", gbd_scenario_wise_lp),
             ("Run scenario-wise GBD on a production problem (convex QCP)", gbd_scenario_wise_convex_qp_production)]
    return run_tests(tests)


def gbd_lp() -> bool:

    engine = AMPLEngine()
    problem = symro.read_ampl("lp.run",
                              working_dir_path=SCRIPT_DIR_PATH,
                              engine=engine,
                              can_clean_script=True)

    engine.solve(solve_options={"solver": "gurobi", "gurobi_options": "outlev=1"})
    v_benchmark = engine.get_obj_value("OBJ")

    gbd = symro.GBDAlgorithm(problem,
                             mp_symbol="Master",
                             complicating_vars=["y"],
                             primal_sp_symbol="PrimalSubproblem",
                             fbl_sp_symbol="FeasibilitySubproblem",
                             init_lb=-100000,
                             init_ub=100000)
    gbd.setup()
    v, y = gbd.run(mp_solve_options={"solver": "gurobi"},
                   sp_solve_options={"solver": "gurobi"},
                   verbosity=VERBOSITY)

    return check_num_result(v, v_benchmark, 0.01)


def gbd_scenario_wise_lp() -> bool:

    engine = AMPLEngine()
    problem = symro.read_ampl("lp_sce.run",
                              working_dir_path=SCRIPT_DIR_PATH,
                              engine=engine,
                              can_clean_script=True)

    engine.solve(solve_options={"solver": "gurobi", "gurobi_options": "outlev=1"})
    v_benchmark = engine.get_obj_value("OBJ")

    gbd = symro.GBDAlgorithm(problem,
                             mp_symbol="Master",
                             complicating_vars=["y"],
                             primal_sp_symbol="PrimalSubproblem",
                             fbl_sp_symbol="FeasibilitySubproblem",
                             primal_sp_obj_symbol="OBJ_SUB",
                             init_lb=-100000,
                             init_ub=100000)
    gbd.add_decomposition_axes(idx_set_defs=["S"])
    gbd.setup()
    v, y = gbd.run(mp_solve_options={"solver": "gurobi"},
                   sp_solve_options={"solver": "gurobi"},
                   verbosity=VERBOSITY)

    return check_num_result(v, v_benchmark, 0.01)


def gbd_scenario_wise_convex_qp_production() -> bool:

    engine = AMPLEngine()
    problem = symro.read_ampl("convex_qp.run",
                              working_dir_path=SCRIPT_DIR_PATH,
                              engine=engine,
                              can_clean_script=True)

    engine.solve(solve_options={"solver": "cplex", "cplex_options": "outlev=1"})
    v_benchmark = -engine.get_obj_value("OBJ")

    gbd = symro.GBDAlgorithm(problem,
                             mp_symbol="Master",
                             complicating_vars=["INLET"],
                             primal_sp_symbol="PrimalSubproblem",
                             fbl_sp_symbol="FeasibilitySubproblem",
                             primal_sp_obj_symbol="OBJ_SUB",
                             init_lb=-1000000,
                             init_ub=1000000)
    gbd.add_decomposition_axes(idx_set_defs=["SCENARIOS"])
    gbd.setup()
    v, y = gbd.run(mp_solve_options={"solver": "cplex"},
                   sp_solve_options={"solver": "cplex"},
                   rel_opt_tol=0.001,
                   verbosity=VERBOSITY)

    return check_num_result(v, v_benchmark, 0.001)


def gbd_scenario_wise_convex_qp_refinery() -> bool:

    engine = AMPLEngine()
    problem = symro.read_ampl("refinery.run",
                              working_dir_path=SCRIPT_DIR_PATH,
                              engine=engine,
                              can_clean_script=True)

    engine.solve(solve_options={"solver": "gurobi", "gurobi_options": "outlev=1"})
    v_benchmark = -engine.get_obj_value("TOTAL_PROFIT")

    gbd = symro.GBDAlgorithm(problem,
                             mp_symbol="Master",
                             complicating_vars=["UNIT_SIZE", "TANK_SIZE", "TANK_LOCATION", "INIT_INVENTORY"],
                             primal_sp_symbol="PrimalSubproblem",
                             fbl_sp_symbol="FeasibilitySubproblem",
                             init_lb=-10000,
                             init_ub=1000000)
    gbd.add_decomposition_axes(["SCENARIOS"])
    gbd.setup()
    v, y = gbd.run(mp_solve_options={"solver": "knitro"},
                   sp_solve_options={"solver": "cplex"},
                   max_iter_count=10000,
                   verbosity=2)

    return check_num_result(v, v_benchmark, 0.01)
