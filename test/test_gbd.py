import pytest

import symro
from symro.execution.amplengine import AMPLEngine
from .test_util import *


VERBOSITY = 1


def test_gbd_lp():

    engine = AMPLEngine()
    problem = symro.read_ampl(
        "lp.run", working_dir_path=SCRIPT_DIR_PATH, engine=engine, can_clean_script=True
    )
    print(problem.primal_to_dat("sol.dat"))

    engine.solve(solve_options={"solver": "gurobi", "gurobi_options": "outlev=1"})
    v_benchmark = engine.get_obj_value("OBJ")

    gbd = symro.GBDAlgorithm(
        problem,
        mp_symbol="Master",
        complicating_vars=["y"],
        primal_sp_symbol="PrimalSubproblem",
        fbl_sp_symbol="FeasibilitySubproblem",
        init_lb=-100000,
        init_ub=100000,
    )
    gbd.setup()
    v, y = gbd.run(
        mp_solve_options={"solver": "gurobi"},
        sp_solve_options={"solver": "gurobi"},
        verbosity=VERBOSITY,
    )

    assert v == pytest.approx(v_benchmark, 0.01)


def test_gbd_scenario_wise_lp():

    engine = AMPLEngine()
    problem = symro.read_ampl(
        "lp_sce.run",
        working_dir_path=SCRIPT_DIR_PATH,
        engine=engine,
        can_clean_script=True,
    )

    engine.solve(solve_options={"solver": "gurobi", "gurobi_options": "outlev=1"})
    v_benchmark = engine.get_obj_value("OBJ")

    gbd = symro.GBDAlgorithm(
        problem,
        mp_symbol="Master",
        complicating_vars=["y"],
        primal_sp_symbol="PrimalSubproblem",
        fbl_sp_symbol="FeasibilitySubproblem",
        primal_sp_obj_symbol="OBJ_SUB",
        init_lb=-100000,
        init_ub=100000,
    )
    gbd.add_decomposition_axes(idx_set_defs=["S"])
    gbd.setup()
    v, y = gbd.run(
        mp_solve_options={"solver": "gurobi"},
        sp_solve_options={"solver": "gurobi"},
        verbosity=VERBOSITY,
    )

    assert v == pytest.approx(v_benchmark, 0.01)


def test_gbd_scenario_wise_convex_qp_production():

    engine = AMPLEngine()
    problem = symro.read_ampl(
        "convex_qp_s3_t1.run",
        working_dir_path=SCRIPT_DIR_PATH,
        engine=engine,
        can_clean_script=True,
    )

    engine.solve(solve_options={"solver": "cplex", "cplex_options": "outlev=1"})
    v_benchmark = -engine.get_obj_value("OBJ")

    gbd = symro.GBDAlgorithm(
        problem,
        mp_symbol="Master",
        complicating_vars=[
            "{r in RAW_MATERIALS, t in TIMEPERIODS, s in SCENARIOS} NODE[r,t,s]"
        ],
        primal_sp_symbol="PrimalSubproblem",
        fbl_sp_symbol="FeasibilitySubproblem",
        primal_sp_obj_symbol="OBJ_SUB",
        init_lb=-1000000,
        init_ub=1000000,
    )
    gbd.add_decomposition_axes(idx_set_defs=["SCENARIOS"])
    gbd.setup()
    v, y = gbd.run(
        mp_solve_options={"solver": "cplex"},
        sp_solve_options={"solver": "cplex"},
        rel_opt_tol=0.001,
        verbosity=VERBOSITY,
    )

    assert v == pytest.approx(v_benchmark, 0.01)


def test_gbd_scenario_temporal_convex_qp_production():

    engine = AMPLEngine()
    problem = symro.read_ampl(
        "convex_qp_s3_t3.run",
        working_dir_path=SCRIPT_DIR_PATH,
        engine=engine,
        can_clean_script=True,
    )

    engine.solve(solve_options={"solver": "cplex", "cplex_options": "outlev=1"})
    v_benchmark = -engine.get_obj_value("OBJ")

    gbd = symro.GBDAlgorithm(
        problem,
        mp_symbol="Master",
        complicating_vars=[
            "{r in RAW_MATERIALS, t in TIMEPERIODS, s in SCENARIOS} NODE[r,t,s]"
        ],
        primal_sp_symbol="PrimalSubproblem",
        fbl_sp_symbol="FeasibilitySubproblem",
        primal_sp_obj_symbol="OBJ_SUB",
        init_lb=-1000000,
        init_ub=1000000,
    )
    gbd.add_decomposition_axes(idx_set_defs=["SCENARIOS", "TIMEPERIODS"])
    gbd.setup()
    v, y = gbd.run(
        mp_solve_options={"solver": "cplex"},
        sp_solve_options={"solver": "cplex"},
        rel_opt_tol=0.001,
        verbosity=VERBOSITY,
    )

    assert v == pytest.approx(v_benchmark, 0.01)


@pytest.mark.skip(reason="runtime is too long")
def test_gbd_scenario_wise_convex_qp_refinery():

    engine = AMPLEngine()
    problem = symro.read_ampl(
        "refinery.run",
        working_dir_path=SCRIPT_DIR_PATH,
        engine=engine,
        can_clean_script=True,
    )

    engine.solve(solve_options={"solver": "gurobi", "gurobi_options": "outlev=1"})
    v_benchmark = -engine.get_obj_value("TOTAL_PROFIT")

    gbd = symro.GBDAlgorithm(
        problem,
        mp_symbol="Master",
        complicating_vars=["UNIT_SIZE", "TANK_SIZE", "TANK_LOCATION", "INIT_INVENTORY"],
        primal_sp_symbol="PrimalSubproblem",
        fbl_sp_symbol="FeasibilitySubproblem",
        init_lb=-10000,
        init_ub=1000000,
    )
    gbd.add_decomposition_axes(["SCENARIOS"])
    gbd.setup()
    v, y = gbd.run(
        mp_solve_options={"solver": "knitro"},
        sp_solve_options={"solver": "cplex"},
        max_iter_count=10000,
        verbosity=2,
    )

    assert v == pytest.approx(v_benchmark, 0.01)


def test_ngbd_lp():

    engine = AMPLEngine()
    problem = symro.read_ampl(
        "lp.run", working_dir_path=SCRIPT_DIR_PATH, engine=engine, can_clean_script=True
    )
    print(problem.primal_to_dat("sol.dat"))

    engine.solve(solve_options={"solver": "gurobi", "gurobi_options": "outlev=1"})
    v_benchmark = engine.get_obj_value("OBJ")

    convexifier = symro.Convexifier()
    convex_relaxation = convexifier.convexify_problem(
        problem=problem,
    )

    ngbd = symro.NGBDAlgorithm(
        problem,
        convex_relaxation=convex_relaxation,
        mp_symbol="Master",
        complicating_vars=["y"],
        primal_sp_symbol="PrimalSubproblem",
        fbl_sp_symbol="FeasibilitySubproblem",
        init_lb=-100000,
        init_ub=100000,
    )
    ngbd.setup()
    v, y = ngbd.run(
        mp_solve_options={"solver": "gurobi"},
        c_sp_solve_options={"solver": "gurobi"},
        nc_sp_solve_options={"solver": "gurobi"},
        verbosity=VERBOSITY,
    )

    assert v == pytest.approx(v_benchmark, 0.01)


def test_ngbd_scenario_wise_lp():

    engine = AMPLEngine()
    problem = symro.read_ampl(
        "lp_sce.run",
        working_dir_path=SCRIPT_DIR_PATH,
        engine=engine,
        can_clean_script=True,
    )

    engine.solve(solve_options={"solver": "gurobi", "gurobi_options": "outlev=1"})
    v_benchmark = engine.get_obj_value("OBJ")

    convexifier = symro.Convexifier()
    convex_relaxation = convexifier.convexify_problem(
        problem=problem,
    )

    ngbd = symro.NGBDAlgorithm(
        problem,
        convex_relaxation=convex_relaxation,
        mp_symbol="Master",
        complicating_vars=["y"],
        primal_sp_symbol="PrimalSubproblem",
        fbl_sp_symbol="FeasibilitySubproblem",
        init_lb=-100000,
        init_ub=100000,
    )
    ngbd.add_decomposition_axes(idx_set_defs=["S"])
    ngbd.setup()
    v, y = ngbd.run(
        mp_solve_options={"solver": "gurobi"},
        c_sp_solve_options={"solver": "gurobi"},
        nc_sp_solve_options={"solver": "gurobi"},
        verbosity=4,
    )

    assert v == pytest.approx(v_benchmark, 0.01)


def test_ngbd_scenario_wise_convex_qp_production():

    engine = AMPLEngine()
    problem = symro.read_ampl(
        "convex_qp_s3_t1.run",
        working_dir_path=SCRIPT_DIR_PATH,
        engine=engine,
        can_clean_script=True,
    )

    engine.solve(solve_options={"solver": "cplex", "cplex_options": "outlev=1"})
    v_benchmark = -engine.get_obj_value("OBJ")

    convexifier = symro.Convexifier()
    convex_relaxation = convexifier.convexify_problem(
        problem=problem,
    )

    ngbd = symro.NGBDAlgorithm(
        problem,
        convex_relaxation=convex_relaxation,
        mp_symbol="Master",
        complicating_vars=[
            "{r in RAW_MATERIALS, t in TIMEPERIODS, s in SCENARIOS} NODE[r,t,s]"
        ],
        primal_sp_symbol="PrimalSubproblem",
        fbl_sp_symbol="FeasibilitySubproblem",
        primal_sp_obj_symbol="OBJ_SUB",
        init_lb=-1000000,
        init_ub=1000000,
    )
    ngbd.add_decomposition_axes(idx_set_defs=["SCENARIOS"])
    ngbd.setup()
    v, y = ngbd.run(
        mp_solve_options={"solver": "cplex"},
        c_sp_solve_options={"solver": "cplex"},
        nc_sp_solve_options={"solver": "cplex"},
        rel_opt_tol=0.001,
        verbosity=2,
    )

    assert v == pytest.approx(v_benchmark, 0.01)


def test_ngbd_scenario_temporal_convex_qp_production():

    engine = AMPLEngine()
    problem = symro.read_ampl(
        "convex_qp_s3_t3.run",
        working_dir_path=SCRIPT_DIR_PATH,
        engine=engine,
        can_clean_script=True,
    )

    engine.solve(solve_options={"solver": "cplex", "cplex_options": "outlev=1"})
    v_benchmark = -engine.get_obj_value("OBJ")

    convexifier = symro.Convexifier()
    convex_relaxation = convexifier.convexify_problem(
        problem=problem,
    )

    ngbd = symro.NGBDAlgorithm(
        problem,
        convex_relaxation=convex_relaxation,
        mp_symbol="Master",
        complicating_vars=[
            "{r in RAW_MATERIALS, t in TIMEPERIODS, s in SCENARIOS} NODE[r,t,s]"
        ],
        primal_sp_symbol="PrimalSubproblem",
        fbl_sp_symbol="FeasibilitySubproblem",
        primal_sp_obj_symbol="OBJ_SUB",
        init_lb=-1000000,
        init_ub=1000000,
    )
    ngbd.add_decomposition_axes(idx_set_defs=["SCENARIOS", "TIMEPERIODS"])
    ngbd.setup()
    v, y = ngbd.run(
        mp_solve_options={"solver": "cplex"},
        c_sp_solve_options={"solver": "cplex"},
        nc_sp_solve_options={"solver": "cplex"},
        rel_opt_tol=0.001,
        verbosity=2,
    )

    assert v == pytest.approx(v_benchmark, 0.01)


def test_ngbd_nonconvex_qp():

    engine = AMPLEngine()
    problem = symro.read_ampl(
        "nonconvex_qp.run",
        working_dir_path=SCRIPT_DIR_PATH,
        engine=engine,
        can_clean_script=True,
    )

    engine.solve(solve_options={"solver": "ipopt", "gurobi_options": "outlev=1"})
    v_benchmark = engine.get_obj_value("OBJ")

    convexifier = symro.Convexifier()
    convex_relaxation = convexifier.convexify_problem(
        problem=problem,
    )

    ngbd = symro.NGBDAlgorithm(
        problem,
        convex_relaxation=convex_relaxation,
        mp_symbol="Master",
        complicating_vars=["y"],
        primal_sp_symbol="PrimalSubproblem",
        fbl_sp_symbol="FeasibilitySubproblem",
        init_lb=-100000,
        init_ub=100000,
    )
    ngbd.setup()
    v, y = ngbd.run(
        mp_solve_options={"solver": "gurobi"},
        c_sp_solve_options={"solver": "ipopt"},
        nc_sp_solve_options={"solver": "ipopt"},
        verbosity=2,
    )

    assert v == pytest.approx(v_benchmark, 0.01)
