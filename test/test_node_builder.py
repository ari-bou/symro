import symro
from symro.core.handlers.nodebuilder import NodeBuilder
from symro.test.test_util import *


# Scripts
# ----------------------------------------------------------------------------------------------------------------------

IDX_SET_EDGE_SCRIPT = """
set NUM_SET = {1, 2, 3};
set ALPHA_SET = {'A', 'B', 'C'};
set GREEKALPHA_SET = {"alpha", "beta", "gamma"};

set NUM_ALPHA_SET = {(1, 'A'), (1, 'B'), (2, 'A'), (2, 'C'), (3, 'B'), (3, 'C')};

set INDEXED_SET{i in NUM_SET} = 1..i;
set INDEXED_SET_2{i in NUM_SET} = {(i,j) in NUM_ALPHA_SET};

var VAR_1{NUM_SET} >= 0;
var VAR_2{i in NUM_SET, (i,j) in NUM_ALPHA_SET};
var VAR_3{INDEXED_SET[1]};

minimize OBJ: 0;
"""


# Tests
# ----------------------------------------------------------------------------------------------------------------------

def run_node_builder_test_group():
    tests = [("Build indexing set nodes for meta-entities", node_builder_entity_idx_set_node_test)]
    return run_tests(tests)


def node_builder_entity_idx_set_node_test():

    problem = symro.build_problem(script_literal=IDX_SET_EDGE_SCRIPT,
                                  working_dir_path=SCRIPT_DIR_PATH)
    node_builder = NodeBuilder(problem)

    results = []

    var_1 = problem.get_meta_entity("VAR_1")
    idx_set_node = node_builder.build_entity_idx_set_node(var_1)
    results.append(check_result(idx_set_node, "{n in NUM_SET}"))

    var_2 = problem.get_meta_entity("VAR_2")

    idx_set_node = node_builder.build_entity_idx_set_node(var_2)
    results.append(check_result(idx_set_node, "{i in NUM_SET, (i,j) in NUM_ALPHA_SET}"))

    idx_set_node = node_builder.build_entity_idx_set_node(var_2,
                                                          custom_dummy_syms={"NUM_SET": 'j'})
    results.append(check_result(idx_set_node, "{j in NUM_SET, (j,j1) in NUM_ALPHA_SET}"))

    var_3 = problem.get_meta_entity("VAR_3")
    idx_set_node = node_builder.build_entity_idx_set_node(var_3)
    results.append(check_result(idx_set_node, "{i in INDEXED_SET[1]}"))

    return results
