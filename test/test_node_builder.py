import symro
import symro.src.handlers.nodebuilder as nb
from symro.test.test_util import *


# Scripts
# ----------------------------------------------------------------------------------------------------------------------

SCRIPT = """
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

def test_node_builder_entity_idx_set_node():

    problem = symro.read_ampl(script_literal=SCRIPT,
                              working_dir_path=SCRIPT_DIR_PATH)

    var_1 = problem.get_meta_entity("VAR_1")
    idx_set_node = nb.build_entity_idx_set_node(problem, var_1)
    assert str(idx_set_node) == "{n in NUM_SET}"

    var_2 = problem.get_meta_entity("VAR_2")

    idx_set_node = nb.build_entity_idx_set_node(problem, var_2)
    assert str(idx_set_node) == "{i in NUM_SET, (i,j) in NUM_ALPHA_SET}"

    idx_set_node = nb.build_entity_idx_set_node(problem, var_2, custom_dummy_syms={"NUM_SET": 'j'})
    assert str(idx_set_node) == "{j in NUM_SET, (j,j1) in NUM_ALPHA_SET}"

    var_3 = problem.get_meta_entity("VAR_3")
    idx_set_node = nb.build_entity_idx_set_node(problem, var_3)
    assert str(idx_set_node) == "{i in INDEXED_SET[1]}"
