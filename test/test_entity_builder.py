import symro
import symro.src.handlers.metaentitybuilder as eb
from symro.src.parsing.amplparser import AMPLParser
from symro.test.test_util import *


# Scripts
# ----------------------------------------------------------------------------------------------------------------------

SUB_SET_SCRIPT = """set NUM_SET = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
set EVEN_SET = {0, 2, 4, 6, 8};
set LETTER_SET = {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'};
set VOWEL_SET = {'A', 'E', 'I'};
set NUM_LETTER_SET = {NUM_SET, LETTER_SET};
set INDEXED_SET{i in NUM_SET} = 0..i;
set INDEXED_SET_2{i in NUM_SET} = {(i,j) in NUM_LETTER_SET};
var VAR_1{i in NUM_SET} >= 0;
var VAR_2{i in NUM_SET, j in LETTER_SET} >= 0;
var VAR_test{i in NUM_SET: 1 in union{i1 in NUM_SET}{1..1: i == 5}};
minimize OBJ: 0;
display {i in NUM_SET: 1 in union{i1 in NUM_SET}{1..1: i == 5}};
"""


# Tests
# ----------------------------------------------------------------------------------------------------------------------


def run_entity_builder_test_group():
    tests = [("Build sub-meta-entities", sub_meta_entity_builder_test)]
    return run_tests(tests)


def sub_meta_entity_builder_test():

    problem = symro.read_ampl(script_literal=SUB_SET_SCRIPT,
                              working_dir_path=SCRIPT_DIR_PATH)

    ampl_parser = AMPLParser(problem)

    results = []

    mv_1 = problem.get_meta_entity("VAR_1")
    mv_2 = problem.get_meta_entity("VAR_2")

    # test 1: {i in NUM_SET} VAR_1[i]
    idx_node = ampl_parser.parse_entity_index("[i]")
    sub_meta_entity = eb.build_sub_meta_entity(
        problem=problem,
        idx_subset_node=mv_1.idx_set_node,
        meta_entity=mv_1,
        entity_idx_node=idx_node)
    results.append(check_str_result(sub_meta_entity, "var VAR_1{i in NUM_SET}"))

    # test 2: {i in EVEN_SET} VAR_1[i]
    idx_subset_node = ampl_parser.parse_indexing_set_definition("{i in EVEN_SET}")
    idx_node = ampl_parser.parse_entity_index("[i]")
    sub_meta_entity = eb.build_sub_meta_entity(
        problem=problem,
        idx_subset_node=idx_subset_node,
        meta_entity=mv_1,
        entity_idx_node=idx_node)
    results.append(check_str_result(sub_meta_entity, "var VAR_1{i in NUM_SET: i in {i2 in EVEN_SET}}"))

    # test 3: {i in NUM_SET} VAR_1[5]
    idx_node = ampl_parser.parse_entity_index("[5]")
    sub_meta_entity = eb.build_sub_meta_entity(
        problem=problem,
        idx_subset_node=mv_1.idx_set_node,
        meta_entity=mv_1,
        entity_idx_node=idx_node)
    results.append(check_str_result(sub_meta_entity, "var VAR_1{i in NUM_SET: i == 5}"))

    # test 4: {i in EVEN_SET, j in VOWEL_SET} VAR_2[i,j]
    idx_subset_node = ampl_parser.parse_indexing_set_definition("{i in EVEN_SET, j in VOWEL_SET}")
    idx_node = ampl_parser.parse_entity_index("[i,j]")
    sub_meta_entity = eb.build_sub_meta_entity(
        problem=problem,
        idx_subset_node=idx_subset_node,
        meta_entity=mv_2,
        entity_idx_node=idx_node)
    s = "var VAR_2{i in NUM_SET, j in LETTER_SET: (i,j) in {i3 in EVEN_SET, j1 in VOWEL_SET}}"
    results.append(check_str_result(sub_meta_entity, s))

    # test 5: {i in NUM_SET, j in INDEXED_SET[i]} VAR_1[j]
    idx_subset_node = ampl_parser.parse_indexing_set_definition("{i in NUM_SET, j in INDEXED_SET[i]}")
    idx_node = ampl_parser.parse_entity_index("[j]")
    sub_meta_entity = eb.build_sub_meta_entity(
        problem=problem,
        idx_subset_node=idx_subset_node,
        meta_entity=mv_1,
        entity_idx_node=idx_node)
    s = "var VAR_1{i in NUM_SET: i in union{i4 in NUM_SET}{j2 in INDEXED_SET[i]}}"
    results.append(check_str_result(sub_meta_entity, s))

    # test 6: {i in NUM_SET, (j,k) in INDEXED_SET_2[i]} VAR_2[j,k]
    idx_subset_node = ampl_parser.parse_indexing_set_definition("{i in NUM_SET, (j,k) in INDEXED_SET_2[i]}")
    idx_node = ampl_parser.parse_entity_index("[j,k]")
    sub_meta_entity = eb.build_sub_meta_entity(
        problem=problem,
        idx_subset_node=idx_subset_node,
        meta_entity=mv_2,
        entity_idx_node=idx_node)
    s = "var VAR_2{i in NUM_SET, j in LETTER_SET: (i,j) in union{i5 in NUM_SET}{(j3,k1) in INDEXED_SET_2[i]}}"
    results.append(check_str_result(sub_meta_entity, s))

    # problem.engine.api.reset()
    # problem.engine.api.eval(SUB_SET_SCRIPT)

    return results
