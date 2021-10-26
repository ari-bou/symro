import amplpy
from copy import deepcopy
from ordered_set import OrderedSet
import os
from typing import List, Iterable, Optional, Tuple, Union
import warnings

import symro.src.mat as mat
from symro.src.prob.problem import BaseProblem, Problem
import symro.src.prob.statement as stm
import symro.src.handlers.nodebuilder as nb
import symro.src.handlers.metaentitybuilder as eb
from symro.src.execution.amplengine import AMPLEngine
from symro.src.parsing.amplscriptparser import AMPLScriptParser
import symro.src.util.util as util


# Problem Construction
# ----------------------------------------------------------------------------------------------------------------------

def __complete_meta_entity_construction(problem: Problem):
    eb.build_all_idx_meta_sets(problem)
    __build_declared_subproblems(problem)


# Subproblem Construction
# ----------------------------------------------------------------------------------------------------------------------


def build_subproblem(problem: Problem,
                     subproblem_symbol: str,
                     idx_set_def: str = None,
                     entity_defs: Iterable[str] = None):

    ampl_parser = AMPLScriptParser(problem)

    sp_idx_set_node = None
    if idx_set_def is not None:
        sp_idx_set_node = ampl_parser.parse_indexing_set_definition(idx_set_def)

    entity_nodes = []

    if entity_defs is None:
        meta_entities = []
        meta_entities.extend(problem.model_meta_vars)
        meta_entities.extend(problem.model_meta_objs)
        meta_entities.extend(problem.model_meta_cons)

    else:

        for entity_def in entity_defs:
            idx_set_node, entity_node = ampl_parser.parse_declared_entity_and_idx_set(entity_def)
            entity_nodes.append((idx_set_node, entity_node))

        meta_entities = __build_subproblem_meta_entities_from_nodes(
            problem=problem,
            sp_idx_set_node=sp_idx_set_node,
            entity_nodes=entity_nodes
        )

    sp = BaseProblem(symbol=subproblem_symbol,
                     idx_set_node=sp_idx_set_node)

    for me in meta_entities:
        sp.add_meta_entity(me)

    problem.add_subproblem(sp)

    return sp


def __build_subproblem_from_nodes(problem: Problem,
                                  sp_sym: str,
                                  sp_idx_set_node: mat.CompoundSetNode,
                                  entity_nodes: List[Tuple[Optional[mat.CompoundSetNode], mat.DeclaredEntityNode]]):

    sp = BaseProblem(symbol=sp_sym,
                     idx_set_node=sp_idx_set_node)

    meta_entities = __build_subproblem_meta_entities_from_nodes(
        problem=problem,
        sp_idx_set_node=sp_idx_set_node,
        entity_nodes=entity_nodes
    )

    for me in meta_entities:
        sp.add_meta_entity(me)

    return sp


def __build_subproblem_meta_entities_from_nodes(problem: Problem,
                                                sp_idx_set_node: mat.CompoundSetNode,
                                                entity_nodes: List[Tuple[Optional[mat.CompoundSetNode],
                                                                         mat.DeclaredEntityNode]]):

    meta_entities = []

    for me_idx_set_node, e_node in entity_nodes:

        meta_entity = problem.get_meta_entity(e_node.symbol)

        if sp_idx_set_node is None and me_idx_set_node is None and e_node.idx_node is None:
            meta_entities.append(meta_entity)

        else:
            idx_subset_node = nb.combine_idx_set_nodes([sp_idx_set_node, me_idx_set_node])
            sub_meta_entity = eb.build_sub_meta_entity(
                problem=problem,
                idx_subset_node=deepcopy(idx_subset_node),
                meta_entity=meta_entity,
                entity_idx_node=deepcopy(e_node.idx_node))
            meta_entities.append(sub_meta_entity)

    return meta_entities


def __build_declared_subproblems(problem: Problem):

    prob_statements = []

    for statement in problem.compound_script.main_script.statements:
        if isinstance(statement, stm.ProblemStatement):
            prob_statements.append(statement)

    for script in problem.compound_script.included_scripts.values():
        for statement in script.statements:
            if isinstance(statement, stm.ProblemStatement):
                prob_statements.append(statement)

    for prob_statement in prob_statements:
        sp = __build_subproblem_from_nodes(
            problem=problem,
            sp_sym=prob_statement.prob_node.symbol,
            sp_idx_set_node=prob_statement.idx_set_node,
            entity_nodes=prob_statement.item_nodes)
        problem.add_subproblem(sp)


# AMPL Input
# ----------------------------------------------------------------------------------------------------------------------

def read_ampl(file_name: str = None,
              script_literal: str = None,
              working_dir_path: str = None,
              name: str = None,
              description: str = None,
              engine: AMPLEngine = None,
              can_clean_script: bool = False) -> Optional[Problem]:

    if file_name is None and script_literal is None:
        raise ValueError("Problem builder requires either a file name or a script literal.")

    if working_dir_path is None:
        working_dir_path = os.getcwd()

    if name is None and file_name is not None:
        file_name = os.path.basename(file_name)
        name = os.path.splitext(file_name)[0]

    problem = Problem(symbol=name,
                      description=description,
                      working_dir_path=working_dir_path)

    if script_literal is None:
        script_literal = util.read_file(working_dir_path, file_name)

    # Parse script
    ampl_parser = AMPLScriptParser(problem, working_dir_path=working_dir_path)
    problem.compound_script = ampl_parser.parse_script(script_literal)
    del ampl_parser

    try:

        if engine is None:
            engine = AMPLEngine(problem,
                                can_clean_script=can_clean_script)
        else:
            engine.setup_ampl_engine(problem,
                                     can_clean_script=can_clean_script)

        __retrieve_problem_data_from_ampl_engine(problem, engine)  # retrieve data

    except SystemError as e:
        print(e)
        warnings.warn("Problem builder encountered a system error while setting up the AMPL engine")

    __complete_meta_entity_construction(problem)  # construct indexing meta-sets and subproblems

    return problem


def __retrieve_problem_data_from_ampl_engine(problem: Problem,
                                             ampl_engine: AMPLEngine):
    __retrieve_set_data_from_ampl_engine(problem, ampl_engine)
    __retrieve_param_data_from_ampl_engine(problem, ampl_engine)


def __retrieve_set_data_from_ampl_engine(problem: Problem,
                                         ampl_engine: AMPLEngine):
    sets = ampl_engine.api.getSets()
    for sym, ampl_set in sets:

        ampl_set: amplpy.Set

        # set the dimension of the meta-set
        meta_set = problem.meta_sets[sym]
        meta_set.set_dim(ampl_set.arity())
        if not meta_set.is_init:
            meta_set.initialize()

        # build the set entity

        if ampl_set.indexarity() == 0:  # non-indexed set
            raw_elements = [m for m in ampl_set.members()]
            elements = __process_set_elements(raw_elements)
            problem.state.add_set(
                symbol=sym,
                dim=ampl_set.arity(),
                elements=elements
            )

        else:  # indexed set
            for raw_indices, ampl_set_instance in ampl_set.instances():

                indices = AMPLEngine.standardize_indices(raw_indices)

                raw_elements = [m for m in ampl_set_instance.getValues().toDict().keys()]
                elements = __process_set_elements(raw_elements)

                problem.state.add_set(
                    symbol=sym,
                    idx=indices,
                    dim=ampl_set.arity(),
                    elements=elements
                )


def __retrieve_param_data_from_ampl_engine(problem: Problem,
                                           ampl_engine: AMPLEngine):
    params = ampl_engine.api.getParameters()
    for sym, param in params:

        param: amplpy.Parameter

        # scalar parameter
        if param.indexarity() == 0:
            problem.state.add_parameter(symbol=sym,
                                        value=param.value())

        # indexed parameter
        else:
            for raw_indices, value in param.instances():

                indices = AMPLEngine.standardize_indices(raw_indices)

                if value is None:
                    value = 0

                problem.state.add_parameter(symbol=sym,
                                            idx=indices,
                                            value=value)


def __standardize_indices_from_ampl_engine(raw_indices: Union[int, float, str,
                                                              Tuple[Union[int, float, str], ...],
                                                              List[Union[int, float, str]],
                                                              None]):
    if raw_indices is None:
        return None
    elif isinstance(raw_indices, tuple):
        return raw_indices
    elif isinstance(raw_indices, list):
        return tuple(raw_indices)
    else:
        return tuple([raw_indices])


def __process_set_elements(raw_elements: List[Union[int, float, str, tuple]]):
    elements = OrderedSet()
    for e_raw in raw_elements:
        if not isinstance(e_raw, tuple):
            e_raw = [e_raw]
        e = []
        for e_i in e_raw:
            if isinstance(e_i, float):
                if e_i.is_integer():
                    e_i = int(e_i)
            e.append(e_i)
        elements.append(tuple(e))
    return elements
