from ordered_set import OrderedSet
import os
from typing import List, Optional, Tuple, Union
import warnings

import symro.core.mat as mat
from symro.core.prob.problem import Problem
import symro.core.prob.statement as stm
from symro.core.handlers.entitybuilder import EntityBuilder
from symro.core.execution.amplengine import AMPLEngine
from symro.core.parsing.amplscriptparser import AMPLScriptParser
import symro.core.util.util as util


def build_problem_from_ampl_script(file_name: str = None,
                                   script_literal: str = None,
                                   working_dir_path: str = None,
                                   name: str = None,
                                   description: str = None,
                                   engine: AMPLEngine = None,
                                   can_clean_script: bool = False) -> Optional[Problem]:

    if file_name is None and script_literal is None:
        raise ValueError("Problem builder requires either a file name or a script literal.")

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
            engine = AMPLEngine(problem)
        else:
            engine.setup_ampl_engine(problem,
                                     can_clean_script=can_clean_script)

        __retrieve_problem_data_from_ampl_engine(problem, engine)  # retrieve data

    except SystemError as e:
        print(e)
        warnings.warn("Problem builder encountered a system error while setting up the AMPL engine")

    __complete_meta_entity_construction(problem)  # construct indexing meta-sets and subproblems

    return problem


# Problem Data
# ----------------------------------------------------------------------------------------------------------------------

def __retrieve_problem_data_from_ampl_engine(problem: Problem,
                                             ampl_engine: AMPLEngine):
    __retrieve_set_data_from_ampl_engine(problem, ampl_engine)
    __retrieve_param_data_from_ampl_engine(problem, ampl_engine)


def __retrieve_set_data_from_ampl_engine(problem: Problem,
                                         ampl_engine: AMPLEngine):
    sets = ampl_engine.api.getSets()
    for sym, ampl_set in sets:

        # set the dimension of the meta-set
        meta_set = problem.meta_sets[sym]
        meta_set.dimension = ampl_set.arity()
        if not meta_set.is_init:
            meta_set.initialize()

        # build the set entity

        if ampl_set.indexarity() == 0:  # non-indexed set
            raw_elements = [m for m in ampl_set.members()]
            elements = __process_set_elements(raw_elements)
            aset = mat.SSet(symbol=sym,
                            dim=ampl_set.arity(),
                            elements=elements)
            problem.state.add_set(aset)

        else:  # indexed set
            for raw_indices, ampl_set_instance in ampl_set.instances():
                indices = mat.Entity.standardize_indices(raw_indices)

                raw_elements = [m for m in ampl_set_instance.getValues().toDict().keys()]
                elements = __process_set_elements(raw_elements)

                aset = mat.SSet(symbol=sym,
                                idx=indices,
                                dim=ampl_set.arity(),
                                elements=elements)
                problem.state.add_set(aset)


def __retrieve_param_data_from_ampl_engine(problem: Problem,
                                           ampl_engine: AMPLEngine):
    params = ampl_engine.api.getParameters()
    for sym, param in params:

        if param.indexarity() == 0:
            problem.state.add_parameter(mat.Parameter(symbol=sym,
                                                      value=param.value()))
        else:
            for raw_indices, value in param.instances():
                indices = __standardize_indices_from_api(raw_indices)
                if value is None:
                    value = 0
                problem.state.add_parameter(mat.Parameter(symbol=sym,
                                                          idx=indices,
                                                          value=value))


def __standardize_indices_from_api(raw_indices: Union[int, float, str,
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


# Meta-Entity Construction
# ----------------------------------------------------------------------------------------------------------------------

def __complete_meta_entity_construction(problem: Problem):
    entity_builder = EntityBuilder(problem)
    entity_builder.build_all_idx_meta_sets()
    __build_declared_subproblems(problem, entity_builder)


def __build_declared_subproblems(problem: Problem,
                                 entity_builder: EntityBuilder):

    prob_statements = []

    for statement in problem.compound_script.main_script.statements:
        if isinstance(statement, stm.ProblemStatement):
            prob_statements.append(statement)

    for script in problem.compound_script.included_scripts.values():
        for statement in script.statements:
            if isinstance(statement, stm.ProblemStatement):
                prob_statements.append(statement)

    for prob_statement in prob_statements:
        sp = entity_builder.build_subproblem(prob_sym=prob_statement.prob_node.symbol,
                                             prob_idx_set_node=prob_statement.idx_set_node,
                                             entity_nodes=prob_statement.item_nodes)
        problem.add_subproblem(sp)


# Utility
# ------------------------------------------------------------------------------------------------------------------

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
