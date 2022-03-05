from typing import Iterable, List, Optional, Set, Union

import symro.src.mat as mat
from symro.src.prob.problem import BaseProblem, Problem
import symro.src.scripting.script as scr
import symro.src.scripting.amplstatement as ampl_stm
import symro.src.handlers.nodebuilder as nb


def model_to_ampl(problem: Problem,
                  file_name: str = None,
                  working_dir_path: str = None):

    script_builder = ScriptBuilder()
    script = script_builder.generate_problem_model_script(problem=problem,
                                                          model_file_name=file_name,
                                                          model_file_extension=None)

    if working_dir_path is None:
        working_dir_path = problem.working_dir_path
    script.write(dir_path=working_dir_path, file_name=file_name)


class ScriptBuilder:

    def __init__(self):
        self._script: Optional[scr.Script] = None
        self._defined_syms: Optional[Set[str]] = None

    # Model Script
    # ------------------------------------------------------------------------------------------------------------------

    def generate_problem_model_script(self,
                                      problem: BaseProblem,
                                      model_file_name: str = None,
                                      model_file_extension: Optional[str] = ".mod"):
        if model_file_name is None:
            model_file_name = problem.symbol
        return self.generate_model_script(model_file_name=model_file_name,
                                          model_file_extension=model_file_extension,
                                          meta_sets_params=problem.model_meta_sets_params,
                                          meta_vars=problem.model_meta_vars,
                                          meta_objs=problem.model_meta_objs,
                                          meta_cons=problem.model_meta_cons)

    def generate_model_script(self,
                              model_file_name: str = None,
                              model_file_extension: Optional[str] = ".mod",
                              meta_sets: Iterable[mat.MetaSet] = None,
                              meta_params: Iterable[mat.MetaParameter] = None,
                              meta_sets_params: Iterable[Union[mat.MetaSet, mat.MetaParameter]] = None,
                              meta_vars: Iterable[mat.MetaVariable] = None,
                              meta_objs: Iterable[mat.MetaObjective] = None,
                              meta_cons: Iterable[mat.MetaConstraint] = None,
                              include_sets: bool = True,
                              include_params: bool = True,
                              include_variables: bool = True,
                              include_objectives: bool = True,
                              include_constraints: bool = True) -> scr.Script:

        def transform_entity_collection(entity_collection):
            if entity_collection is None:
                entity_collection = []
            return entity_collection

        meta_sets = transform_entity_collection(meta_sets)
        meta_params = transform_entity_collection(meta_params)
        meta_sets_params = transform_entity_collection(meta_sets_params)
        meta_vars = transform_entity_collection(meta_vars)
        meta_objs = transform_entity_collection(meta_objs)
        meta_cons = transform_entity_collection(meta_cons)

        if len(meta_sets_params) == 0:
            meta_sets_params.extend(meta_sets)
            meta_params.extend(meta_params)

        if model_file_name is None:
            model_file_name = "model"
        if model_file_extension is not None:
            if len(model_file_name) >= 4 and model_file_name[-4:] == ".mod":
                model_file_name = model_file_name[:-4]
            model_file_name += model_file_extension

        self._script = scr.Script(name=model_file_name, script_type=scr.ScriptType.MODEL)
        self._defined_syms = set()

        if (include_sets or include_params) and len(meta_sets_params) > 0:
            self.__generate_region_heading("Sets and Parameters")
            self.__generate_set_and_param_statements(meta_sets_params,
                                                     include_sets,
                                                     include_params)
            self.__generate_region_footer()

        if include_variables and len(meta_vars) > 0:
            self.__generate_region_heading("Variables")
            self.__generate_entity_declarations(meta_vars)
            self.__generate_region_footer()

        if include_objectives and len(meta_objs) > 0:
            self.__generate_region_heading("Objectives")
            self.__generate_entity_declarations(meta_objs)
            self.__generate_region_footer()

        if include_constraints and len(meta_cons) > 0:
            self.__generate_region_heading("Constraints")
            self.__generate_entity_declarations(meta_cons)
            self.__generate_region_footer()

        return self._script

    def __generate_entity_declarations(self, meta_entities: Iterable[mat.MetaEntity]):
        for meta_entity in meta_entities:
            if meta_entity.symbol not in self._defined_syms:
                if meta_entity.is_sub:
                    meta_entity = meta_entity.parent
                self.__add_entity_declaration(meta_entity)

    def __generate_set_and_param_statements(self,
                                            meta_sets_params: Iterable[Union[mat.MetaSet, mat.MetaParameter]],
                                            include_sets: bool,
                                            include_params: bool):

        for meta_entity in meta_sets_params:
            if meta_entity.symbol not in self._defined_syms:

                if meta_entity.is_sub:
                    meta_entity = meta_entity.parent

                if isinstance(meta_entity, mat.MetaSet) and include_sets:
                    self.__add_entity_declaration(meta_entity)
                elif isinstance(meta_entity, mat.MetaParameter) and include_params:
                    self.__add_entity_declaration(meta_entity)

    # Problem Statement
    # ------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def generate_subproblem_declaration(problem: Problem,
                                        subproblem: BaseProblem,
                                        idx_meta_sets: List[mat.MetaSet] = None):

        prob_node = mat.DeclaredEntityNode(symbol=subproblem.symbol,  # problem symbol
                                           type=mat.PROB_TYPE)

        unb_sym_mapping = {}
        idx_set_node = nb.build_idx_set_node(problem=problem,
                                             idx_meta_sets=idx_meta_sets,
                                             unb_sym_mapping=unb_sym_mapping)  # problem indexing set

        custom_dummy_syms = None
        if idx_meta_sets is not None:
            custom_dummy_syms = {ms.symbol: ms.dummy_element for ms in idx_meta_sets}

        # collect the meta-variables, meta-objectives, and meta-constraints of the problem in a single list
        meta_entities = []
        meta_entities.extend(subproblem.model_meta_vars)
        meta_entities.extend(subproblem.model_meta_objs)
        meta_entities.extend(subproblem.model_meta_cons)

        # build tuples of indexing set nodes and entity nodes for the collected meta-entities
        def build_indexing_and_entity_node_tuple(meta_entity: mat.MetaEntity):

            entity_idx_node = nb.build_default_entity_index_node(meta_entity)
            if len(unb_sym_mapping) > 0:
                nb.replace_unbound_symbols(entity_idx_node, unb_sym_mapping)

            entity_node = mat.DeclaredEntityNode(symbol=meta_entity.symbol,
                                                 idx_node=entity_idx_node,
                                                 type=meta_entity.type)

            if meta_entity.idx_set_reduced_dim == 0:
                return None, entity_node
            else:
                # build a new indexing set node with modified dummy symbols
                entity_idx_set_node = nb.build_entity_idx_set_node(
                    problem=problem,
                    meta_entity=meta_entity,
                    remove_sets=idx_meta_sets,
                    custom_dummy_syms=custom_dummy_syms)
                return entity_idx_set_node, entity_node

        node_tuples = [build_indexing_and_entity_node_tuple(me) for me in meta_entities]

        # build the problem declaration
        decl = ampl_stm.ProblemStatement(
            prob_node=prob_node,
            idx_set_node=idx_set_node,
            item_nodes=node_tuples
        )
        return decl

    # Utility
    # ------------------------------------------------------------------------------------------------------------------

    def __add_entity_declaration(self, meta_entity: mat.MetaEntity):
        statement = ampl_stm.ModelEntityDeclaration(meta_entity)
        self.__add_statement(statement)
        self._defined_syms.add(meta_entity.symbol)

    def __generate_region_heading(self, name: str):
        comment = ampl_stm.Comment(
            comment=" {0}".format(name.upper()),
            is_multi=False
        )
        self.__add_statement(comment)

        comment = ampl_stm.Comment(
            comment=" {0}".format('-' * 98),
            is_multi=False
        )
        self.__add_statement(comment)

        comment = ampl_stm.Comment(
            comment=" beginregion",
            is_multi=False
        )
        self.__add_statement(comment)

    def __generate_region_footer(self):
        comment = ampl_stm.Comment(
            comment=" endregion \n\n",
            is_multi=False
        )
        self.__add_statement(comment)

    def __add_statement(self, statement: scr.BaseStatement):
        self._script.statements.append(statement)
