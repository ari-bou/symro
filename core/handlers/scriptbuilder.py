from typing import Dict, List, Optional, Union

import symro.core.constants as const
import symro.core.mat as mat
from symro.core.prob.problem import BaseProblem, Problem
import symro.core.prob.statement as stm
from symro.core.handlers.nodebuilder import NodeBuilder


def model_to_ampl(problem: Problem,
                  file_name: str = None,
                  working_dir_path: str = None):
    script_builder = ScriptBuilder()
    script = script_builder.generate_problem_model_script(problem=problem,
                                                          model_file_name=file_name,
                                                          model_file_extension=None)
    script.write(dir_path=working_dir_path, file_name=file_name)


class ScriptBuilder:

    def __init__(self):
        self._script: Optional[stm.Script] = None

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
                              meta_sets: Union[List[mat.MetaSet], Dict[str, mat.MetaSet]] = None,
                              meta_params: Union[List[mat.MetaParameter], Dict[str, mat.MetaParameter]] = None,
                              meta_sets_params: Union[List[Union[mat.MetaSet, mat.MetaParameter]],
                                                      Dict[str, Union[mat.MetaSet, mat.MetaParameter]]] = None,
                              meta_vars: Union[List[mat.MetaVariable], Dict[str, mat.MetaVariable]] = None,
                              meta_objs: Union[List[mat.MetaObjective], Dict[str, mat.MetaObjective]] = None,
                              meta_cons: Union[List[mat.MetaConstraint], Dict[str, mat.MetaConstraint]] = None,
                              include_sets: bool = True,
                              include_params: bool = True,
                              include_variables: bool = True,
                              include_objectives: bool = True,
                              include_constraints: bool = True) -> stm.Script:

        def transform_entity_collection(entity_collection):
            if entity_collection is None:
                entity_collection = []
            if isinstance(entity_collection, dict):
                entity_collection = list(entity_collection.values())
            return entity_collection

        meta_sets = transform_entity_collection(meta_sets)
        meta_params = transform_entity_collection(meta_params)
        meta_sets_params = transform_entity_collection(meta_sets_params)
        meta_vars = transform_entity_collection(meta_vars)
        meta_objs = transform_entity_collection(meta_objs)
        meta_cons = transform_entity_collection(meta_cons)

        if len(meta_sets_params) == 0:
            meta_sets_params = meta_sets + meta_params

        if model_file_name is None:
            model_file_name = "model"
        if model_file_extension is not None:
            if len(model_file_name) >= 4 and model_file_name[-4:] == ".mod":
                model_file_name = model_file_name[:-4]
            model_file_name += model_file_extension

        self._script = stm.Script(id=model_file_name)

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

    def __generate_entity_declarations(self, meta_entities: List[mat.MetaEntity]):
        for meta_entity in meta_entities:
            self.__add_entity_declaration(meta_entity)

    def __generate_set_and_param_statements(self,
                                            meta_sets_params: List[Union[mat.MetaSet, mat.MetaParameter]],
                                            include_sets: bool,
                                            include_params: bool):
        for meta_entity in meta_sets_params:
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

        node_builder = NodeBuilder(problem)

        prob_node = mat.DeclaredEntityNode(symbol=subproblem.symbol,  # problem symbol
                                           type=const.PROB_TYPE)
        idx_set_node = node_builder.build_idx_set_node(idx_meta_sets=idx_meta_sets)  # problem indexing set
        custom_dummy_syms = None
        if idx_meta_sets is not None:
            custom_dummy_syms = {ms.get_symbol(): ms.get_dummy_element() for ms in idx_meta_sets}

        # collect the meta-variables, meta-objectives, and meta-constraints of the problem in a single list
        meta_entities = []
        meta_entities.extend(subproblem.model_meta_vars)
        meta_entities.extend(subproblem.model_meta_objs)
        meta_entities.extend(subproblem.model_meta_cons)

        # build tuples of indexing set nodes and entity nodes for the collected meta-entities
        def build_indexing_and_entity_node_tuple(meta_entity: mat.MetaEntity):

            entity_idx_node = node_builder.build_default_entity_index_node(meta_entity)
            if node_builder.unb_sym_map is not None:
                node_builder.replace_unbound_symbols(entity_idx_node, node_builder.unb_sym_map)

            entity_node = mat.DeclaredEntityNode(symbol=meta_entity.get_symbol(),
                                                 idx_node=entity_idx_node,
                                                 type=meta_entity.get_type())

            if meta_entity.get_idx_set_reduced_dim() == 0:
                return None, entity_node
            else:
                # build a new indexing set node with modified dummy symbols
                entity_idx_set_node = node_builder.build_entity_idx_set_node(meta_entity=meta_entity,
                                                                             remove_sets=idx_meta_sets,
                                                                             custom_dummy_syms=custom_dummy_syms)
                return entity_idx_set_node, entity_node

        node_tuples = [build_indexing_and_entity_node_tuple(me) for me in meta_entities]

        # build the problem declaration
        decl = stm.ProblemStatement(prob_node=prob_node,
                                    idx_set_node=idx_set_node,
                                    item_nodes=node_tuples)
        return decl

    # Utility
    # ------------------------------------------------------------------------------------------------------------------

    def __add_entity_declaration(self, meta_entity: mat.MetaEntity):
        statement = stm.ModelEntityDeclaration(meta_entity)
        self.__add_statement(statement)

    def __generate_region_heading(self, name: str):
        comment = stm.Comment(comment=" {0}".format(name.upper()),
                              is_multi=False)
        self.__add_statement(comment)

        comment = stm.Comment(comment=" {0}".format('-' * 98),
                              is_multi=False)
        self.__add_statement(comment)

        comment = stm.Comment(comment=" beginregion",
                              is_multi=False)
        self.__add_statement(comment)

    def __generate_region_footer(self):
        comment = stm.Comment(comment=" endregion \n\n",
                              is_multi=False)
        self.__add_statement(comment)

    def __add_statement(self, statement: stm.BaseStatement):
        self._script.statements.append(statement)
