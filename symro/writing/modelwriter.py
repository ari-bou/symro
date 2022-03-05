import os
from typing import Iterable, List, Union

import symro.mat as mat
from symro.prob.problem import BaseProblem, Problem
import symro.util.util as util


class ModelWriter:
    def __init__(self):
        self.script = ""

    def generate_problem_model(
        self,
        problem: Problem,
        model_file_name: str = None,
        model_file_extension: str = ".mod",
    ):
        if model_file_name is None:
            model_file_name = problem.symbol + ".mod"
        return self.generate_model(
            working_dir_path=problem.working_dir_path,
            model_file_name=model_file_name,
            model_file_extension=model_file_extension,
            meta_sets_params=problem.model_meta_sets_params,
            meta_vars=problem.model_meta_vars,
            meta_objs=problem.model_meta_objs,
            meta_cons=problem.model_meta_cons,
        )

    def generate_subproblem_model(
        self,
        problem: Problem,
        subproblem: BaseProblem,
        model_file_name: str = None,
        model_file_extension: str = ".mod",
    ):
        if model_file_name is None:
            model_file_name = subproblem.symbol + ".mod"
        return self.generate_model(
            working_dir_path=problem.working_dir_path,
            model_file_name=model_file_name,
            model_file_extension=model_file_extension,
            meta_sets_params=subproblem.model_meta_sets_params,
            meta_vars=subproblem.model_meta_vars,
            meta_objs=subproblem.model_meta_objs,
            meta_cons=subproblem.model_meta_cons,
        )

    def generate_model(
        self,
        working_dir_path: str = None,
        model_file_name: str = None,
        model_file_extension: str = ".mod",
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
        include_constraints: bool = True,
    ) -> str:
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
            meta_sets_params = meta_sets + meta_params

        if working_dir_path is None:
            working_dir_path = os.getcwd()
        if model_file_name is None:
            model_file_name = "model.mod"
        if "." not in model_file_name:
            model_file_name += ".mod"
        if model_file_extension is not None:
            model_file_name = (
                os.path.splitext(model_file_name)[0] + model_file_extension
            )

        self.script = ""

        if (include_sets or include_params) and len(meta_sets_params) > 0:
            self.__generate_region_heading("Sets and Parameters")
            self.__generate_set_and_param_statements(
                meta_sets_params, include_sets, include_params
            )
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

        util.write_file(working_dir_path, model_file_name, self.script)

        return model_file_name

    def __generate_entity_declarations(self, meta_entities: List[mat.MetaEntity]):
        for meta_entity in meta_entities:
            self.__add_entity_declaration(meta_entity)

    def __generate_set_and_param_statements(
        self,
        meta_sets_params: List[Union[mat.MetaSet, mat.MetaParameter]],
        include_sets: bool,
        include_params: bool,
    ):
        for meta_entity in meta_sets_params:
            if isinstance(meta_entity, mat.MetaSet) and include_sets:
                self.__add_entity_declaration(meta_entity)
            elif isinstance(meta_entity, mat.MetaParameter) and include_params:
                self.__add_entity_declaration(meta_entity)

    def __add_entity_declaration(self, meta_entity: mat.MetaEntity):
        statement = meta_entity.generate_declaration()
        self.__add_statement(statement)

    def __add_statement(self, statement: str):
        self.script += statement + "\n"

    def __generate_region_heading(self, name: str):
        self.script += "# {0} \n#{1}#\n# beginregion \n".format(name.upper(), "-" * 98)

    def __generate_region_footer(self):
        self.script += "# endregion \n\n"
