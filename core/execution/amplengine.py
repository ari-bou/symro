import amplpy as ampl
import os
from typing import List, Tuple, Union, Optional


class AMPLEngine:

    def __init__(self, working_dir_path: str = None):
        self.working_dir_path: str = working_dir_path if working_dir_path is not None else os.getcwd()
        self.api: ampl.AMPL = ampl.AMPL()
        if working_dir_path != "":
            self.api.setOption(name="ampl_include",
                               value=working_dir_path)
            self.api.cd(working_dir_path)
        self.api.reset()

    # Modelling
    # ------------------------------------------------------------------------------------------------------------------

    def read_model(self, model_file_name: str):
        if ' ' in model_file_name:
            model_file_name = '"{0}"'.format(model_file_name)
        self.api.eval("model {0};".format(model_file_name))

    def read_data(self, data_file_name: str):
        if ' ' in data_file_name:
            data_file_name = '"{0}"'.format(data_file_name)
        self.api.eval("data {0};".format(data_file_name))

    # Solve
    # ------------------------------------------------------------------------------------------------------------------

    def solve(self, solver_name: str = None, solver_options: str = None) -> str:
        if solver_name is not None:
            self.api.eval("option solver {0};\n".format(solver_name))
            self.api.eval("option solver {0};\n".format(solver_name))
            if solver_options is not None:
                self.api.eval('option {0}_options "{1}";\n'.format(solver_name, solver_options))
        return self.api.getOutput("solve;")

    def get_solve_result(self) -> str:
        result = self.api.getOutput("display solve_result;")
        result = result.replace("\n", '')
        result = result.split('=')[1].strip()
        return result

    # Accessors and Mutators
    # ------------------------------------------------------------------------------------------------------------------

    def get_entity(self, symbol: str, indices: Union[Tuple, List], etype: str) -> ampl.Entity:
        if etype == "set":
            entity = self.api.getSet(symbol)
        elif etype == "param":
            entity = self.api.getParameter(symbol)
        elif etype == "var":
            entity = self.api.getVariable(symbol)
        elif etype == "obj":
            entity = self.api.getObjective(symbol)
        elif etype == "con":
            entity = self.api.getConstraint(symbol)
        else:
            raise ValueError("Encountered unexpected entity type.")
        return self.get_entity_instance(entity, indices)

    def get_set(self, symbol: str, indices: Union[Tuple, List]) -> ampl.Set:
        entity = self.api.getSet(symbol)
        return self.get_entity_instance(entity, indices)

    def get_param(self, symbol: str, indices: Union[Tuple, List]) -> ampl.Parameter:
        entity = self.api.getParameter(symbol)
        return self.get_entity_instance(entity, indices)

    def get_var(self, symbol: str, indices: Union[Tuple, List]) -> ampl.Variable:
        entity = self.api.getVariable(symbol)
        return self.get_entity_instance(entity, indices)

    def get_obj(self, symbol: str, indices: Union[Tuple, List]) -> ampl.Objective:
        entity = self.api.getObjective(symbol)
        return self.get_entity_instance(entity, indices)

    def get_con(self, symbol: str, indices: Union[Tuple, List]) -> ampl.Constraint:
        entity = self.api.getConstraint(symbol)
        return self.get_entity_instance(entity, indices)

    @staticmethod
    def get_entity_instance(entity: ampl.Entity,
                            indices: Union[Tuple, List] = None
                            ) -> Union[ampl.Set, ampl.Parameter, ampl.Variable, ampl.Objective, ampl.Constraint]:
        if entity.indexarity() > 0:
            entity = entity.get(tuple(indices))
        return entity

    def get_entity_indices(self, symbol: str, etype: str) -> List[Tuple]:

        if etype == "set":
            entity = self.api.getSet(symbol)
        elif etype == "param":
            entity = self.api.getParameter(symbol)
        elif etype == "var":
            entity = self.api.getVariable(symbol)
        elif etype == "obj":
            entity = self.api.getObjective(symbol)
        elif etype == "con":
            entity = self.api.getConstraint(symbol)
        else:
            raise ValueError("Encountered unexpected entity type.")

        if entity.indexarity() == 0:
            return []
        elif entity.indexarity() == 1:
            return [tuple([k]) for k, v in entity.instances()]
        else:
            return [k for k, v in entity.instances()]

    def get_param_value(self,
                        symbol: str,
                        indices: Union[Tuple, List] = None):
        param = self.api.getParameter(symbol)
        if param.indexarity() > 0:
            param.get(indices)
        else:
            return param.value()

    def get_var_value(self,
                      symbol: str,
                      indices: Union[Tuple, List] = None):
        var = self.api.getVariable(symbol)
        var = self.get_entity_instance(var, indices)
        return var.value()

    def get_obj_value(self,
                      symbol: str,
                      indices: Union[Tuple, List] = None):
        obj = self.api.getObjective(symbol)
        obj = self.get_entity_instance(obj, indices)
        return obj.value()

    def get_con_value(self,
                      symbol: str,
                      indices: Union[Tuple, List] = None):
        con = self.api.getConstraint(symbol)
        con = self.get_entity_instance(con, indices)
        return con.value()

    def get_entity_value(self,
                         symbol: str,
                         indices: Union[Tuple, List],
                         etype: str) -> float:
        if etype == "param":
            return self.get_param_value(symbol, indices)
        elif etype == "var":
            return self.get_var_value(symbol, indices)
        elif etype == "obj":
            return self.get_obj_value(symbol, indices)
        elif etype == "con":
            return self.get_con_value(symbol, indices)
        else:
            raise ValueError("Encountered unexpected symbol.")

    def get_con_dual(self, symbol: str, indices: Union[Tuple, List] = None) -> float:
        con = self.api.getConstraint(symbol)
        if con.indexarity() == 1:
            return con.getValues().toDict()[indices[0]]
        elif con.indexarity() > 1:
            return con.getValues().toDict()[tuple(indices)]
        else:
            return con.dual()

    def set_param_value(self, symbol: str, indices: Optional[Union[tuple, list]], value: float):
        param = self.api.getParameter(symbol)
        if param.indexarity() > 0:
            param.set(indices, value)
        else:
            param.set(value)

    def set_var_value(self, symbol: str, indices: Union[tuple, list], value: float):
        var = self.api.getVariable(symbol)
        var = self.get_entity_instance(var, indices)
        var.set(value)

    def set_entity_value(self,
                         symbol: str,
                         indices: list,
                         value: float,
                         etype: str):
        if etype == "param":
            self.set_param_value(symbol, indices, value)
        elif etype == "var":
            self.set_var_value(symbol, indices, value)
        else:
            raise ValueError("Encountered unexpected symbol.")
