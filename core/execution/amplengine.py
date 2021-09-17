import amplpy as ampl
import os
from typing import List, Tuple, Union, Optional

import symro.core.mat as mat
from symro.core.prob.problem import Problem, BaseProblem
from symro.core.execution.engine import Engine


class AMPLEngine(Engine):

    # Construction
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, problem: Problem = None):

        super(AMPLEngine, self).__init__(problem)

        self.api: Optional[ampl.AMPL] = None

        if problem is not None:
            self.setup_ampl_engine(problem)

    def setup_ampl_engine(self, problem: Problem):

        self.problem = problem
        self.api = ampl.AMPL()

        if self.problem.working_dir_path is not None:
            working_dir_path = self.problem.working_dir_path
        else:
            working_dir_path = os.getcwd()

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

    def set_active_problem(self,
                           problem_symbol: str = None,
                           problem_idx: mat.Element = None):

        Engine.set_active_problem(self,
                                  problem_symbol=problem_symbol,
                                  problem_idx=problem_idx)

        problem_literal = problem_symbol
        if problem_idx is not None:
            problem_idx = [mat.get_element_literal(s) for s in problem_idx]
            problem_literal += "[{0}]".format(','.join(problem_idx))

        self.api.eval("problem {0};".format(problem_literal))

    # Solve
    # ------------------------------------------------------------------------------------------------------------------

    def solve(self,
              solver_name: str = None,
              solver_options: str = None):

        if solver_name is not None:
            self.api.eval("option solver {0};\n".format(solver_name))
            self.api.eval("option solver {0};\n".format(solver_name))
            if solver_options is not None:
                self.api.eval('option {0}_options "{1}";\n'.format(solver_name, solver_options))

        self._solver_output = self.api.getOutput("solve;")

        if self.can_store_soln:
            self.__store_solution()

    def __store_solution(self):

        # retrieve active problem

        if self._active_problem_sym is None:
            p = self.problem

        else:
            p = self.problem.subproblems[self._active_problem_sym]

        # TODO: filter indexing set of each meta-entity using the index of the active problem

        # retrieve variable data
        for meta_var in p.model_meta_vars:

            if not meta_var.is_indexed():
                idx_set = [None]
            else:
                idx_set = meta_var.idx_set_node.evaluate(self.problem.state)[0]

            for idx in idx_set:
                var = self.get_var(symbol=meta_var.symbol, idx=idx)
                self._store_var(symbol=meta_var.symbol,
                                idx=idx,
                                value=var.value(),
                                lb=var.lb(),
                                ub=var.ub())

        # retrieve objective data
        for meta_obj in p.model_meta_objs:

            if not meta_obj.is_indexed():
                idx_set = [None]
            else:
                idx_set = meta_obj.idx_set_node.evaluate(self.problem.state)[0]

            for idx in idx_set:
                obj = self.get_obj(symbol=meta_obj.symbol, idx=idx)
                self._store_obj(symbol=meta_obj.symbol,
                                idx=idx,
                                value=obj.value())

        # retrieve constraint data
        for meta_con in p.model_meta_cons:

            if not meta_con.is_indexed():
                idx_set = [None]
            else:
                idx_set = meta_con.idx_set_node.evaluate(self.problem.state)[0]

            for idx in idx_set:
                con = self.get_con(symbol=meta_con.symbol, idx=idx)
                self._store_con(symbol=meta_con.symbol,
                                idx=idx,
                                value=con.body(),
                                lb=con.lb(),
                                ub=con.ub(),
                                dual=con.dual())

    def get_status(self) -> str:
        result = self.api.getOutput("display solve_result;")
        result = result.replace("\n", '')
        result = result.split('=')[1].strip()
        return result

    def get_solver_output(self) -> str:
        return self._solver_output

    # Accessors and Mutators
    # ------------------------------------------------------------------------------------------------------------------

    def get_entity(self, symbol: str, idx: Union[Tuple, List], etype: str) -> ampl.Entity:
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
        return self.get_entity_instance(entity, idx)

    def get_set(self, symbol: str, idx: Union[Tuple, List] = None) -> ampl.Set:
        entity = self.api.getSet(symbol)
        return self.get_entity_instance(entity, idx)

    def get_param(self, symbol: str, idx: Union[Tuple, List] = None) -> ampl.Parameter:
        entity = self.api.getParameter(symbol)
        return self.get_entity_instance(entity, idx)

    def get_var(self, symbol: str, idx: Union[Tuple, List] = None) -> ampl.Variable:
        entity = self.api.getVariable(symbol)
        return self.get_entity_instance(entity, idx)

    def get_obj(self, symbol: str, idx: Union[Tuple, List] = None) -> ampl.Objective:
        entity = self.api.getObjective(symbol)
        return self.get_entity_instance(entity, idx)

    def get_con(self, symbol: str, idx: Union[Tuple, List] = None) -> ampl.Constraint:
        entity = self.api.getConstraint(symbol)
        return self.get_entity_instance(entity, idx)

    @staticmethod
    def get_entity_instance(entity: ampl.Entity,
                            idx: Union[Tuple, List] = None
                            ) -> Union[ampl.Set, ampl.Parameter, ampl.Variable, ampl.Objective, ampl.Constraint]:
        if entity.indexarity() > 0:
            entity = entity.get(tuple(idx))
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
