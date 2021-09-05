import os
import pickle
from typing import Any, Dict, Iterable, List, Optional, Set, Union
import warnings

import symro.core.mat as mat
from symro.core.prob.specialcommand import SpecialCommand
from symro.core.prob.statement import CompoundScript
from symro.core.execution.amplengine import AMPLEngine


class BaseProblem:

    def __init__(self,
                 symbol: str = None,
                 description: str = None,
                 idx_set_node: mat.CompoundSetNode = None,
                 model_meta_entities: Iterable[mat.MetaEntity] = None):

        self.symbol: str = symbol if symbol is not None else "Initial"
        self.description = description if description is not None else ""

        self.idx_set_node: mat.CompoundSetNode = idx_set_node

        self.model_meta_sets_params: List[Union[mat.MetaSet, mat.MetaParameter]] = []
        self.model_meta_vars: List[mat.MetaVariable] = []
        self.model_meta_objs: List[mat.MetaObjective] = []
        self.model_meta_cons: List[mat.MetaConstraint] = []

        if model_meta_entities is not None:
            for me in model_meta_entities:
                self.add_meta_entity_to_model(me)

    def copy(self, source: "BaseProblem"):

        self.symbol = source.symbol
        self.description = source.description

        self.model_meta_sets_params = list(source.model_meta_sets_params)
        self.model_meta_vars = list(source.model_meta_vars)
        self.model_meta_objs = list(source.model_meta_objs)
        self.model_meta_cons = list(source.model_meta_cons)

    def add_meta_entity_to_model(self, meta_entity: mat.MetaEntity):
        if isinstance(meta_entity, mat.MetaSet):
            self.add_meta_set_to_model(meta_entity)
        elif isinstance(meta_entity, mat.MetaParameter):
            self.add_meta_parameter_to_model(meta_entity)
        elif isinstance(meta_entity, mat.MetaVariable):
            self.add_meta_variable_to_model(meta_entity)
        elif isinstance(meta_entity, mat.MetaObjective):
            self.add_meta_objective_to_model(meta_entity)
        elif isinstance(meta_entity, mat.MetaConstraint):
            self.add_meta_constraint_to_model(meta_entity)

    def add_meta_set_to_model(self, meta_set: mat.MetaSet):
        self.model_meta_sets_params.append(meta_set)

    def add_meta_parameter_to_model(self, meta_param: mat.MetaParameter):
        self.model_meta_sets_params.append(meta_param)

    def add_meta_variable_to_model(self, meta_var: mat.MetaVariable):
        self.model_meta_vars.append(meta_var)

    def add_meta_objective_to_model(self, meta_obj: mat.MetaObjective):
        self.model_meta_objs.append(meta_obj)

    def add_meta_constraint_to_model(self, meta_con: mat.MetaConstraint):
        self.model_meta_cons.append(meta_con)


class Problem(BaseProblem):

    def __init__(self,
                 symbol: str = None,
                 description: str = None,
                 file_name: str = None,
                 working_dir_path: str = None,
                 engine: AMPLEngine = None):

        # --- Name ---
        if symbol is None and file_name is not None:
            file_name = os.path.basename(file_name)
            symbol = os.path.splitext(file_name)[0]

        super(Problem, self).__init__(symbol, description)

        # --- I/O ---
        self.working_dir_path: str = working_dir_path

        # --- Script ---
        self.run_script_literal: str = ""
        self.compound_script: Optional[CompoundScript] = None
        self.script_commands: Dict[str, List[SpecialCommand]] = {}  # Key: flag. Value: list of script commands.

        # --- Meta-Entities ---
        self.symbols: Set[str] = set()
        self.meta_sets: Dict[str, Optional[mat.MetaSet]] = {}
        self.meta_params: Dict[str, Optional[mat.MetaParameter]] = {}
        self.meta_vars: Dict[str, Optional[mat.MetaVariable]] = {}
        self.meta_objs: Dict[str, Optional[mat.MetaObjective]] = {}
        self.meta_cons: Dict[str, Optional[mat.MetaConstraint]] = {}
        self.meta_tables: Dict[str, Any] = {}
        self.subproblems: Dict[str, BaseProblem] = {}

        # --- State ---
        self.state: mat.State = mat.State()

        # --- Engine ---
        self.engine: Optional[AMPLEngine] = engine

        # --- Miscellaneous ---
        self.__free_node_id: int = 0

    def copy(self, source: "Problem"):

        BaseProblem.copy(self, source)

        self.run_script_literal = source.run_script_literal
        self.script_commands = dict(source.script_commands)

        self.compound_script: Optional[CompoundScript] = CompoundScript()
        self.compound_script.copy(source.compound_script)

        self.symbols = set(source.symbols)

        self.meta_sets = dict(source.meta_sets)
        self.meta_params = dict(source.meta_params)
        self.meta_vars = dict(source.meta_vars)
        self.meta_objs = dict(source.meta_objs)
        self.meta_cons = dict(source.meta_cons)
        self.meta_tables = dict(source.meta_tables)

        self.subproblems = {}
        for sym, sp in source.subproblems.items():
            self.subproblems[sym] = BaseProblem()
            self.subproblems[sym].copy(sp)

        self.state = source.state

        self.engine = source.engine

        self.__free_node_id = source.__free_node_id

    def generate_unique_symbol(self, base_symbol: str = None) -> str:
        if base_symbol is None:
            base_symbol = "ENTITY"
        symbol = base_symbol
        i = 0
        while symbol in self.symbols:
            i += 1
            symbol = base_symbol + str(i)
        return symbol

    def add_subproblem(self, sp: BaseProblem):
        self.symbols.add(sp.symbol)
        self.subproblems[sp.symbol] = sp

    def is_meta_entity_in_model(self, meta_entity: mat.MetaEntity) -> bool:
        if isinstance(meta_entity, mat.MetaSet):
            return meta_entity.symbol in self.model_meta_sets_params
        elif isinstance(meta_entity, mat.MetaParameter):
            return meta_entity.symbol in self.model_meta_sets_params
        elif isinstance(meta_entity, mat.MetaVariable):
            return meta_entity.symbol in self.model_meta_vars
        elif isinstance(meta_entity, mat.MetaObjective):
            return meta_entity.symbol in self.model_meta_objs
        elif isinstance(meta_entity, mat.MetaConstraint):
            return meta_entity.symbol in self.model_meta_cons
        return False

    def get_meta_entity(self, symbol: str) -> Optional[mat.MetaEntity]:
        if symbol in self.meta_sets:
            return self.meta_sets[symbol]
        elif symbol in self.meta_params:
            return self.meta_params[symbol]
        elif symbol in self.meta_vars:
            return self.meta_vars[symbol]
        elif symbol in self.meta_objs:
            return self.meta_objs[symbol]
        elif symbol in self.meta_cons:
            return self.meta_cons[symbol]
        else:
            warnings.warn("Definition of entity '{0}' is unavailable".format(symbol))
            return None

    def add_meta_entity(self, meta_entity: mat.MetaEntity, is_in_model: bool = True):
        if isinstance(meta_entity, mat.MetaSet):
            self.add_meta_set(meta_entity, is_in_model)
        elif isinstance(meta_entity, mat.MetaParameter):
            self.add_meta_parameter(meta_entity, is_in_model)
        elif isinstance(meta_entity, mat.MetaVariable):
            self.add_meta_variable(meta_entity, is_in_model)
        elif isinstance(meta_entity, mat.MetaObjective):
            self.add_meta_objective(meta_entity, is_in_model)
        elif isinstance(meta_entity, mat.MetaConstraint):
            self.add_meta_constraint(meta_entity, is_in_model)

    def add_meta_set(self, meta_set: mat.MetaSet, is_in_model: bool = True):
        self.symbols.add(meta_set.symbol)
        self.meta_sets[meta_set.symbol] = meta_set
        if is_in_model:
            BaseProblem.add_meta_set_to_model(self, meta_set)

    def add_meta_parameter(self, meta_param: mat.MetaParameter, is_in_model: bool = True):
        self.symbols.add(meta_param.symbol)
        self.meta_params[meta_param.symbol] = meta_param
        if is_in_model:
            BaseProblem.add_meta_parameter_to_model(self, meta_param)

    def add_meta_variable(self, meta_var: mat.MetaVariable, is_in_model: bool = True):
        self.symbols.add(meta_var.symbol)
        self.meta_vars[meta_var.symbol] = meta_var
        if is_in_model:
            BaseProblem.add_meta_variable_to_model(self, meta_var)

    def add_meta_objective(self, meta_obj: mat.MetaObjective, is_in_model: bool = True):
        self.symbols.add(meta_obj.symbol)
        self.meta_objs[meta_obj.symbol] = meta_obj
        if is_in_model:
            BaseProblem.add_meta_objective_to_model(self, meta_obj)

    def add_meta_constraint(self, meta_con: mat.MetaConstraint, is_in_model: bool = True):
        self.symbols.add(meta_con.symbol)
        self.meta_cons[meta_con.symbol] = meta_con
        if is_in_model:
            BaseProblem.add_meta_constraint_to_model(self, meta_con)

    def add_script_command(self, script_command: SpecialCommand):
        if script_command.symbol in self.script_commands:
            self.script_commands[script_command.symbol].append(script_command)
        else:
            self.script_commands[script_command.symbol] = [script_command]

    def contains_script_command(self, symbol: str) -> bool:
        if self.script_commands is not None:
            return symbol in self.script_commands
        else:
            return False

    def save(self, dir_path: str = ""):
        file_name = self.symbol.lower().replace(' ', '_')
        with open(os.path.join(dir_path, file_name + ".p"), "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, problem_name: str, dir_path: str = "") -> "Problem":
        file_name = problem_name.lower().replace(' ', '_')
        with open(os.path.join(dir_path, file_name + ".p"), "rb") as f:
            problem = pickle.load(f)
        return problem

    def generate_free_node_id(self) -> int:
        free_node_id = self.__free_node_id
        self.__free_node_id += 1
        return free_node_id

    def seed_free_node_id(self, node_id: int):
        self.__free_node_id = max(node_id, self.__free_node_id)
