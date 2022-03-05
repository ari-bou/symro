from copy import copy, deepcopy
import os
import pickle
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Union
import warnings

import symro.mat as mat
from symro.scripting.specialcommand import SpecialCommand
import symro.scripting.script as scr
import symro.scripting.amplstatement as ampl_stm


class BaseProblem:
    def __init__(
        self,
        symbol: str = None,
        description: str = None,
        idx_set_node: mat.CompoundSetNode = None,
        model_meta_entities: Iterable[mat.MetaEntity] = None,
    ):

        self._symbol: str = symbol if symbol is not None else "Initial"
        self.description = description if description is not None else ""

        self.idx_set_node: mat.CompoundSetNode = idx_set_node

        self._model_meta_sets_params: mat.OrderedSet[
            Union[mat.MetaSet, mat.MetaParameter]
        ] = mat.OrderedSet([])
        self._model_meta_vars: mat.OrderedSet[mat.MetaVariable] = mat.OrderedSet([])
        self._model_meta_objs: mat.OrderedSet[mat.MetaObjective] = mat.OrderedSet([])
        self._model_meta_cons: mat.OrderedSet[mat.MetaConstraint] = mat.OrderedSet([])

        if model_meta_entities is not None:
            for me in model_meta_entities:
                self.add_meta_entity(me)

    def __copy__(self):
        clone = BaseProblem()
        self.copy(self, clone)
        return clone

    def __deepcopy__(self):
        clone = BaseProblem()
        self.deepcopy(self, clone)
        return clone

    # Copying
    # ------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def copy(source: "BaseProblem", clone: "BaseProblem"):

        clone.symbol = source.symbol
        clone.description = source.description
        clone.idx_set_node = source.idx_set_node

        clone.model_meta_sets_params = source.model_meta_sets_params
        clone.model_meta_vars = source.model_meta_vars
        clone.model_meta_objs = source.model_meta_objs
        clone.model_meta_cons = source.model_meta_cons

    @staticmethod
    def deepcopy(source: "BaseProblem", clone: "BaseProblem"):

        clone.symbol = source.symbol
        clone.description = source.description
        clone.idx_set_node = deepcopy(source.idx_set_node)

        clone.model_meta_sets_params = deepcopy(source.model_meta_sets_params)
        clone.model_meta_vars = deepcopy(source.model_meta_vars)
        clone.model_meta_objs = deepcopy(source.model_meta_objs)
        clone.model_meta_cons = deepcopy(source.model_meta_cons)

    # Properties
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def model_meta_sets_params(self):
        return self._model_meta_sets_params

    @model_meta_sets_params.setter
    def model_meta_sets_params(self, iterable: Iterable):
        self._model_meta_sets_params = mat.OrderedSet(iterable)

    @property
    def model_meta_vars(self):
        return self._model_meta_vars

    @model_meta_vars.setter
    def model_meta_vars(self, iterable: Iterable):
        self._model_meta_vars = mat.OrderedSet(iterable)

    @property
    def model_meta_objs(self):
        return self._model_meta_objs

    @model_meta_objs.setter
    def model_meta_objs(self, iterable: Iterable):
        self._model_meta_objs = mat.OrderedSet(iterable)

    @property
    def model_meta_cons(self):
        return self._model_meta_cons

    @model_meta_cons.setter
    def model_meta_cons(self, iterable: Iterable):
        self._model_meta_cons = mat.OrderedSet(iterable)

    @property
    def symbol(self) -> str:
        return self._symbol

    @symbol.setter
    def symbol(self, symbol):
        self._symbol = symbol

    @property
    def entity_symbols(self) -> Set[str]:
        return (
            self.set_param_symbols
            | self.variable_symbols
            | self.objective_symbols
            | self.constraint_symbols
        )

    @property
    def set_param_symbols(self) -> Set[str]:
        return set([mv.symbol for mv in self.model_meta_sets_params])

    @property
    def variable_symbols(self) -> Set[str]:
        return set([mv.symbol for mv in self.model_meta_vars])

    @property
    def objective_symbols(self) -> Set[str]:
        return set([mo.symbol for mo in self.model_meta_objs])

    @property
    def constraint_symbols(self) -> Set[str]:
        return set([mc.symbol for mc in self.model_meta_cons])

    # Addition
    # ------------------------------------------------------------------------------------------------------------------

    def add_meta_entity(self, meta_entity: mat.MetaEntity, is_auxiliary: bool = False):

        if isinstance(meta_entity, mat.MetaSet):
            self.model_meta_sets_params.append(meta_entity)

        elif isinstance(meta_entity, mat.MetaParameter):
            self.model_meta_sets_params.append(meta_entity)

        elif isinstance(meta_entity, mat.MetaVariable):
            self.model_meta_vars.append(meta_entity)

        elif isinstance(meta_entity, mat.MetaObjective):
            self.model_meta_objs.append(meta_entity)

        elif isinstance(meta_entity, mat.MetaConstraint):
            self.model_meta_cons.append(meta_entity)

    # Replacement
    # ------------------------------------------------------------------------------------------------------------------

    def replace_model_meta_constraint(
        self, old_symbol: str, new_meta_cons: List[mat.MetaConstraint]
    ):
        self._replace_model_meta_entity(
            old_symbol=old_symbol,
            new_meta_entities=new_meta_cons,
            model_meta_entities=self.model_meta_cons,
        )

    def _replace_model_meta_entity(
        self,
        old_symbol: str,
        new_meta_entities: List[mat.MetaEntity],
        model_meta_entities: mat.OrderedSet[mat.MetaEntity],
    ):

        if len(new_meta_entities) == 1 and old_symbol == new_meta_entities[0].symbol:
            return

        i = 0
        while i < len(model_meta_entities):

            old_me = self.model_meta_cons[i]

            if old_me.symbol == old_symbol:

                model_meta_entities.pop(i)

                if not old_me.is_sub:
                    replacements = new_meta_entities
                else:
                    replacements = [
                        new_mc.build_sub_entity(old_me.idx_set_node)
                        for new_mc in new_meta_entities
                    ]

                for rep_me in replacements:
                    model_meta_entities.insert(i, rep_me)

                i += len(replacements) - 1

            i += 1


class Problem(BaseProblem):

    # Construction
    # ------------------------------------------------------------------------------------------------------------------
    def __init__(
        self, symbol: str = None, description: str = None, working_dir_path: str = None
    ):

        super(Problem, self).__init__(symbol, description)

        # --- I/O ---
        self.working_dir_path: str = (
            working_dir_path if working_dir_path is not None else os.getcwd()
        )

        # --- Script ---
        self.run_script_literal: str = ""
        self.compound_script: Optional[scr.CompoundScript] = None
        self.script_commands: Dict[
            str, List[SpecialCommand]
        ] = {}  # Key: flag. Value: list of script commands.

        # --- Symbols ---
        self.symbols: Set[str] = set()
        self.unbound_symbols: Set[str] = set()

        # --- Meta-Entities ---

        self.meta_sets: Dict[str, Optional[mat.MetaSet]] = {}
        self.meta_params: Dict[str, Optional[mat.MetaParameter]] = {}
        self.meta_vars: Dict[str, Optional[mat.MetaVariable]] = {}
        self.meta_objs: Dict[str, Optional[mat.MetaObjective]] = {}
        self.meta_cons: Dict[str, Optional[mat.MetaConstraint]] = {}
        self.meta_tables: Dict[str, Any] = {}

        self.origin_to_std_con_map: Optional[Dict[str, List[mat.MetaConstraint]]] = {}

        self.subproblems: Dict[str, BaseProblem] = {}

        # --- State ---
        self.state: mat.State = mat.State()

    def __copy__(self):
        clone = Problem()
        self.copy(self, clone)
        return clone

    def __deepcopy__(self):
        clone = Problem()
        self.deepcopy(self, clone)
        return clone

    def __getitem__(self, item: Sequence):

        sym = item[0]
        idx = item[1:]

        if sym in self.meta_sets:
            entity = self.state.get_set(sym, idx)
            return entity.get_value()

        elif sym in self.meta_params:
            entity = self.state.get_parameter(sym, idx)
            return entity.get_value()

        elif sym in self.meta_vars:
            entity = self.state.get_variable(sym, idx)
            return entity.get_value()

        elif sym in self.meta_objs:
            entity = self.state.get_objective(sym, idx)
            return entity.get_value()

        elif sym in self.meta_cons:
            entity = self.state.get_constraint(sym, idx)
            return entity.get_value()

        else:
            raise ValueError("Symbol '{0}' is undefined".format(sym))

    def __setitem__(
        self, key: Sequence, value: Union[int, float, str, mat.IndexingSet]
    ):

        sym = key[0]
        idx = key[1:]

        if sym in self.meta_sets:
            entity = self.state.get_set(sym, idx)
            entity.elements = mat.OrderedSet(value)

        elif sym in self.meta_params:
            entity = self.state.get_parameter(sym, idx)
            entity.value = value

        elif sym in self.meta_vars:
            entity = self.state.get_variable(sym, idx)
            entity.value = value

        elif sym in self.meta_objs:
            entity = self.state.get_objective(sym, idx)
            entity.value = value

        elif sym in self.meta_cons:
            entity = self.state.get_constraint(sym, idx)
            entity.dual = value

        else:
            raise ValueError("Symbol '{0}' is undefined".format(sym))

    @staticmethod
    def copy(source: "Problem", clone: "Problem"):

        BaseProblem.copy(source, clone)

        clone.working_dir_path = source.working_dir_path

        clone.run_script_literal = source.run_script_literal
        clone.script_commands = dict(source.script_commands)

        clone.compound_script = scr.CompoundScript()
        clone.compound_script.copy(source.compound_script)

        clone.symbols = set(source.symbols)
        clone.unbound_symbols = set(source.unbound_symbols)

        clone.meta_sets = dict(source.meta_sets)
        clone.meta_params = dict(source.meta_params)
        clone.meta_vars = dict(source.meta_vars)
        clone.meta_objs = dict(source.meta_objs)
        clone.meta_cons = dict(source.meta_cons)
        clone.meta_tables = dict(source.meta_tables)

        clone.subproblems = {sym: copy(sp) for sym, sp in source.subproblems.items()}

        clone.state = source.state

    @staticmethod
    def deepcopy(source: "Problem", clone: "Problem"):

        clone.symbol = source.symbol
        clone.description = source.description

        clone.working_dir_path = source.working_dir_path

        clone.run_script_literal = source.run_script_literal
        clone.script_commands = deepcopy(source.script_commands)

        clone.compound_script = scr.CompoundScript()
        clone.compound_script.copy(source.compound_script)

        clone.symbols = deepcopy(source.symbols)
        clone.unbound_symbols = deepcopy(source.unbound_symbols)

        clone.meta_sets = deepcopy(source.meta_sets)
        clone.meta_params = deepcopy(source.meta_params)
        clone.meta_vars = deepcopy(source.meta_vars)
        clone.meta_objs = deepcopy(source.meta_objs)
        clone.meta_cons = deepcopy(source.meta_cons)
        clone.meta_tables = deepcopy(source.meta_tables)

        for me in source.model_meta_sets_params:
            if isinstance(me, mat.MetaSet):
                clone.model_meta_sets_params.append(clone.meta_sets[me.symbol])
            elif isinstance(me, mat.MetaParameter):
                clone.model_meta_sets_params.append(clone.meta_params[me.symbol])

        for me in source.model_meta_vars:
            clone.model_meta_vars.append(clone.meta_vars[me.symbol])

        for me in source.model_meta_objs:
            clone.model_meta_objs.append(clone.meta_objs[me.symbol])

        for me in source.model_meta_cons:
            clone.model_meta_cons.append(clone.meta_cons[me.symbol])

        for sym, sp_source in source.subproblems.items():

            sp_clone = BaseProblem(
                symbol=sym,
                description=sp_source.description,
                idx_set_node=deepcopy(sp_source.idx_set_node),
            )

            for me in sp_source.model_meta_sets_params:
                if isinstance(me, mat.MetaSet):
                    sp_clone.model_meta_sets_params.append(clone.meta_sets[me.symbol])
                elif isinstance(me, mat.MetaParameter):
                    sp_clone.model_meta_sets_params.append(clone.meta_params[me.symbol])

            for me in source.model_meta_vars:
                if not me.is_sub:
                    sp_clone.model_meta_vars.append(clone.meta_vars[me.symbol])
                else:
                    sp_clone.model_meta_vars.append(deepcopy(me))

            for me in source.model_meta_objs:
                if not me.is_sub:
                    sp_clone.model_meta_objs.append(clone.meta_objs[me.symbol])
                else:
                    sp_clone.model_meta_objs.append(deepcopy(me))

            for me in source.model_meta_cons:
                if not me.is_sub:
                    sp_clone.model_meta_cons.append(clone.meta_cons[me.symbol])
                else:
                    sp_clone.model_meta_cons.append(deepcopy(me))

        clone.state = deepcopy(source.state)

    # Checkers
    # ------------------------------------------------------------------------------------------------------------------

    def contains_script_command(self, symbol: str) -> bool:
        if self.script_commands is not None:
            return symbol in self.script_commands
        else:
            return False

    # Accessors
    # ------------------------------------------------------------------------------------------------------------------

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

    # Addition
    # ------------------------------------------------------------------------------------------------------------------

    def add_subproblem(self, sp: BaseProblem):
        self.symbols.add(sp.symbol)
        self.subproblems[sp.symbol] = sp

    def add_meta_entity(self, meta_entity: mat.MetaEntity, is_auxiliary: bool = False):
        if isinstance(meta_entity, mat.MetaSet):
            self.add_meta_set(meta_entity, is_auxiliary)
        elif isinstance(meta_entity, mat.MetaParameter):
            self.add_meta_parameter(meta_entity, is_auxiliary)
        elif isinstance(meta_entity, mat.MetaVariable):
            self.add_meta_variable(meta_entity, is_auxiliary)
        elif isinstance(meta_entity, mat.MetaObjective):
            self.add_meta_objective(meta_entity, is_auxiliary)
        elif isinstance(meta_entity, mat.MetaConstraint):
            self.add_meta_constraint(meta_entity, is_auxiliary)

    def add_meta_set(self, meta_set: mat.MetaSet, is_auxiliary: bool = False):
        self.symbols.add(meta_set.symbol)
        self.meta_sets[meta_set.symbol] = meta_set
        if not is_auxiliary:
            self.model_meta_sets_params.append(meta_set)
        self._evaluate_default_or_defined_set(meta_set)

    def _evaluate_default_or_defined_set(self, meta_set: mat.MetaSet):

        default_value = meta_set.default_value_node
        defined_value = meta_set.defined_value_node

        if default_value is not None:
            set_node = default_value
        elif defined_value is not None:
            set_node = defined_value
        else:
            set_node = None

        if set_node is not None:

            symbol = meta_set.symbol  # symbol of the meta-set
            m = 0  # length of indexing set
            idx_set: Optional[mat.IndexingSet] = None  # indexing set
            dummy_element = None  # dummy element of indexing set

            if meta_set.idx_set_node is not None:
                try:
                    idx_set = meta_set.idx_set_node.evaluate(self.state)[0]
                except (ValueError, KeyError):
                    return
                dummy_element = meta_set.idx_set_node.get_dummy_element(self.state)
                m = len(idx_set)

            try:
                dim = set_node.get_dim(self.state)
                sets = set_node.evaluate(
                    state=self.state, idx_set=idx_set, dummy_element=dummy_element
                )

            except (ValueError, KeyError):
                return

            for i in range(m):

                idx: Optional[mat.Element] = None
                if idx_set is not None:
                    idx = idx_set[i]

                self.state.add_set(symbol=symbol, idx=idx, dim=dim, elements=sets[i])

    def add_meta_parameter(
        self, meta_param: mat.MetaParameter, is_auxiliary: bool = False
    ):
        self.symbols.add(meta_param.symbol)
        self.meta_params[meta_param.symbol] = meta_param
        if not is_auxiliary:
            self.model_meta_sets_params.append(meta_param)
        self._evaluate_default_or_defined_param(meta_param)

    def _evaluate_default_or_defined_param(self, meta_param: mat.MetaParameter):

        default_value = meta_param.default_value_node
        defined_value = meta_param.defined_value_node

        if default_value is not None:
            expr_node = default_value
        elif defined_value is not None:
            expr_node = defined_value
        else:
            expr_node = None

        if expr_node is not None:

            symbol = meta_param.symbol  # symbol of the meta-set
            m = 0  # length of indexing set
            idx_set: Optional[mat.IndexingSet] = None  # indexing set
            dummy_element = None  # dummy element of indexing set

            if meta_param.idx_set_node is not None:
                try:
                    idx_set = meta_param.idx_set_node.evaluate(self.state)[0]
                except (ValueError, KeyError):
                    return
                dummy_element = meta_param.idx_set_node.get_dummy_element(self.state)
                m = len(idx_set)

            try:
                values = expr_node.evaluate(
                    state=self.state, idx_set=idx_set, dummy_element=dummy_element
                )

            except (ValueError, KeyError):
                return

            for i in range(m):

                idx: Optional[mat.Element] = None
                if idx_set is not None:
                    idx = idx_set[i]

                self.state.add_parameter(symbol=symbol, idx=idx, value=values[i])

    def add_meta_variable(self, meta_var: mat.MetaVariable, is_auxiliary: bool = False):
        self.symbols.add(meta_var.symbol)
        self.meta_vars[meta_var.symbol] = meta_var
        if not is_auxiliary:
            self.model_meta_vars.append(meta_var)

    def add_meta_objective(
        self, meta_obj: mat.MetaObjective, is_auxiliary: bool = False
    ):
        self.symbols.add(meta_obj.symbol)
        self.meta_objs[meta_obj.symbol] = meta_obj
        if not is_auxiliary:
            self.model_meta_objs.append(meta_obj)

    def add_meta_constraint(
        self, meta_con: mat.MetaConstraint, is_auxiliary: bool = False
    ):
        self.symbols.add(meta_con.symbol)
        self.meta_cons[meta_con.symbol] = meta_con
        if not is_auxiliary:
            self.model_meta_cons.append(meta_con)

    def add_script_command(self, script_command: SpecialCommand):
        if script_command.symbol in self.script_commands:
            self.script_commands[script_command.symbol].append(script_command)
        else:
            self.script_commands[script_command.symbol] = [script_command]

    # Replacement
    # ------------------------------------------------------------------------------------------------------------------

    def replace_meta_constraint(
        self, old_symbol: str, new_meta_cons: List[mat.MetaConstraint]
    ):

        if len(new_meta_cons) == 1 and old_symbol == new_meta_cons[0].symbol:
            return

        self.symbols.remove(old_symbol)
        self.meta_cons.pop(old_symbol)

        for new_mc in new_meta_cons:
            self.add_meta_constraint(new_mc)

        self.replace_model_meta_constraint(
            old_symbol=old_symbol, new_meta_cons=new_meta_cons
        )

        for sp in self.subproblems.values():
            sp.replace_model_meta_constraint(
                old_symbol=old_symbol, new_meta_cons=new_meta_cons
            )

    # Identifier Generation
    # ------------------------------------------------------------------------------------------------------------------

    def generate_unique_symbol(
        self, base_symbol: str = None, symbol_blacklist: Iterable[str] = None
    ) -> str:
        """
        Generate a unique entity symbol that has not been assigned to a previously declared entity.

        :param base_symbol: prefix of the symbol
        :param symbol_blacklist: string literals to omit when eliciting a unique symbol
        :return: unique symbol
        """

        if base_symbol is None:
            base_symbol = "ENTITY"
        if symbol_blacklist is None:
            symbol_blacklist = set()

        i = 1
        sym = base_symbol
        while (
            sym in symbol_blacklist
            or sym in self.symbols
            or sym in self.unbound_symbols
        ):
            sym = base_symbol + str(i)
            i += 1

        return sym

    # File I/O
    # ------------------------------------------------------------------------------------------------------------------

    def save(self, dir_path: str = ""):
        file_name = self.symbol.lower().replace(" ", "_")
        with open(os.path.join(dir_path, file_name + ".p"), "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, problem_name: str, dir_path: str = "") -> "Problem":
        file_name = problem_name.lower().replace(" ", "_")
        with open(os.path.join(dir_path, file_name + ".p"), "rb") as f:
            problem = pickle.load(f)
        return problem

    def primal_to_dat(
        self,
        file_name: str,
        problem_symbol: str = None,
        problem_idx: mat.Element = None,
        include_defined: bool = False,
        include_default: bool = False,
    ):

        # TODO: filter variables by subproblem

        # generate data script
        dat_script = scr.Script(name=file_name, script_type=scr.ScriptType.DATA)

        for mv in self.model_meta_vars:

            sym = mv.symbol

            can_include = True
            if mv.is_defined and not include_defined:
                can_include = False
            if mv.has_default and not include_default:
                can_include = False

            if can_include:

                # retrieve values
                values = {
                    k[1:]: v.value
                    for k, v in self.state.variables.items()
                    if k[0] == sym
                }

                # generate data statement
                data_statement = ampl_stm.ParameterDataStatement(
                    symbol=sym, type="var", values=values
                )

                dat_script.statements.append(
                    data_statement
                )  # append statement to script

        return dat_script

    def dual_to_dat(
        self,
        file_name: str,
        problem_symbol: str = None,
        problem_idx: mat.Element = None,
    ):

        # TODO: filter variables by subproblem

        # generate data script
        dat_script = scr.Script(name=file_name, script_type=scr.ScriptType.DATA)

        for mc in self.model_meta_cons:

            sym = mc.symbol

            # retrieve duals
            duals = {
                k[1:]: v.dual for k, v in self.state.constraints.items() if k[0] == sym
            }

            # generate data statement
            data_statement = ampl_stm.ParameterDataStatement(
                symbol=sym, type="var", values=duals
            )

            dat_script.statements.append(data_statement)  # append statement to script

        return dat_script
