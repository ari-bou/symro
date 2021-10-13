from typing import Dict, Sequence

from symro.src.mat.util import *
from symro.src.mat.entity import *


class State:

    def __init__(self):

        self.sets: Dict[str, Dict[Element, SSet]] = {}
        self.parameters: Dict[str, Dict[Element, Parameter]] = {}
        self.variables: Dict[str, Dict[Element, Variable]] = {}
        self.objectives: Dict[str, Dict[Element, Objective]] = {}
        self.constraints: Dict[str, Dict[Element, Constraint]] = {}

    def __getitem__(self, item: Sequence):

        sym = item[0]
        idx = item[1:]

        if sym in self.sets:
            return self.get_set(sym, idx).get_value()
        elif sym in self.parameters:
            return self.get_parameter(sym, idx).get_value()
        elif sym in self.variables:
            return self.get_variable(sym, idx).get_value()
        elif sym in self.objectives:
            return self.get_objective(sym, idx).get_value()
        elif sym in self.constraints:
            return self.get_constraint(sym, idx).get_value()
        else:
            raise ValueError("State does not own an entity with id '{0}'".format(item))

    def __setitem__(self, key: Sequence, value: Union[int, float, str, Iterable]):

        sym = key[0]
        idx = key[1:]

        if sym in self.sets:
            self.get_set(sym, idx).elements = OrderedSet(value)
        elif sym in self.parameters:
            self.get_parameter(sym, idx).value = value
        elif sym in self.variables:
            self.get_variable(sym, idx).value = value
        elif sym in self.objectives:
            self.get_objective(sym, idx).value = value
        elif sym in self.constraints:
            self.get_constraint(sym, idx).dual = value
        else:
            raise ValueError("State does not own an entity with id '{0}'".format(key))

    # Checkers
    # ------------------------------------------------------------------------------------------------------------------

    def entity_exists(self, symbol: str, idx: Element = None) -> bool:
        """
        Check if an entity with the supplied symbol and indexing element is stored within the state.

        :param symbol: unique declared symbol that identifies the entity
        :param idx: indexing set element that identifies the entity if indexed, None otherwise
        :return: True if the entity exists within the state
        """

        if symbol in self.sets and idx in self.sets[symbol]:
            return True
        elif symbol in self.parameters and idx in self.parameters[symbol]:
            return True
        elif symbol in self.variables and idx in self.variables[symbol]:
            return True
        elif symbol in self.objectives and idx in self.objectives[symbol]:
            return True
        elif symbol in self.constraints and idx in self.constraints[symbol]:
            return True
        else:
            return False

    # Accessors and Mutators
    # ------------------------------------------------------------------------------------------------------------------

    def get_entity(self, symbol: str, idx: Optional[Element], entity_type: str) -> Entity:

        if entity_type == SET_TYPE:
            return self.get_set(symbol, idx)

        elif entity_type == PARAM_TYPE:
            return self.get_parameter(symbol, idx)

        elif entity_type == VAR_TYPE:
            return self.get_variable(symbol, idx)

        elif entity_type == OBJ_TYPE:
            return self.get_objective(symbol, idx)

        elif entity_type == CON_TYPE:
            return self.get_constraint(symbol, idx)

        else:
            raise ValueError("Unable to resolve '{0}' as an entity type".format(entity_type))

    def get_set(self, symbol: str, idx: Element = None) -> SSet:
        entity_dict = self.sets.setdefault(symbol, {})
        entity = entity_dict.get(idx, None)
        if entity is None:
            entity = self.add_set(symbol, idx)
        return entity

    def get_parameter(self, symbol: str, idx: Element = None) -> Parameter:
        entity_dict = self.parameters.setdefault(symbol, {})
        entity = entity_dict.get(idx, None)
        if entity is None:
            entity = self.add_parameter(symbol, idx)
        return entity

    def get_variable(self, symbol: str, idx: Element = None) -> Variable:
        entity_dict = self.variables.setdefault(symbol, {})
        entity = entity_dict.get(idx, None)
        if entity is None:
            entity = self.add_variable(symbol, idx)
        return entity

    def get_objective(self, symbol: str, idx: Element = None) -> Objective:
        entity_dict = self.objectives.setdefault(symbol, {})
        entity = entity_dict.get(idx, None)
        if entity is None:
            entity = self.add_objective(symbol, idx)
        return entity

    def get_constraint(self, symbol: str, idx: Element = None) -> Constraint:
        entity_dict = self.constraints.setdefault(symbol, {})
        entity = entity_dict.get(idx, None)
        if entity is None:
            entity = self.add_constraint(symbol, idx)
        return entity

    # Entity Construction
    # ------------------------------------------------------------------------------------------------------------------

    def add_entity(self,
                   symbol: str,
                   idx: Optional[Element],
                   entity_type: str) -> Entity:
        if entity_type == SET_TYPE:
            return self.add_set(symbol, idx)
        elif entity_type == PARAM_TYPE:
            return self.add_parameter(symbol, idx)
        elif entity_type == VAR_TYPE:
            return self.add_variable(symbol, idx)
        elif entity_type == OBJ_TYPE:
            return self.add_objective(symbol, idx)
        elif entity_type == CON_TYPE:
            return self.add_constraint(symbol, idx)
        else:
            raise ValueError("Unable to resolve '{0}' as an entity type".format(entity_type))

    def add_set(self,
                symbol: str,
                idx: Element = None,
                dim: int = 0,
                elements: IndexingSet = None) -> SSet:
        sset = SSet(
            symbol=symbol,
            idx=idx,
            dim=dim,
            elements=elements
        )
        self.sets.setdefault(symbol, {})[idx] = sset
        return sset

    def add_parameter(self,
                      symbol: str,
                      idx: Element = None,
                      value: float = 0) -> Parameter:
        param = Parameter(
            symbol=symbol,
            idx=idx,
            value=value
        )
        self.parameters.setdefault(symbol, {})[idx] = param
        return param

    def add_variable(self,
                     symbol: str,
                     idx: Element = None,
                     value: float = 0,
                     lb: float = 0,
                     ub: float = 0) -> Variable:
        var = Variable(
            symbol=symbol,
            idx=idx,
            value=value,
            lb=lb,
            ub=ub
        )
        self.variables.setdefault(symbol, {})[idx] = var
        return var

    def add_objective(self,
                      symbol: str,
                      idx: Element = None,
                      value: float = 0) -> Objective:
        obj = Objective(
            symbol=symbol,
            idx=idx,
            value=value
        )
        self.objectives.setdefault(symbol, {})[idx] = obj
        return obj

    def add_constraint(self,
                       symbol: str,
                       idx: Element = None,
                       body: float = 0,
                       lb: float = 0,
                       ub: float = 0,
                       dual: float = 0) -> Constraint:
        con = Constraint(
            symbol=symbol,
            idx=idx,
            value=body,
            lb=lb,
            ub=ub,
            dual=dual
        )
        self.constraints.setdefault(symbol, {})[idx] = con
        return con
