from typing import Dict, Sequence

from symro.src.mat.util import *
from symro.src.mat.entity import *


class State:
    """
    class State

    A container object that manages the data of a problem instance, including:
        - the elements of all defined sets,
        - the dimensions of all defined sets,
        - the values of all defined parameters,
        - the values and bounds of all defined variables,
        - the values of all defined objectives,
        - the bodies, bounds, and dual values of all defined constraints.
    The data associated with a particular problem entity is stored in a corresponding Entity object.
    Attempting to manipulate a set or a parameter for which no data has been supplied raises a value error. Otherwise,
    attempting to manipulate a variable, an objective, or a constraint will result in the automatic construction of the
    appropriate Entity object with all properties defaulting to 0.
    """

    def __init__(self):

        self.sets: Dict[tuple, SSet] = {}
        self.parameters: Dict[tuple, Parameter] = {}
        self.variables: Dict[tuple, Variable] = {}
        self.objectives: Dict[tuple, Objective] = {}
        self.constraints: Dict[tuple, Constraint] = {}

        self.set_dims: Dict[str, int] = {}

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
            raise ValueError("Entity with id {0} does not exist in the problem state".format(item))

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
            raise ValueError("Entity with id {0} does not exist in the problem state".format(key))

    # Checkers
    # ------------------------------------------------------------------------------------------------------------------

    def entity_exists(self, symbol: str, idx: Element = None) -> bool:
        """
        Check if an entity with the supplied symbol and indexing element is stored within the state.

        :param symbol: unique declared symbol that identifies the entity
        :param idx: indexing set element that identifies the entity if indexed, None otherwise
        :return: True if the entity exists within the state
        """

        entity_id = Entity.generate_entity_id(symbol, idx)

        if entity_id in self.sets:
            return True
        elif entity_id in self.parameters:
            return True
        elif entity_id in self.variables:
            return True
        elif entity_id in self.objectives:
            return True
        elif entity_id in self.constraints:
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

        entity_id = Entity.generate_entity_id(symbol, idx)
        entity = self.sets.get(entity_id, None)

        if entity is None:
            if idx is None:
                set_literal = symbol
            else:
                set_literal = "{0}[{1}]".format(symbol, ','.join([str(i) for i in idx]))
            raise ValueError("Instance of set {0} does not exist in the state".format(set_literal))

        return entity

    def get_parameter(self, symbol: str, idx: Element = None) -> Parameter:

        entity_id = Entity.generate_entity_id(symbol, idx)
        entity = self.parameters.get(entity_id, None)

        if entity is None:
            if idx is None:
                param_literal = symbol
            else:
                param_literal = "{0}[{1}]".format(symbol, ','.join([str(i) for i in idx]))
            raise ValueError("Instance of parameter {0} does not exist in the state".format(param_literal))

        return entity

    def get_variable(self, symbol: str, idx: Element = None) -> Variable:
        entity_id = Entity.generate_entity_id(symbol, idx)
        entity = self.variables.get(entity_id, None)
        if entity is None:
            entity = self.add_variable(symbol, idx)
        return entity

    def get_objective(self, symbol: str, idx: Element = None) -> Objective:
        entity_id = Entity.generate_entity_id(symbol, idx)
        entity = self.objectives.get(entity_id, None)
        if entity is None:
            entity = self.add_objective(symbol, idx)
        return entity

    def get_constraint(self, symbol: str, idx: Element = None) -> Constraint:
        entity_id = Entity.generate_entity_id(symbol, idx)
        entity = self.constraints.get(entity_id, None)
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
        self.sets[sset.entity_id] = sset
        if symbol not in self.set_dims:
            self.set_dims[symbol] = dim
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
        self.parameters[param.entity_id] = param
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
        self.variables[var.entity_id] = var
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
        self.objectives[obj.entity_id] = obj
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
        self.constraints[con.entity_id] = con
        return con
