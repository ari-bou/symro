from typing import Dict, Sequence

from symro.src.mat.util import *
from symro.src.mat.entity import *


class State:

    def __init__(self):

        self.sets: Dict[tuple, SSet] = {}
        self.parameters: Dict[tuple, Parameter] = {}
        self.variables: Dict[tuple, Variable] = {}
        self.objectives: Dict[tuple, Objective] = {}
        self.constraints: Dict[tuple, Constraint] = {}

    def __getitem__(self, item: Sequence):

        entity_id = Entity.generate_entity_id(item[0], item[1:])

        if entity_id in self.sets:
            return self.sets[entity_id].get_value()
        elif entity_id in self.parameters:
            return self.parameters[entity_id].get_value()
        elif entity_id in self.variables:
            return self.variables[entity_id].get_value()
        elif entity_id in self.objectives:
            return self.objectives[entity_id].get_value()
        elif entity_id in self.constraints:
            return self.constraints[entity_id].get_value()
        else:
            raise ValueError("State does not own an entity with id '{0}'".format(entity_id))

    def __setitem__(self, key: Sequence, value: Union[int, float, str, Iterable]):

        entity_id = Entity.generate_entity_id(key[0], key[1:])

        if entity_id in self.sets:
            self.sets[entity_id].elements = OrderedSet(value)
        elif entity_id in self.parameters:
            self.parameters[entity_id].value = value
        elif entity_id in self.variables:
            self.variables[entity_id].value = value
        elif entity_id in self.objectives:
            self.objectives[entity_id].value = value
        elif entity_id in self.constraints:
            self.constraints[entity_id].dual = value
        else:
            raise ValueError("State does not own an entity with id '{0}'".format(entity_id))

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

    def set_exists(self, symbol: str, idx: Element = None) -> bool:
        entity_id = Entity.generate_entity_id(symbol, idx)
        return entity_id in self.sets

    def parameter_exists(self, symbol: str, idx: Element = None) -> bool:
        entity_id = Entity.generate_entity_id(symbol, idx)
        return entity_id in self.parameters

    def variable_exists(self, symbol: str, idx: Element = None) -> bool:
        entity_id = Entity.generate_entity_id(symbol, idx)
        return entity_id in self.variables

    def objective_exists(self, symbol: str, idx: Element = None) -> bool:
        entity_id = Entity.generate_entity_id(symbol, idx)
        return entity_id in self.objectives

    def constraint_exists(self, symbol: str, idx: Element = None) -> bool:
        entity_id = Entity.generate_entity_id(symbol, idx)
        return entity_id in self.constraints

    # Accessors and Mutators
    # ------------------------------------------------------------------------------------------------------------------

    def get_entity(self, symbol: str, idx: Element = None) -> Entity:

        entity_id = Entity.generate_entity_id(symbol, idx)

        if entity_id in self.sets:
            return self.sets[entity_id]
        elif entity_id in self.parameters:
            return self.parameters[entity_id]
        elif entity_id in self.variables:
            return self.variables[entity_id]
        elif entity_id in self.objectives:
            return self.objectives[entity_id]
        elif entity_id in self.constraints:
            return self.constraints[entity_id]
        else:
            raise ValueError("Entity '{0}' does not exist".format(Entity.generate_entity_id(symbol, idx)))

    def get_set(self, symbol: str, idx: Element = None) -> SSet:
        return self.sets[Entity.generate_entity_id(symbol, idx)]

    def get_parameter(self, symbol: str, idx: Element = None) -> Parameter:
        return self.parameters[Entity.generate_entity_id(symbol, idx)]

    def get_variable(self, symbol: str, idx: Element = None) -> Variable:
        return self.variables[Entity.generate_entity_id(symbol, idx)]

    def get_objective(self, symbol: str, idx: Element = None) -> Objective:
        return self.objectives[Entity.generate_entity_id(symbol, idx)]

    def get_constraint(self, symbol: str, idx: Element = None) -> Constraint:
        return self.constraints[Entity.generate_entity_id(symbol, idx)]

    def set_variable_value(self, value: Union[int, float, str], symbol: str, idx: Element = None):
        entity_id = Entity.generate_entity_id(symbol, idx)
        self.variables[entity_id].value = value

    # Entity Construction
    # ------------------------------------------------------------------------------------------------------------------

    def build_entity(self,
                     symbol: str,
                     idx: Optional[Element],
                     type: str) -> Entity:
        if type == SET_TYPE:
            return self.build_set(symbol, idx)
        elif type == PARAM_TYPE:
            return self.build_parameter(symbol, idx)
        elif type == VAR_TYPE:
            return self.build_variable(symbol, idx)
        elif type == OBJ_TYPE:
            return self.build_objective(symbol, idx)
        elif type == CON_TYPE:
            return self.build_constraint(symbol, idx)
        else:
            raise ValueError("Unable to resolve '{0}' as an entity type".format(type))

    def build_set(self,
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
        return sset

    def build_parameter(self,
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

    def build_variable(self,
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

    def build_objective(self,
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

    def build_constraint(self,
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
