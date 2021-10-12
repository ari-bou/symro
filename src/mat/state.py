from symro.src.mat.util import *
from symro.src.mat.entity import *


class State:

    def __init__(self):

        self.sets: Dict[tuple, SSet] = {}
        self.params: Dict[tuple, Parameter] = {}
        self.vars: Dict[tuple, Variable] = {}
        self.objs: Dict[tuple, Objective] = {}
        self.cons: Dict[tuple, Constraint] = {}

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
        elif entity_id in self.params:
            return True
        elif entity_id in self.vars:
            return True
        elif entity_id in self.objs:
            return True
        elif entity_id in self.cons:
            return True
        else:
            return False

    def set_exists(self, symbol: str, idx: Element = None) -> bool:
        entity_id = Entity.generate_entity_id(symbol, idx)
        return entity_id in self.sets

    def param_exists(self, symbol: str, idx: Element = None) -> bool:
        entity_id = Entity.generate_entity_id(symbol, idx)
        return entity_id in self.params

    def var_exists(self, symbol: str, idx: Element = None) -> bool:
        entity_id = Entity.generate_entity_id(symbol, idx)
        return entity_id in self.vars

    def obj_exists(self, symbol: str, idx: Element = None) -> bool:
        entity_id = Entity.generate_entity_id(symbol, idx)
        return entity_id in self.objs

    def con_exists(self, symbol: str, idx: Element = None) -> bool:
        entity_id = Entity.generate_entity_id(symbol, idx)
        return entity_id in self.cons

    # Accessors
    # ------------------------------------------------------------------------------------------------------------------

    def get_entity(self, symbol: str, idx: Element = None) -> Entity:

        entity_id = Entity.generate_entity_id(symbol, idx)

        if entity_id in self.sets:
            return self.sets[entity_id]
        elif entity_id in self.params:
            return self.params[entity_id]
        elif entity_id in self.vars:
            return self.vars[entity_id]
        elif entity_id in self.objs:
            return self.objs[entity_id]
        elif entity_id in self.cons:
            return self.cons[entity_id]
        else:
            raise ValueError("Entity '{0}' does not exist".format(Entity.generate_entity_id(symbol, idx)))

    def get_set(self, symbol: str, idx: Element = None) -> SSet:
        return self.sets[Entity.generate_entity_id(symbol, idx)]

    def get_parameter(self, symbol: str, idx: Element = None) -> Parameter:
        return self.params[Entity.generate_entity_id(symbol, idx)]

    def get_variable(self, symbol: str, idx: Element = None) -> Variable:
        return self.vars[Entity.generate_entity_id(symbol, idx)]

    def get_objective(self, symbol: str, idx: Element = None) -> Objective:
        return self.objs[Entity.generate_entity_id(symbol, idx)]

    def get_constraint(self, symbol: str, idx: Element = None) -> Constraint:
        return self.cons[Entity.generate_entity_id(symbol, idx)]

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

        if not self.set_exists(symbol, idx):

            sset = SSet(
                symbol=symbol,
                idx=idx,
                dim=dim,
                elements=elements
            )

            self.sets[sset.entity_id] = sset

            return sset

        else:
            return self.get_set(symbol, idx)

    def build_parameter(self,
                        symbol: str,
                        idx: Element = None,
                        value: float = 0) -> Parameter:

        if not self.param_exists(symbol, idx):

            param = Parameter(
                symbol=symbol,
                idx=idx,
                value=value
            )

            self.params[param.entity_id] = param

            return param

        else:
            return self.get_parameter(symbol, idx)

    def build_variable(self,
                       symbol: str,
                       idx: Element = None,
                       value: float = 0,
                       lb: float = 0,
                       ub: float = 0) -> Variable:

        if not self.var_exists(symbol, idx):

            var = Variable(
                symbol=symbol,
                idx=idx,
                value=value,
                lb=lb,
                ub=ub
            )

            self.vars[var.entity_id] = var

            return var

        else:
            return self.get_variable(symbol, idx)

    def build_objective(self,
                        symbol: str,
                        idx: Element = None,
                        value: float = 0) -> Objective:

        if not self.obj_exists(symbol, idx):

            obj = Objective(
                symbol=symbol,
                idx=idx,
                value=value
            )

            self.objs[obj.entity_id] = obj

            return obj

        else:
            return self.get_objective(symbol, idx)

    def build_constraint(self,
                         symbol: str,
                         idx: Element = None,
                         body: float = 0,
                         lb: float = 0,
                         ub: float = 0,
                         dual: float = 0) -> Constraint:

        if not self.con_exists(symbol, idx):

            con = Constraint(
                symbol=symbol,
                idx=idx,
                value=body,
                lb=lb,
                ub=ub,
                dual=dual
            )

            self.cons[con.entity_id] = con

            return con

        else:
            return self.get_constraint(symbol, idx)
