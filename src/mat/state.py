from symro.src.mat.util import *
from symro.src.mat.entity import *


class State:

    def __init__(self):

        self.sets: Dict[str, SSet] = {}
        self.params: Dict[str, Parameter] = {}
        self.vars: Dict[str, Variable] = {}
        self.objs: Dict[str, Objective] = {}
        self.cons: Dict[str, Constraint] = {}

        self.set_collections: Dict[str, EntityCollection] = {}
        self.param_collections: Dict[str, EntityCollection] = {}
        self.var_collections: Dict[str, EntityCollection] = {}
        self.obj_collections: Dict[str, EntityCollection] = {}
        self.con_collections: Dict[str, EntityCollection] = {}

    # Checkers
    # ------------------------------------------------------------------------------------------------------------------

    def entity_exists(self, symbol: str, idx: Element = None) -> bool:
        """
        Check if an entity with the supplied symbol and indexing element is stored within the state.

        :param symbol: unique declared symbol that identifies the entity
        :param idx: indexing set element that identifies the entity if indexed, None otherwise
        :return: True if the entity exists within the state
        """
        if symbol in self.set_collections and idx in self.set_collections[symbol].entity_map:
            return True
        elif symbol in self.param_collections and idx in self.param_collections[symbol].entity_map:
            return True
        elif symbol in self.var_collections and idx in self.var_collections[symbol].entity_map:
            return True
        elif symbol in self.obj_collections and idx in self.obj_collections[symbol].entity_map:
            return True
        elif symbol in self.con_collections and idx in self.con_collections[symbol].entity_map:
            return True
        else:
            return False

    def set_exists(self, symbol: str, idx: Element = None) -> bool:
        return symbol in self.set_collections and idx in self.set_collections[symbol].entity_map

    def param_exists(self, symbol: str, idx: Element = None) -> bool:
        return symbol in self.param_collections and idx in self.param_collections[symbol].entity_map

    def var_exists(self, symbol: str, idx: Element = None) -> bool:
        return symbol in self.var_collections and idx in self.var_collections[symbol].entity_map

    def obj_exists(self, symbol: str, idx: Element = None) -> bool:
        return symbol in self.obj_collections and idx in self.obj_collections[symbol].entity_map

    def con_exists(self, symbol: str, idx: Element = None) -> bool:
        return symbol in self.con_collections and idx in self.con_collections[symbol].entity_map

    # Accessors
    # ------------------------------------------------------------------------------------------------------------------

    def get_entity(self, symbol: str, idx: Element = None) -> Optional[Entity]:
        if symbol in self.set_collections:
            return self.set_collections[symbol].entity_map[idx]
        elif symbol in self.param_collections:
            return self.param_collections[symbol].entity_map[idx]
        elif symbol in self.var_collections:
            return self.var_collections[symbol].entity_map[idx]
        elif symbol in self.obj_collections:
            return self.obj_collections[symbol].entity_map[idx]
        elif symbol in self.con_collections:
            return self.con_collections[symbol].entity_map[idx]
        else:
            raise ValueError("Entity '{0}' does not exist".format(Entity.generate_entity_id(symbol, idx)))

    def get_collection(self, symbol: str) -> Optional[EntityCollection]:
        if symbol in self.set_collections:
            return self.set_collections[symbol]
        elif symbol in self.param_collections:
            return self.param_collections[symbol]
        elif symbol in self.var_collections:
            return self.var_collections[symbol]
        elif symbol in self.obj_collections:
            return self.obj_collections[symbol]
        elif symbol in self.con_collections:
            return self.con_collections[symbol]
        else:
            return None

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

            if sset.symbol in self.set_collections:
                self.set_collections[sset.symbol].entity_map[sset.idx] = sset
            else:
                collection = EntityCollection(sset.symbol)
                collection.entity_map[sset.idx] = sset
                self.set_collections[sset.symbol] = collection

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

            if param.symbol in self.param_collections:
                self.param_collections[param.symbol].entity_map[param.idx] = param
            else:
                collection = EntityCollection(param.symbol)
                collection.entity_map[param.idx] = param
                self.param_collections[param.symbol] = collection

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

            if var.symbol in self.var_collections:
                self.var_collections[var.symbol].entity_map[var.idx] = var
            else:
                collection = EntityCollection(var.symbol)
                collection.entity_map[var.idx] = var
                self.var_collections[var.symbol] = collection

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

            if obj.symbol in self.obj_collections:
                self.obj_collections[obj.symbol].entity_map[obj.idx] = obj
            else:
                collection = EntityCollection(obj.symbol)
                collection.entity_map[obj.idx] = obj
                self.obj_collections[obj.symbol] = collection

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

            if con.symbol in self.con_collections:
                self.con_collections[con.symbol].entity_map[con.idx] = con
            else:
                collection = EntityCollection(con.symbol)
                collection.entity_map[con.idx] = con
                self.con_collections[con.symbol] = collection

            return con

        else:
            return self.get_constraint(symbol, idx)
