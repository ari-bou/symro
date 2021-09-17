from typing import Dict, Optional, Tuple, Union
import warnings

from symro.core.mat.entity import Entity, EntityCollection, SSet, Parameter, Variable, Objective, Constraint
from symro.core.mat.util import Element


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
        if symbol in self.set_collections:
            return idx in self.set_collections[symbol].entity_map
        elif symbol in self.param_collections:
            return idx in self.param_collections[symbol].entity_map
        elif symbol in self.var_collections:
            return idx in self.var_collections[symbol].entity_map
        elif symbol in self.obj_collections:
            return idx in self.obj_collections[symbol].entity_map
        elif symbol in self.con_collections:
            return idx in self.con_collections[symbol].entity_map
        else:
            return False

    # Accessors
    # ------------------------------------------------------------------------------------------------------------------

    def get_entity(self, symbol: str, idx: Element) -> Optional[Entity]:
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
            warnings.warn("Entity '{0}' does not exist".format(Entity.generate_entity_id(symbol, idx)))
            return None

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

    def get_set(self, set_id: str) -> SSet:
        set_id = set_id.strip().replace("'", "")
        return self.sets.get(set_id, None)

    def get_parameter(self, param_id: str) -> Parameter:
        param_id = param_id.strip().replace("'", "")
        return self.params.get(param_id, None)

    def get_variable(self, var_id: str) -> Variable:
        var_id = var_id.strip().replace("'", "")
        return self.vars.get(var_id, None)

    def get_objective(self, obj_id: str) -> Objective:
        obj_id = obj_id.strip().replace("'", "")
        return self.objs.get(obj_id, None)

    def get_constraint(self, con_id: str) -> Constraint:
        con_id = con_id.strip().replace("'", "")
        return self.cons.get(con_id, None)

    # Object Addition
    # ------------------------------------------------------------------------------------------------------------------

    def add_set(self, sset: SSet):
        self.sets[sset.entity_id] = sset
        if sset.symbol in self.set_collections:
            self.set_collections[sset.symbol].entity_map[sset.idx] = sset
        else:
            collection = EntityCollection(sset.symbol)
            collection.entity_map[sset.idx] = sset
            self.set_collections[sset.symbol] = collection

    def add_parameter(self, param: Parameter):
        self.params[param.entity_id] = param
        if param.symbol in self.param_collections:
            self.param_collections[param.symbol].entity_map[param.idx] = param
        else:
            collection = EntityCollection(param.symbol)
            collection.entity_map[param.idx] = param
            self.param_collections[param.symbol] = collection

    def add_variable(self, var: Variable):
        self.vars[var.entity_id] = var
        if var.symbol in self.var_collections:
            self.var_collections[var.symbol].entity_map[var.idx] = var
        else:
            collection = EntityCollection(var.symbol)
            collection.entity_map[var.idx] = var
            self.var_collections[var.symbol] = collection

    def add_objective(self, obj: Objective):
        self.objs[obj.entity_id] = obj
        if obj.symbol in self.obj_collections:
            self.obj_collections[obj.symbol].entity_map[obj.idx] = obj
        else:
            collection = EntityCollection(obj.symbol)
            collection.entity_map[obj.idx] = obj
            self.obj_collections[obj.symbol] = collection

    def add_constraint(self, con: Constraint):
        self.cons[con.entity_id] = con
        if con.symbol in self.con_collections:
            self.con_collections[con.symbol].entity_map[con.idx] = con
        else:
            collection = EntityCollection(con.symbol)
            collection.entity_map[con.idx] = con
            self.con_collections[con.symbol] = collection
