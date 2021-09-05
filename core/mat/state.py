from typing import Dict, Optional, Tuple, Union
import warnings

from symro.core.mat.entity import Entity, EntityCollection, ASet, Parameter, Variable, Objective, Constraint


class State:

    def __init__(self):

        self.sets: Dict[str, ASet] = {}
        self.params: Dict[str, Parameter] = {}
        self.vars: Dict[str, Variable] = {}
        self.objs: Dict[str, Objective] = {}
        self.cons: Dict[str, Constraint] = {}

        self.set_collections: Dict[str, EntityCollection] = {}
        self.param_collections: Dict[str, EntityCollection] = {}
        self.var_collections: Dict[str, EntityCollection] = {}
        self.obj_collections: Dict[str, EntityCollection] = {}
        self.con_collections: Dict[str, EntityCollection] = {}

    def get_entity(self, symbol: str, indices: Tuple[Union[int, float, str], ...]) -> Optional[Entity]:
        if symbol in self.set_collections:
            return self.set_collections[symbol].entity_map[indices]
        elif symbol in self.param_collections:
            return self.param_collections[symbol].entity_map[indices]
        elif symbol in self.var_collections:
            return self.var_collections[symbol].entity_map[indices]
        elif symbol in self.obj_collections:
            return self.obj_collections[symbol].entity_map[indices]
        elif symbol in self.con_collections:
            return self.con_collections[symbol].entity_map[indices]
        else:
            warnings.warn("Entity '{0}' does not exist".format(Entity.generate_entity_id(symbol, indices)))
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

    def get_set(self, set_id: str) -> ASet:
        set_id = set_id.strip().replace("'", "")
        return self.sets.get(set_id, None)

    def get_parameter(self, param_id: str) -> Parameter:
        param_id = param_id.strip().replace("'", "")
        return self.params.get(param_id, None)

    def get_variable(self, var_id: str) -> Variable:
        var_id = var_id.strip().replace("'", "")
        return self.vars.get(var_id, None)

    def get_constraint(self, con_id: str) -> Constraint:
        con_id = con_id.strip().replace("'", "")
        return self.cons.get(con_id, None)

    def add_set(self, aset: ASet):
        self.sets[aset.entity_id] = aset
        if aset.symbol in self.set_collections:
            self.set_collections[aset.symbol].entity_map[aset.indices] = aset
        else:
            collection = EntityCollection(aset.symbol)
            collection.entity_map[aset.indices] = aset
            self.set_collections[aset.symbol] = collection

    def add_parameter(self, param: Parameter):
        self.params[param.entity_id] = param
        if param.symbol in self.param_collections:
            self.param_collections[param.symbol].entity_map[param.indices] = param
        else:
            collection = EntityCollection(param.symbol)
            collection.entity_map[param.indices] = param
            self.param_collections[param.symbol] = collection

    def add_variable(self, var: Variable):
        self.vars[var.entity_id] = var
        if var.symbol in self.var_collections:
            self.var_collections[var.symbol].entity_map[var.indices] = var
        else:
            collection = EntityCollection(var.symbol)
            collection.entity_map[var.indices] = var
            self.var_collections[var.symbol] = collection

    def add_objective(self, obj: Objective):
        self.objs[obj.entity_id] = obj
        if obj.symbol in self.obj_collections:
            self.obj_collections[obj.symbol].entity_map[obj.indices] = obj
        else:
            collection = EntityCollection(obj.symbol)
            collection.entity_map[obj.indices] = obj
            self.obj_collections[obj.symbol] = collection

    def add_constraint(self, con: Constraint):
        self.cons[con.entity_id] = con
        if con.symbol in self.con_collections:
            self.con_collections[con.symbol].entity_map[con.indices] = con
        else:
            collection = EntityCollection(con.symbol)
            collection.entity_map[con.indices] = con
            self.con_collections[con.symbol] = collection
