from abc import ABC, abstractmethod
import re
from typing import Dict, List, Optional, Tuple, Union

from symro.src.mat.util import IndexingSet, Element


class Entity(ABC):

    def __init__(self,
                 symbol: str,
                 idx: Element = None,
                 is_dim_aggregated: List[bool] = None):
        """
        Constructor of the Entity class.

        :param symbol: unique declared symbol that identifies the entity
        :param idx: unique indexing set element that identifies the entity if indexed, None otherwise
        :param is_dim_aggregated: list of Boolean flags that indicates which dimensions are aggregated
        """

        self.symbol: str = symbol
        self.idx: Optional[Element] = idx  # length: dim
        self.entity_id: str = self.generate_entity_id(self.symbol, self.idx)
        self.is_dim_aggregated: List[bool] = is_dim_aggregated  # length: dim

        if self.is_dim_aggregated is None:
            self.is_dim_aggregated = [False] * self.get_dim()

    def __str__(self):
        literal = self.symbol
        if self.get_dim() > 0:
            index_literals = []
            for j, idx in enumerate(self.idx):
                if isinstance(idx, int) or isinstance(idx, float):
                    index_literals.append(str(idx))
                elif isinstance(idx, str):
                    if self.is_dim_aggregated[j]:
                        index_literals.append(idx)
                    else:
                        index_literals.append("'{0}'".format(idx))
                else:
                    index_literals.append(str(idx))
            literal += "[{0}]".format(','.join(index_literals))
        return literal

    def is_aggregate(self) -> bool:
        if self.get_dim() == 0:
            return False
        else:
            return any(self.is_dim_aggregated)

    def get_dim(self) -> int:
        if self.idx is None:
            return 0
        else:
            return len(self.idx)

    @abstractmethod
    def get_value(self):
        pass

    @staticmethod
    def generate_entity_id(symbol: str, idx: Element) -> str:

        # indexed entity
        if idx is not None and len(idx) > 0:
            idx = [str(i) for i in idx]
            return "{0}[{1}]".format(symbol, ','.join(idx))

        # scalar entity
        else:
            return symbol


class SSet(Entity):

    def __init__(self,
                 symbol: str,
                 idx: Element = None,
                 is_dim_aggregated: List[bool] = None,
                 dim: int = 0,
                 elements: IndexingSet = None):
        super(SSet, self).__init__(symbol=symbol,
                                   idx=idx,
                                   is_dim_aggregated=is_dim_aggregated)
        self.dim: int = dim
        self.elements: IndexingSet = elements

    def __eq__(self, other):
        if isinstance(other, SSet) and self.entity_id == other.entity_id:
            return True
        return False

    def __len__(self):
        return len(self.elements)

    def get_value(self):
        return None


class Parameter(Entity):

    def __init__(self,
                 symbol: str,
                 idx: Element = None,
                 is_dim_aggregated: List[bool] = None,
                 value: float = 0):
        super(Parameter, self).__init__(symbol=symbol,
                                        idx=idx,
                                        is_dim_aggregated=is_dim_aggregated)
        self.value: Union[float, str] = value

    def __eq__(self, other):
        if isinstance(other, Parameter) and self.entity_id == other.entity_id:
            return True
        return False

    def get_value(self):
        return self.value


class Variable(Entity):

    def __init__(self,
                 symbol: str,
                 idx: Element = None,
                 is_dim_aggregated: List[bool] = None,
                 value: float = 0,
                 lb: float = 0,
                 ub: float = 0):
        super(Variable, self).__init__(symbol=symbol,
                                       idx=idx,
                                       is_dim_aggregated=is_dim_aggregated)
        self.value: Union[float, str] = value
        self.lb: float = float(lb)
        self.ub: float = float(ub)

    def __eq__(self, other):
        if isinstance(other, Variable) and self.entity_id == other.entity_id:
            return True
        return False

    def get_value(self):
        return self.value


class Objective(Entity):

    def __init__(self,
                 symbol: str,
                 idx: Element = None,
                 is_dim_aggregated: List[bool] = None,
                 value: float = 0):
        super(Objective, self).__init__(symbol=symbol,
                                        idx=idx,
                                        is_dim_aggregated=is_dim_aggregated)
        self.value: Union[float, str] = value

    def __eq__(self, other):
        if isinstance(other, Objective) and self.entity_id == other.entity_id:
            return True
        return False

    def get_value(self):
        return self.value


class Constraint(Entity):

    def __init__(self,
                 symbol: str,
                 idx: Element = None,
                 is_dim_aggregated: List[bool] = None,
                 value: float = 0,
                 lb: float = 0,
                 ub: float = 0,
                 dual: float = 0):
        super(Constraint, self).__init__(symbol=symbol,
                                         idx=idx,
                                         is_dim_aggregated=is_dim_aggregated)
        self.value: Union[float, str] = value  # body
        self.lb: float = float(lb)
        self.ub: float = float(ub)
        self.dual: float = dual

    def __eq__(self, other):
        if isinstance(other, Constraint) and self.entity_id == other.entity_id:
            return True
        return False

    def get_value(self):
        return self.value


# Entity Collection
# ----------------------------------------------------------------------------------------------------------------------

class EntityCollection:

    def __init__(self, symbol):

        self.symbol: str = symbol  # name of entity collection

        # key: index of entity instance; value: entity.
        # key is None if the entity is scalar
        self.entity_map: Dict[Element, Entity] = {}

    def filter(self, indices_filter: Tuple[Optional[str], ...]):

        filtered_map = {}

        for entity_indices, entity in self.entity_map.items():
            index_in_filter = []
            for entity_index, index_filter in zip(entity_indices, indices_filter):
                if index_filter is not None and index_filter != "":
                    index_in_filter.append(entity_index == index_filter)
                else:
                    index_in_filter.append(True)
            entity_in_filter = all(index_in_filter)
            if entity_in_filter:
                filtered_map[entity_indices] = entity

        return filtered_map

    def generate_value_dict(self):
        self.entity_map: Dict[Element, Union[Parameter, Variable, Objective, Constraint]]
        return {k: v.value for k, v in self.entity_map.items()}

    def generate_dual_dict(self):
        self.entity_map: Dict[Element, Union[Constraint]]
        return {k: v.dual for k, v in self.entity_map.items()}
