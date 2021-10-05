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
    def generate_entity_id(symbol: str, indices: Tuple[Union[int, float, str], ...]) -> str:
        if indices is not None and len(indices) > 0:
            indices = [str(i) for i in indices]
            return "{0}[{1}]".format(symbol, ','.join(indices))
        else:
            return symbol

    @classmethod
    def parse_entity_id_literal(cls,
                                entity_id_literal: str,
                                ) -> Tuple[str, List[Union[int, float, str]]]:

        if '[' in entity_id_literal:

            symbol = entity_id_literal.split('[')[0]

            multi_index = re.search(r"\[(.+)\]", entity_id_literal).group()
            multi_index = multi_index.replace('[', '').replace(']', '')
            indices = multi_index.split(',')
            indices = cls.standardize_str_indices(indices)

        else:
            symbol = entity_id_literal
            indices = []

        return symbol, indices

    @staticmethod
    def parse_index_literal(index_literal: str) -> List[str]:
        if index_literal == "":
            return []
        index_literal = index_literal.replace('[', "").replace(']', "").strip()
        symbols = index_literal.split(',')
        return symbols

    @staticmethod
    def standardize_indices(raw_indices: Union[int, float, str,
                                               Tuple[Union[int, float, str], ...],
                                               List[Union[int, float, str]]]
                            ) -> Tuple[Union[int, float, str], ...]:

        if not isinstance(raw_indices, tuple) and not isinstance(raw_indices, list):
            raw_indices = [raw_indices]

        indices = list(raw_indices)
        for i, raw_index in enumerate(raw_indices):

            if isinstance(raw_index, str):
                if len(raw_index) > 0:
                    if raw_index[0] in ["'", '"']:
                        raw_index = raw_index[1:]
                if len(raw_index) > 0:
                    if raw_index[len(raw_index) - 1] in ["'", '"']:
                        raw_index = raw_index[:-1]
                indices[i] = raw_index

            elif isinstance(raw_index, float):
                if raw_index.is_integer():
                    indices[i] = int(raw_index)

        return tuple(indices)

    @classmethod
    def standardize_entity_str_identifiers(cls,
                                           symbol: str,
                                           indices: Union[List[str],
                                                          Tuple[str, ...]],
                                           ) -> Tuple[str, str, Tuple[Union[int, float, str], ...]]:
        if symbol[0] in ["'", '"']:
            symbol = symbol[1:]
        if len(indices) > 0:
            indices = cls.standardize_str_indices(indices)
            entity_id = "{0}[{1}]".format(symbol, ','.join([str(idx) for idx in indices]))
        else:
            entity_id = symbol
        return entity_id, symbol, tuple(indices)

    @staticmethod
    def standardize_str_indices(raw_indices: Union[str,
                                                   Tuple[str, ...],
                                                   List[str]]
                                ) -> Tuple[Union[int, float, str], ...]:

        if not isinstance(raw_indices, tuple) and not isinstance(raw_indices, list):
            raw_indices = [raw_indices]

        indices = list(raw_indices)
        for i, raw_index in enumerate(raw_indices):

            if isinstance(raw_index, str):
                if raw_index.isnumeric():
                    raw_index = float(raw_index)
                    if raw_index.is_integer():
                        raw_index = int(raw_index)
                else:
                    if len(raw_index) > 0:
                        if raw_index[0] in ["'", '"']:
                            raw_index = raw_index[1:]
                    if len(raw_index) > 0:
                        if raw_index[len(raw_index) - 1] in ["'", '"']:
                            raw_index = raw_index[:-1]

            indices[i] = raw_index

        return tuple(indices)


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
                 idx: Element,
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
                 idx: Element,
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
                 idx: Element,
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
