from abc import ABC
import re
from typing import Dict, List, Optional, Tuple, Union

from symro.core.mat.util import IndexSet


class Entity(ABC):

    def __init__(self,
                 symbol: str,
                 indices: Tuple[Union[int, float, str], ...] = None,
                 is_dim_aggregated: List[bool] = None,
                 value: Union[float, str] = 0,
                 lb: float = 0,
                 ub: float = 0):

        self.symbol: str = symbol
        self.indices: Tuple[Union[int, float, str], ...] = indices if indices is not None else tuple([])  # length: dim
        self.entity_id: str = self.generate_entity_id(self.symbol, self.indices)
        self.is_dim_aggregated: List[bool] = is_dim_aggregated  # length: dim
        self.value: Union[float, str] = value
        self.lb: float = float(lb)
        self.ub: float = float(ub)

        if self.indices is None:
            self.indices = tuple([])
        if self.is_dim_aggregated is None:
            self.is_dim_aggregated = [False] * len(self.indices)

    def is_aggregate(self) -> bool:
        return any(self.is_dim_aggregated)

    def get_dim(self) -> int:
        return len(self.indices)

    def __str__(self):
        literal = self.symbol
        if len(self.indices) > 0:
            index_literals = []
            for j, idx in enumerate(self.indices):
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


class ASet(Entity):

    def __init__(self,
                 symbol: str,
                 indices: Tuple[Union[int, float, str], ...] = None,
                 is_dim_aggregated: List[bool] = None,
                 dim: int = 0,
                 elements: IndexSet = None):
        super(ASet, self).__init__(symbol=symbol,
                                   indices=indices,
                                   is_dim_aggregated=is_dim_aggregated)
        self.dim: int = dim
        self.elements: IndexSet = elements

    def __eq__(self, other):
        if isinstance(other, ASet) and self.entity_id == other.entity_id:
            return True
        return False

    def __len__(self):
        return len(self.elements)


class Parameter(Entity):

    def __init__(self,
                 symbol: str,
                 indices: Tuple[Union[int, float, str], ...] = None,
                 is_dim_aggregated: List[bool] = None,
                 value: float = 0):
        super(Parameter, self).__init__(symbol=symbol,
                                        indices=indices,
                                        is_dim_aggregated=is_dim_aggregated,
                                        value=value)

    def __eq__(self, other):
        if isinstance(other, Parameter) and self.entity_id == other.entity_id:
            return True
        return False


class Variable(Entity):

    def __init__(self,
                 symbol: str,
                 indices: Tuple[Union[int, float, str], ...],
                 is_dim_aggregated: List[bool] = None,
                 value: float = 0,
                 lb: float = 0,
                 ub: float = 0):
        super(Variable, self).__init__(symbol=symbol,
                                       indices=indices,
                                       is_dim_aggregated=is_dim_aggregated,
                                       value=value,
                                       lb=lb,
                                       ub=ub)

    def __eq__(self, other):
        if isinstance(other, Variable) and self.entity_id == other.entity_id:
            return True
        return False


class Objective(Entity):

    def __init__(self,
                 symbol: str,
                 indices: Tuple[Union[int, float, str], ...],
                 is_dim_aggregated: List[bool] = None,
                 value: float = 0):
        super(Objective, self).__init__(symbol=symbol,
                                        indices=indices,
                                        is_dim_aggregated=is_dim_aggregated,
                                        value=value)

    def __eq__(self, other):
        if isinstance(other, Objective) and self.entity_id == other.entity_id:
            return True
        return False


class Constraint(Entity):

    def __init__(self,
                 symbol: str,
                 indices: Tuple[Union[int, float, str], ...],
                 is_dim_aggregated: List[bool] = None,
                 value: float = 0,
                 lb: float = 0,
                 ub: float = 0):
        super(Constraint, self).__init__(symbol=symbol,
                                         indices=indices,
                                         is_dim_aggregated=is_dim_aggregated,
                                         value=value,
                                         lb=lb,
                                         ub=ub)

    def __eq__(self, other):
        if isinstance(other, Constraint) and self.entity_id == other.entity_id:
            return True
        return False


# Entity Collection
# ----------------------------------------------------------------------------------------------------------------------

class EntityCollection:

    def __init__(self, symbol):
        self.symbol: str = symbol  # name of entity collection
        # Key: index of entity instance; Value: indexed entity.
        self.entity_map: Dict[Tuple[Union[int, float, str], ...], Entity] = {}

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
