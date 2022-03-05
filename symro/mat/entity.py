from abc import ABC
from typing import List, Optional, Union

from .types import IndexingSet, Element
from .orderedset import OrderedSet


class Entity(ABC):
    def __init__(
        self, symbol: str, idx: Element = None, is_dim_aggregated: List[bool] = None
    ):
        """
        Constructor of the Entity class.

        :param symbol: unique declared symbol that identifies the entity
        :param idx: unique indexing set element that identifies the entity if indexed, None otherwise
        :param is_dim_aggregated: list of Boolean flags that indicates which dimensions are aggregated
        """

        self.symbol: str = symbol
        self.idx: Optional[Element] = idx  # length: dim
        self.entity_id: tuple = self.generate_entity_id(self.symbol, self.idx)
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
            literal += "[{0}]".format(",".join(index_literals))
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

    def get_value(self):
        raise TypeError(
            "Entity type '{0}' does not possess a value property".format(type(self))
        )

    def get_lb(self):
        raise TypeError(
            "Entity type '{0}' does not possess a lower bound property".format(
                type(self)
            )
        )

    def get_ub(self):
        raise TypeError(
            "Entity type '{0}' does not possess an upper bound property".format(
                type(self)
            )
        )

    def get_body(self):
        raise TypeError(
            "Entity type '{0}' does not possess a body property".format(type(self))
        )

    @staticmethod
    def generate_entity_id(symbol: str, idx: Element = None) -> tuple:
        if idx is None:
            return tuple([symbol])
        else:
            return tuple([symbol]) + idx


class SSet(Entity):
    def __init__(
        self,
        symbol: str,
        idx: Element = None,
        is_dim_aggregated: List[bool] = None,
        dim: int = 0,
        elements: IndexingSet = None,
    ):
        super(SSet, self).__init__(
            symbol=symbol, idx=idx, is_dim_aggregated=is_dim_aggregated
        )
        self.dim: int = dim
        self.elements: IndexingSet = elements

        if self.elements is None:
            self.elements = OrderedSet()

    def __len__(self):
        return len(self.elements)

    def __eq__(self, other):
        if isinstance(other, SSet):
            return self.elements == other.elements
        else:
            return False

    def __getitem__(self, item: int):
        return self.elements[item]

    def get_value(self):
        return self.elements


class Parameter(Entity):
    def __init__(
        self,
        symbol: str,
        idx: Element = None,
        is_dim_aggregated: List[bool] = None,
        value: float = 0,
    ):
        super(Parameter, self).__init__(
            symbol=symbol, idx=idx, is_dim_aggregated=is_dim_aggregated
        )
        self.value: Union[float, str] = value

    def __eq__(self, other):
        if isinstance(other, Parameter):
            return self.value == other.value
        else:
            return False

    def get_value(self):
        return self.value


class Variable(Entity):
    def __init__(
        self,
        symbol: str,
        idx: Element = None,
        is_dim_aggregated: List[bool] = None,
        value: float = 0,
        lb: float = 0,
        ub: float = 0,
    ):
        super(Variable, self).__init__(
            symbol=symbol, idx=idx, is_dim_aggregated=is_dim_aggregated
        )
        self.value: Union[float, str] = value
        self.lb: float = float(lb)
        self.ub: float = float(ub)

    def __eq__(self, other):
        if isinstance(other, Variable):
            return self.value == other.value
        else:
            return False

    def get_value(self):
        return self.value

    def get_lb(self):
        return self.lb

    def get_ub(self):
        return self.ub


class Objective(Entity):
    def __init__(
        self,
        symbol: str,
        idx: Element = None,
        is_dim_aggregated: List[bool] = None,
        value: float = 0,
    ):
        super(Objective, self).__init__(
            symbol=symbol, idx=idx, is_dim_aggregated=is_dim_aggregated
        )
        self.value: Union[float, str] = value

    def __eq__(self, other):
        if isinstance(other, Objective):
            return self.value == other.value
        else:
            return False

    def get_value(self):
        return self.value


class Constraint(Entity):
    def __init__(
        self,
        symbol: str,
        idx: Element = None,
        is_dim_aggregated: List[bool] = None,
        value: float = 0,
        lb: float = 0,
        ub: float = 0,
        dual: float = 0,
    ):
        super(Constraint, self).__init__(
            symbol=symbol, idx=idx, is_dim_aggregated=is_dim_aggregated
        )
        self.body: Union[float, str] = value  # body
        self.lb: float = float(lb)
        self.ub: float = float(ub)
        self.dual: float = dual

    def __eq__(self, other):
        if isinstance(other, Constraint):
            return self.dual == other.dual
        else:
            return False

    def get_value(self):
        return self.dual

    def get_lb(self):
        return self.lb

    def get_ub(self):
        return self.ub

    def get_body(self):
        return self.body
