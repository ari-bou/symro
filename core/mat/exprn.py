from abc import ABC, abstractmethod
from copy import deepcopy
from ordered_set import OrderedSet
from typing import Dict, List, Optional, Tuple, Union

from symro.core.mat.entity import Parameter, Variable
from symro.core.mat.util import IndexingSet, Element
from symro.core.mat.state import State


# Expression Node
# ----------------------------------------------------------------------------------------------------------------------

class ExpressionNode(ABC):

    def __init__(self, id: int = 0):
        self.id: int = id
        self.parent: Optional[ExpressionNode] = None
        self.is_prioritized: bool = False

    def __str__(self):
        return self.get_literal()

    def __deepcopy__(self, memo):
        cls = self.__class__
        clone = cls.__new__(cls)
        memo[id(self)] = clone
        clone.parent = None
        for k, v in self.__dict__.items():
            if k != "parent":
                setattr(clone, k, deepcopy(v, memo))
        return clone

    @abstractmethod
    def evaluate(self,
                 state: State,
                 idx_set: IndexingSet = None,
                 dummy_symbols: Tuple[str, ...] = None):
        pass

    def collect_declared_entities(self,
                                  state: State,
                                  idx_set: IndexingSet = None,
                                  dummy_symbols: Tuple[str, ...] = None) -> Dict[str, Union[Parameter, Variable]]:
        return {}

    def to_lambda(self,
                  state: State,
                  idx_set_member: Element = None,
                  dummy_symbols: Tuple[str, ...] = None):
        pass

    def is_constant(self) -> bool:
        are_operands_const = [o.is_constant() for o in self.get_children()]
        return all(are_operands_const)

    def is_null(self) -> bool:
        are_operands_null = [o.is_null() for o in self.get_children()]
        return all(are_operands_null)

    def is_controlled(self, dummy_syms: List[str] = None) -> bool:
        """
        Returns True if the node contains a dummy node whose symbol is in the dummy_syms argument. If dummy_syms is
        None, then returns True if the node contains a dummy node with any symbol.
        :param dummy_syms: list of dummy symbols
        :return: bool
        """
        return any([o.is_controlled(dummy_syms) for o in self.get_children()])

    def get_node(self, id: int):
        if self.id == id:
            return self
        for child in self.get_children():
            n = child.get_node(id)
            if n is not None:
                return n
        return None

    def get_free_id(self, id: int = 0):
        id = max(id, self.id + 1)
        for child in self.get_children():
            id = child.get_free_id(id)
        return id

    @abstractmethod
    def get_children(self) -> List["ExpressionNode"]:
        pass

    @abstractmethod
    def set_children(self, operands: List["ExpressionNode"]):
        pass

    @abstractmethod
    def get_literal(self) -> str:
        return ""


# Fundamental Expression Nodes
# ----------------------------------------------------------------------------------------------------------------------

class LogicalExpressionNode(ExpressionNode, ABC):

    def __init__(self, id: int = 0):
        super().__init__(id)

    def evaluate(self,
                 state: State,
                 idx_set: IndexingSet = None,
                 dummy_symbols: Tuple[str, ...] = None) -> List[bool]:
        return [False]


class SetExpressionNode(ExpressionNode, ABC):

    def __init__(self, id: int = 0):
        super().__init__(id)

    @abstractmethod
    def get_dim(self, state: State) -> int:
        pass

    def get_dummy_component_nodes(self, state: State) -> list:
        dummy_nodes = [None] * self.get_dim(state)
        return dummy_nodes

    def get_dummy_elements(self, state: State) -> Tuple[Union[int, float, str, tuple, None], ...]:
        dummy_syms = [None] * self.get_dim(state)
        return tuple(dummy_syms)

    def evaluate(self,
                 state: State,
                 idx_set: IndexingSet = None,
                 dummy_symbols: Tuple[str, ...] = None
                 ) -> List[IndexingSet]:
        return [OrderedSet()]


class ArithmeticExpressionNode(ExpressionNode, ABC):

    def __init__(self, id: int = 0):
        super().__init__(id)

    def collect_declared_entities(self,
                                  state: State,
                                  idx_set: IndexingSet = None,
                                  dummy_symbols: Tuple[str, ...] = None) -> Dict[str, Union[Parameter, Variable]]:
        return {}

    def evaluate(self,
                 state: State,
                 idx_set: IndexingSet = None,
                 dummy_symbols: Tuple[str, ...] = None) -> List[float]:
        return [0]


class StringExpressionNode(ExpressionNode, ABC):

    def __init__(self, id: int = 0):
        super().__init__(id)

    def evaluate(self,
                 state: State,
                 idx_set: IndexingSet = None,
                 dummy_symbols: Tuple[str, ...] = None) -> List[str]:
        return [""]
