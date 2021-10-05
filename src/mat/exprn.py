from abc import ABC, abstractmethod
from copy import deepcopy
import numpy as np
from ordered_set import OrderedSet
from typing import Callable, Dict, List, Optional, Tuple, Union

from symro.src.mat.util import Element, IndexingSet
from symro.src.mat.entity import Parameter, Variable
from symro.src.mat.state import State


# Expression Node
# ----------------------------------------------------------------------------------------------------------------------

class ExpressionNode(ABC):

    def __init__(self):
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
                 dummy_element: Element = None) -> np.ndarray:
        pass

    def collect_declared_entities(self,
                                  state: State,
                                  idx_set: IndexingSet = None,
                                  dummy_element: Element = None) -> Dict[str, Union[Parameter, Variable]]:
        return {}

    def to_lambda(self,
                  state: State,
                  idx_set_member: Element = None,
                  dummy_element: Tuple[str, ...] = None) -> Callable:
        """
        Generate an anonymous function that evaluates the expression at a unique static indexing set member.
        :param state: object containing the state of the problem
        :param idx_set_member: member of the indexing set for which the anonymous function will be generated; 
        None if scalar
        :param dummy_element: the unbound symbols of the indexing set of an indexed expression; None if scalar
        :return: callable
        """
        pass

    def is_constant(self) -> bool:
        are_operands_const = [o.is_constant() for o in self.get_children()]
        return all(are_operands_const)

    def is_controlled(self, dummy_element: List[str] = None) -> bool:
        """
        Returns True if the node contains a dummy node whose symbol is in the dummy_syms argument. If dummy_syms is
        None, then returns True if the node contains a dummy node with any symbol.
        :param dummy_element: list of dummy symbols
        :return: bool
        """
        return any([o.is_controlled(dummy_element) for o in self.get_children()])

    def get_children(self) -> List["ExpressionNode"]:
        return []

    def set_children(self, operands: List["ExpressionNode"]):
        pass

    @abstractmethod
    def get_literal(self) -> str:
        return ""


# Fundamental Expression Nodes
# ----------------------------------------------------------------------------------------------------------------------

class LogicalExpressionNode(ExpressionNode, ABC):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def __invert__(self):
        pass

    @abstractmethod
    def __and__(self, other: "LogicalExpressionNode"):
        pass

    @abstractmethod
    def __or__(self, other: "LogicalExpressionNode"):
        pass

    def evaluate(self,
                 state: State,
                 idx_set: IndexingSet = None,
                 dummy_element: Element = None) -> np.ndarray:
        return np.array([False])


class SetExpressionNode(ExpressionNode, ABC):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def __and__(self, other: "SetExpressionNode"):
        pass

    @abstractmethod
    def __or__(self, other: "SetExpressionNode"):
        pass

    @abstractmethod
    def __sub__(self, other: "SetExpressionNode"):
        pass

    @abstractmethod
    def __xor__(self, other: "SetExpressionNode"):
        pass

    @abstractmethod
    def get_dim(self, state: State) -> int:
        pass

    def get_dummy_component_nodes(self, state: State) -> list:
        dummy_nodes = [None] * self.get_dim(state)
        return dummy_nodes

    def get_dummy_element(self, state: State) -> Element:
        dummy_syms = [None] * self.get_dim(state)
        return tuple(dummy_syms)

    def evaluate(self,
                 state: State,
                 idx_set: IndexingSet = None,
                 dummy_element: Element = None
                 ) -> np.ndarray:
        return np.array([OrderedSet()])


class ArithmeticExpressionNode(ExpressionNode, ABC):

    def __init__(self):
        super().__init__()

    def __pos__(self):
        return self

    @abstractmethod
    def __neg__(self):
        pass

    @abstractmethod
    def __add__(self, other: "ArithmeticExpressionNode"):
        pass

    @abstractmethod
    def __sub__(self, other: "ArithmeticExpressionNode"):
        pass

    @abstractmethod
    def __mul__(self, other: "ArithmeticExpressionNode"):
        pass

    @abstractmethod
    def __truediv__(self, other: "ArithmeticExpressionNode"):
        pass

    @abstractmethod
    def __pow__(self, power: "ArithmeticExpressionNode", modulo=None):
        pass

    @abstractmethod
    def __eq__(self, other: "ArithmeticExpressionNode"):
        pass

    @abstractmethod
    def __ne__(self, other: "ArithmeticExpressionNode"):
        pass

    @abstractmethod
    def __lt__(self, other: "ArithmeticExpressionNode"):
        pass

    @abstractmethod
    def __le__(self, other: "ArithmeticExpressionNode"):
        pass

    @abstractmethod
    def __gt__(self, other: "ArithmeticExpressionNode"):
        pass

    @abstractmethod
    def __ge__(self, other: "ArithmeticExpressionNode"):
        pass

    def collect_declared_entities(self,
                                  state: State,
                                  idx_set: IndexingSet = None,
                                  dummy_element: Element = None) -> Dict[str, Union[Parameter, Variable]]:
        return {}

    def evaluate(self,
                 state: State,
                 idx_set: IndexingSet = None,
                 dummy_element: Element = None) -> np.ndarray:
        return np.array([0])


class StringExpressionNode(ExpressionNode, ABC):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def __and__(self, other: "StringExpressionNode"):
        pass

    @abstractmethod
    def __eq__(self, other: "StringExpressionNode"):
        pass

    @abstractmethod
    def __ne__(self, other: "StringExpressionNode"):
        pass

    @abstractmethod
    def __lt__(self, other: "StringExpressionNode"):
        pass

    @abstractmethod
    def __le__(self, other: "StringExpressionNode"):
        pass

    @abstractmethod
    def __gt__(self, other: "StringExpressionNode"):
        pass

    @abstractmethod
    def __ge__(self, other: "StringExpressionNode"):
        pass

    def evaluate(self,
                 state: State,
                 idx_set: IndexingSet = None,
                 dummy_element: Element = None) -> np.ndarray:
        return np.array([""])
