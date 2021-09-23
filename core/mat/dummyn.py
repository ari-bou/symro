from abc import ABC, abstractmethod
from functools import partial
import numpy as np
from typing import Callable, List, Tuple, Union

from symro.core.mat.util import IndexingSet, Element
from symro.core.mat.state import State
from symro.core.mat.exprn import ExpressionNode, ArithmeticExpressionNode, StringExpressionNode


class BaseDummyNode(ExpressionNode, ABC):

    def __init__(self, id: int = 0):
        super().__init__(id)

    def is_constant(self) -> bool:
        return True

    def is_null(self) -> bool:
        return False

    @staticmethod
    @abstractmethod
    def get_dim() -> int:
        pass

    @abstractmethod
    def get_unbound_symbols(self) -> List[str]:
        pass

    def get_children(self) -> list:
        return []

    def set_children(self, children: list):
        pass


class DummyNode(BaseDummyNode):

    def __init__(self, symbol: str, id: int = 0):
        super().__init__(id)
        self.symbol: str = symbol

    def evaluate(self,
                 state: State,
                 idx_set: IndexingSet = None,
                 dummy_element: Element = None
                 ) -> np.ndarray:

        if idx_set is None:
            return np.array([self.symbol])

        else:
            mp = len(idx_set)
            y = np.ndarray(shape=(mp,), dtype=object)
            for ip in range(mp):
                y[ip] = self.__control_dummy(idx_set[ip], dummy_element)
            return y

    def to_lambda(self,
                  state: State,
                  idx_set_member: Element = None,
                  dummy_element: Element = None) -> Callable:
        dummy = self.__control_dummy(idx_set_member, dummy_element)
        return partial(lambda d: d, dummy)

    def __control_dummy(self,
                        idx_set_member: Element,
                        dummy_symbols: Element) -> Union[int, float, str]:
        if self.symbol in dummy_symbols:
            pos = dummy_symbols.index(self.symbol)
            return idx_set_member[pos]
        else:
            return self.symbol

    def is_constant(self) -> bool:
        return True

    def is_null(self) -> bool:
        return False

    def is_controlled(self, dummy_element: List[str] = None) -> bool:
        if dummy_element is None:
            return True
        else:
            return self.symbol in dummy_element

    @staticmethod
    def get_dim() -> int:
        return 1

    def get_unbound_symbols(self) -> List[str]:
        return [self.symbol]

    def get_children(self) -> list:
        return []

    def set_children(self, children: list):
        pass

    def get_literal(self) -> str:
        if not self.is_prioritized:
            return self.symbol
        else:
            return "({0})".format(self.symbol)


class CompoundDummyNode(BaseDummyNode):

    def __init__(self,
                 component_nodes: List[Union[DummyNode,
                                             StringExpressionNode,
                                             ArithmeticExpressionNode]],
                 id: int = 0):
        super().__init__(id)
        self.component_nodes: List[Union[DummyNode,
                                         StringExpressionNode,
                                         ArithmeticExpressionNode]] = component_nodes

    def evaluate(self,
                 state: State,
                 idx_set: IndexingSet = None,
                 dummy_element: Element = None
                 ) -> np.ndarray:

        component_sub_elements = [c.evaluate(state, idx_set, dummy_element) for c in self.component_nodes]

        mp = 1
        if idx_set is not None:
            mp = len(idx_set)

        y = np.ndarray(shape=(mp,), dtype=object)
        for ip in range(mp):
            element_ic = tuple([component[ip] for component in component_sub_elements])
            y[ip] = element_ic

        return y

    def to_lambda(self,
                  state: State,
                  idx_set_member: Element = None,
                  dummy_element: Element = None):
        x = tuple([c.to_lambda(state, idx_set_member, dummy_element) for c in self.component_nodes])
        return partial(lambda d: tuple([x_i() for x_i in x]), x)

    def is_constant(self) -> bool:
        return True

    def is_null(self) -> bool:
        return False

    def is_controlled(self, dummy_element: List[str] = None) -> bool:
        return any([c.is_controlled(dummy_element) for c in self.component_nodes])

    def get_dim(self) -> int:
        return len(self.component_nodes)

    def get_unbound_symbols(self) -> List[str]:
        syms = []
        for component_node in self.component_nodes:
            if isinstance(component_node, DummyNode):
                syms.append(component_node.symbol)
        return syms

    def get_children(self) -> list:
        return list(self.component_nodes)

    def set_children(self, children: list):
        self.component_nodes = list(children)

    def get_literal(self) -> str:
        return "({0})".format(','.join([str(n) for n in self.component_nodes]))
