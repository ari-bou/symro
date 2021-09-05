from abc import ABC, abstractmethod
from functools import partial
from typing import List, Tuple, Union

from symro.core.mat.util import IndexSet, IndexSetMember
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

    def __init__(self, dummy: str, id: int = 0):
        super().__init__(id)
        self.dummy: str = dummy

    def evaluate(self,
                 state: State,
                 idx_set: IndexSet = None,
                 dummy_symbols: Tuple[str, ...] = None
                 ) -> List[Union[int, float, str]]:

        if idx_set is None:
            return [self.dummy]

        else:
            results = []
            for element_ip in idx_set:
                results.append(self.__control_dummy(element_ip, dummy_symbols))
            return results

    def to_lambda(self,
                  state: State,
                  idx_set_member: IndexSetMember = None,
                  dummy_symbols: Tuple[str, ...] = None):
        dummy = self.__control_dummy(idx_set_member, dummy_symbols)
        return partial(lambda d: d, dummy)

    def __control_dummy(self,
                        idx_set_member: IndexSetMember,
                        dummy_symbols: Tuple[str, ...]) -> Union[int, float, str]:
        if self.dummy in dummy_symbols:
            pos = dummy_symbols.index(self.dummy)
            return idx_set_member[pos]
        else:
            return self.dummy

    def is_constant(self) -> bool:
        return True

    def is_null(self) -> bool:
        return False

    def is_controlled(self, dummy_syms: List[str] = None) -> bool:
        if dummy_syms is None:
            return True
        else:
            return self.dummy in dummy_syms

    @staticmethod
    def get_dim() -> int:
        return 1

    def get_unbound_symbols(self) -> List[str]:
        return [self.dummy]

    def get_children(self) -> list:
        return []

    def set_children(self, children: list):
        pass

    def get_literal(self) -> str:
        if not self.is_prioritized:
            return self.dummy
        else:
            return "({0})".format(self.dummy)


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
                 idx_set: IndexSet = None,
                 dummy_symbols: Tuple[str, ...] = None
                 ) -> List[Tuple[Union[int, float, str], ...]]:

        component_sub_elements = [c.evaluate(state, idx_set, dummy_symbols) for c in self.component_nodes]

        count_ip = 1
        if idx_set is not None:
            count_ip = len(idx_set)

        results = []
        for ip in range(count_ip):
            element_ic = tuple([component[ip] for component in component_sub_elements])
            results.append(element_ic)

        return results

    def to_lambda(self,
                  state: State,
                  idx_set_member: IndexSetMember = None,
                  dummy_symbols: Tuple[str, ...] = None):
        dummy = tuple([c.to_lambda(state, idx_set_member, dummy_symbols)() for c in self.component_nodes])
        return partial(lambda d: d, dummy)

    def is_constant(self) -> bool:
        return True

    def is_null(self) -> bool:
        return False

    def is_controlled(self, dummy_syms: List[str] = None) -> bool:
        return any([c.is_controlled(dummy_syms) for c in self.component_nodes])

    def get_dim(self) -> int:
        return len(self.component_nodes)

    def get_unbound_symbols(self) -> List[str]:
        syms = []
        for component_node in self.component_nodes:
            if isinstance(component_node, DummyNode):
                syms.append(component_node.dummy)
        return syms

    def get_children(self) -> list:
        return list(self.component_nodes)

    def set_children(self, children: list):
        self.component_nodes = list(children)

    def get_literal(self) -> str:
        return "({0})".format(','.join([str(n) for n in self.component_nodes]))
