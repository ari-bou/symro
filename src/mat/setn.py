from abc import ABC, abstractmethod

import numpy as np
from ordered_set import OrderedSet
from typing import List, Optional, Set, Tuple, Union

from symro.src.mat.util import Element, IndexingSet, cartesian_product, remove_set_dimensions
from symro.src.mat.exprn import ExpressionNode, LogicalExpressionNode, SetExpressionNode, ArithmeticExpressionNode, \
    StringExpressionNode
from symro.src.mat.opern import SetOperationNode
from symro.src.mat.dummyn import BaseDummyNode, DummyNode, CompoundDummyNode
from symro.src.mat.entity import Entity
from symro.src.mat.state import State


class BaseSetNode(SetExpressionNode, ABC):

    def __init__(self):
        super().__init__()

    def __and__(self, other: SetExpressionNode):
        return SetOperationNode.intersection(self, other)

    def __or__(self, other: SetExpressionNode):
        return SetOperationNode.union(self, other)

    def __sub__(self, other: SetExpressionNode):
        return SetOperationNode.difference(self, other)

    def __xor__(self, other: SetExpressionNode):
        return SetOperationNode.symmetric_difference(self, other)

    @abstractmethod
    def get_dim(self, state: State) -> int:
        pass

    def is_constant(self) -> bool:
        return True

    def is_null(self) -> bool:
        return False

    def get_children(self) -> list:
        return []

    def set_children(self, operands: list):
        pass


class SetNode(BaseSetNode):

    def __init__(self,
                 symbol: str,
                 entity_index_node: CompoundDummyNode = None,
                 suffix: str = None):
        super().__init__()
        self.symbol: str = symbol
        self.entity_index_node: CompoundDummyNode = entity_index_node
        self.suffix: str = suffix

    def evaluate(self,
                 state: State,
                 idx_set: IndexingSet = None,
                 dummy_element: Element = None
                 ) -> np.ndarray:

        elements = OrderedSet()
        if self.symbol in state.sets:
            a_set = state.sets[self.symbol]
            for element in a_set.elements:
                elements.add(element)

        mp = 1
        if idx_set is not None:
            mp = len(idx_set)

        y = np.ndarray(shape=(mp,), dtype=object)
        for ip in range(mp):
            y[ip] = elements

        return y

    def is_indexed(self) -> bool:
        return self.entity_index_node is not None

    def is_constant(self) -> bool:
        return True

    def is_null(self) -> bool:
        return False

    def get_entity_id(self, state: State) -> str:
        indices = None
        if self.is_indexed():
            indices = self.entity_index_node.evaluate(state)[0]
            if not isinstance(indices, tuple):
                indices = [indices]
        return Entity.generate_entity_id(self.symbol, indices)

    def get_dim(self, state: State) -> int:
        return state.sets[self.get_entity_id(state)].dim

    def get_children(self) -> list:
        return []

    def set_children(self, operands: list):
        pass

    def get_literal(self) -> str:
        literal = self.symbol
        if self.is_indexed():
            literal += "[{0}]".format(','.join([str(n) for n in self.entity_index_node.component_nodes]))
        if self.suffix is not None:
            literal += ".{0}".format(self.suffix)
        if self.is_prioritized:
            literal = '(' + literal + ')'
        return literal


class OrderedSetNode(BaseSetNode):

    def __init__(self,
                 start_node: ArithmeticExpressionNode = None,
                 end_node: ArithmeticExpressionNode = None):
        super().__init__()
        self.start_node: ArithmeticExpressionNode = start_node
        self.end_node: ArithmeticExpressionNode = end_node

    def evaluate(self,
                 state: State,
                 idx_set: IndexingSet = None,
                 dummy_element: Element = None
                 ) -> np.ndarray:

        start_elements = self.start_node.evaluate(state, idx_set, dummy_element)
        end_elements = self.end_node.evaluate(state, idx_set, dummy_element)

        mp = 1
        if idx_set is not None:
            mp = len(idx_set)

        y = np.ndarray(shape=(mp,), dtype=object)

        for ip in range(mp):
            x_start = start_elements[ip]
            x_end = end_elements[ip]
            x = list(range(x_start, x_end + 1))
            elements = OrderedSet([(x_i,) for x_i in x])
            y[ip] = elements

        return y

    def get_dim(self, state: State) -> int:
        return 1

    def get_children(self) -> List[ArithmeticExpressionNode]:
        return [self.start_node, self.end_node]

    def set_children(self, operands: list):
        if len(operands) > 0:
            self.start_node = operands[0]
            if len(operands) > 1:
                self.end_node = operands[1]

    def get_literal(self) -> str:
        literal = "{0}..{1}".format(self.start_node, self.end_node)
        if self.is_prioritized:
            literal = '(' + literal + ')'
        return literal


class EnumeratedSet(BaseSetNode):

    def __init__(self,
                 element_nodes: List[Union[ArithmeticExpressionNode,
                                           StringExpressionNode]] = None):

        super().__init__()

        self.element_nodes: List[Union[ArithmeticExpressionNode, StringExpressionNode]] = element_nodes

        if self.element_nodes is None:
            self.element_nodes = []

    def evaluate(self,
                 state: State,
                 idx_set: IndexingSet = None,
                 dummy_element: Element = None
                 ) -> np.ndarray:

        _np = len(self.element_nodes)  # number of elements in the set

        elements = [e.evaluate(state, idx_set, dummy_element) for e in self.element_nodes]

        mp = 1  # number of elements in the parent set
        if idx_set is not None:
            mp = len(idx_set)

        y = np.ndarray(shape=(mp,), dtype=object)

        for ip in range(mp):

            elements_ip = OrderedSet()

            for jp in range(_np):
                e = elements[jp][ip]
                if not isinstance(e, tuple):
                    e = tuple([e])  # each element must be a tuple
                elements_ip.add(e)

            y[ip] = elements_ip

        return y

    def get_dim(self, state: State) -> int:
        if len(self.element_nodes) > 0:
            element = self.element_nodes[0]
            if isinstance(element, CompoundDummyNode):
                return element.get_dim()
            else:
                return 1
        else:
            return 0

    def get_children(self) -> List[Union[ArithmeticExpressionNode, StringExpressionNode]]:
        return list(self.element_nodes)

    def set_children(self, operands: list):
        self.element_nodes = list(operands)

    def get_literal(self) -> str:
        literal = '{' + "{0}".format(','.join([e.get_literal() for e in self.element_nodes])) + '}'
        if self.is_prioritized:
            literal = '(' + literal + ')'
        return literal


class IndexingSetNode(SetExpressionNode):

    def __init__(self,
                 dummy_node: BaseDummyNode,
                 set_node: SetExpressionNode):
        super().__init__()
        self.dummy_node: BaseDummyNode = dummy_node
        self.set_node: SetExpressionNode = set_node

    def __and__(self, other: SetExpressionNode):
        return SetOperationNode.intersection(self, other)

    def __or__(self, other: SetExpressionNode):
        return SetOperationNode.union(self, other)

    def __sub__(self, other: SetExpressionNode):
        return SetOperationNode.difference(self, other)

    def __xor__(self, other: SetExpressionNode):
        return SetOperationNode.symmetric_difference(self, other)

    def evaluate(self,
                 state: State,
                 idx_set: IndexingSet = None,
                 dummy_element: Element = None
                 ) -> np.ndarray:

        dim_c = self.dummy_node.get_dim()  # length nc

        challenge_elements = self.dummy_node.evaluate(state, idx_set, dummy_element)
        if self.dummy_node.get_dim() == 1:
            challenge_elements = [tuple([e]) for e in challenge_elements]

        sets_c = self.set_node.evaluate(state, idx_set, dummy_element)

        # identify fixed dimensions
        is_dim_fixed = []
        if self.dummy_node.get_dim() > 1:

            dummy_node = self.dummy_node
            if not isinstance(dummy_node, CompoundDummyNode):
                raise ValueError("Encountered an unexpected expression node"
                                 " while processing the dummy node of an indexed set")

            for jc, cmpt in enumerate(dummy_node.component_nodes):
                if isinstance(cmpt, DummyNode):  # component is a dummy node
                    if dummy_element is not None:
                        if cmpt.symbol in dummy_element:
                            is_dim_fixed.append(True)  # dummy is controlled
                        else:
                            is_dim_fixed.append(False)  # dummy is not controlled
                    else:
                        is_dim_fixed.append(False)  # dummy is not controlled
                else:
                    is_dim_fixed.append(True)  # component is a numeric constant or a string node

        mp = 1
        if idx_set is not None:
            mp = len(idx_set)

        y = np.ndarray(shape=(mp,), dtype=object)

        for ip in range(mp):

            set_c_ip = sets_c[ip]

            if self.dummy_node.get_dim() == 1:
                y[ip] = set_c_ip

            else:

                filtered_set = OrderedSet()
                challenge_element_ip = challenge_elements[ip]

                for element_c_ip_ic in set_c_ip:
                    is_member = True
                    for jc in range(dim_c):
                        if is_dim_fixed[jc] and challenge_element_ip[jc] != element_c_ip_ic[jc]:
                            is_member = False
                            break
                    if is_member:
                        filtered_set.add(element_c_ip_ic)

                y[ip] = filtered_set

        return y

    def get_dim(self, state: State) -> int:
        return self.dummy_node.get_dim()

    def get_dummy_element(self, state: State) -> Tuple[Union[int, float, str, tuple], ...]:
        if isinstance(self.dummy_node, DummyNode):
            return tuple([self.dummy_node.symbol])
        elif isinstance(self.dummy_node, CompoundDummyNode):
            dummy_syms = []
            for component_dummy_node in self.dummy_node.component_nodes:
                dummy_syms.append(component_dummy_node.get_literal())
            return tuple(dummy_syms)

    def get_dummy_component_nodes(self, state: State) -> list:
        if isinstance(self.dummy_node, DummyNode):
            return [self.dummy_node]
        elif isinstance(self.dummy_node, CompoundDummyNode):
            dummy_nodes = []
            for component_dummy_node in self.dummy_node.component_nodes:
                dummy_nodes.append(component_dummy_node)
            return dummy_nodes

    def get_children(self) -> list:
        return [self.dummy_node, self.set_node]

    def set_children(self, operands: list):
        if len(operands) > 0:
            self.dummy_node = operands[0]
            if len(operands) > 1:
                self.set_node = operands[1]

    def get_literal(self) -> str:
        literal = "{0} in {1}".format(self.dummy_node, self.set_node)
        if self.is_prioritized:
            literal = '(' + literal + ')'
        return literal


class CompoundSetNode(BaseSetNode):

    def __init__(self,
                 set_nodes: List[SetExpressionNode],
                 constraint_node: LogicalExpressionNode = None):
        super().__init__()
        self.set_nodes: List[SetExpressionNode] = set_nodes if set_nodes is not None else []  # length o
        self.constraint_node: LogicalExpressionNode = constraint_node
        self.combined_dummy_element: Optional[Tuple[Union[int, float, str, tuple, None], ...]] = None

    def evaluate(self,
                 state: State,
                 idx_set: IndexingSet = None,
                 dummy_element: Element = None
                 ) -> np.ndarray:

        combined_sets = self.combine_indexing_and_component_sets(state, idx_set, dummy_element)

        mp = 1
        if idx_set is not None:
            mp = len(idx_set)

        y = np.ndarray(shape=(mp,), dtype=object)

        for ip in range(mp):

            combined_set_ip = combined_sets[ip]

            if self.constraint_node is not None:
                filtered_set = self.__filter_set(state, combined_set_ip)
                if idx_set is not None:
                    filtered_set = remove_set_dimensions(filtered_set, list(range(len(dummy_element))))
                y[ip] = filtered_set

            else:
                if idx_set is not None:
                    combined_set_ip = remove_set_dimensions(combined_set_ip, list(range(len(dummy_element))))
                y[ip] = combined_set_ip

        return y

    def combine_indexing_and_component_sets(self,
                                            state: State,
                                            idx_set: IndexingSet = None,  # length mp
                                            dummy_element: Element = None):
        """
        Combine the indexing sets and the component sets together.
        Note that the indexing set constraint is not applied.
        :param state: State
        :param idx_set: IndexSet of length mp
        :param dummy_element: tuple of unbound symbols of length np
        :return: combined indexing sets for each index of the outer scope
        """

        mp = 1
        if idx_set is not None:
            mp = len(idx_set)

        # Combine dummy sub-elements from the indexing set and each component set
        combined_dummy_sub_elements = None if dummy_element is None else list(dummy_element)
        for set_node in self.set_nodes:
            component_dummy_syms = list(set_node.get_dummy_element(state))
            if combined_dummy_sub_elements is None:
                combined_dummy_sub_elements = component_dummy_syms
            else:
                combined_dummy_sub_elements = combined_dummy_sub_elements + component_dummy_syms
        self.combined_dummy_element = tuple(combined_dummy_sub_elements)

        combine_idx_sets = np.ndarray(shape=(mp,), dtype=object)  # length mp
        for ip in range(mp):

            idx_set_ip = None if idx_set is None else OrderedSet(idx_set[[ip]])
            sub_combined_unb_syms = dummy_element

            # Retrieve and combine indexing sets for each component set
            for set_node in self.set_nodes:

                component_idx_sets_ip = set_node.evaluate(state, idx_set_ip, sub_combined_unb_syms)
                component_idx_set_ip = OrderedSet().union(*component_idx_sets_ip)

                if idx_set_ip is None:
                    idx_set_ip = component_idx_set_ip
                    sub_combined_unb_syms = set_node.get_dummy_element(state)
                else:
                    idx_set_ip = cartesian_product([idx_set_ip, component_idx_set_ip])
                    sub_combined_unb_syms += set_node.get_dummy_element(state)

            combine_idx_sets[ip] = idx_set_ip

        return combine_idx_sets

    def __filter_set(self,
                     state: State,
                     combined_set_ip: IndexingSet):
        in_set = self.constraint_node.evaluate(state,
                                               combined_set_ip,
                                               self.combined_dummy_element)
        filtered_set = OrderedSet()
        for element, b in zip(combined_set_ip, in_set):
            if b:
                filtered_set.add(element)
        return filtered_set

    def get_dim(self, state: State) -> int:
        return sum([set_node.get_dim(state) for set_node in self.set_nodes])

    def get_dummy_component_nodes(self, state: State) -> List[Union[DummyNode,
                                                                    ArithmeticExpressionNode,
                                                                    StringExpressionNode]]:
        dummy_nodes = []
        for set_node in self.set_nodes:
            dummy_nodes.extend(set_node.get_dummy_component_nodes(state))
        return dummy_nodes

    def get_dummy_element(self, state: State) -> Element:
        dummy_syms = []
        for set_node in self.set_nodes:
            dummy_syms.extend(set_node.get_dummy_element(state))
        return tuple(dummy_syms)

    def get_defined_unbound_symbols(self, outer_unb_syms: Set[str] = None) -> OrderedSet[str]:
        """
        Retrieve ordered set of unbound symbols defined by the compound set node. The order corresponds to the order in
        which the unbound symbols are defined.
        :param outer_unb_syms: set of unbound symbols defined in the outer scope (exclusive filter)
        :return: ordered set of defined unbound symbols
        """

        unb_syms = OrderedSet()

        if outer_unb_syms is None:
            outer_unb_syms = set()

        for cmpt_set_node in self.set_nodes:

            if isinstance(cmpt_set_node, IndexingSetNode):
                dummy_node = cmpt_set_node.dummy_node

                if isinstance(dummy_node, DummyNode):
                    if dummy_node.symbol not in outer_unb_syms:
                        unb_syms.add(dummy_node.symbol)

                elif isinstance(dummy_node, CompoundDummyNode):
                    for cmpt_node in dummy_node.component_nodes:
                        if cmpt_node.symbol not in outer_unb_syms:
                            unb_syms.add(cmpt_node.symbol)

        return unb_syms

    def get_children(self) -> List[ExpressionNode]:
        children = []
        children.extend(self.set_nodes)
        if self.constraint_node is not None:
            children.append(self.constraint_node)
        return children

    def set_children(self, operands: List[ExpressionNode]):
        self.set_nodes.clear()
        for child in operands:
            if isinstance(child, SetExpressionNode):
                self.set_nodes.append(child)
            else:
                self.constraint_node = child

    def get_literal(self) -> str:
        definition = ", ".join([n.get_literal() for n in self.set_nodes])
        if self.constraint_node is not None:
            definition += ": {0}".format(self.constraint_node)
        literal = '{' + definition + '}'
        if self.is_prioritized:
            literal = '(' + literal + ')'
        return literal