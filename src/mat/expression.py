from queue import Queue
from typing import Optional

from symro.src.mat.util import *
from symro.src.mat.exprn import ExpressionNode
from symro.src.mat.setn import CompoundSetNode
from symro.src.mat.aexprn import DeclaredEntityNode, ArithmeticOperationNode, ArithmeticTransformationNode
from symro.src.mat.state import State


# Node Utility Functions
# ----------------------------------------------------------------------------------------------------------------------

def is_constant(root_node: ExpressionNode) -> bool:

    queue = Queue()
    queue.put(root_node)

    while not queue.empty():

        node: ExpressionNode = queue.get()

        if isinstance(node, DeclaredEntityNode) and not node.is_constant():
            return False

        for child in node.get_children():
            queue.put(child)

    return True


def is_linear(root_node: ExpressionNode) -> bool:

    queue = Queue()
    queue.put(root_node)

    while not queue.empty():

        node: ExpressionNode = queue.get()

        if isinstance(node, ArithmeticOperationNode) and node.operator == MULTIPLICATION_OPERATOR:  # multiplication
            is_const = [is_constant(child) for child in node.get_children()]
            if len(is_const) > 1:
                return False

        elif isinstance(node, ArithmeticOperationNode) and node.operator == DIVISION_OPERATOR:  # division
            if not is_constant(node.operands[1]):
                return False

        elif isinstance(node, ArithmeticOperationNode) \
                and node.operator == EXPONENTIATION_OPERATOR:  # exponentiation
            if not is_constant(node.operands[0]):
                return False
            elif not is_constant(node.operands[1]):
                return False

        elif isinstance(node, ArithmeticTransformationNode):   # transformation
            if node.symbol != "sum":
                is_const = [is_constant(child) for child in node.get_children()]
                if len(is_const) > 0:
                    return False

        for child in node.get_children():
            queue.put(child)

    return True


def is_univariate(node: ExpressionNode,
                  state: State,
                  idx_set: IndexingSet,
                  dummy_element: Element) -> bool:

    var_nodes = get_var_nodes(node)

    # exponent node contains only 1 variable node
    if len(var_nodes) == 1:
        return True

    # exponent node contains more than 1 variable node
    else:

        # retrieve the symbol of the first variable node
        var_sym_0 = var_nodes[0].symbol

        # retrieve the index of the first variable node
        var_idx_set_0 = None
        if var_nodes[0].idx_node is not None:
            var_idx_set_0 = var_nodes[0].idx_node.evaluate(
                state=state,
                idx_set=idx_set,
                dummy_element=dummy_element
            )

        # check whether the variable nodes are identical
        for i in range(1, len(var_nodes)):

            # check if the variable nodes reference different meta-variables
            if var_sym_0 != var_nodes[i].symbol:  # different variable symbols
                return False  # function is multivariate

            # check if the variable nodes reference different instances of the same meta-variable
            elif var_idx_set_0 is not None:
                var_idx_set_i = var_nodes[i].idx_node.evaluate(
                    state=state,
                    idx_set=idx_set,
                    dummy_element=dummy_element
                )
                if (var_idx_set_0 != var_idx_set_i).any():
                    return False  # function is multivariate

        return True  # function is univariate


def get_node(root_node: ExpressionNode, node_id: int):

    queue = Queue()
    queue.put(root_node)

    while not queue.empty():

        node = queue.get()

        if id(node) == node_id:
            return node

        for child in node.get_children():
            queue.put(child)

    return None


def get_param_nodes(root_node: ExpressionNode) -> List[DeclaredEntityNode]:

    param_nodes = []

    queue = Queue()
    queue.put(root_node)

    while not queue.empty():

        node: ExpressionNode = queue.get()

        if isinstance(node, DeclaredEntityNode):
            if node.is_constant():
                param_nodes.append(node)

        else:
            for child in node.get_children():
                queue.put(child)

    return param_nodes


def get_var_nodes(root_node: ExpressionNode) -> List[DeclaredEntityNode]:

    var_nodes = []

    queue = Queue()
    queue.put(root_node)

    while not queue.empty():

        node: ExpressionNode = queue.get()

        if isinstance(node, DeclaredEntityNode):
            if not node.is_constant():
                var_nodes.append(node)

        else:
            for child in node.get_children():
                queue.put(child)

    return var_nodes


# Expression
# ----------------------------------------------------------------------------------------------------------------------
class Expression:

    def __init__(self,
                 root_node: ExpressionNode,
                 indexing_set_node: CompoundSetNode = None,
                 id: str = ""):
        self.id: str = id
        self.root_node: ExpressionNode = root_node
        self.indexing_set_node: Optional[CompoundSetNode] = indexing_set_node
        self.link_nodes()

    def __str__(self):
        return self.get_literal()

    def to_lambda(self,
                  state: State,
                  idx_set_member: Element = None):

        if idx_set_member is not None:
            if self.indexing_set_node is None:
                raise ValueError("Non-indexed expression cannot be evaluated at an index")
            idx_set = OrderedSet([idx_set_member])
        else:
            if self.indexing_set_node is not None:
                idx_set = self.indexing_set_node.evaluate(state)[0]
            else:
                idx_set = None

        dummy_symbols = None
        if self.indexing_set_node is not None:
            self.indexing_set_node.combine_indexing_and_component_sets(state)
            dummy_symbols = self.indexing_set_node.combined_dummy_element

        if idx_set is None:
            return [self.root_node.to_lambda(state)]
        else:
            fcns = []
            for idx_set_member in idx_set:
                fcn = self.root_node.to_lambda(state, idx_set_member, dummy_symbols)
                fcns.append(fcn)
            return fcns

    def link_nodes(self):

        def link(node: ExpressionNode):
            children: List[ExpressionNode] = node.get_children()
            for child in children:
                child.parent = node
                link(child)

        link(self.root_node)

    def get_node_count(self) -> int:
        count = [0]

        def descend(node: ExpressionNode):
            count[0] += 1
            for child in node.get_children():
                descend(child)

        descend(self.root_node)
        return count[0]

    def get_declared_entity_nodes(self) -> List[DeclaredEntityNode]:
        nodes = []

        def descend(node: ExpressionNode):
            if isinstance(node, DeclaredEntityNode):
                nodes.append(node)
            for child in node.get_children():
                descend(child)

        descend(self.root_node)
        return nodes

    def get_param_nodes(self) -> List[DeclaredEntityNode]:
        return get_param_nodes(self.root_node)

    def get_var_nodes(self) -> List[DeclaredEntityNode]:
        return get_var_nodes(self.root_node)

    def get_declared_entity_syms(self) -> List[str]:
        return [n.symbol for n in self.get_declared_entity_nodes()]

    def get_param_syms(self) -> List[str]:
        return [n.symbol for n in self.get_param_nodes()]

    def get_var_syms(self) -> List[str]:
        return [n.symbol for n in self.get_var_nodes()]

    def get_literal(self) -> str:
        return str(self.root_node)
