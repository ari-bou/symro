import numpy as np
from typing import Callable, Optional

from symro.src.mat.util import *
from symro.src.mat.exprn import ExpressionNode, LogicalExpressionNode, SetExpressionNode, ArithmeticExpressionNode, \
    StringExpressionNode
from symro.src.mat.opern import LogicalOperationNode
from symro.src.mat.setn import CompoundSetNode
from symro.src.mat.state import State


class SetMembershipOperationNode(LogicalExpressionNode):

    def __init__(self,
                 operator: str,
                 member_node: Union[ArithmeticExpressionNode, StringExpressionNode] = None,
                 set_node: SetExpressionNode = None):
        super().__init__()
        self.operator: str = operator  # 'in' or 'not in'
        self.member_node: Optional[Union[ArithmeticExpressionNode, StringExpressionNode]] = member_node
        self.set_node: Optional[SetExpressionNode] = set_node

    def __invert__(self):
        return LogicalOperationNode.invert(self)

    def __and__(self, other: LogicalExpressionNode):
        return LogicalOperationNode.conjunction(self, other)

    def __or__(self, other: LogicalExpressionNode):
        return LogicalOperationNode.disjunction(self, other)

    def evaluate(self,
                 state: State,
                 idx_set: IndexingSet = None,
                 dummy_element: Element = None) -> np.ndarray:

        if idx_set is None:
            raise ValueError("Indexing set of a set membership operation cannot be null")

        challenge_elements = self.member_node.evaluate(state, idx_set, dummy_element)
        challenge_elements = np.array([tuple([e]) if not isinstance(e, tuple) else e for e in challenge_elements])

        sets_c = self.set_node.evaluate(state, idx_set, dummy_element)

        mp = 1
        if idx_set is not None:
            mp = len(idx_set)

        y = np.ndarray(shape=(mp,), dtype=bool)

        for ip in range(mp):
            y[ip] = challenge_elements[ip] in sets_c[ip]

        if self.operator == "not in":
            y = ~y

        return y

    def add_operand(self, operand: Optional[ExpressionNode]):
        if self.member_node is None:
            self.member_node = operand
        else:
            self.set_node = operand

    def get_children(self) -> List:
        return [self.member_node, self.set_node]

    def set_children(self, operands: list):
        if len(operands) > 0:
            self.member_node = operands[0]
        if len(operands) > 1:
            self.set_node = operands[1]

    def get_literal(self) -> str:
        literal = "{0} {1} {2}".format(self.member_node, self.operator, self.set_node)
        if self.is_prioritized:
            return '(' + literal + ')'
        return literal


class SetComparisonOperationNode(LogicalExpressionNode):

    def __init__(self,
                 operator: str,
                 lhs_operand: SetExpressionNode = None,
                 rhs_operand: SetExpressionNode = None):
        super().__init__()
        self.operator: str = operator
        self.lhs_operand: Optional[SetExpressionNode] = lhs_operand
        self.rhs_operand: Optional[SetExpressionNode] = rhs_operand

    def __invert__(self):
        return LogicalOperationNode.invert(self)

    def __and__(self, other: LogicalExpressionNode):
        return LogicalOperationNode.conjunction(self, other)

    def __or__(self, other: LogicalExpressionNode):
        return LogicalOperationNode.disjunction(self, other)

    def evaluate(self,
                 state: State,
                 idx_set: IndexingSet = None,
                 dummy_element: Element = None) -> np.ndarray:
        raise NotImplementedError("evaluate method of Set Comparison Operation Node has not been implemented")

    def add_operand(self, operand: Optional[ExpressionNode]):
        if self.lhs_operand is None:
            self.lhs_operand = operand
        else:
            self.rhs_operand = operand

    def get_children(self) -> List:
        return [self.lhs_operand, self.rhs_operand]

    def set_children(self, operands: list):
        if len(operands) > 0:
            self.lhs_operand = operands[0]
        if len(operands) > 1:
            self.rhs_operand = operands[1]

    def get_literal(self) -> str:
        literal = "{0} {1} {2}".format(self.lhs_operand, self.operator, self.rhs_operand)
        if self.is_prioritized:
            return '(' + literal + ')'
        return literal


class LogicalReductionOperationNode(LogicalExpressionNode):

    def __init__(self,
                 symbol: str,
                 idx_set_node: CompoundSetNode,
                 operand: ExpressionNode = None):

        super().__init__()
        self.symbol: str = symbol
        self.idx_set_node: CompoundSetNode = idx_set_node
        self.operand: ExpressionNode = operand

    def __invert__(self):
        return LogicalOperationNode.invert(self)

    def __and__(self, other: LogicalExpressionNode):
        return LogicalOperationNode.conjunction(self, other)

    def __or__(self, other: LogicalExpressionNode):
        return LogicalOperationNode.disjunction(self, other)

    def get_children(self) -> List[ExpressionNode]:
        if self.idx_set_node is None:
            return [self.operand]
        else:
            return [self.operand, self.idx_set_node]

    def set_children(self, operands: List[ExpressionNode]):
        if len(operands) > 0:
            self.operand = operands[0]
            if len(operands) > 1:
                self.idx_set_node = operands[1]

    def get_literal(self) -> str:
        literal = self.symbol + str(self.idx_set_node) + str(self.operand)
        if self.is_prioritized:
            literal = '(' + literal + ')'
        return literal


class BooleanNode(LogicalExpressionNode):

    def __init__(self, value: bool):
        super().__init__()
        self.value: bool = value

    def __invert__(self):
        self.value = not self.value

    def __and__(self, other: LogicalExpressionNode):

        if isinstance(other, BooleanNode):
            return BooleanNode(self.value and other.value)

        elif isinstance(other, LogicalOperationNode) and other.operator == CONJUNCTION_OPERATOR:

            for operand in other.operands:
                if isinstance(operand, BooleanNode):
                    operand.value = self.value and operand.value
                    return other

            other.operands.insert(0, self)
            return other

        else:
            return LogicalOperationNode.conjunction(self, other)

    def __or__(self, other: LogicalExpressionNode):

        if isinstance(other, BooleanNode):
            return BooleanNode(self.value or other.value)

        elif isinstance(other, LogicalOperationNode) and other.operator == DISJUNCTION_OPERATOR:

            for operand in other.operands:
                if isinstance(operand, BooleanNode):
                    operand.value = self.value or operand.value
                    return other

            other.operands.insert(0, self)
            return other

        else:
            return LogicalOperationNode.disjunction(self, other)

    def evaluate(self,
                 state: State,
                 idx_set: IndexingSet = None,
                 dummy_element: Element = None) -> np.ndarray:

        mp = 1
        if idx_set is not None:
            mp = len(idx_set)

        return np.full(shape=(mp,), fill_value=self.value, dtype=bool)

    def to_lambda(self,
                  state: State,
                  idx_set_member: Element = None,
                  dummy_element: Element = None) -> Callable:
        return lambda: self.value

    def get_literal(self) -> str:
        literal = str(int(self.value))
        if self.is_prioritized:
            return '(' + literal + ')'
        return literal
