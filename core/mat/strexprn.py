from functools import partial
from typing import List, Tuple, Union

from symro.core.mat.exprn import StringExpressionNode
from symro.core.mat.dummyn import BaseDummyNode
from symro.core.mat.util import IndexSet, IndexSetMember
from symro.core.mat.state import State


class StringNode(StringExpressionNode):

    def __init__(self, literal: str, delimiter: str, id: int = 0):
        super().__init__(id)
        self.literal: str = literal  # string literal without delimiter
        self.delimiter: str = delimiter

    def evaluate(self,
                 state: State,
                 idx_set: IndexSet = None,
                 dummy_symbols: Tuple[str, ...] = None) -> List[str]:
        count_p = 1 if idx_set is None else len(idx_set)
        return [self.literal] * count_p

    def to_lambda(self,
                  state: State,
                  idx_set_member: IndexSetMember = None,
                  dummy_symbols: Tuple[str, ...] = None):
        return partial(lambda l: l, self.literal)

    def get_children(self) -> List:
        return []

    def set_children(self, operands: list):
        pass

    def get_literal(self) -> str:
        return self.delimiter + self.literal + self.delimiter


class BinaryStringOperationNode(StringExpressionNode):

    CONCAT_OPERATOR = '&'

    def __init__(self,
                 operator: str,
                 lhs_operand: Union[StringExpressionNode, BaseDummyNode] = None,
                 rhs_operand: Union[StringExpressionNode, BaseDummyNode] = None,
                 is_prioritized: bool = False,
                 id: int = 0):
        super().__init__(id)
        self.operator: str = operator
        self.lhs_operand: Union[StringExpressionNode, BaseDummyNode] = lhs_operand
        self.rhs_operand: Union[StringExpressionNode, BaseDummyNode] = rhs_operand
        self.is_prioritized = is_prioritized

    def evaluate(self,
                 state: State,
                 idx_set: IndexSet = None,
                 dummy_symbols: Tuple[str, ...] = None
                 ) -> List[str]:

        x_lhs = self.lhs_operand.evaluate(state, idx_set, dummy_symbols)
        x_rhs = self.rhs_operand.evaluate(state, idx_set, dummy_symbols)

        if self.operator == self.CONCAT_OPERATOR:  # Concatenation
            y = [str(x_lhs_i) + str(x_rhs_i) for x_lhs_i, x_rhs_i in zip(x_lhs, x_rhs)]
        else:
            raise ValueError("Unable to resolve symbol '{0}'"
                             " as a binary string operator".format(self.operator))
        return y

    def to_lambda(self,
                  state: State,
                  idx_set_member: IndexSetMember = None,
                  dummy_symbols: Tuple[str, ...] = None):
        x_lhs = self.lhs_operand.to_lambda(state, idx_set_member, dummy_symbols)
        x_rhs = self.rhs_operand.to_lambda(state, idx_set_member, dummy_symbols)
        if self.operator == self.CONCAT_OPERATOR:  # Concatenation
            return partial(lambda l, r: str(l()) + str(r()), x_lhs, x_rhs)
        else:
            raise ValueError("Unable to resolve symbol '{0}'"
                             " as a binary arithmetic operator".format(self.operator))

    def get_children(self) -> List:
        return [self.lhs_operand, self.rhs_operand]

    def set_children(self, operands: list):
        if len(operands) > 0:
            self.lhs_operand = operands[0]
        if len(operands) > 1:
            self.rhs_operand = operands[1]

    def get_literal(self) -> str:
        literal = "{0} {1} {2}".format(self.lhs_operand,
                                       self.operator,
                                       self.rhs_operand)
        if self.is_prioritized:
            literal = '(' + literal + ')'
        return literal
