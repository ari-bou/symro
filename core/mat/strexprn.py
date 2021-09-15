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


class MultiStringOperationNode(StringExpressionNode):

    CONCAT_OPERATOR = '&'

    def __init__(self,
                 operator: str,
                 operands: List[Union[StringExpressionNode, BaseDummyNode]] = None,
                 is_prioritized: bool = False,
                 id: int = 0):
        super().__init__(id)
        self.operator: str = operator
        self.operands: List[Union[StringExpressionNode, BaseDummyNode]] = operands
        self.is_prioritized = is_prioritized

    def evaluate(self,
                 state: State,
                 idx_set: IndexSet = None,
                 dummy_symbols: Tuple[str, ...] = None
                 ) -> List[str]:

        x_lhs = self.operands[0].evaluate(state, idx_set, dummy_symbols)
        y = x_lhs

        for i in range(1, len(self.operands)):

            x_rhs = self.operands[i].evaluate(state, idx_set, dummy_symbols)

            if self.operator == self.CONCAT_OPERATOR:  # Concatenation
                y = [x_lhs_i + x_rhs_i for x_lhs_i, x_rhs_i in zip(x_lhs, x_rhs)]
            else:
                raise ValueError("Unable to resolve symbol '{0}'"
                                 " as a string operator".format(self.operator))

            x_lhs = y

        return y

    def to_lambda(self,
                  state: State,
                  idx_set_member: IndexSetMember = None,
                  dummy_symbols: Tuple[str, ...] = None):

        args_all = []
        for operand in self.operands:
            args_all.append(operand.to_lambda(state, idx_set_member, dummy_symbols))

        if self.operator == self.CONCAT_OPERATOR:  # Concatenation
            return partial(lambda x: ''.join([x_i() for x_i in x]), args_all)
        else:
            raise ValueError("Unable to resolve symbol '{0}'".format(self.operator)
                             + " as a string operator")

    def get_children(self) -> List:
        return list(self.operands)

    def set_children(self, operands: list):
        self.operands = list(operands)

    def get_literal(self) -> str:
        s = ""
        for i, operand in enumerate(self.operands):
            if i == 0:
                s = operand.get_literal()
            else:
                s += " {0} {1}".format(self.operator, operand)
        if self.is_prioritized:
            return '(' + s + ')'
        else:
            return s
