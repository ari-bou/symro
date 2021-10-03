import numpy as np

from symro.core.mat.exprn import SetExpressionNode
from symro.core.mat.util import *
from symro.core.mat.state import State


class SetOperationNode(SetExpressionNode):

    def __init__(self,
                 operator: int,
                 lhs_operand: SetExpressionNode = None,
                 rhs_operand: SetExpressionNode = None):
        super().__init__()
        self.operator: int = operator
        self.lhs_operand: SetExpressionNode = lhs_operand
        self.rhs_operand: SetExpressionNode = rhs_operand

    def __and__(self, other: SetExpressionNode):
        return self.intersection(self, other)

    def __or__(self, other: SetExpressionNode):
        return self.union(self, other)

    def __sub__(self, other: SetExpressionNode):
        return self.difference(self, other)

    def __xor__(self, other: SetExpressionNode):
        return self.symmetric_difference(self, other)

    def evaluate(self,
                 state: State,
                 idx_set: IndexingSet = None,
                 dummy_element: Element = None
                 ) -> np.ndarray:

        x_lhs = self.lhs_operand.evaluate(state, idx_set, dummy_element)
        x_rhs = self.rhs_operand.evaluate(state, idx_set, dummy_element)

        if self.operator == UNION_OPERATOR:
            return x_lhs | x_rhs

        elif self.operator == INTERSECTION_OPERATOR:
            return x_lhs & x_rhs

        elif self.operator == DIFFERENCE_OPERATOR:
            return x_lhs - x_rhs

        elif self.operator == SYMMETRIC_DIFFERENCE_OPERATOR:
            return x_lhs ^ x_rhs

        else:
            raise ValueError("Unable to resolve operator '{0}' as a set operator".format(self.operator))

    @staticmethod
    def union(lhs_operand: SetExpressionNode, rhs_operand: SetExpressionNode):
        return SetOperationNode(operator=UNION_OPERATOR,
                                lhs_operand=lhs_operand,
                                rhs_operand=rhs_operand)

    @staticmethod
    def intersection(lhs_operand: SetExpressionNode, rhs_operand: SetExpressionNode):
        return SetOperationNode(operator=INTERSECTION_OPERATOR,
                                lhs_operand=lhs_operand,
                                rhs_operand=rhs_operand)

    @staticmethod
    def difference(lhs_operand: SetExpressionNode, rhs_operand: SetExpressionNode):
        return SetOperationNode(operator=DIFFERENCE_OPERATOR,
                                lhs_operand=lhs_operand,
                                rhs_operand=rhs_operand)

    @staticmethod
    def symmetric_difference(lhs_operand: SetExpressionNode, rhs_operand: SetExpressionNode):
        return SetOperationNode(operator=SYMMETRIC_DIFFERENCE_OPERATOR,
                                lhs_operand=lhs_operand,
                                rhs_operand=rhs_operand)

    def get_dim(self, state: State) -> int:
        return self.lhs_operand.get_dim(state)

    def get_children(self) -> list:
        return [self.lhs_operand, self.rhs_operand]

    def set_children(self, operands: list):
        if len(operands) > 0:
            self.lhs_operand = operands[0]
            if len(operands) > 1:
                self.rhs_operand = operands[1]

    def get_literal(self) -> str:
        literal = "{0} {1} {2}".format(self.lhs_operand, AMPL_OPERATOR_SYMBOLS[self.operator], self.rhs_operand)
        if self.is_prioritized:
            return '(' + literal + ')'
        return literal
