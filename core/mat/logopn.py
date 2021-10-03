from functools import partial
import numpy as np
from typing import Callable, Optional

from symro.core.mat.util import *
from symro.core.mat.exprn import ExpressionNode, LogicalExpressionNode, ArithmeticExpressionNode
from symro.core.mat.state import State


class LogicalOperationNode(LogicalExpressionNode):

    def __init__(self,
                 operator: int,
                 operands: List[LogicalExpressionNode] = None):

        super().__init__()

        self.operator: int = operator
        self.operands: Optional[List[LogicalExpressionNode]] = operands

        if self.operands is None:
            self.operands = []

    def __invert__(self):
        return self.invert(self)

    def __and__(self, other: LogicalExpressionNode):
        return self.conjunction(self, other)

    def __or__(self, other: LogicalExpressionNode):
        return self.disjunction(self, other)

    def evaluate(self,
                 state: State,
                 idx_set: IndexingSet = None,
                 dummy_element: Element = None
                 ) -> np.ndarray:

        x_lhs = self.operands[0].evaluate(state, idx_set, dummy_element)
        y = x_lhs

        # logical inversion
        if self.operator == UNARY_INVERSION_OPERATOR:
            return ~y

        # n-ary operation
        else:

            for i in range(1, len(self.operands)):

                x_rhs = self.operands[i].evaluate(state, idx_set, dummy_element)

                if self.operator == CONJUNCTION_OPERATOR:  # logical conjunction
                    y = x_lhs & x_rhs

                elif self.operator == DISJUNCTION_OPERATOR:  # logical disjunction
                    y = x_lhs | x_rhs

                else:
                    raise ValueError("Unable to resolve symbol '{0}'".format(self.operator)
                                     + " as an n-ary logical operator")

                x_lhs = y

            return y

    def to_lambda(self,
                  state: State,
                  idx_set_member: Element = None,
                  dummy_element: Element = None) -> Callable:

        args = np.array([o.to_lambda(state, idx_set_member, dummy_element) for o in self.operands])

        # logical inversion
        if self.operator == UNARY_INVERSION_OPERATOR:
            return partial(lambda x: not x(), args[0])

        # n-ary operation
        else:

            if self.operator == CONJUNCTION_OPERATOR:  # logical conjunction
                return partial(lambda x: all([x_i() for x_i in x]), args)

            elif self.operator == DISJUNCTION_OPERATOR:  # logical disjunction
                return partial(lambda x: any([x_i() for x_i in x]), args)

            else:
                raise ValueError("Unable to resolve symbol '{0}'".format(self.operator)
                                 + " as an n-ary logical operator")

    @staticmethod
    def invert(operand: LogicalExpressionNode):
        return LogicalOperationNode(operator=UNARY_INVERSION_OPERATOR,
                                    operands=[operand])

    @staticmethod
    def conjunction(lhs_operand: LogicalExpressionNode, rhs_operand: LogicalExpressionNode):

        if isinstance(lhs_operand, LogicalOperationNode) and lhs_operand.operator == CONJUNCTION_OPERATOR:

            for operand in lhs_operand.operands:
                rhs_operand = operand & rhs_operand

            return rhs_operand

        elif isinstance(rhs_operand, LogicalOperationNode) and rhs_operand.operator == CONJUNCTION_OPERATOR:

            for operand in rhs_operand.operands:
                lhs_operand = operand & lhs_operand

            return lhs_operand

        else:
            return LogicalOperationNode(operator=CONJUNCTION_OPERATOR,
                                        operands=[lhs_operand, rhs_operand])

    @staticmethod
    def disjunction(lhs_operand: LogicalExpressionNode, rhs_operand: LogicalExpressionNode):

        if isinstance(lhs_operand, LogicalOperationNode) and lhs_operand.operator == DISJUNCTION_OPERATOR:

            for operand in lhs_operand.operands:
                rhs_operand = operand | rhs_operand

            return rhs_operand

        elif isinstance(rhs_operand, LogicalOperationNode) and rhs_operand.operator == DISJUNCTION_OPERATOR:

            for operand in rhs_operand.operands:
                lhs_operand = operand | lhs_operand

            return lhs_operand

        else:
            return LogicalOperationNode(operator=DISJUNCTION_OPERATOR,
                                        operands=[lhs_operand, rhs_operand])

    def get_lhs_operand(self):
        return self.operands[0]

    def set_lhs_operand(self, operand: ArithmeticExpressionNode):
        self.operands[0] = operand

    def get_rhs_operand(self):
        return self.operands[1]

    def set_rhs_operand(self, operand: ArithmeticExpressionNode):
        self.operands[1] = operand

    def get_children(self) -> List[ExpressionNode]:
        return self.operands

    def set_children(self, operands: list):
        self.operands.clear()
        self.operands.extend(operands)

    def get_literal(self) -> str:

        s = ""

        if self.operator == UNARY_INVERSION_OPERATOR:
            s = "{0} {1}".format(AMPL_OPERATOR_SYMBOLS[self.operator], self.operands[0])

        else:
            for i, operand in enumerate(self.operands):
                if i == 0:
                    s = operand.get_literal()
                else:
                    s += " {0} {1}".format(AMPL_OPERATOR_SYMBOLS[self.operator], operand)

        if self.is_prioritized:
            return '(' + s + ')'

        else:
            return s
