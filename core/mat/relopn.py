from functools import partial

import numpy as np
from typing import Callable, Dict, Optional

from symro.core.mat.entity import Parameter, Variable
from symro.core.mat.exprn import ExpressionNode, LogicalExpressionNode, ArithmeticExpressionNode, StringExpressionNode
from symro.core.mat.logopn import LogicalOperationNode
from symro.core.mat.util import *
from symro.core.mat.state import State


class RelationalOperationNode(LogicalExpressionNode):

    def __init__(self,
                 operator: int,
                 lhs_operand: Union[ArithmeticExpressionNode,
                                    StringExpressionNode] = None,
                 rhs_operand: Union[ArithmeticExpressionNode,
                                    StringExpressionNode] = None):
        super().__init__()
        self.operator: int = operator
        self.lhs_operand: Optional[Union[ArithmeticExpressionNode,
                                         StringExpressionNode]] = lhs_operand
        self.rhs_operand: Optional[Union[ArithmeticExpressionNode,
                                         StringExpressionNode]] = rhs_operand

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

        x_lhs = self.lhs_operand.evaluate(state, idx_set, dummy_element)
        x_rhs = self.rhs_operand.evaluate(state, idx_set, dummy_element)

        # equality
        if self.operator == EQUALITY_OPERATOR:
            return x_lhs == x_rhs

        # strict inequality
        elif self.operator == STRICT_INEQUALITY_OPERATOR:
            return x_lhs != x_rhs

        # greater than
        elif self.operator == GREATER_INEQUALITY_OPERATOR:
            return x_lhs > x_rhs

        # greater than or equal to
        elif self.operator == GREATER_EQUAL_INEQUALITY_OPERATOR:
            return x_lhs >= x_rhs

        # less than
        elif self.operator == LESS_INEQUALITY_OPERATOR:
            return x_lhs < x_rhs

        # less than or equal to
        elif self.operator == LESS_EQUAL_INEQUALITY_OPERATOR:
            return x_lhs <= x_rhs

        else:
            raise ValueError("Unable to resolve operator '{0}' as a relational operator".format(self.operator))

    def to_lambda(self,
                  state: State,
                  idx_set_member: Element = None,
                  dummy_element: Element = None) -> Callable:

        x_lhs = self.lhs_operand.to_lambda(state, idx_set_member, dummy_element)
        x_rhs = self.rhs_operand.to_lambda(state, idx_set_member, dummy_element)

        # equality
        if self.operator == EQUALITY_OPERATOR:
            return partial(lambda l, r: l() == r(), x_lhs, x_rhs)

        # strict inequality
        elif self.operator == STRICT_INEQUALITY_OPERATOR:
            return partial(lambda l, r: l() != r(), x_lhs, x_rhs)

        # greater than
        elif self.operator == GREATER_INEQUALITY_OPERATOR:
            return partial(lambda l, r: l() > r(), x_lhs, x_rhs)

        # greater than or equal to
        elif self.operator == GREATER_EQUAL_INEQUALITY_OPERATOR:
            return partial(lambda l, r: l() >= r(), x_lhs, x_rhs)

        # less than
        elif self.operator == LESS_INEQUALITY_OPERATOR:
            return partial(lambda l, r: l() < r(), x_lhs, x_rhs)

        # less than or equal to
        elif self.operator == LESS_EQUAL_INEQUALITY_OPERATOR:
            return partial(lambda l, r: l() <= r(), x_lhs, x_rhs)

        else:
            raise ValueError("Unable to resolve operator '{0}' as a relational operator".format(self.operator))

    @staticmethod
    def equal(lhs_operand: Union[ArithmeticExpressionNode, StringExpressionNode],
              rhs_operand: Union[ArithmeticExpressionNode, StringExpressionNode]):
        return RelationalOperationNode(operator=EQUALITY_OPERATOR,
                                       lhs_operand=lhs_operand,
                                       rhs_operand=rhs_operand)

    @staticmethod
    def not_equal(lhs_operand: Union[ArithmeticExpressionNode, StringExpressionNode],
                  rhs_operand: Union[ArithmeticExpressionNode, StringExpressionNode]):
        return RelationalOperationNode(operator=STRICT_INEQUALITY_OPERATOR,
                                       lhs_operand=lhs_operand,
                                       rhs_operand=rhs_operand)

    @staticmethod
    def less_than(lhs_operand: Union[ArithmeticExpressionNode, StringExpressionNode],
                  rhs_operand: Union[ArithmeticExpressionNode, StringExpressionNode]):
        return RelationalOperationNode(operator=LESS_INEQUALITY_OPERATOR,
                                       lhs_operand=lhs_operand,
                                       rhs_operand=rhs_operand)

    @staticmethod
    def less_than_or_equal(lhs_operand: Union[ArithmeticExpressionNode, StringExpressionNode],
                           rhs_operand: Union[ArithmeticExpressionNode, StringExpressionNode]):
        return RelationalOperationNode(operator=LESS_EQUAL_INEQUALITY_OPERATOR,
                                       lhs_operand=lhs_operand,
                                       rhs_operand=rhs_operand)

    @staticmethod
    def greater_than(lhs_operand: Union[ArithmeticExpressionNode, StringExpressionNode],
                     rhs_operand: Union[ArithmeticExpressionNode, StringExpressionNode]):
        return RelationalOperationNode(operator=GREATER_INEQUALITY_OPERATOR,
                                       lhs_operand=lhs_operand,
                                       rhs_operand=rhs_operand)

    @staticmethod
    def greater_than_or_equal(lhs_operand: Union[ArithmeticExpressionNode, StringExpressionNode],
                              rhs_operand: Union[ArithmeticExpressionNode, StringExpressionNode]):
        return RelationalOperationNode(operator=GREATER_EQUAL_INEQUALITY_OPERATOR,
                                       lhs_operand=lhs_operand,
                                       rhs_operand=rhs_operand)

    def collect_declared_entities(self,
                                  state: State,
                                  idx_set: IndexingSet = None,
                                  dummy_element: Element = None) -> Dict[str, Union[Parameter, Variable]]:
        entities = self.lhs_operand.collect_declared_entities(state, idx_set, dummy_element)
        entities.update(self.rhs_operand.collect_declared_entities(state, idx_set, dummy_element))
        return entities

    def get_children(self) -> List[ExpressionNode]:
        return [self.lhs_operand, self.rhs_operand]

    def set_children(self, operands: list):
        if len(operands) > 0:
            self.lhs_operand = operands[0]
        if len(operands) > 1:
            self.rhs_operand = operands[1]

    def get_literal(self) -> str:
        if self.operator == '<=':
            x = 2
        literal = "{0} {1} {2}".format(self.lhs_operand, AMPL_OPERATOR_SYMBOLS[self.operator], self.rhs_operand)
        if self.is_prioritized:
            return '(' + literal + ')'
        return literal
