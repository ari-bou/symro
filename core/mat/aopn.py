from functools import partial
import numpy as np
from typing import Callable, Dict

from symro.core.mat.util import *
from symro.core.mat.entity import Parameter, Variable
from symro.core.mat.exprn import ArithmeticExpressionNode
from symro.core.mat.relopn import RelationalOperationNode
from symro.core.mat.state import State


class ArithmeticOperationNode(ArithmeticExpressionNode):

    def __init__(self,
                 operator: int,
                 operands: List[ArithmeticExpressionNode],
                 is_prioritized: bool = False):
        super().__init__()
        self.operator: int = operator
        self.operands: List[ArithmeticExpressionNode] = operands
        self.is_prioritized = is_prioritized

    def __neg__(self):
        return self.negation(self)

    def __add__(self, other: ArithmeticExpressionNode):
        return self.addition(self, other)

    def __sub__(self, other: ArithmeticExpressionNode):
        return self.subtraction(self, other)

    def __mul__(self, other: ArithmeticExpressionNode):
        return self.multiplication(self, other)

    def __truediv__(self, other: ArithmeticExpressionNode):
        return self.division(self, other)

    def __pow__(self, power: ArithmeticExpressionNode, modulo=None):
        return self.exponentiation(self, power)

    def __eq__(self, other: ArithmeticExpressionNode):
        return RelationalOperationNode.equal(self, other)

    def __ne__(self, other: ArithmeticExpressionNode):
        return RelationalOperationNode.not_equal(self, other)

    def __lt__(self, other: ArithmeticExpressionNode):
        return RelationalOperationNode.less_than(self, other)

    def __le__(self, other: ArithmeticExpressionNode):
        return RelationalOperationNode.less_than_or_equal(self, other)

    def __gt__(self, other: ArithmeticExpressionNode):
        return RelationalOperationNode.greater_than(self, other)

    def __ge__(self, other: ArithmeticExpressionNode):
        return RelationalOperationNode.greater_than_or_equal(self, other)

    def evaluate(self,
                 state: State,
                 idx_set: IndexingSet = None,
                 dummy_element: Element = None
                 ) -> np.ndarray:

        x_lhs = self.operands[0].evaluate(state, idx_set, dummy_element)
        y = x_lhs

        # unary operation
        if self.operator in (UNARY_POSITIVE_OPERATOR, UNARY_NEGATION_OPERATOR):

            if self.operator == UNARY_POSITIVE_OPERATOR:
                return y

            elif self.operator == UNARY_NEGATION_OPERATOR:
                return -y

            else:
                raise ValueError("Unable to resolve operator '{0}'".format(self.operator)
                                 + " as a unary arithmetic operator")

        # n-ary operation
        else:

            for i in range(1, len(self.operands)):

                x_rhs = self.operands[i].evaluate(state, idx_set, dummy_element)

                if self.operator == ADDITION_OPERATOR:  # addition
                    y = x_lhs + x_rhs
                elif self.operator == SUBTRACTION_OPERATOR:  # subtraction
                    y = x_lhs - x_rhs
                elif self.operator == MULTIPLICATION_OPERATOR:  # multiplication
                    y = x_lhs * x_rhs
                elif self.operator == DIVISION_OPERATOR:  # division
                    y = x_lhs / x_rhs
                elif self.operator == EXPONENTIATION_OPERATOR:  # exponentiation
                    y = x_lhs ** x_rhs
                else:
                    raise ValueError("Unable to resolve operator '{0}'".format(self.operator)
                                     + " as an n-ary arithmetic operator")

                x_lhs = y

            return y

    def to_lambda(self,
                  state: State,
                  idx_set_member: Element = None,
                  dummy_element: Element = None) -> Callable:

        args = []
        for operand in self.operands:
            args.append(operand.to_lambda(state, idx_set_member, dummy_element))

        # unary operation
        if self.operator in (UNARY_POSITIVE_OPERATOR, UNARY_NEGATION_OPERATOR):

            if self.operator == UNARY_POSITIVE_OPERATOR:
                return args[0]

            elif self.operator == UNARY_NEGATION_OPERATOR:
                return lambda: -args[0]()

            else:
                raise ValueError("Unable to resolve operator '{0}'".format(self.operator)
                                 + " as a unary arithmetic operator")

        # n-ary operation
        else:

            if self.operator == ADDITION_OPERATOR:  # addition
                return partial(lambda x: sum([x_i() for x_i in x]), args)

            elif self.operator == SUBTRACTION_OPERATOR:  # subtraction
                return partial(lambda x: x[0]() - x[1](), args)

            elif self.operator == MULTIPLICATION_OPERATOR:  # multiplication
                return partial(lambda x: np.prod([x_i() for x_i in x]), args)

            elif self.operator == DIVISION_OPERATOR:  # division
                return partial(lambda x: x[0]() / x[1](), args)

            elif self.operator == EXPONENTIATION_OPERATOR:  # exponentiation
                return partial(lambda x: x[0]() ** x[1](), args)

            else:
                raise ValueError("Unable to resolve operator '{0}'".format(self.operator)
                                 + " as a multi arithmetic operator")

    def collect_declared_entities(self,
                                  state: State,
                                  idx_set: IndexingSet = None,
                                  dummy_element: Element = None) -> Dict[str, Union[Parameter, Variable]]:
        vars = {}
        for operand in self.operands:
            vars.update(operand.collect_declared_entities(state, idx_set, dummy_element))
        return vars

    @staticmethod
    def negation(operand: ArithmeticExpressionNode):
        return ArithmeticOperationNode(operator=UNARY_NEGATION_OPERATOR,
                                       operands=[operand])

    @staticmethod
    def addition(lhs_operand: ArithmeticExpressionNode, rhs_operand: ArithmeticExpressionNode):

        if isinstance(lhs_operand, ArithmeticOperationNode) and lhs_operand.operator == ADDITION_OPERATOR:

            for term in lhs_operand.operands:
                rhs_operand = term + rhs_operand

            return rhs_operand

        elif isinstance(rhs_operand, ArithmeticOperationNode) and rhs_operand.operator == ADDITION_OPERATOR:

            for term in rhs_operand.operands:
                lhs_operand = term + lhs_operand

            return lhs_operand

        else:
            return ArithmeticOperationNode(operator=ADDITION_OPERATOR,
                                           operands=[lhs_operand, rhs_operand])

    @staticmethod
    def subtraction(lhs_operand: ArithmeticExpressionNode, rhs_operand: ArithmeticExpressionNode):

        sub_op_node = ArithmeticOperationNode(operator=SUBTRACTION_OPERATOR,
                                              operands=[lhs_operand, rhs_operand])

        if isinstance(rhs_operand, ArithmeticOperationNode):
            if rhs_operand.operator == ADDITION_OPERATOR:
                rhs_operand.is_prioritized = True

        return sub_op_node

    @staticmethod
    def multiplication(lhs_operand: ArithmeticExpressionNode, rhs_operand: ArithmeticExpressionNode):

        if isinstance(lhs_operand, ArithmeticOperationNode) and lhs_operand.operator == MULTIPLICATION_OPERATOR:

            for factor in lhs_operand.operands:
                rhs_operand = factor * rhs_operand

            return rhs_operand

        elif isinstance(rhs_operand, ArithmeticOperationNode) and rhs_operand.operator == MULTIPLICATION_OPERATOR:

            for factor in rhs_operand.operands:
                lhs_operand = factor * lhs_operand

            return lhs_operand

        else:
            return ArithmeticOperationNode(operator=MULTIPLICATION_OPERATOR,
                                           operands=[lhs_operand, rhs_operand])

    @staticmethod
    def division(lhs_operand: ArithmeticExpressionNode, rhs_operand: ArithmeticExpressionNode):

        div_op_node = ArithmeticOperationNode(operator=DIVISION_OPERATOR,
                                              operands=[lhs_operand, rhs_operand])

        if isinstance(lhs_operand, ArithmeticOperationNode):
            if lhs_operand.operator in (ADDITION_OPERATOR, SUBTRACTION_OPERATOR):
                lhs_operand.is_prioritized = True

        if isinstance(rhs_operand, ArithmeticOperationNode):
            if rhs_operand.operator in (ADDITION_OPERATOR, SUBTRACTION_OPERATOR):
                rhs_operand.is_prioritized = True

        return div_op_node

    @staticmethod
    def exponentiation(lhs_operand: ArithmeticExpressionNode, rhs_operand: ArithmeticExpressionNode):

        exp_op_node = ArithmeticOperationNode(operator=EXPONENTIATION_OPERATOR,
                                              operands=[lhs_operand, rhs_operand])

        if isinstance(lhs_operand, ArithmeticOperationNode):
            lhs_operand.is_prioritized = True

        if isinstance(rhs_operand, ArithmeticOperationNode):
            rhs_operand.is_prioritized = True

        return exp_op_node

    def get_arity(self) -> int:
        return len(self.operands)

    def get_lhs_operand(self):
        return self.operands[0]

    def set_lhs_operand(self, operand: ArithmeticExpressionNode):
        self.operands[0] = operand

    def get_rhs_operand(self):
        return self.operands[1]

    def set_rhs_operand(self, operand: ArithmeticExpressionNode):
        self.operands[1] = operand

    def get_children(self) -> List:
        return self.operands

    def set_children(self, operands: list):
        self.operands.clear()
        self.operands.extend(operands)

    def get_literal(self) -> str:

        s = ""

        if self.operator == UNARY_POSITIVE_OPERATOR:
            s = str(self.operands[0])

        elif self.operator == UNARY_NEGATION_OPERATOR:
            s = "{0}{1}".format(AMPL_OPERATOR_SYMBOLS[self.operator], self.operands[0])

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
