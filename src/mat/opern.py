from functools import partial
import numpy as np
from typing import Callable, Dict, Optional

from symro.src.mat.util import *
from symro.src.mat.entity import Parameter, Variable
from symro.src.mat.exprn import ExpressionNode, LogicalExpressionNode, SetExpressionNode, ArithmeticExpressionNode, \
    StringExpressionNode
from symro.src.mat.state import State


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
                                  dummy_element: Element = None) -> Dict[tuple, Union[Parameter, Variable]]:
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
        literal = "{0} {1} {2}".format(self.lhs_operand, AMPL_OPERATOR_SYMBOLS[self.operator], self.rhs_operand)
        if self.is_prioritized:
            return '(' + literal + ')'
        return literal


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

    def to_lambda(self,
                  state: State,
                  idx_set_member: Element = None,
                  dummy_element: Tuple[str, ...] = None) -> Callable:
        raise NotImplementedError("to_lambda method has not yet been implemented for '{0}'".format(type(self)))

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
                                  dummy_element: Element = None) -> Dict[tuple, Union[Parameter, Variable]]:
        vars = {}
        for operand in self.operands:
            vars.update(operand.collect_declared_entities(state, idx_set, dummy_element))
        return vars

    @staticmethod
    def negation(operand: ArithmeticExpressionNode):

        if isinstance(operand, ArithmeticOperationNode):

            if operand.operator in (ADDITION_OPERATOR, SUBTRACTION_OPERATOR):

                # distribute negative to each term
                for i, term in enumerate(operand.operands):
                    operand.operands[i] = -term

                return operand

            elif operand.operator in (MULTIPLICATION_OPERATOR, DIVISION_OPERATOR):
                operand.operands[0] = -operand.operands[0]  # negate first factor
                return operand

        return ArithmeticOperationNode(operator=UNARY_NEGATION_OPERATOR,
                                       operands=[operand])

    @staticmethod
    def addition(lhs_operand: ArithmeticExpressionNode, rhs_operand: ArithmeticExpressionNode):

        if isinstance(lhs_operand, ArithmeticOperationNode) and lhs_operand.operator == ADDITION_OPERATOR:
            return rhs_operand + lhs_operand

        elif isinstance(rhs_operand, ArithmeticOperationNode) and rhs_operand.operator == ADDITION_OPERATOR:
            rhs_operand.operands.insert(0, lhs_operand)
            return rhs_operand

        else:
            return ArithmeticOperationNode(operator=ADDITION_OPERATOR,
                                           operands=[lhs_operand, rhs_operand])

    @staticmethod
    def subtraction(lhs_operand: ArithmeticExpressionNode, rhs_operand: ArithmeticExpressionNode):

        rhs_operand.is_prioritized = True
        rhs_operand = -rhs_operand

        if isinstance(lhs_operand, ArithmeticOperationNode) and lhs_operand.operator == ADDITION_OPERATOR:
            return rhs_operand + lhs_operand

        elif isinstance(rhs_operand, ArithmeticOperationNode) and rhs_operand.operator == ADDITION_OPERATOR:
            rhs_operand.operands.insert(0, lhs_operand)
            return rhs_operand

        else:
            return ArithmeticOperationNode(operator=ADDITION_OPERATOR,
                                           operands=[lhs_operand, rhs_operand])

    @staticmethod
    def multiplication(lhs_operand: ArithmeticExpressionNode, rhs_operand: ArithmeticExpressionNode):

        if isinstance(lhs_operand, ArithmeticOperationNode) and lhs_operand.operator == MULTIPLICATION_OPERATOR:
            return rhs_operand * lhs_operand

        elif isinstance(rhs_operand, ArithmeticOperationNode) and rhs_operand.operator == MULTIPLICATION_OPERATOR:
            rhs_operand.operands.insert(0, lhs_operand)
            return rhs_operand

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


class AdditionNode(ArithmeticOperationNode):

    def __init__(self,
                 operands: List[ArithmeticExpressionNode],
                 is_prioritized: bool = False):
        super().__init__(operator=ADDITION_OPERATOR,
                         operands=operands,
                         is_prioritized=is_prioritized)


class SubtractionNode(ArithmeticOperationNode):

    def __init__(self,
                 lhs_operand: ArithmeticExpressionNode = None,
                 rhs_operand: ArithmeticExpressionNode = None,
                 is_prioritized: bool = False):
        super().__init__(operator=SUBTRACTION_OPERATOR,
                         operands=[lhs_operand, rhs_operand],
                         is_prioritized=is_prioritized)


class MultiplicationNode(ArithmeticOperationNode):

    def __init__(self,
                 operands: List[ArithmeticExpressionNode],
                 is_prioritized: bool = False):
        super().__init__(operator=MULTIPLICATION_OPERATOR,
                         operands=operands,
                         is_prioritized=is_prioritized)


class DivisionNode(ArithmeticOperationNode):

    def __init__(self,
                 lhs_operand: ArithmeticExpressionNode = None,
                 rhs_operand: ArithmeticExpressionNode = None,
                 is_prioritized: bool = False):
        super().__init__(operator=DIVISION_OPERATOR,
                         operands=[lhs_operand, rhs_operand],
                         is_prioritized=is_prioritized)


class ExponentiationNode(ArithmeticOperationNode):

    def __init__(self,
                 lhs_operand: ArithmeticExpressionNode = None,
                 rhs_operand: ArithmeticExpressionNode = None,
                 is_prioritized: bool = False):
        super().__init__(operator=EXPONENTIATION_OPERATOR,
                         operands=[lhs_operand, rhs_operand],
                         is_prioritized=is_prioritized)


class StringOperationNode(StringExpressionNode):

    def __init__(self,
                 operator: int,
                 operands: List[Union[StringExpressionNode]] = None,
                 is_prioritized: bool = False):
        super().__init__()
        self.operator: int = operator
        self.operands: List[Union[StringExpressionNode]] = operands
        self.is_prioritized = is_prioritized

    def __and__(self, other: StringExpressionNode):
        return self.concatenate(self, other)

    def __eq__(self, other: StringExpressionNode):
        return RelationalOperationNode.equal(self, other)

    def __ne__(self, other: StringExpressionNode):
        return RelationalOperationNode.not_equal(self, other)

    def __lt__(self, other: StringExpressionNode):
        return RelationalOperationNode.less_than(self, other)

    def __le__(self, other: StringExpressionNode):
        return RelationalOperationNode.less_than_or_equal(self, other)

    def __gt__(self, other: StringExpressionNode):
        return RelationalOperationNode.greater_than(self, other)

    def __ge__(self, other: StringExpressionNode):
        return RelationalOperationNode.greater_than_or_equal(self, other)

    def evaluate(self,
                 state: State,
                 idx_set: IndexingSet = None,
                 dummy_element: Element = None
                 ) -> np.ndarray:

        x_lhs = self.operands[0].evaluate(state, idx_set, dummy_element)
        y = x_lhs

        for i in range(1, len(self.operands)):

            x_rhs = self.operands[i].evaluate(state, idx_set, dummy_element)

            if self.operator == CONCATENATION_OPERATOR:  # concatenation
                y = np.char.add(x_lhs, x_rhs)
            else:
                raise ValueError("Unable to resolve symbol '{0}'".format(self.operator)
                                 + " as a string operator")

            x_lhs = y

        return y

    def to_lambda(self,
                  state: State,
                  idx_set_member: Element = None,
                  dummy_element: Element = None) -> Callable:

        args = np.array([o.to_lambda(state, idx_set_member, dummy_element) for o in self.operands])

        if self.operator == CONCATENATION_OPERATOR:  # concatenation
            return partial(lambda x: ''.join([x_i() for x_i in x]), args)
        else:
            raise ValueError("Unable to resolve symbol '{0}'".format(self.operator)
                             + " as a string operator")

    @staticmethod
    def concatenate(lhs_operand: StringExpressionNode, rhs_operand: StringExpressionNode):

        if isinstance(lhs_operand, StringOperationNode) and lhs_operand.operator == CONCATENATION_OPERATOR:

            if isinstance(rhs_operand, StringOperationNode) and rhs_operand.operator == CONCATENATION_OPERATOR:
                lhs_operand.operands.extend(rhs_operand.operands)
            else:
                lhs_operand.operands.append(rhs_operand)

            return lhs_operand

        elif isinstance(rhs_operand, StringOperationNode) and rhs_operand.operator == CONCATENATION_OPERATOR:
            rhs_operand.operands.insert(0, lhs_operand)
            return rhs_operand

        else:
            return StringOperationNode(operator=CONCATENATION_OPERATOR,
                                       operands=[lhs_operand, rhs_operand])

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
                s += " {0} {1}".format(AMPL_OPERATOR_SYMBOLS[self.operator], operand)
        if self.is_prioritized:
            return '(' + s + ')'
        else:
            return s
