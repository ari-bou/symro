from functools import partial
import numpy as np
from typing import Callable, Dict, List, Optional, Tuple, Union

from symro.core.mat.entity import Parameter, Variable
from symro.core.mat.exprn import ExpressionNode, LogicalExpressionNode, SetExpressionNode, ArithmeticExpressionNode, \
    StringExpressionNode
from symro.core.mat.util import IndexingSet, Element
from symro.core.mat.dummyn import BaseDummyNode, DummyNode
from symro.core.mat.setn import CompoundSetNode
from symro.core.mat.state import State


class RelationalOperationNode(LogicalExpressionNode):

    def __init__(self,
                 operator: str,
                 lhs_operand: Union[ArithmeticExpressionNode,
                                    StringExpressionNode,
                                    DummyNode] = None,
                 rhs_operand: Union[ArithmeticExpressionNode,
                                    StringExpressionNode,
                                    DummyNode] = None):
        super().__init__()
        self.operator: str = operator
        self.lhs_operand: Optional[Union[ArithmeticExpressionNode,
                                         StringExpressionNode,
                                         DummyNode]] = lhs_operand
        self.rhs_operand: Optional[Union[ArithmeticExpressionNode,
                                         StringExpressionNode,
                                         DummyNode]] = rhs_operand

    def evaluate(self,
                 state: State,
                 idx_set: IndexingSet = None,
                 dummy_element: Element = None) -> np.ndarray:

        x_lhs = self.lhs_operand.evaluate(state, idx_set, dummy_element)
        x_rhs = self.rhs_operand.evaluate(state, idx_set, dummy_element)

        # equality
        if self.operator in ['=', "=="]:
            return x_lhs == x_rhs

        # strict inequality
        elif self.operator in ['!=', "<>"]:
            return x_lhs != x_rhs

        # greater than
        elif self.operator == '>':
            return x_lhs > x_rhs

        # greater than or equal to
        elif self.operator == '>=':
            return x_lhs >= x_rhs

        # less than
        elif self.operator == '<':
            return x_lhs < x_rhs

        # less than or equal to
        elif self.operator == '<=':
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
        if self.operator in ['=', "=="]:
            return partial(lambda l, r: l() == r(), x_lhs, x_rhs)

        # strict inequality
        elif self.operator in ['!=', "<>"]:
            return partial(lambda l, r: l() != r(), x_lhs, x_rhs)

        # greater than
        elif self.operator == '>':
            return partial(lambda l, r: l() > r(), x_lhs, x_rhs)

        # greater than or equal to
        elif self.operator == '>=':
            return partial(lambda l, r: l() >= r(), x_lhs, x_rhs)

        # less than
        elif self.operator == '<':
            return partial(lambda l, r: l() < r(), x_lhs, x_rhs)

        # less than or equal to
        elif self.operator == '<=':
            return partial(lambda l, r: l() <= r(), x_lhs, x_rhs)

        else:
            raise ValueError("Unable to resolve operator '{0}' as a relational operator".format(self.operator))

    def collect_declared_entities(self,
                                  state: State,
                                  idx_set: IndexingSet = None,
                                  dummy_element: Element = None) -> Dict[str, Union[Parameter, Variable]]:
        entities = self.lhs_operand.collect_declared_entities(state, idx_set, dummy_element)
        entities.update(self.rhs_operand.collect_declared_entities(state, idx_set, dummy_element))
        return entities

    def add_operand(self, operand: Union[ArithmeticExpressionNode,
                                         StringExpressionNode,
                                         DummyNode] = None):
        if self.lhs_operand is None:
            self.lhs_operand = operand
        else:
            self.rhs_operand = operand

    def get_children(self) -> List[ExpressionNode]:
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


class SetMembershipOperationNode(LogicalExpressionNode):

    def __init__(self,
                 operator: str,
                 member_node: Union[BaseDummyNode, ArithmeticExpressionNode, StringExpressionNode] = None,
                 set_node: SetExpressionNode = None):
        super().__init__()
        self.operator: str = operator  # 'in' or 'not in'
        self.member_node: Optional[Union[BaseDummyNode, ArithmeticExpressionNode, StringExpressionNode]] = member_node
        self.set_node: Optional[SetExpressionNode] = set_node

    def evaluate(self,
                 state: State,
                 idx_set: IndexingSet = None,
                 dummy_element: Element = None) -> np.ndarray:

        if idx_set is None:
            raise ValueError("Indexing set of a set membership operation cannot be null")

        challenge_elements = self.member_node.evaluate(state, idx_set, dummy_element)
        if self.member_node.get_dim() == 1:
            challenge_elements = np.array([tuple([e]) for e in challenge_elements])

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


class UnaryLogicalOperationNode(LogicalExpressionNode):

    def __init__(self,
                 operator: str,
                 operand: LogicalExpressionNode = None):
        super().__init__()
        self.operator: str = operator
        self.operand: Optional[LogicalExpressionNode] = operand

    def evaluate(self,
                 state: State,
                 idx_set: IndexingSet = None,
                 dummy_element: Element = None) -> np.ndarray:

        x = self.operand.evaluate(state, idx_set, dummy_element)

        # logical negation
        if self.operator in ['!', "not"]:
            return ~x

        else:
            raise ValueError("Unable to resolve operator '{0}' as a unary logical operator".format(self.operator))

    def to_lambda(self,
                  state: State,
                  idx_set_member: Element = None,
                  dummy_element: Element = None) -> Callable:

        x = self.operand.to_lambda(state, idx_set_member, dummy_element)

        # logical negation
        if self.operator in ['!', "not"]:
            return partial(lambda o: not o(), x)

        else:
            raise ValueError("Unable to resolve operator '{0}' as a unary logical operator".format(self.operator))

    def add_operand(self, operand: Optional[LogicalExpressionNode]):
        self.operand = operand

    def get_children(self) -> List:
        return [self.operand]

    def set_children(self, operands: list):
        if len(operands) > 0:
            self.operand = operands[0]

    def get_literal(self) -> str:
        literal = "{0} {1}".format(self.operator, self.operand)
        if self.is_prioritized:
            return '(' + literal + ')'
        return literal


class BinaryLogicalOperationNode(LogicalExpressionNode):

    def __init__(self,
                 operator: str,
                 lhs_operand: LogicalExpressionNode = None,
                 rhs_operand: LogicalExpressionNode = None):
        super().__init__()
        self.operator: str = operator
        self.lhs_operand: Optional[LogicalExpressionNode] = lhs_operand
        self.rhs_operand: Optional[LogicalExpressionNode] = rhs_operand

    def evaluate(self,
                 state: State,
                 idx_set: IndexingSet = None,
                 dummy_element: Element = None) -> np.ndarray:

        x_lhs = self.lhs_operand.evaluate(state, idx_set, dummy_element)
        x_rhs = self.rhs_operand.evaluate(state, idx_set, dummy_element)

        # logical conjunction
        if self.operator in ["&&", "and"]:
            return x_lhs & x_rhs

        # logical disjunction
        elif self.operator in ["||", "or"]:
            return x_lhs | x_rhs

        else:
            raise ValueError("Unable to resolve operator '{0}' as a binary logical operator".format(self.operator))

    def to_lambda(self,
                  state: State,
                  idx_set_member: Element = None,
                  dummy_element: Tuple[str, ...] = None):

        x_lhs = self.lhs_operand.to_lambda(state, idx_set_member, dummy_element)
        x_rhs = self.rhs_operand.to_lambda(state, idx_set_member, dummy_element)

        if self.operator in ["&&", "and"]:  # logical conjunction
            return partial(lambda l, r: l() and r(), x_lhs, x_rhs)
        elif self.operator in ["||", "or"]:  # logical disjunction
            return partial(lambda l, r: l() or r(), x_lhs, x_rhs)
        else:
            raise ValueError("Unable to resolve operator '{0}' as a binary logical operator".format(self.operator))

    def add_operand(self, operand: Optional[ExpressionNode]):
        if self.lhs_operand is None:
            self.lhs_operand = operand
        else:
            self.rhs_operand = operand

    def get_children(self) -> List[ExpressionNode]:
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


class MultiLogicalOperationNode(LogicalExpressionNode):

    def __init__(self,
                 operator: str,
                 operands: List[LogicalExpressionNode] = None):
        super().__init__()
        self.operator: str = operator
        self.operands: Optional[List[LogicalExpressionNode]] = operands

    def evaluate(self,
                 state: State,
                 idx_set: IndexingSet = None,
                 dummy_element: Element = None
                 ) -> np.ndarray:

        x_lhs = self.operands[0].evaluate(state, idx_set, dummy_element)
        y = x_lhs

        for i in range(1, len(self.operands)):

            x_rhs = self.operands[i].evaluate(state, idx_set, dummy_element)

            if self.operator in ["&&", "and"]:  # logical conjunction
                y = x_lhs & x_rhs
            elif self.operator in ["||", "or"]:  # logical disjunction
                y = x_lhs | x_rhs
            else:
                raise ValueError("Unable to resolve symbol '{0}'".format(self.operator)
                                 + " as a multi logical operator")

            x_lhs = y

        return y

    def to_lambda(self,
                  state: State,
                  idx_set_member: Element = None,
                  dummy_element: Element = None) -> Callable:

        args = np.array([o.to_lambda(state, idx_set_member, dummy_element) for o in self.operands])

        if self.operator in ["&&", "and"]:  # logical conjunction
            return partial(lambda x: all([x_i() for x_i in x]), args)
        elif self.operator in ["||", "or"]:  # logical disjunction
            return partial(lambda x: any([x_i() for x_i in x]), args)
        else:
            raise ValueError("Unable to resolve symbol '{0}'".format(self.operator)
                             + " as a multi logical operator")

    def get_children(self) -> List[ExpressionNode]:
        return self.operands

    def set_children(self, operands: list):
        self.operands.clear()
        self.operands.extend(operands)

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
