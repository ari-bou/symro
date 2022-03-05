from typing import Callable, List, Optional, Tuple, Union

import numpy as np

from symro.src.mat.constants import *
from symro.src.mat.types import Element, IndexingSet
from symro.src.mat.exprn import ExpressionNode, LogicalExpressionNode, SetExpressionNode
from symro.src.mat.opern import SetOperationNode
from symro.src.mat.setn import CompoundSetNode
from symro.src.mat.state import State


class SetConditionalNode(SetExpressionNode):

    def __init__(self,
                 operands: List[SetExpressionNode],
                 conditions: List[Optional[LogicalExpressionNode]]):
        super().__init__()
        self.operands: List[SetExpressionNode] = operands
        self.conditions: List[Optional[LogicalExpressionNode]] = conditions

    def __and__(self, other: SetExpressionNode):
        return SetOperationNode.intersection(self, other)

    def __or__(self, other: SetExpressionNode):
        return SetOperationNode.union(self, other)

    def __sub__(self, other: SetExpressionNode):
        return SetOperationNode.difference(self, other)

    def __xor__(self, other: SetExpressionNode):
        return SetOperationNode.symmetric_difference(self, other)

    def evaluate(self,
                 state: State,
                 idx_set: IndexingSet = None,
                 dummy_element: Element = None) -> np.ndarray:
        raise NotImplementedError("evaluate method has not yet been implemented for '{0}'".format(type(self)))

    def to_lambda(self,
                  state: State,
                  idx_set_member: Element = None,
                  dummy_element: Tuple[str, ...] = None) -> Callable:
        raise NotImplementedError("to_lambda method has not yet been implemented for '{0}'".format(type(self)))

    @staticmethod
    def generate_negated_combined_condition(conditions: List[str]):
        other_conditions_par = ["(" + c + ")" for c in conditions]
        combined_condition = " or ".join(other_conditions_par)
        return "!({0})".format(combined_condition)

    def add_operand(self, operand: SetExpressionNode):
        self.operands.append(operand)

    def add_condition(self, condition: LogicalExpressionNode = None):
        if condition is not None:
            self.conditions.append(condition)

    def has_trailing_else_clause(self) -> bool:
        return len(self.operands) == len(self.conditions) + 1

    def get_dim(self, state: State) -> int:
        return self.operands[0].get_dim(state)

    def get_children(self) -> List[Union[SetExpressionNode, LogicalExpressionNode]]:
        children = []
        children.extend(self.operands)
        for condition in self.conditions:
            if condition is not None:
                children.append(condition)
        return children

    def set_children(self, operands: List[Union[SetExpressionNode, LogicalExpressionNode]]):
        count = len(operands)
        if count % 2 == 0:
            operand_count = int(count / 2)
        else:
            operand_count = int((count + 1) / 2)
        self.operands = operands[:operand_count]
        self.conditions = operands[operand_count:]

    def get_literal(self) -> str:
        rhs = ""
        for i, operand in enumerate(self.operands):
            if i == 0:
                rhs += "if {0} then {1}".format(self.conditions[i], operand)
            elif i == len(self.operands) - 1 and self.has_trailing_else_clause():
                rhs += " else {0}".format(operand)
            else:
                rhs += " else if {0} then {1}".format(self.conditions[i], operand)
        return rhs


class SetReductionNode(SetExpressionNode):

    def __init__(self,
                 operator: int,
                 idx_set_node: CompoundSetNode,
                 operand: SetExpressionNode = None):

        super().__init__()
        self.operator: int = operator
        self.idx_set_node: CompoundSetNode = idx_set_node
        self.operand: SetExpressionNode = operand

    def __and__(self, other: SetExpressionNode):
        return SetOperationNode.intersection(self, other)

    def __or__(self, other: SetExpressionNode):
        return SetOperationNode.union(self, other)

    def __sub__(self, other: SetExpressionNode):
        return SetOperationNode.difference(self, other)

    def __xor__(self, other: SetExpressionNode):
        return SetOperationNode.symmetric_difference(self, other)

    def evaluate(self,
                 state: State,
                 idx_set: IndexingSet = None,
                 dummy_element: Element = None) -> np.ndarray:
        raise NotImplementedError("evaluate method has not yet been implemented for '{0}'".format(type(self)))

    def to_lambda(self,
                  state: State,
                  idx_set_member: Element = None,
                  dummy_element: Tuple[str, ...] = None) -> Callable:
        raise NotImplementedError("to_lambda method has not yet been implemented for '{0}'".format(type(self)))

    def get_dim(self, state: State) -> int:
        return self.operand.get_dim(state)

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
        literal = AMPL_OPERATOR_SYMBOLS[self.operator] + str(self.idx_set_node) + str(self.operand)
        if self.is_prioritized:
            literal = '(' + literal + ')'
        return literal
