from typing import List, Optional, Tuple, Union

from symro.core.mat.exprn import ExpressionNode, LogicalExpressionNode, SetExpressionNode
from symro.core.mat.setn import CompoundSetNode
from symro.core.mat.util import IndexSet
from symro.core.mat.state import State


class ConditionalSetExpressionNode(SetExpressionNode):

    def __init__(self, id: int = 0):
        super().__init__(id)
        self.operands: List[SetExpressionNode] = []
        self.conditions: List[Optional[LogicalExpressionNode]] = []

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
        else:
            self.conditions.append(None)

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
        self.conditions = [None] * operand_count
        self.conditions[:operand_count] = operands[operand_count:]

    def get_literal(self) -> str:
        rhs = ""
        for i, operand in enumerate(self.operands):
            if i == 0:
                rhs += "if {0} then {1}".format(self.conditions[i], operand)
            elif i == len(self.operands) - 1 and self.conditions[i] is None:
                rhs += " else {0}".format(operand)
            else:
                rhs += " else if {0} then {1}".format(self.conditions[i], operand)
        return rhs


class SetReductionOperationNode(SetExpressionNode):

    def __init__(self,
                 symbol: str,
                 idx_set_node: CompoundSetNode,
                 operand: SetExpressionNode = None,
                 id: int = 0):

        super().__init__(id)
        self.symbol: str = symbol
        self.idx_set_node: CompoundSetNode = idx_set_node
        self.operand: SetExpressionNode = operand

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
        literal = self.symbol + str(self.idx_set_node) + str(self.operand)
        if self.is_prioritized:
            literal = '(' + literal + ')'
        return literal


class BinarySetOperationNode(SetExpressionNode):

    def __init__(self,
                 operator: str,
                 lhs_operand: SetExpressionNode = None,
                 rhs_operand: SetExpressionNode = None,
                 id: int = 0):
        super().__init__(id)
        self.operator: str = operator
        self.lhs_operand: SetExpressionNode = lhs_operand
        self.rhs_operand: SetExpressionNode = rhs_operand

    def is_constant(self) -> bool:
        return True

    def is_null(self) -> bool:
        return False

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
        literal = "{0} {1} {2}".format(self.lhs_operand, self.operator, self.rhs_operand)
        if self.is_prioritized:
            return '(' + literal + ')'
        return literal

    def evaluate(self,
                 state: State,
                 idx_set: IndexSet = None,
                 dummy_symbols: Tuple[str, ...] = None
                 ) -> List[IndexSet]:

        lhs_arg = self.lhs_operand.evaluate(state, idx_set, dummy_symbols)
        rhs_arg = self.lhs_operand.evaluate(state, idx_set, dummy_symbols)

        if self.operator == "union":
            return [l.union(r) for l, r in zip(lhs_arg, rhs_arg)]

        elif self.operator == "inter":
            return [l.intersection(r) for l, r in zip(lhs_arg, rhs_arg)]

        elif self.operator == "diff":
            return [l.difference(r) for l, r in zip(lhs_arg, rhs_arg)]

        else:
            raise ValueError("Unable to resolve operator '{0}' as a binary set operator".format(self.operator))
