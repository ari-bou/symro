from functools import partial
from typing import Dict, List, Optional, Tuple, Union

from symro.core.mat.entity import Parameter, Variable
from symro.core.mat.exprn import ExpressionNode, LogicalExpressionNode, SetExpressionNode, ArithmeticExpressionNode, \
    StringExpressionNode
from symro.core.mat.util import IndexSet, IndexSetMember
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
                                    DummyNode] = None,
                 id: int = 0):
        super().__init__(id)
        self.operator: str = operator
        self.lhs_operand: Optional[Union[ArithmeticExpressionNode,
                                         StringExpressionNode,
                                         DummyNode]] = lhs_operand
        self.rhs_operand: Optional[Union[ArithmeticExpressionNode,
                                         StringExpressionNode,
                                         DummyNode]] = rhs_operand

    def evaluate(self,
                 state: State,
                 idx_set: IndexSet = None,
                 dummy_symbols: Tuple[str, ...] = None) -> List[bool]:

        lhs_arg = self.lhs_operand.evaluate(state, idx_set, dummy_symbols)
        rhs_arg = self.rhs_operand.evaluate(state, idx_set, dummy_symbols)

        # Equality
        if self.operator in ['=', "=="]:
            return [l == r for l, r in zip(lhs_arg, rhs_arg)]

        # Inequality
        elif self.operator in ['!=', "<>"]:
            return [l != r for l, r in zip(lhs_arg, rhs_arg)]

        # Greater than
        elif self.operator == '>':
            return [l > r for l, r in zip(lhs_arg, rhs_arg)]

        # Greater than or equal to
        elif self.operator == '>=':
            return [l >= r for l, r in zip(lhs_arg, rhs_arg)]

        # Less than
        elif self.operator == '<':
            return [l < r for l, r in zip(lhs_arg, rhs_arg)]

        # Less than or equal to
        elif self.operator == '<=':
            return [l <= r for l, r in zip(lhs_arg, rhs_arg)]

        else:
            raise ValueError("Unable to resolve operator '{0}' as a relational operator".format(self.operator))

    def to_lambda(self,
                  state: State,
                  idx_set_member: IndexSetMember = None,
                  dummy_symbols: Tuple[str, ...] = None):

        lhs_arg = self.lhs_operand.to_lambda(state, idx_set_member, dummy_symbols)
        rhs_arg = self.rhs_operand.to_lambda(state, idx_set_member, dummy_symbols)

        # Equality
        if self.operator in ['=', "=="]:
            return partial(lambda l, r: l() == r(), lhs_arg, rhs_arg)

        # Inequality
        elif self.operator in ['!=', "<>"]:
            return partial(lambda l, r: l() != r(), lhs_arg, rhs_arg)

        # Greater than
        elif self.operator == '>':
            return partial(lambda l, r: l() > r(), lhs_arg, rhs_arg)

        # Greater than or equal to
        elif self.operator == '>=':
            return partial(lambda l, r: l() >= r(), lhs_arg, rhs_arg)

        # Less than
        elif self.operator == '<':
            return partial(lambda l, r: l() < r(), lhs_arg, rhs_arg)

        # Less than or equal to
        elif self.operator == '<=':
            return partial(lambda l, r: l() <= r(), lhs_arg, rhs_arg)

        else:
            raise ValueError("Unable to resolve operator '{0}' as a relational operator".format(self.operator))

    def collect_declared_entities(self,
                                  state: State,
                                  idx_set: IndexSet = None,
                                  dummy_symbols: Tuple[str, ...] = None) -> Dict[str, Union[Parameter, Variable]]:
        entities = self.lhs_operand.collect_declared_entities(state, idx_set, dummy_symbols)
        entities.update(self.rhs_operand.collect_declared_entities(state, idx_set, dummy_symbols))
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
                 set_node: SetExpressionNode = None,
                 id: int = 0):
        super().__init__(id)
        self.operator: str = operator  # 'in' or 'not in'
        self.member_node: Optional[Union[BaseDummyNode, ArithmeticExpressionNode, StringExpressionNode]] = member_node
        self.set_node: Optional[SetExpressionNode] = set_node

    def evaluate(self,
                 state: State,
                 idx_set: IndexSet = None,
                 dummy_symbols: Tuple[str, ...] = None) -> List[bool]:

        if idx_set is None:
            raise ValueError("Indexing set of a set membership operation cannot be null")

        challenge_elements = self.member_node.evaluate(state, idx_set, dummy_symbols)
        if self.member_node.get_dim() == 1:
            challenge_elements = [tuple([e]) for e in challenge_elements]

        sets_c = self.set_node.evaluate(state, idx_set, dummy_symbols)

        result = []
        for chlg_element, set_c in zip(challenge_elements, sets_c):
            result.append(chlg_element in set_c)

        if self.operator == "not in":
            result = [not r for r in result]

        return result

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
                 rhs_operand: SetExpressionNode = None,
                 id: int = 0):
        super().__init__(id)
        self.operator: str = operator
        self.lhs_operand: Optional[SetExpressionNode] = lhs_operand
        self.rhs_operand: Optional[SetExpressionNode] = rhs_operand

    def evaluate(self,
                 state: State,
                 idx_set: IndexSet = None,
                 dummy_symbols: Tuple[str, ...] = None) -> List[bool]:
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
                 operand: ExpressionNode = None,
                 id: int = 0):

        super().__init__(id)
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
                 operand: LogicalExpressionNode = None,
                 id: int = 0):
        super().__init__(id)
        self.operator: str = operator
        self.operand: Optional[LogicalExpressionNode] = operand

    def evaluate(self,
                 state: State,
                 idx_set: IndexSet = None,
                 dummy_symbols: Tuple[str, ...] = None) -> List[bool]:

        arg = self.operand.evaluate(state, idx_set, dummy_symbols)

        # Logical Negation
        if self.operator in ['!', "not"]:
            return [not r for r in arg]

        else:
            raise ValueError("Unable to resolve operator '{0}' as a unary logical operator".format(self.operator))

    def to_lambda(self,
                  state: State,
                  idx_set_member: IndexSetMember = None,
                  dummy_symbols: Tuple[str, ...] = None):

        arg = self.operand.to_lambda(state, idx_set_member, dummy_symbols)

        # Logical Negation
        if self.operator in ['!', "not"]:
            return partial(lambda o: not o(), arg)

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
                 rhs_operand: LogicalExpressionNode = None,
                 id: int = 0):
        super().__init__(id)
        self.operator: str = operator
        self.lhs_operand: Optional[LogicalExpressionNode] = lhs_operand
        self.rhs_operand: Optional[LogicalExpressionNode] = rhs_operand

    def evaluate(self,
                 state: State,
                 idx_set: IndexSet = None,
                 dummy_symbols: Tuple[str, ...] = None) -> List[bool]:

        lhs_arg = self.lhs_operand.evaluate(state, idx_set, dummy_symbols)
        rhs_arg = self.rhs_operand.evaluate(state, idx_set, dummy_symbols)

        # Logical Conjunction
        if self.operator in ["&&", "and"]:
            return [l and r for l, r in zip(lhs_arg, rhs_arg)]

        # Logical Disjunction
        elif self.operator in ["||", "or"]:
            return [l or r for l, r in zip(lhs_arg, rhs_arg)]

        else:
            raise ValueError("Unable to resolve operator '{0}' as a binary logical operator".format(self.operator))

    def to_lambda(self,
                  state: State,
                  idx_set_member: IndexSetMember = None,
                  dummy_symbols: Tuple[str, ...] = None):

        lhs_arg = self.lhs_operand.to_lambda(state, idx_set_member, dummy_symbols)
        rhs_arg = self.rhs_operand.to_lambda(state, idx_set_member, dummy_symbols)

        if self.operator in ["&&", "and"]:  # Logical Conjunction
            return partial(lambda l, r: l() and r(), lhs_arg, rhs_arg)
        elif self.operator in ["||", "or"]:  # Logical Disjunction
            return partial(lambda l, r: l() or r(), lhs_arg, rhs_arg)
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
                 operands: List[LogicalExpressionNode] = None,
                 id: int = 0):
        super().__init__(id)
        self.operator: str = operator
        self.operands: Optional[List[LogicalExpressionNode]] = operands

    def evaluate(self,
                 state: State,
                 idx_set: IndexSet = None,
                 dummy_symbols: Tuple[str, ...] = None
                 ) -> List[float]:

        x_lhs = self.operands[0].evaluate(state, idx_set, dummy_symbols)
        y = x_lhs

        for i in range(1, len(self.operands)):

            x_rhs = self.operands[i].evaluate(state, idx_set, dummy_symbols)

            if self.operator in ["&&", "and"]:  # Logical Conjunction
                y = [x_lhs_i and x_rhs_i for x_lhs_i, x_rhs_i in zip(x_lhs, x_rhs)]
            elif self.operator in ["||", "or"]:  # Logical Disjunction
                y = [x_lhs_i or x_rhs_i for x_lhs_i, x_rhs_i in zip(x_lhs, x_rhs)]
            else:
                raise ValueError("Unable to resolve symbol '{0}'"
                                 " as a multi logical operator".format(self.operator))

            x_lhs = y

        return y

    def to_lambda(self,
                  state: State,
                  idx_set_member: IndexSetMember = None,
                  dummy_symbols: Tuple[str, ...] = None):

        args_all = []
        for operand in self.operands:
            args_all.append(operand.to_lambda(state, idx_set_member, dummy_symbols))

        if self.operator in ["&&", "and"]:  # Logical Conjunction
            return partial(lambda x: all([x_i() for x_i in x]), args_all)
        elif self.operator in ["||", "or"]:  # Logical Disjunction
            return partial(lambda x: any([x_i() for x_i in x]), args_all)
        else:
            raise ValueError("Unable to resolve symbol '{0}'"
                             " as a multi logical operator".format(self.operator))

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
