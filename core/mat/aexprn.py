from functools import partial
import numpy as np
from ordered_set import OrderedSet
from typing import Dict, List, Optional, Tuple, Union

from symro.core.mat.entity import Parameter, Variable, Objective, Constraint
from symro.core.mat.exprn import LogicalExpressionNode, ArithmeticExpressionNode
from symro.core.mat.dummyn import CompoundDummyNode
from symro.core.mat.setn import CompoundSetNode
from symro.core.mat.util import IndexingSet, Element
from symro.core.mat.util import cartesian_product
from symro.core.mat.state import State
import symro.core.constants as const


class NumericNode(ArithmeticExpressionNode):

    def __init__(self,
                 value: Union[int, float, str],
                 sci_not: bool = False,
                 coeff_sym: str = None,
                 power_sign: str = None,
                 power_sym: str = None,
                 id: int = 0):

        super().__init__(id)

        self.value: Union[int, float] = 0
        self.sci_not: bool = sci_not
        self.__is_null: bool = False

        if not self.sci_not:
            self.value = float(value)
        else:
            coeff = float(coeff_sym)
            power = float(power_sign + power_sym)
            self.value = coeff * (10 ** power)
        if isinstance(self.value, float):
            if self.value.is_integer():
                self.value = int(self.value)

    def evaluate(self,
                 state: State,
                 idx_set: IndexingSet = None,
                 dummy_symbols: Tuple[str, ...] = None
                 ) -> List[float]:
        mp = 1
        if idx_set is not None:
            mp = len(idx_set)
        return [self.value] * mp

    def to_lambda(self,
                  state: State,
                  idx_set_member: Element = None,
                  dummy_symbols: Tuple[str, ...] = None):
        return partial(lambda c: c, self.value)

    def is_constant(self) -> bool:
        return True

    def is_null(self) -> bool:
        return self.__is_null

    def nullify(self):
        self.value = 0
        self.__is_null = True

    def get_children(self) -> list:
        return []

    def set_children(self, children: list):
        pass

    def get_literal(self) -> str:
        if self.value == np.inf:
            literal = "Infinity"
        elif self.value == -np.inf:
            literal = "-Infinity"
        elif not self.sci_not:
            literal = str(self.value)
        else:
            literal = "{:e}".format(self.value)
        if self.is_prioritized:
            return '(' + literal + ')'
        return literal


class DeclaredEntityNode(ArithmeticExpressionNode):

    def __init__(self,
                 symbol: str,
                 entity_index_node: CompoundDummyNode = None,
                 suffix: str = None,
                 type: str = None,
                 id: int = 0):

        super().__init__(id)

        self.symbol: str = symbol
        self.idx_node: CompoundDummyNode = entity_index_node
        self.suffix: str = suffix
        self.__entity_type: str = type
        self.__is_null: bool = False

    def evaluate(self,
                 state: State,
                 idx_set: IndexingSet = None,
                 dummy_symbols: Tuple[str, ...] = None
                 ) -> List[float]:

        indices = None
        if self.is_indexed():
            indices = self.idx_node.evaluate(state, idx_set, dummy_symbols)

        count_p = 1
        if idx_set is not None:
            count_p = len(idx_set)

        values = []
        for ip in range(count_p):
            index = None
            if indices is not None:
                index = indices[ip]
            entity: Union[Parameter, Variable, Objective, Constraint] = self.__get_entity(state, index)
            values.append(entity.value)

        return values

    def __get_entity(self,
                     state: State,
                     entity_index: Tuple[Union[int, float, str], ...] = None):
        if entity_index is None:
            entity = state.get_entity(self.symbol)
        else:
            entity = state.get_entity(self.symbol, entity_index)
        return entity

    def to_lambda(self,
                  state: State,
                  idx_set_member: Element = None,
                  dummy_symbols: Tuple[str, ...] = None):
        index = None
        if self.is_indexed():
            index = self.idx_node.to_lambda(state, idx_set_member, dummy_symbols)()
        entity = self.__get_entity(state, index)
        return partial(lambda e: e.value, entity)

    def collect_declared_entities(self,
                                  state: State,
                                  idx_set: IndexingSet = None,
                                  dummy_symbols: Tuple[str, ...] = None) -> Dict[str, Union[Parameter, Variable]]:
        entities = {}

        entity_indices = None
        if self.is_indexed():
            entity_indices = self.idx_node.evaluate(state, idx_set, dummy_symbols)

        mp = 1
        if idx_set is not None:
            mp = len(idx_set)

        for ip in range(mp):  # Iterate over elements in indexing set

            # Scalar entity
            if entity_indices is None:
                entity_index = tuple([])
                is_dim_aggregated = []

            # Indexed entity
            else:
                entity_index_list = list(entity_indices[ip])
                is_dim_aggregated = [False] * len(entity_index_list)
                for j, idx in enumerate(entity_index_list):
                    if isinstance(idx, tuple):
                        entity_index_list[j] = idx[0]
                        is_dim_aggregated[j] = True
                entity_index = tuple(entity_index_list)

            if not self.is_constant():
                var = Variable(symbol=self.symbol,
                               idx=entity_index,
                               is_dim_aggregated=is_dim_aggregated)
                entities[var.entity_id] = var
            else:
                param = Parameter(symbol=self.symbol,
                                  idx=entity_index,
                                  is_dim_aggregated=is_dim_aggregated)
                entities[param.entity_id] = param

        return entities

    def is_indexed(self) -> bool:
        return self.idx_node is not None

    def is_constant(self) -> bool:
        return self.__entity_type == const.PARAM_TYPE

    def is_null(self) -> bool:
        return self.__is_null

    def nullify(self):
        self.__is_null = True

    def get_type(self) -> str:
        return self.__entity_type

    def set_type(self, entity_type: str):
        self.__entity_type = entity_type

    def get_children(self) -> list:
        if self.idx_node is not None:
            return [self.idx_node]
        return []

    def set_children(self, children: list):
        if len(children) > 0:
            self.idx_node = children[0]

    def get_literal(self) -> str:
        literal = self.symbol
        if self.is_indexed():
            literal += "[{0}]".format(','.join([str(n) for n in self.idx_node.component_nodes]))
        if self.suffix is not None:
            literal += ".{0}".format(self.suffix)
        if self.is_prioritized:
            literal = '(' + literal + ')'
        return literal


class ArithmeticTransformationNode(ArithmeticExpressionNode):

    def __init__(self,
                 symbol: str,
                 idx_set_node: CompoundSetNode = None,
                 operands: Union[ArithmeticExpressionNode, List[ArithmeticExpressionNode]] = None,
                 id: int = 0):

        super().__init__(id)
        self.symbol: str = symbol
        self.idx_set_node: Optional[CompoundSetNode] = idx_set_node
        self.operands: List[ArithmeticExpressionNode] = []

        if self.symbol in ["sum", "prod"]:
            self.is_prioritized = True

        if operands is not None:
            if isinstance(operands, ArithmeticExpressionNode):
                self.operands.append(operands)
            else:
                self.operands = operands

    def evaluate(self,
                 state: State,
                 idx_set: IndexingSet = None,
                 dummy_symbols: Tuple[str, ...] = None
                 ) -> List[float]:
        if self.is_reductive():
            return self.__evaluate_reductive_function(state, idx_set, dummy_symbols)
        else:
            return self.__evaluate_non_reductive_function(state, idx_set, dummy_symbols)

    def __evaluate_reductive_function(self,
                                      state: State,
                                      idx_set: IndexingSet = None,
                                      dummy_symbols: Tuple[str, ...] = None) -> List[float]:

        combined_idx_sets = self.combine_indexing_sets(state, idx_set, dummy_symbols)  # length mp
        combined_dummy_syms = self.idx_set_node.combined_dummy_syms

        mp = 1
        if idx_set is not None:
            mp = len(idx_set)

        result_values = []

        for ip in range(mp):
            x = self.operands[0].evaluate(state, combined_idx_sets[ip], combined_dummy_syms)
            if self.symbol == "sum":  # Reductive Summation
                y = sum(x)
            elif self.symbol == "prod":  # Reductive Multiplication
                y = np.prod(x)
            else:
                raise ValueError("Unable to resolve symbol '{0}'"
                                 " as a reductive arithmetic function".format(self.symbol))
            result_values.append(y)
        return result_values

    def __evaluate_non_reductive_function(self,
                                          state: State,
                                          idx_set: IndexingSet = None,
                                          dummy_symbols: Tuple[str, ...] = None) -> List[float]:

        x = [o.evaluate(state, idx_set, dummy_symbols) for o in self.operands]

        # Single Argument
        if len(self.operands) == 1:
            x_0 = x[0]
            if self.symbol == "div":
                y = np.divide(x[0], x[1])
                y = np.around(y)
            elif self.symbol == "mod":
                y = np.mod(x[0], x[1])
            elif self.symbol == "sin":  # Sine
                y = np.sin(x_0)
            elif self.symbol == "cos":  # Cosine
                y = np.cos(x_0)
            elif self.symbol == "exp":  # Exponential
                y = np.exp(x_0)
            elif self.symbol == "log":  # Natural Logarithm
                y = np.log(x_0)
            elif self.symbol == "log10":  # Logarithm Base 10
                y = np.log10(x_0)
            else:
                raise ValueError("Unable to resolve symbol '{0}'"
                                 " as a single-argument arithmetic function".format(self.symbol))

        # Multiple Arguments
        else:
            if self.symbol == "max":  # Maximum
                y = max(*x)
            elif self.symbol == "min":  # Minimum
                y = min(*x)
            else:
                raise ValueError("Unable to resolve symbol '{0}'"
                                 " as a multi-argument arithmetic function".format(self.symbol))

        return list(y)

    def to_lambda(self,
                  state: State,
                  idx_set_member: Element = None,
                  dummy_symbols: Tuple[str, ...] = None):

        if self.is_reductive():

            if idx_set_member is None:
                idx_set = None
            else:
                idx_set = OrderedSet([idx_set_member])
            combined_idx_sets = self.combine_indexing_sets(state, idx_set, dummy_symbols)  # length 1
            combined_dummy_syms = self.idx_set_node.combined_dummy_syms

            arg_0 = []
            for combined_idx_set_member in combined_idx_sets[0]:
                arg_0.append(self.operands[0].to_lambda(state,
                                                        combined_idx_set_member,
                                                        combined_dummy_syms))

            if self.symbol == "sum":  # Reductive Summation
                return partial(lambda x: sum([x_i() for x_i in x]), arg_0)
            elif self.symbol == "prod":  # Reductive Multiplication
                return partial(lambda x: np.prod([x_i() for x_i in x]), arg_0)
            else:
                raise ValueError("Unable to resolve symbol '{0}'"
                                 " as a reductive arithmetic function".format(self.symbol))

        else:

            args = [o.to_lambda(state, idx_set_member, dummy_symbols) for o in self.operands]

            # Single Argument
            if len(self.operands) == 1:
                arg_0 = args[0]
                if self.symbol == "div":
                    return partial(lambda x1, x2: int(x1() / x2()), arg_0, args[1])
                elif self.symbol == "mod":
                    return partial(lambda x1, x2: x1() % x2(), arg_0, args[1])
                elif self.symbol == "sin":  # Sin
                    return partial(lambda x: np.sin(x()), arg_0)
                elif self.symbol == "cos":  # Cos
                    return partial(lambda x: np.cos(x()), arg_0)
                elif self.symbol == "exp":  # Exponential
                    return partial(lambda x: np.exp(x()), arg_0)
                elif self.symbol == "log":  # Natural Logarithm
                    return partial(lambda x: np.log(x()), arg_0)
                elif self.symbol == "log10":  # Logarithm Base 10
                    return partial(lambda x: np.log10(x()), arg_0)
                else:
                    raise ValueError("Unable to resolve symbol '{0}'"
                                     " as a single-argument arithmetic function".format(self.symbol))

            # Multiple Arguments
            else:
                if self.symbol == "max":  # Maximum
                    return partial(lambda x: max(*[x_i() for x_i in x]), args)
                elif self.symbol == "min":  # Minimum
                    return partial(lambda x: min(*[x_i() for x_i in x]), args)
                raise ValueError("Unable to resolve symbol '{0}'"
                                 " as a multi-argument arithmetic function".format(self.symbol))

    def collect_declared_entities(self,
                                  state: State,
                                  idx_set: IndexingSet = None,
                                  dummy_symbols: Tuple[str, ...] = None) -> Dict[str, Union[Parameter, Variable]]:

        if self.is_reductive():

            combined_idx_sets = self.combine_indexing_sets(state, idx_set, dummy_symbols)  # length mp
            combined_dummy_syms = self.idx_set_node.combined_dummy_syms  # length mp

            mp = 1
            if idx_set is not None:
                mp = len(idx_set)

            vars = {}
            for ip in range(mp):
                sub_vars = self.operands[0].collect_declared_entities(state,
                                                                      combined_idx_sets[ip],
                                                                      combined_dummy_syms)
                vars.update(sub_vars)
            return vars

        else:
            entities = {}
            for o in self.operands:
                entities.update(o.collect_declared_entities(state, idx_set, dummy_symbols))
            return entities

    def combine_indexing_sets(self,
                              state: State,
                              idx_set: IndexingSet = None,  # length mp
                              dummy_symbols: Tuple[str, ...] = None):

        fcn_idx_sets = self.idx_set_node.evaluate(state, idx_set, dummy_symbols)  # length mp

        if idx_set is not None:
            combined_idx_sets = []
            for element_p, fcn_set_c in zip(idx_set, fcn_idx_sets):  # loop from ip = 0 ... mp
                set_ip = OrderedSet([element_p])
                combined_idx_set = cartesian_product([set_ip, fcn_set_c])
                combined_idx_sets.append(combined_idx_set)

        else:
            combined_idx_sets = fcn_idx_sets

        return combined_idx_sets  # length mp

    def is_reductive(self) -> bool:
        return self.idx_set_node is not None

    def get_children(self) -> list:
        if self.idx_set_node is None:
            return self.operands
        else:
            children = []
            children.extend(self.operands)
            children.append(self.idx_set_node)
            return children

    def set_children(self, operands: list):
        self.operands = []
        self.idx_set_node = None
        for operand in operands:
            if isinstance(operand, CompoundSetNode):
                self.idx_set_node = operand
            else:
                self.operands.append(operand)

    def get_literal(self) -> str:

        # reductive transformation
        if self.is_reductive():
            literal = "{0} {1} {2}".format(self.symbol, self.idx_set_node, self.operands[0])

        # non-reductive transformation
        else:

            if self.symbol == "div" or self.symbol == "mod":
                arguments = [o.get_literal() for o in self.operands]
                literal = "{0} {1} {2}".format(arguments[0], self.symbol, arguments[1])

            else:
                arguments = [o.get_literal() for o in self.operands]
                literal = "{0}({1})".format(self.symbol, ', '.join(arguments))

        if self.is_prioritized:
            literal = '(' + literal + ')'

        return literal


class BinaryArithmeticOperationNode(ArithmeticExpressionNode):

    ADDITION_OPERATOR = '+'
    SUBTRACTION_OPERATOR = '-'
    MULTIPLICATION_OPERATOR = '*'
    DIVISION_OPERATOR = '/'
    EXPONENTIATION_OPERATOR = '^'

    def __init__(self,
                 operator: str,
                 lhs_operand: ArithmeticExpressionNode = None,
                 rhs_operand: ArithmeticExpressionNode = None,
                 is_prioritized: bool = False,
                 id: int = 0):
        super().__init__(id)
        self.operator: str = operator
        self.lhs_operand: ArithmeticExpressionNode = lhs_operand
        self.rhs_operand: ArithmeticExpressionNode = rhs_operand
        self.is_prioritized = is_prioritized

    def evaluate(self,
                 state: State,
                 idx_set: IndexingSet = None,
                 dummy_symbols: Tuple[str, ...] = None
                 ) -> List[float]:

        x_lhs = self.lhs_operand.evaluate(state, idx_set, dummy_symbols)
        x_rhs = self.rhs_operand.evaluate(state, idx_set, dummy_symbols)

        if self.operator == self.ADDITION_OPERATOR:  # Addition
            y = [x_lhs_i + x_rhs_i for x_lhs_i, x_rhs_i in zip(x_lhs, x_rhs)]
        elif self.operator == self.SUBTRACTION_OPERATOR:  # Subtraction
            y = [x_lhs_i - x_rhs_i for x_lhs_i, x_rhs_i in zip(x_lhs, x_rhs)]
        elif self.operator == self.MULTIPLICATION_OPERATOR:  # Multiplication
            y = [x_lhs_i * x_rhs_i for x_lhs_i, x_rhs_i in zip(x_lhs, x_rhs)]
        elif self.operator == self.DIVISION_OPERATOR:  # Division
            y = [x_lhs_i / x_rhs_i for x_lhs_i, x_rhs_i in zip(x_lhs, x_rhs)]
        elif self.operator == self.EXPONENTIATION_OPERATOR:  # Exponentiation
            y = [x_lhs_i ** x_rhs_i for x_lhs_i, x_rhs_i in zip(x_lhs, x_rhs)]
        else:
            raise ValueError("Unable to resolve symbol '{0}'"
                             " as a binary arithmetic operator".format(self.operator))
        return y

    def to_lambda(self,
                  state: State,
                  idx_set_member: Element = None,
                  dummy_symbols: Tuple[str, ...] = None):
        x_lhs = self.lhs_operand.to_lambda(state, idx_set_member, dummy_symbols)
        x_rhs = self.rhs_operand.to_lambda(state, idx_set_member, dummy_symbols)
        if self.operator == self.ADDITION_OPERATOR:  # Addition
            return partial(lambda l, r: l() + r(), x_lhs, x_rhs)
        elif self.operator == self.SUBTRACTION_OPERATOR:  # Subtraction
            return partial(lambda l, r: l() - r(), x_lhs, x_rhs)
        elif self.operator == self.MULTIPLICATION_OPERATOR:  # Multiplication
            return partial(lambda l, r: l() * r(), x_lhs, x_rhs)
        elif self.operator == self.DIVISION_OPERATOR:  # Division
            return partial(lambda l, r: l() / r(), x_lhs, x_rhs)
        elif self.operator == self.EXPONENTIATION_OPERATOR:  # Exponentiation
            return partial(lambda l, r: l() ** r(), x_lhs, x_rhs)
        else:
            raise ValueError("Unable to resolve symbol '{0}'"
                             " as a binary arithmetic operator".format(self.operator))

    def collect_declared_entities(self,
                                  state: State,
                                  idx_set: IndexingSet = None,
                                  dummy_symbols: Tuple[str, ...] = None) -> Dict[str, Union[Parameter, Variable]]:
        vars = {}
        vars.update(self.lhs_operand.collect_declared_entities(state, idx_set, dummy_symbols))
        vars.update(self.rhs_operand.collect_declared_entities(state, idx_set, dummy_symbols))
        return vars

    def get_children(self) -> List:
        return [self.lhs_operand, self.rhs_operand]

    def set_children(self, operands: list):
        if len(operands) > 0:
            self.lhs_operand = operands[0]
        if len(operands) > 1:
            self.rhs_operand = operands[1]

    def get_literal(self) -> str:
        literal = "{0} {1} {2}".format(self.lhs_operand,
                                       self.operator,
                                       self.rhs_operand)
        if self.is_prioritized:
            literal = '(' + literal + ')'
        return literal


class MultiArithmeticOperationNode(ArithmeticExpressionNode):

    ADDITION_OPERATOR = '+'
    MULTIPLICATION_OPERATOR = '*'

    def __init__(self,
                 operator: str,
                 operands: List[ArithmeticExpressionNode],
                 id: int = 0):
        super().__init__(id)
        self.operator: str = operator
        self.operands: List[ArithmeticExpressionNode] = operands

    def evaluate(self,
                 state: State,
                 idx_set: IndexingSet = None,
                 dummy_symbols: Tuple[str, ...] = None
                 ) -> List[float]:

        x_lhs = self.operands[0].evaluate(state, idx_set, dummy_symbols)
        y = x_lhs

        for i in range(1, len(self.operands)):

            x_rhs = self.operands[i].evaluate(state, idx_set, dummy_symbols)

            if self.operator == self.ADDITION_OPERATOR:  # Addition
                y = [x_lhs_i + x_rhs_i for x_lhs_i, x_rhs_i in zip(x_lhs, x_rhs)]
            elif self.operator == self.MULTIPLICATION_OPERATOR:  # Multiplication
                y = [x_lhs_i * x_rhs_i for x_lhs_i, x_rhs_i in zip(x_lhs, x_rhs)]
            else:
                raise ValueError("Unable to resolve symbol '{0}'"
                                 " as a multi arithmetic operator".format(self.operator))

            x_lhs = y

        return y

    def to_lambda(self,
                  state: State,
                  idx_set_member: Element = None,
                  dummy_symbols: Tuple[str, ...] = None):

        args_all = []
        for operand in self.operands:
            args_all.append(operand.to_lambda(state, idx_set_member, dummy_symbols))

        if self.operator == self.ADDITION_OPERATOR:  # Addition
            return partial(lambda x: sum([x_i() for x_i in x]), args_all)
        elif self.operator == self.MULTIPLICATION_OPERATOR:  # Multiplication
            return partial(lambda x: np.prod([x_i() for x_i in x]), args_all)
        else:
            raise ValueError("Unable to resolve symbol '{0}'"
                             " as a multi arithmetic operator".format(self.operator))

    def collect_declared_entities(self,
                                  state: State,
                                  idx_set: IndexingSet = None,
                                  dummy_symbols: Tuple[str, ...] = None) -> Dict[str, Union[Parameter, Variable]]:
        vars = {}
        for operand in self.operands:
            vars.update(operand.collect_declared_entities(state, idx_set, dummy_symbols))
        return vars

    def get_children(self) -> List:
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


class AdditionNode(MultiArithmeticOperationNode):

    def __init__(self,
                 operands: List[ArithmeticExpressionNode],
                 id: int = 0):
        super().__init__(id=id,
                         operator='+',
                         operands=operands)


class SubtractionNode(BinaryArithmeticOperationNode):

    def __init__(self,
                 lhs_operand: ArithmeticExpressionNode = None,
                 rhs_operand: ArithmeticExpressionNode = None,
                 is_prioritized: bool = False,
                 id: int = 0):
        super().__init__(id=id,
                         operator='-',
                         lhs_operand=lhs_operand,
                         rhs_operand=rhs_operand,
                         is_prioritized=is_prioritized)


class MultiplicationNode(MultiArithmeticOperationNode):

    def __init__(self,
                 operands: List[ArithmeticExpressionNode],
                 id: int = 0):
        super().__init__(id=id,
                         operator='*',
                         operands=operands)


class DivisionNode(BinaryArithmeticOperationNode):

    def __init__(self,
                 lhs_operand: ArithmeticExpressionNode = None,
                 rhs_operand: ArithmeticExpressionNode = None,
                 is_prioritized: bool = False,
                 id: int = 0):
        super().__init__(id=id,
                         operator='/',
                         lhs_operand=lhs_operand,
                         rhs_operand=rhs_operand,
                         is_prioritized=is_prioritized)


class ExponentiationNode(BinaryArithmeticOperationNode):

    def __init__(self,
                 lhs_operand: ArithmeticExpressionNode = None,
                 rhs_operand: ArithmeticExpressionNode = None,
                 is_prioritized: bool = False,
                 id: int = 0):
        super().__init__(id=id,
                         operator='^',
                         lhs_operand=lhs_operand,
                         rhs_operand=rhs_operand,
                         is_prioritized=is_prioritized)


class UnaryArithmeticOperationNode(ArithmeticExpressionNode):

    UNARY_PLUS_OPERATOR = '+'
    UNARY_NEGATION_OPERATOR = '-'

    def __init__(self,
                 operator: str,
                 operand: ArithmeticExpressionNode = None,
                 id: int = 0):
        super().__init__(id)
        self.operator: str = operator
        self.operand: ArithmeticExpressionNode = operand

    def evaluate(self,
                 state: State,
                 idx_set: IndexingSet = None,
                 dummy_symbols: Tuple[str, ...] = None
                 ) -> List[float]:
        x = self.operand.evaluate(state, idx_set, dummy_symbols)
        if self.operator == self.UNARY_PLUS_OPERATOR:  # Unary Plus
            y = x
        elif self.operator == self.UNARY_NEGATION_OPERATOR:  # Unary Negation
            y = [-x_i for x_i in x]
        else:
            raise ValueError("Unable to resolve symbol '{0}'"
                             " as a unary arithmetic operator".format(self.operator))
        return y

    def to_lambda(self,
                  state: State,
                  idx_set_member: Element = None,
                  dummy_symbols: Tuple[str, ...] = None):
        arg = self.operand.evaluate(state, idx_set_member, dummy_symbols)
        if self.operator == self.UNARY_PLUS_OPERATOR:  # Unary Plus
            return arg
        elif self.operator == self.UNARY_NEGATION_OPERATOR:  # Unary Negation
            return partial(lambda x: -x(), arg)
        else:
            raise ValueError("Unable to resolve symbol '{0}'"
                             " as a unary arithmetic operator".format(self.operator))

    def collect_declared_entities(self,
                                  state: State,
                                  idx_set: IndexingSet = None,
                                  dummy_symbols: Tuple[str, ...] = None) -> Dict[str, Union[Parameter, Variable]]:
        return self.operand.collect_declared_entities(state, idx_set, dummy_symbols)

    def get_children(self) -> List:
        return [self.operand]

    def set_children(self, operands: list):
        if len(operands) > 0:
            self.operand = operands[0]

    def get_literal(self) -> str:
        literal = "{0}{1}".format(self.operator, self.operand)
        if self.is_prioritized:
            literal = '(' + literal + ')'
        return literal


class ConditionalArithmeticExpressionNode(ArithmeticExpressionNode):

    def __init__(self, id: int = 0):
        super().__init__(id)
        self.operands: List[ArithmeticExpressionNode] = []
        self.conditions: List[LogicalExpressionNode] = []

    def evaluate(self,
                 state: State,
                 idx_set: IndexingSet = None,
                 dummy_symbols: Tuple[str, ...] = None
                 ) -> List[float]:

        count_p = 1
        if idx_set is not None:
            count_p = len(idx_set)

        clause_count = len(self.operands)
        results = []
        for ip in range(count_p):

            sub_idx_set = None
            if idx_set is not None:
                sub_idx_set = OrderedSet(idx_set[[ip]])

            result = 0
            for k in range(clause_count):

                can_evaluate_operand = True

                if k < clause_count - 1 or not self.has_trailing_else_clause():
                    can_evaluate_operand = self.conditions[k].evaluate(state, sub_idx_set, dummy_symbols)[0]

                if can_evaluate_operand:
                    result = self.operands[k].evaluate(state, sub_idx_set, dummy_symbols)[0]
                    break

            results.append(result)

        return results

    def to_lambda(self,
                  state: State,
                  idx_set_member: Element = None,
                  dummy_symbols: Tuple[str, ...] = None):

        clause_count = len(self.operands)

        idx_set = None
        if idx_set_member is not None:
            idx_set = OrderedSet([idx_set_member])

        for k in range(clause_count):

            can_evaluate_operand = True

            if k < clause_count - 1 or not self.has_trailing_else_clause():
                can_evaluate_operand = self.conditions[k].evaluate(state, idx_set, dummy_symbols)[0]

            if can_evaluate_operand:
                arg = self.operands[k].to_lambda(state, idx_set_member, dummy_symbols)
                return partial(lambda o: o(), arg)

        return lambda: 0

    def collect_declared_entities(self,
                                  state: State,
                                  idx_set: IndexingSet = None,
                                  dummy_symbols: Tuple[str, ...] = None) -> Dict[str, Union[Variable, Parameter]]:
        count_p = 1
        if idx_set is not None:
            count_p = len(idx_set)

        clause_count = len(self.operands)
        entities = {}
        for ip in range(count_p):

            sub_idx_set = None
            if idx_set is not None:
                sub_idx_set = OrderedSet(idx_set[[ip]])

            entities_ip = {}
            for k in range(clause_count):

                can_evaluate_operand = True

                if k < clause_count - 1 or not self.has_trailing_else_clause():
                    can_evaluate_operand = self.conditions[k].evaluate(state, sub_idx_set, dummy_symbols)[0]

                if can_evaluate_operand:
                    entities_ip = self.operands[k].collect_declared_entities(state, sub_idx_set, dummy_symbols)
                    break

            entities.update(entities_ip)

        return entities

    @staticmethod
    def generate_negated_combined_condition(conditions: List[str]):
        other_conditions_par = ["(" + c + ")" for c in conditions]
        combined_condition = " or ".join(other_conditions_par)
        return "!({0})".format(combined_condition)

    def add_operand(self, operand: ArithmeticExpressionNode):
        self.operands.append(operand)

    def add_condition(self, condition: LogicalExpressionNode = None):
        if condition is not None and condition != "":
            self.conditions.append(condition)

    def has_trailing_else_clause(self) -> bool:
        return len(self.operands) == len(self.conditions) + 1

    def get_children(self) -> List[Union[ArithmeticExpressionNode, LogicalExpressionNode]]:
        children = []
        children.extend(self.operands)
        for condition in self.conditions:
            children.append(condition)
        return children

    def set_children(self, operands: List[Union[ArithmeticExpressionNode, LogicalExpressionNode]]):
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
