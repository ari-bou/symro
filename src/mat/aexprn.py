from functools import partial
from numbers import Number
import numpy as np
from typing import Callable, Dict, List, Optional, Union

from symro.src.mat.constants import *
from symro.src.mat.types import Element, IndexingSet
from symro.src.mat.orderedset import OrderedSet
from symro.src.mat.entity import Parameter, Variable
from symro.src.mat.state import State
from symro.src.mat.exprn import LogicalExpressionNode, ArithmeticExpressionNode
from symro.src.mat.opern import RelationalOperationNode, ArithmeticOperationNode
from symro.src.mat.lexprn import BooleanNode
from symro.src.mat.dummyn import CompoundDummyNode
from symro.src.mat.setn import CompoundSetNode


class ArithmeticTransformationNode(ArithmeticExpressionNode):

    def __init__(self,
                 fcn: int,
                 idx_set_node: CompoundSetNode = None,
                 operands: Union[ArithmeticExpressionNode, List[ArithmeticExpressionNode]] = None):

        super().__init__()
        self.fcn: int = fcn
        self.idx_set_node: Optional[CompoundSetNode] = idx_set_node
        self.operands: List[ArithmeticExpressionNode] = []

        if self.fcn in [SUMMATION_FUNCTION, PRODUCT_FUNCTION]:
            self.is_prioritized = True

        if operands is not None:
            if isinstance(operands, ArithmeticExpressionNode):
                self.operands.append(operands)
            else:
                self.operands.extend(operands)

    def __neg__(self):
        return ArithmeticOperationNode.negation(self)

    def __add__(self, other: ArithmeticExpressionNode):
        return ArithmeticOperationNode.addition(self, other)

    def __sub__(self, other: ArithmeticExpressionNode):
        return ArithmeticOperationNode.subtraction(self, other)

    def __mul__(self, other: ArithmeticExpressionNode):
        return ArithmeticOperationNode.multiplication(self, other)

    def __truediv__(self, other: ArithmeticExpressionNode):
        return ArithmeticOperationNode.division(self, other)

    def __pow__(self, power: ArithmeticExpressionNode, modulo=None):
        return ArithmeticOperationNode.exponentiation(self, power)

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
        if self.is_reductive():
            return self.__evaluate_reductive_function(state, idx_set, dummy_element)
        else:
            return self.__evaluate_non_reductive_function(state, idx_set, dummy_element)

    def __evaluate_reductive_function(self,
                                      state: State,
                                      idx_set: IndexingSet = None,
                                      dummy_symbols: Element = None) -> np.ndarray:

        combined_idx_sets = self.combine_indexing_sets(state, idx_set, dummy_symbols)  # length mp
        combined_dummy_syms = self.idx_set_node.combined_dummy_element

        mp = 1
        if idx_set is not None:
            mp = len(idx_set)

        y = np.zeros(shape=mp)

        for ip in range(mp):
            x = self.operands[0].evaluate(state, combined_idx_sets[ip], combined_dummy_syms)
            if self.fcn == SUMMATION_FUNCTION:  # Reductive Summation
                y_ip = sum(x)
            elif self.fcn == PRODUCT_FUNCTION:  # Reductive Multiplication
                y_ip = np.prod(x)
            else:
                raise ValueError("Unable to resolve symbol '{0}'"
                                 " as a reductive arithmetic transformation".format(self.fcn))
            y[ip] = y_ip

        return y

    def __evaluate_non_reductive_function(self,
                                          state: State,
                                          idx_set: IndexingSet = None,
                                          dummy_symbols: Element = None) -> np.ndarray:

        x = np.array([o.evaluate(state, idx_set, dummy_symbols) for o in self.operands])

        # Single Argument
        if len(self.operands) == 1:
            x_0 = x[0]
            if self.fcn == INTEGER_DIVISION_FUNCTION:
                y = np.divide(x[0], x[1])
                y = np.around(y)
            elif self.fcn == MODULUS_FUNCTION:
                y = np.mod(x[0], x[1])
            elif self.fcn == SINE_FUNCTION:  # Sine
                y = np.sin(x_0)
            elif self.fcn == COSINE_FUNCTION:  # Cosine
                y = np.cos(x_0)
            elif self.fcn == EXPONENTIAL_FUNCTION:  # Exponential
                y = np.exp(x_0)
            elif self.fcn == NATURAL_LOGARITHM_FUNCTION:  # Natural Logarithm
                y = np.log(x_0)
            elif self.fcn == BASE_10_LOGARITHM_FUNCTION:  # Logarithm Base 10
                y = np.log10(x_0)
            else:
                raise ValueError("Unable to resolve symbol '{0}'"
                                 " as a single-argument arithmetic function".format(self.fcn))

        # Multiple Arguments
        else:
            if self.fcn == MAXIMUM_FUNCTION:  # Maximum
                y = np.max(x, axis=0)
            elif self.fcn == MINIMUM_FUNCTION:  # Minimum
                y = np.min(x, axis=0)
            else:
                raise ValueError("Unable to resolve symbol '{0}'"
                                 " as a multi-argument arithmetic function".format(self.fcn))

        return y

    def to_lambda(self,
                  state: State,
                  idx_set_member: Element = None,
                  dummy_element: Element = None) -> Callable:

        if self.is_reductive():

            if idx_set_member is None:
                idx_set = None
            else:
                idx_set = OrderedSet([idx_set_member])
            combined_idx_set = self.combine_indexing_sets(state, idx_set, dummy_element)[0]  # length mc
            combined_dummy_syms = self.idx_set_node.combined_dummy_element

            arg_0 = np.array([self.operands[0].to_lambda(state, idx, combined_dummy_syms)
                              for idx in combined_idx_set])

            if self.fcn == SUMMATION_FUNCTION:  # reductive summation
                return partial(lambda x: np.sum([x_i() for x_i in x]), arg_0)
            elif self.fcn == PRODUCT_FUNCTION:  # reductive multiplication
                return partial(lambda x: np.prod([x_i() for x_i in x]), arg_0)
            else:
                raise ValueError("Unable to resolve symbol '{0}'".format(self.fcn)
                                 + " as a reductive arithmetic transformation")

        else:

            args = np.array([o.to_lambda(state, idx_set_member, dummy_element) for o in self.operands])

            # Single Argument
            if len(self.operands) == 1:
                arg_0 = args[0]
                if self.fcn == INTEGER_DIVISION_FUNCTION:
                    return partial(lambda x1, x2: int(x1() / x2()), arg_0, args[1])
                elif self.fcn == MODULUS_FUNCTION:
                    return partial(lambda x1, x2: x1() % x2(), arg_0, args[1])
                elif self.fcn == SINE_FUNCTION:  # Sin
                    return partial(lambda x: np.sin(x()), arg_0)
                elif self.fcn == COSINE_FUNCTION:  # Cos
                    return partial(lambda x: np.cos(x()), arg_0)
                elif self.fcn == EXPONENTIAL_FUNCTION:  # Exponential
                    return partial(lambda x: np.exp(x()), arg_0)
                elif self.fcn == NATURAL_LOGARITHM_FUNCTION:  # Natural Logarithm
                    return partial(lambda x: np.log(x()), arg_0)
                elif self.fcn == BASE_10_LOGARITHM_FUNCTION:  # Logarithm Base 10
                    return partial(lambda x: np.log10(x()), arg_0)
                else:
                    raise ValueError("Unable to resolve symbol '{0}'".format(self.fcn)
                                     + " as a single-argument arithmetic transformation")

            # Multiple Arguments
            else:
                if self.fcn == MAXIMUM_FUNCTION:  # Maximum
                    return partial(lambda x: np.max([x_i() for x_i in x]), args)
                elif self.fcn == MINIMUM_FUNCTION:  # Minimum
                    return partial(lambda x: np.min([x_i() for x_i in x]), args)
                raise ValueError("Unable to resolve symbol '{0}'".format(self.fcn)
                                 + " as a multi-argument arithmetic transformation")

    def collect_declared_entities(self,
                                  state: State,
                                  idx_set: IndexingSet = None,
                                  dummy_element: Element = None) -> Dict[tuple, Union[Parameter, Variable]]:

        if self.is_reductive():

            combined_idx_sets = self.combine_indexing_sets(state, idx_set, dummy_element)  # length mp
            combined_dummy_syms = self.idx_set_node.combined_dummy_element  # length mp

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
                entities.update(o.collect_declared_entities(state, idx_set, dummy_element))
            return entities

    def combine_indexing_sets(self,
                              state: State,
                              idx_set: IndexingSet = None,  # length mp
                              dummy_element: Element = None):
        return self.idx_set_node.generate_combined_idx_sets(  # length mp
            state=state,
            idx_set=idx_set,
            dummy_element=dummy_element,
            can_reduce=False
        )

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
            literal = "{0} {1} {2}".format(AMPL_FUNCTION_SYMBOLS[self.fcn], self.idx_set_node, self.operands[0])

        # non-reductive transformation
        else:

            if self.fcn == INTEGER_DIVISION_FUNCTION or self.fcn == MODULUS_FUNCTION:
                arguments = [o.get_literal() for o in self.operands]
                literal = "{0} {1} {2}".format(arguments[0], AMPL_FUNCTION_SYMBOLS[self.fcn], arguments[1])

            else:
                arguments = [o.get_literal() for o in self.operands]
                literal = "{0}({1})".format(AMPL_FUNCTION_SYMBOLS[self.fcn], ', '.join(arguments))

        if self.is_prioritized:
            literal = '(' + literal + ')'

        return literal


class ArithmeticConditionalNode(ArithmeticExpressionNode):

    def __init__(self,
                 operands: List[ArithmeticExpressionNode],
                 conditions: List[LogicalExpressionNode],
                 is_prioritized: bool = False):
        super().__init__()
        self.operands: List[ArithmeticExpressionNode] = operands
        self.conditions: List[LogicalExpressionNode] = conditions
        self.is_prioritized = is_prioritized

    def __neg__(self):
        return ArithmeticOperationNode.negation(self)

    def __add__(self, other: ArithmeticExpressionNode):
        return ArithmeticOperationNode.addition(self, other)

    def __sub__(self, other: ArithmeticExpressionNode):
        return ArithmeticOperationNode.subtraction(self, other)

    def __mul__(self, other: ArithmeticExpressionNode):
        return ArithmeticOperationNode.multiplication(self, other)

    def __truediv__(self, other: ArithmeticExpressionNode):
        return ArithmeticOperationNode.division(self, other)

    def __pow__(self, power: ArithmeticExpressionNode, modulo=None):
        return ArithmeticOperationNode.exponentiation(self, power)

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

        mp = 1
        if idx_set is not None:
            mp = len(idx_set)

        clause_count = len(self.operands)

        y = np.zeros(shape=mp)

        for ip in range(mp):

            sub_idx_set = None
            if idx_set is not None:
                sub_idx_set = OrderedSet(idx_set[[ip]])

            y_ip = 0
            for k in range(clause_count):

                can_evaluate_operand = True

                if k < clause_count - 1 or not self.has_trailing_else_clause():
                    can_evaluate_operand = self.conditions[k].evaluate(state, sub_idx_set, dummy_element)[0]

                if can_evaluate_operand:
                    y_ip = self.operands[k].evaluate(state, sub_idx_set, dummy_element)[0]
                    break

            y[ip] = y_ip

        return y

    def to_lambda(self,
                  state: State,
                  idx_set_member: Element = None,
                  dummy_element: Element = None) -> Callable:

        clause_count = len(self.operands)

        idx_set = None
        if idx_set_member is not None:
            idx_set = OrderedSet([idx_set_member])

        for k in range(clause_count):

            can_evaluate_operand = True

            if k < clause_count - 1 or not self.has_trailing_else_clause():
                can_evaluate_operand = self.conditions[k].evaluate(state, idx_set, dummy_element)[0]

            if can_evaluate_operand:
                arg = self.operands[k].to_lambda(state, idx_set_member, dummy_element)
                return partial(lambda o: o(), arg)

        # else
        return lambda: 0

    def collect_declared_entities(self,
                                  state: State,
                                  idx_set: IndexingSet = None,
                                  dummy_element: Element = None) -> Dict[tuple, Union[Variable, Parameter]]:
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
                    can_evaluate_operand = self.conditions[k].evaluate(state, sub_idx_set, dummy_element)[0]

                if can_evaluate_operand:
                    entities_ip = self.operands[k].collect_declared_entities(state, sub_idx_set, dummy_element)
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

        literal = ""

        for i, operand in enumerate(self.operands):
            if i == 0:
                literal += "if {0} then {1}".format(self.conditions[i], operand)
            elif i == len(self.operands) - 1 and self.has_trailing_else_clause():
                literal += " else {0}".format(operand)
            else:
                literal += " else if {0} then {1}".format(self.conditions[i], operand)

        if self.is_prioritized:
            literal = "({0})".format(literal)

        return literal


class DeclaredEntityNode(ArithmeticExpressionNode):

    def __init__(self,
                 symbol: str,
                 idx_node: CompoundDummyNode = None,
                 suffix: str = None,
                 type: str = None):

        super().__init__()

        self.symbol: str = symbol
        self.idx_node: CompoundDummyNode = idx_node
        self.suffix: str = suffix
        self.__entity_type: str = type

    def __neg__(self):
        return ArithmeticOperationNode.negation(self)

    def __add__(self, other: ArithmeticExpressionNode):
        return ArithmeticOperationNode.addition(self, other)

    def __sub__(self, other: ArithmeticExpressionNode):
        return ArithmeticOperationNode.subtraction(self, other)

    def __mul__(self, other: ArithmeticExpressionNode):
        return ArithmeticOperationNode.multiplication(self, other)

    def __truediv__(self, other: ArithmeticExpressionNode):
        return ArithmeticOperationNode.division(self, other)

    def __pow__(self, power: ArithmeticExpressionNode, modulo=None):
        return ArithmeticOperationNode.exponentiation(self, power)

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

        indices = None
        if self.is_indexed:
            indices = self.idx_node.evaluate(state, idx_set, dummy_element)

        mp = 1
        if idx_set is not None:
            mp = len(idx_set)

        entities = []

        for ip in range(mp):

            idx = None
            if indices is not None:
                idx = indices[ip]

            # build the entity if nonexistent and retrieve it
            entity = state.get_entity(self.symbol, idx, self.__entity_type)
            entities.append(entity)

        if self.suffix is None:
            return np.array([e.get_value() for e in entities])

        elif self.suffix == "lb":
            return np.array([e.get_lb() for e in entities])

        elif self.suffix == "ub":
            return np.array([e.get_ub() for e in entities])

        elif self.suffix == "val":
            return np.array([e.get_value() for e in entities])

        elif self.suffix == "body":
            return np.array([e.get_body() for e in entities])

        elif self.suffix == "dual":
            return np.array([e.get_value() for e in entities])

        else:
            raise ValueError("Unable to resolve suffix '{0}'".format(self.suffix)
                             + " of declared entity node '{0}'".format(self))

    def to_lambda(self,
                  state: State,
                  idx_set_member: Element = None,
                  dummy_element: Element = None) -> Callable:

        idx = None
        if self.is_indexed:
            idx = self.idx_node.evaluate(state=state,
                                         idx_set=OrderedSet([idx_set_member]),
                                         dummy_element=dummy_element)[0]

        # build the entity if nonexistent and retrieve it
        entity = state.get_entity(self.symbol, idx, self.__entity_type)

        if self.suffix is None:
            return partial(lambda e: e.get_value(), entity)

        elif self.suffix == "lb":
            return partial(lambda e: e.lb, entity)

        elif self.suffix == "ub":
            return partial(lambda e: e.ub, entity)

        elif self.suffix == "val":
            return partial(lambda e: e.dual, entity)

        elif self.suffix == "body":
            return partial(lambda e: e.body, entity)

        elif self.suffix == "dual":
            return partial(lambda e: e.dual, entity)

        else:
            raise ValueError("Unable to resolve suffix '{0}'".format(self.suffix)
                             + " of declared entity node '{0}'".format(self))

    def collect_declared_entities(self,
                                  state: State,
                                  idx_set: IndexingSet = None,
                                  dummy_element: Element = None) -> Dict[tuple, Union[Parameter, Variable]]:
        entities = {}

        entity_indices = None
        if self.is_indexed:
            entity_indices = self.idx_node.evaluate(state, idx_set, dummy_element)

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

            if not self.is_constant:
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

    @property
    def is_indexed(self) -> bool:
        return self.idx_node is not None

    @property
    def is_constant(self) -> bool:
        return self.__entity_type == PARAM_TYPE

    @property
    def type(self) -> str:
        return self.__entity_type

    @type.setter
    def type(self, type: str):
        self.__entity_type = type

    def get_children(self) -> list:
        if self.idx_node is not None:
            return [self.idx_node]
        return []

    def set_children(self, children: list):
        if len(children) > 0:
            self.idx_node = children[0]

    def get_literal(self) -> str:
        literal = self.symbol
        if self.is_indexed:
            literal += "[{0}]".format(','.join([str(n) for n in self.idx_node.component_nodes]))
        if self.suffix is not None:
            literal += ".{0}".format(self.suffix)
        return literal


class NumericNode(ArithmeticExpressionNode):

    def __init__(self,
                 value: Union[Number, str],
                 sci_not: bool = False,
                 coeff_sym: str = None,
                 power_sign: str = None,
                 power_sym: str = None):

        super().__init__()

        self.value: Number = 0
        self.sci_not: bool = sci_not  # True if scientific notation is used
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

    def __neg__(self):
        self.value *= -1
        return self

    def __add__(self, other: ArithmeticExpressionNode):

        if isinstance(other, NumericNode):
            return NumericNode(self.value + other.value)

        elif isinstance(other, ArithmeticOperationNode) and other.operator == ADDITION_OPERATOR:

            for i, term in enumerate(other.operands):
                if isinstance(term, NumericNode):
                    other.operands[i] = NumericNode(term.value + self.value)
                    return other

            other.operands.append(self)
            return other

        else:
            return ArithmeticOperationNode.addition(self, other)

    def __sub__(self, other: ArithmeticExpressionNode):
        if isinstance(other, NumericNode):
            return NumericNode(self.value - other.value)
        else:
            return ArithmeticOperationNode.subtraction(self, other)

    def __mul__(self, other: ArithmeticExpressionNode):

        if isinstance(other, NumericNode):
            return NumericNode(self.value * other.value)

        elif isinstance(other, ArithmeticOperationNode) and other.operator == MULTIPLICATION_OPERATOR:

            for i, factor in enumerate(other.operands):
                if isinstance(factor, NumericNode):
                    other.operands[i] = NumericNode(factor.value * self.value)
                    return other

            other.operands.insert(0, self)
            return other

        else:
            return ArithmeticOperationNode.addition(self, other)

    def __truediv__(self, other: ArithmeticExpressionNode):
        if isinstance(other, NumericNode):
            return NumericNode(self.value / other.value)
        else:
            return ArithmeticOperationNode.division(self, other)

    def __pow__(self, power: ArithmeticExpressionNode, modulo=None):
        if isinstance(power, NumericNode):
            return NumericNode(self.value ** power.value)
        else:
            return ArithmeticOperationNode.exponentiation(self, power)

    def __eq__(self, other: ArithmeticExpressionNode):
        if isinstance(other, NumericNode):
            return BooleanNode(value=self.value == other.value)
        else:
            return RelationalOperationNode.equal(self, other)

    def __ne__(self, other: ArithmeticExpressionNode):
        if isinstance(other, NumericNode):
            return BooleanNode(value=self.value != other.value)
        else:
            return RelationalOperationNode.not_equal(self, other)

    def __lt__(self, other: ArithmeticExpressionNode):
        if isinstance(other, NumericNode):
            return BooleanNode(value=self.value < other.value)
        else:
            return RelationalOperationNode.less_than(self, other)

    def __le__(self, other: ArithmeticExpressionNode):
        if isinstance(other, NumericNode):
            return BooleanNode(value=self.value <= other.value)
        else:
            return RelationalOperationNode.less_than_or_equal(self, other)

    def __gt__(self, other: ArithmeticExpressionNode):
        if isinstance(other, NumericNode):
            return BooleanNode(value=self.value > other.value)
        else:
            return RelationalOperationNode.greater_than(self, other)

    def __ge__(self, other: ArithmeticExpressionNode):
        if isinstance(other, NumericNode):
            return BooleanNode(value=self.value >= other.value)
        else:
            return RelationalOperationNode.greater_than_or_equal(self, other)

    def evaluate(self,
                 state: State,
                 idx_set: IndexingSet = None,
                 dummy_element: Element = None
                 ) -> np.ndarray:
        mp = 1
        if idx_set is not None:
            mp = len(idx_set)
        return np.full(shape=mp, fill_value=self.value)

    def to_lambda(self,
                  state: State,
                  idx_set_member: Element = None,
                  dummy_element: Element = None):
        return lambda: self.value

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
