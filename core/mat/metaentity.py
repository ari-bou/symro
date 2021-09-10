from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union

import symro.core.constants as const
from symro.core.mat.entity import Entity
from symro.core.mat.state import State
from symro.core.mat.setn import BaseSetNode, CompoundSetNode
from symro.core.mat.lexprn import RelationalOperationNode
from symro.core.mat.exprn import ExpressionNode
from symro.core.mat.expression import Expression


# TODO: consider assigning unique numeric ids to each meta-entity

class MetaEntity(ABC):

    def __init__(self,
                 symbol: str,
                 alias: str = None,
                 idx_meta_sets: Union[List["MetaSet"], Dict[str, "MetaSet"]] = None,
                 idx_set_node: CompoundSetNode = None):

        if idx_meta_sets is None:
            idx_meta_sets = []
        elif isinstance(idx_meta_sets, dict):
            idx_meta_sets = [ms for ms in idx_meta_sets.values()]

        self.symbol: str = symbol
        self.alias: str = alias
        self.idx_meta_sets: List["MetaSet"] = idx_meta_sets
        self.idx_set_node: CompoundSetNode = idx_set_node
        self.is_sub: bool = False  # designates whether the meta-entity is a subset of another meta-entity

    def __eq__(self, other):
        return str(self) == str(other)

    def get_dummy_symbols(self) -> List[str]:
        dummy_symbols = []
        for meta_set in self.idx_meta_sets:
            dummy_symbols.extend(meta_set.reduced_dummy_symbols)
        return dummy_symbols

    def get_dimension(self) -> int:
        return sum([meta_set.dimension for meta_set in self.idx_meta_sets])

    def get_reduced_dimension(self) -> int:
        return sum([meta_set.reduced_dimension for meta_set in self.idx_meta_sets])

    def is_indexed_with(self, meta_set: "MetaSet") -> bool:
        return any(map(lambda ms: ms.symbol == meta_set.symbol, self.idx_meta_sets))

    def get_first_reduced_dim_index_of_idx_set(self, meta_set: "MetaSet") -> int:
        if self.is_indexed_with(meta_set):
            counter = 0
            for ms in self.idx_meta_sets:
                if ms.symbol == meta_set.symbol:
                    return counter
                counter += ms.reduced_dimension
        else:
            return -1

    def get_indexing_set_by_position(self, pos: int) -> "MetaSet":
        if pos >= self.get_reduced_dimension():
            raise ValueError("Position is out of range")
        p = 0
        for ms in self.idx_meta_sets:
            p += ms.reduced_dimension
            if p > pos:
                return ms

    def get_idx_set_con_literal(self) -> Optional[str]:
        if self.idx_set_node is not None:
            if self.idx_set_node.constraint_node is not None:
                return self.idx_set_node.constraint_node.get_literal()
        return None

    def is_owner(self, entity: Entity, state: State):
        """
        Returns True if the meta-entity is the owner of the entity argument.
        :param entity: algebraic entity
        :param state: problem state
        :return: bool
        """

        # Check whether symbols are identical
        if self.symbol != entity.symbol:
            return False

        # Check whether entity index is a member of the meta-entity indexing set
        if self.idx_set_node is not None:
            idx_set = self.idx_set_node.evaluate(state)[0]
            if entity.indices not in idx_set:
                return False

        return True

    @abstractmethod
    def get_type(self) -> str:
        pass

    @abstractmethod
    def get_declaration(self) -> str:
        pass


class MetaSet(MetaEntity):

    def __init__(self,
                 symbol: str,
                 alias: str = None,
                 idx_meta_sets: Union[List["MetaSet"], Dict[str, "MetaSet"]] = None,
                 idx_set_node: CompoundSetNode = None,
                 dimension: int = None,
                 reduced_dimension: int = None,
                 dummy_symbols: List[str] = None,
                 reduced_dummy_symbols: List[str] = None,
                 is_dim_fixed: List[bool] = None,
                 super_set_node: BaseSetNode = None,
                 set_node: BaseSetNode = None):

        super(MetaSet, self).__init__(symbol=symbol,
                                      alias=alias,
                                      idx_meta_sets=idx_meta_sets,
                                      idx_set_node=idx_set_node)

        self.dimension: int = dimension
        self.reduced_dimension: int = reduced_dimension  # set dimension after set constraint is applied

        self.dummy_symbols: List[str] = dummy_symbols  # length: dimension
        self.reduced_dummy_symbols: List[str] = reduced_dummy_symbols  # length: reduced dimension

        self.is_init: bool = False
        self.is_dim_fixed: List[bool] = is_dim_fixed  # length: dimension

        self.super_set_node: BaseSetNode = super_set_node
        self.set_node: BaseSetNode = set_node

        self.initialize()

    def __str__(self):
        return "set {0}".format(self.symbol)

    def initialize(self):

        if self.dimension is not None:

            if self.reduced_dimension is None:
                if self.reduced_dummy_symbols is not None:
                    self.reduced_dimension = len(self.reduced_dummy_symbols)
                elif self.is_dim_fixed is not None:
                    self.reduced_dimension = self.is_dim_fixed.count(False)
                else:
                    self.reduced_dimension = self.dimension

            if self.is_dim_fixed is None:
                self.is_dim_fixed = [False] * self.dimension
                if self.reduced_dimension < self.dimension:
                    delta_dim = self.dimension - self.reduced_dimension
                    self.is_dim_fixed[:delta_dim] = [True] * delta_dim

            if self.dummy_symbols is None:
                self.dummy_symbols = []
                base_index_symbol = self.symbol[0].lower()
                if self.dimension == 1:
                    self.dummy_symbols.append(base_index_symbol)
                else:
                    for i in range(self.dimension):
                        self.dummy_symbols.append(base_index_symbol + str(i))

            if self.reduced_dummy_symbols is None:
                if self.dimension == self.reduced_dimension:
                    self.reduced_dummy_symbols = list(self.dummy_symbols)
                else:
                    self.reduced_dummy_symbols = []
                    base_index_symbol = self.symbol[0].lower()
                    for i in range(self.dimension):
                        if not self.is_dim_fixed[i]:
                            if self.dummy_symbols is not None:
                                self.reduced_dummy_symbols.append(self.dummy_symbols[i])
                            else:
                                self.reduced_dummy_symbols.append(base_index_symbol + str(i))

            self.is_init = True

        else:
            self.is_init = False

    def get_type(self) -> str:
        return const.SET_TYPE

    def get_declaration(self) -> str:
        declaration = "set {0}".format(self.symbol)
        if self.alias is not None:
            declaration += " {0} ".format(self.alias)
        if self.idx_set_node is not None:
            declaration += str(self.idx_set_node)
        if self.super_set_node is not None:
            declaration += " within {0}".format(self.super_set_node)
        if self.set_node is not None:
            declaration += " := {0}".format(self.set_node)
        declaration += ';'
        return declaration

    def get_idx_def_literal(self) -> str:
        if self.dimension == 0:
            return self.symbol
        elif self.dimension == 1:
            return "{0} in {1}".format(self.dummy_symbols[0], self.symbol)
        else:
            index = '(' + ','.join(self.dummy_symbols) + ')'
            return "{0} in {1}".format(index, self.symbol)


class MetaParameter(MetaEntity):

    def __init__(self,
                 symbol: str,
                 alias: str = None,
                 idx_meta_sets: Union[List[MetaSet], Dict[str, MetaSet]] = None,
                 idx_set_node: CompoundSetNode = None,
                 is_binary: bool = False,
                 is_integer: bool = False,
                 is_symbolic: bool = False,
                 default_value: ExpressionNode = None,
                 super_set_node: BaseSetNode = None,
                 relational_constraints: Dict[str, ExpressionNode] = None):

        super(MetaParameter, self).__init__(symbol=symbol,
                                            alias=alias,
                                            idx_meta_sets=idx_meta_sets,
                                            idx_set_node=idx_set_node)

        self.is_binary: bool = is_binary
        self.is_integer: bool = is_integer
        self.is_symbolic: bool = is_symbolic
        self.default_value: ExpressionNode = default_value
        self.super_set_node: BaseSetNode = super_set_node
        self.relational_constraints: Dict[str, ExpressionNode] = relational_constraints \
            if relational_constraints is not None else {}

    def __str__(self):
        declaration = "param {0}".format(self.symbol)
        if self.get_dimension() > 0 and self.idx_set_node is not None:
            declaration += str(self.idx_set_node)
        return declaration

    def get_type(self) -> str:
        return const.PARAM_TYPE

    def get_declaration(self) -> str:

        declaration = "param {0}".format(self.symbol)
        if self.alias is not None:
            declaration += " {0} ".format(self.alias)
        if self.idx_set_node is not None:
            declaration += str(self.idx_set_node)

        attributes = []
        if self.is_binary:
            attributes.append("binary")
        if self.is_integer:
            attributes.append("integer")
        if self.is_symbolic:
            attributes.append("symbolic")
        if self.default_value is not None:
            attributes.append("default {0}".format(self.default_value))
        if self.super_set_node is not None:
            attributes.append("in {0}".format(self.super_set_node))
        for rel_opr, node in self.relational_constraints.items():
            attributes.append("{0} {1}".format(rel_opr, node))

        if len(attributes) > 0:
            declaration += ' ' + ", ".join(attributes)

        declaration += ";"

        return declaration


class MetaVariable(MetaEntity):

    def __init__(self,
                 symbol: str,
                 alias: str = None,
                 idx_meta_sets: Union[List[MetaSet], Dict[str, MetaSet]] = None,
                 idx_set_node: CompoundSetNode = None,
                 is_binary: bool = False,
                 is_integer: bool = False,
                 is_symbolic: bool = False,
                 default_value: ExpressionNode = None,
                 defined_value: ExpressionNode = None,
                 lower_bound: ExpressionNode = None,
                 upper_bound: ExpressionNode = None):

        super(MetaVariable, self).__init__(symbol=symbol,
                                           alias=alias,
                                           idx_meta_sets=idx_meta_sets,
                                           idx_set_node=idx_set_node)

        self.is_binary: bool = is_binary
        self.is_integer: bool = is_integer
        self.is_symbolic: bool = is_symbolic
        self.default_value: ExpressionNode = default_value
        self.defined_value: ExpressionNode = defined_value
        self.lower_bound: ExpressionNode = lower_bound
        self.upper_bound: ExpressionNode = upper_bound

    def __str__(self):
        declaration = "var {0}".format(self.symbol)
        if self.get_dimension() > 0 and self.idx_set_node is not None:
            declaration += str(self.idx_set_node)
        return declaration

    def is_defined(self) -> bool:
        return self.defined_value is not None

    def get_type(self) -> str:
        return const.VAR_TYPE

    def get_declaration(self) -> str:

        declaration = "var {0}".format(self.symbol)
        if self.alias is not None:
            declaration += " {0} ".format(self.alias)
        if self.idx_set_node is not None:
            declaration += str(self.idx_set_node)

        attributes = []
        if self.is_binary:
            attributes.append("binary")
        if self.is_integer:
            attributes.append("integer")
        if self.is_symbolic:
            attributes.append("symbolic")
        if self.default_value is not None:
            attributes.append(":= {0}".format(self.default_value))
        if self.defined_value is not None:
            attributes.append("= {0}".format(self.defined_value))
        if self.lower_bound is not None:
            attributes.append(">= {0}".format(self.lower_bound))
        if self.upper_bound is not None:
            attributes.append("<= {0}".format(self.upper_bound))

        if len(attributes) > 0:
            declaration += ' ' + ", ".join(attributes)

        declaration += ";"

        return declaration


class MetaObjective(MetaEntity):

    MINIMIZE_DIRECTION = "minimize"
    MAXIMIZE_DIRECTION = "maximize"

    def __init__(self,
                 symbol: str,
                 alias: str = None,
                 idx_meta_sets: Union[List[MetaSet], Dict[str, MetaSet]] = None,
                 idx_set_node: CompoundSetNode = None,
                 direction: str = "minimize",
                 expression: Expression = None):
        super(MetaObjective, self).__init__(symbol=symbol,
                                            alias=alias,
                                            idx_meta_sets=idx_meta_sets,
                                            idx_set_node=idx_set_node)

        if direction not in ["minimize", "maximize"]:
            direction = "minimize"

        self.direction: str = direction
        self.expression: Expression = expression

    def __str__(self):
        declaration = "{0} {1}".format(self.direction, self.symbol)
        if self.get_dimension() > 0 and self.idx_set_node is not None:
            declaration += str(self.idx_set_node)
        return declaration

    def get_type(self) -> str:
        return const.OBJ_TYPE

    def get_declaration(self) -> str:
        declaration = "{0} {1}".format(self.direction, self.symbol)
        if self.alias is not None:
            declaration += " {0} ".format(self.alias)
        if self.idx_set_node is not None:
            declaration += str(self.idx_set_node)
        if self.expression is not None:
            declaration += ": {0}".format(self.expression)
        declaration += ";"
        return declaration


class MetaConstraint(MetaEntity):

    EQUALITY_TYPE = "eq"
    INEQUALITY_TYPE = "ineq"
    DOUBLE_INEQUALITY_TYPE = "dbl_ineq"

    def __init__(self,
                 symbol: str,
                 alias: str = None,
                 idx_meta_sets: Union[List[MetaSet], Dict[str, MetaSet]] = None,
                 idx_set_node: CompoundSetNode = None,
                 expression: Expression = None,
                 ctype: str = None):
        super(MetaConstraint, self).__init__(symbol=symbol,
                                             alias=alias,
                                             idx_meta_sets=idx_meta_sets,
                                             idx_set_node=idx_set_node)
        self.expression: Expression = expression
        self.ctype: str = ctype

    def __str__(self):
        declaration = "subject to {0}".format(self.symbol)
        if self.get_dimension() > 0 and self.idx_set_node is not None:
            declaration += str(self.idx_set_node)
        return declaration

    def elicit_constraint_type(self):
        expr_node = self.expression.expression_node
        if isinstance(expr_node, RelationalOperationNode):
            if expr_node.operator in ['=', "=="]:
                self.ctype = self.EQUALITY_TYPE
            elif expr_node.operator in ['<', '<=', '>=', '>']:
                if not isinstance(expr_node.rhs_operand, RelationalOperationNode):
                    self.ctype = self.INEQUALITY_TYPE
                else:
                    self.ctype = self.DOUBLE_INEQUALITY_TYPE
            return self.ctype
        raise ValueError("Meta-constraint expected an equality or an inequality expression"
                         " while eliciting the constraint type")

    def get_type(self) -> str:
        return const.CON_TYPE

    def get_declaration(self) -> str:
        declaration = "{0}".format(self.symbol)
        if self.alias is not None:
            declaration += " {0} ".format(self.alias)
        if self.idx_set_node is not None:
            declaration += str(self.idx_set_node)
        if self.expression is not None:
            declaration += ": {0}".format(self.expression)
        declaration += ";"
        return declaration
