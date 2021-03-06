from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union

from .constants import *
from .types import Element, IndexingSet
from .util import remove_set_dimensions
from .entity import Entity
from .state import State
from .setn import BaseSetNode, CompoundSetNode
from .opern import RelationalOperationNode
from .exprn import ExpressionNode
from .expression import Expression, get_var_nodes, get_param_nodes


class MetaEntity(ABC):
    def __init__(
        self,
        symbol: str,
        alias: str = None,
        idx_meta_sets: Union[List["MetaSet"], Dict[str, "MetaSet"]] = None,
        idx_set_node: CompoundSetNode = None,
        parent: "MetaEntity" = None,
    ):

        if idx_meta_sets is None:
            idx_meta_sets = []
        elif isinstance(idx_meta_sets, dict):
            idx_meta_sets = [ms for ms in idx_meta_sets.values()]

        self._symbol: str = symbol
        self._non_std_symbol = None

        self._alias: str = alias

        self._idx_meta_sets: List["MetaSet"] = idx_meta_sets
        self.idx_set_node: CompoundSetNode = idx_set_node

        self._parent: "MetaEntity" = parent

    def __hash__(self):
        return id(self)

    def __str__(self):
        return self.generate_declaration()

    # Type
    # ------------------------------------------------------------------------------------------------------------------

    @property
    @abstractmethod
    def type(self) -> str:
        pass

    # Properties
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def symbol(self) -> str:
        if self._parent is None:
            return self._symbol
        else:
            return self._parent.symbol

    @symbol.setter
    def symbol(self, symbol: str):
        if self._parent is None:
            self._symbol = symbol
        else:
            self._parent.symbol = symbol

    @property
    def non_std_symbol(self) -> str:
        if self._parent is None:
            return self._non_std_symbol
        else:
            return self._parent.non_std_symbol

    @non_std_symbol.setter
    def non_std_symbol(self, symbol: str):
        if self._parent is None:
            self._non_std_symbol = symbol
        else:
            self._parent.non_std_symbol = symbol

    @property
    def alias(self) -> str:
        if self._parent is None:
            return self._alias
        else:
            return self._parent.alias

    @alias.setter
    def alias(self, alias: str):
        if self._parent is None:
            self._alias = alias
        else:
            self._parent.alias = alias

    @property
    def parent(self) -> Optional["MetaEntity"]:
        if self._parent is not None:
            return self._parent
        else:
            return None

    @property
    @abstractmethod
    def expression_nodes(self) -> List[ExpressionNode]:
        pass

    # Indexing
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def is_indexed(self) -> bool:
        return self.idx_set_node is not None

    def is_indexed_with(self, meta_set: "MetaSet") -> bool:
        return any(map(lambda ms: ms.symbol == meta_set.symbol, self.idx_meta_sets))

    @property
    def idx_set_dim(self) -> int:
        return sum([meta_set.dim for meta_set in self.idx_meta_sets])

    @property
    def idx_set_reduced_dim(self) -> int:
        return sum([meta_set.reduced_dim for meta_set in self.idx_meta_sets])

    def is_idx_set_dim_fixed(self) -> List[bool]:

        combined_fixed_dim_flags = []

        for idx_meta_set in self.idx_meta_sets:
            combined_fixed_dim_flags += [
                idx_meta_set.is_dim_fixed(i) for i in range(idx_meta_set.dim)
            ]

        return combined_fixed_dim_flags

    @property
    def idx_set_dummy_element(self) -> List[str]:
        dummy_element = []
        for meta_set in self.idx_meta_sets:
            dummy_element.extend(meta_set.dummy_element)
        return dummy_element

    @property
    def idx_set_reduced_dummy_element(self) -> List[str]:
        dummy_element = []
        for meta_set in self.idx_meta_sets:
            dummy_element.extend(meta_set.reduced_dummy_element)
        return dummy_element

    @property
    def idx_meta_sets(self) -> List["MetaSet"]:
        if self._parent is None:
            return self._idx_meta_sets
        else:
            return self._parent.idx_meta_sets

    @idx_meta_sets.setter
    def idx_meta_sets(self, idx_meta_sets: List["MetaSet"]):
        if self._parent is None:
            self._idx_meta_sets = idx_meta_sets
        else:
            self._parent.idx_meta_sets = idx_meta_sets

    @property
    def idx_set_con_literal(self) -> Optional[str]:
        if self.idx_set_node is not None:
            if self.idx_set_node.constraint_node is not None:
                return self.idx_set_node.constraint_node.get_literal()
        return None

    def evaluate_reduced_idx_set(self, state: State) -> Optional[IndexingSet]:

        if not self.is_indexed:  # scalar entity
            return None

        else:  # indexed entity

            idx_set = self.idx_set_node.evaluate(state=state)[0]

            if self.idx_set_reduced_dim < self.idx_set_dim:
                fixed_dim_flags = self.is_idx_set_dim_fixed()
                fixed_dim_pos = [
                    i for i, is_fixed in enumerate(fixed_dim_flags) if is_fixed
                ]
                idx_set = remove_set_dimensions(
                    set_in=idx_set, dim_positions=fixed_dim_pos
                )

            return idx_set

    def get_first_reduced_dim_index_of_idx_set(
        self, meta_set: "MetaSet"
    ) -> Optional[int]:
        """
        Get the positional index of the first dimension of the meta-entity's indexing set controlled by the supplied
        meta-set.

        :param meta_set: meta-set for the first positional index is returned
        :return: positional index or None if the meta-entity is not indexed with respect to the meta-set
        """

        # meta-entity is indexed with respect to the supplied meta-set
        if self.is_indexed_with(meta_set):
            counter = 0
            for ms in self.idx_meta_sets:
                if ms.symbol == meta_set.symbol:
                    return counter
                counter += ms.reduced_dim

        # meta-entity is not indexed with respect to the supplied meta-set
        else:
            return None  # return

    def get_indexing_set_by_position(self, pos: int) -> "MetaSet":
        if pos >= self.idx_set_reduced_dim:
            raise ValueError("Position is out of range")
        p = 0
        for ms in self.idx_meta_sets:
            p += ms.reduced_dim
            if p > pos:
                return ms

    # Entity Ownership
    # ------------------------------------------------------------------------------------------------------------------

    def is_owner(self, entity: Entity, state: State):
        """
        Check whether the meta-entity owns the supplied entity.

        :param entity: algebraic entity
        :param state: problem state
        :return: True if the meta-entity is the owner of the entity argument
        """

        # Check whether symbols are identical
        if self.symbol != entity.symbol:
            return False

        # Check whether the entity index is a member of the meta-entity indexing set
        if self.idx_set_node is not None:
            idx_set = self.evaluate_reduced_idx_set(state)
            if entity.idx not in idx_set:
                return False

        return True

    # Sub-Entity
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def is_sub(self) -> bool:
        return self._parent is not None

    @abstractmethod
    def build_sub_entity(self, idx_set_node: CompoundSetNode = None) -> "MetaEntity":
        pass

    # Writing
    # ------------------------------------------------------------------------------------------------------------------

    @abstractmethod
    def generate_declaration(self) -> str:
        pass


class MetaSet(MetaEntity):
    def __init__(
        self,
        symbol: str = None,
        alias: str = None,
        idx_meta_sets: Union[List["MetaSet"], Dict[str, "MetaSet"]] = None,
        idx_set_node: CompoundSetNode = None,
        dimension: int = None,
        reduced_dimension: int = None,
        dummy_symbols: Element = None,
        reduced_dummy_symbols: Element = None,
        is_dim_fixed: List[bool] = None,
        super_set_node: BaseSetNode = None,
        defined_value_node: BaseSetNode = None,
        default_value_node: BaseSetNode = None,
        parent: "MetaSet" = None,
    ):

        super(MetaSet, self).__init__(
            symbol=symbol,
            alias=alias,
            idx_meta_sets=idx_meta_sets,
            idx_set_node=idx_set_node,
        )

        self._dim: int = dimension
        self._reduced_dim: int = (
            reduced_dimension  # set dimension after set constraint is applied
        )

        self._dummy_symbols: Element = dummy_symbols  # length: dimension
        self._reduced_dummy_symbols: Element = (
            reduced_dummy_symbols  # length: reduced dimension
        )

        self.is_init: bool = False
        self._is_dim_fixed: List[bool] = is_dim_fixed  # length: dimension

        self._super_set_node: BaseSetNode = super_set_node
        self._defined_value_node: BaseSetNode = defined_value_node
        self._default_value_node: BaseSetNode = default_value_node

        self._parent: "MetaSet" = parent

        self.initialize()

    def __str__(self):
        return "set {0}".format(self.symbol)

    # Initialization
    # ------------------------------------------------------------------------------------------------------------------

    def initialize(self):

        if self._dim is not None:

            if self._reduced_dim is None:
                if self._reduced_dummy_symbols is not None:
                    self._reduced_dim = len(self._reduced_dummy_symbols)
                elif self._is_dim_fixed is not None:
                    self._reduced_dim = self._is_dim_fixed.count(False)
                else:
                    self._reduced_dim = self._dim

            if self._is_dim_fixed is None:
                self._is_dim_fixed = [False] * self._dim
                if self._reduced_dim < self._dim:
                    delta_dim = self._dim - self._reduced_dim
                    self._is_dim_fixed[:delta_dim] = [True] * delta_dim

            if self._dummy_symbols is None:
                dummy_symbols = []
                base_index_symbol = self._symbol[0].lower()
                if self._dim == 1:
                    dummy_symbols.append(base_index_symbol)
                else:
                    for i in range(self._dim):
                        dummy_symbols.append(base_index_symbol + str(i))
                self._dummy_symbols = tuple(dummy_symbols)

            if self._reduced_dummy_symbols is None:

                if self._dim == self._reduced_dim:
                    reduced_dummy_symbols = list(self._dummy_symbols)

                else:
                    reduced_dummy_symbols = []
                    base_index_symbol = self._symbol[0].lower()
                    for i in range(self._dim):
                        if not self._is_dim_fixed[i]:
                            if self._dummy_symbols is not None:
                                reduced_dummy_symbols.append(self._dummy_symbols[i])
                            else:
                                reduced_dummy_symbols.append(base_index_symbol + str(i))

                self._reduced_dummy_symbols = reduced_dummy_symbols

            self.is_init = True

        else:
            self.is_init = False

    # Type
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def type(self) -> str:
        return SET_TYPE

    # Properties
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def dim(self) -> int:
        if self._parent is None:
            return self._dim
        else:
            return self._parent.dim

    @dim.setter
    def dim(self, dim: int):
        if self._parent is None:
            self._dim = dim
        else:
            self._parent.dim = dim

    @property
    def reduced_dim(self) -> int:
        if self._parent is None:
            return self._reduced_dim
        else:
            return self._parent.reduced_dim

    @reduced_dim.setter
    def reduced_dim(self, reduced_dim: int):
        if self._parent is None:
            self._reduced_dim = reduced_dim
        else:
            self._parent.reduced_dim = reduced_dim

    def is_dim_fixed(self, pos: int) -> bool:
        if self._parent is None:
            return self._is_dim_fixed[pos]
        else:
            return self._parent.is_dim_fixed(pos)

    @property
    def dummy_element(self) -> Element:
        if self._parent is None:
            return self._dummy_symbols
        else:
            return self._parent.dummy_element

    @dummy_element.setter
    def dummy_element(self, dummy_element: Element):
        if self._parent is None:
            self._dummy_symbols = dummy_element
        else:
            self._parent.dummy_element = dummy_element

    @property
    def reduced_dummy_element(self) -> Element:
        if self._parent is None:
            return self._reduced_dummy_symbols
        else:
            return self._parent.reduced_dummy_element

    @reduced_dummy_element.setter
    def reduced_dummy_element(self, dummy_element: Element):
        if self._parent is None:
            self._reduced_dummy_symbols = dummy_element
        else:
            self._parent.reduced_dummy_element = dummy_element

    @property
    def super_set_node(self) -> BaseSetNode:
        if self._parent is None:
            return self._super_set_node
        else:
            return self._parent.super_set_node

    @property
    def defined_value_node(self) -> BaseSetNode:
        if self._parent is None:
            return self._defined_value_node
        else:
            return self._parent.defined_value_node

    @property
    def default_value_node(self) -> BaseSetNode:
        if self._parent is None:
            return self._default_value_node
        else:
            return self._parent.default_value_node

    @property
    def expression_nodes(self) -> List[ExpressionNode]:

        expr_nodes = []

        if self.idx_set_node is not None:
            expr_nodes.append(self.idx_set_node)

        nodes = (self.defined_value_node, self.default_value_node, self.super_set_node)

        for node in nodes:
            if node is not None:
                expr_nodes.append(node)

        return expr_nodes

    # Sub-Entity
    # ------------------------------------------------------------------------------------------------------------------

    def build_sub_entity(self, idx_set_node: CompoundSetNode = None) -> "MetaSet":
        return MetaSet(parent=self, idx_set_node=idx_set_node)

    # Writing
    # ------------------------------------------------------------------------------------------------------------------

    def generate_declaration(self) -> str:

        declaration = "set {0}".format(self.symbol)

        alias = self.alias
        if alias is not None:
            declaration += " {0} ".format(alias)

        if self.idx_set_node is not None:
            declaration += str(self.idx_set_node)

        super_set_node = self.super_set_node
        if super_set_node is not None:
            declaration += " within {0}".format(super_set_node)

        defined_value_node = self.defined_value_node
        if defined_value_node is not None:
            declaration += " = {0}".format(defined_value_node)

        default_value_node = self.default_value_node
        if default_value_node is not None:
            declaration += " default {0}".format(default_value_node)

        declaration += ";"
        return declaration

    def generate_idx_set_literal(self) -> str:
        if self._dim == 0:
            return self._symbol
        elif self._dim == 1:
            return "{0} in {1}".format(self._dummy_symbols[0], self._symbol)
        else:
            index = "(" + ",".join([str(d) for d in self._dummy_symbols]) + ")"
            return "{0} in {1}".format(index, self._symbol)


class MetaParameter(MetaEntity):
    def __init__(
        self,
        symbol: str = None,
        alias: str = None,
        idx_meta_sets: Union[List[MetaSet], Dict[str, MetaSet]] = None,
        idx_set_node: CompoundSetNode = None,
        is_binary: bool = False,
        is_integer: bool = False,
        is_symbolic: bool = False,
        defined_value: ExpressionNode = None,
        default_value: ExpressionNode = None,
        super_set_node: BaseSetNode = None,
        relational_constraints: Dict[str, ExpressionNode] = None,
        parent: "MetaParameter" = None,
    ):

        super(MetaParameter, self).__init__(
            symbol=symbol,
            alias=alias,
            idx_meta_sets=idx_meta_sets,
            idx_set_node=idx_set_node,
        )

        self._is_binary: bool = is_binary
        self._is_integer: bool = is_integer
        self._is_symbolic: bool = is_symbolic
        self._defined_value: ExpressionNode = defined_value
        self._default_value: ExpressionNode = default_value
        self._super_set_node: BaseSetNode = super_set_node
        self._relational_constraints: Dict[str, ExpressionNode] = (
            relational_constraints if relational_constraints is not None else {}
        )

        self._parent: "MetaParameter" = parent

    def __str__(self):
        declaration = "param {0}".format(self.symbol)
        if self.idx_set_dim > 0 and self.idx_set_node is not None:
            declaration += str(self.idx_set_node)
        return declaration

    # Type
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def type(self) -> str:
        return PARAM_TYPE

    # Properties
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def is_binary(self) -> bool:
        if self._parent is None:
            return self._is_binary
        else:
            return self._parent.is_binary

    @property
    def is_integer(self) -> bool:
        if self._parent is None:
            return self._is_integer
        else:
            return self._parent.is_integer

    @property
    def is_symbolic(self) -> bool:
        if self._parent is None:
            return self._is_symbolic
        else:
            return self._parent.is_symbolic

    @property
    def defined_value_node(self) -> ExpressionNode:
        if self._parent is None:
            return self._defined_value
        else:
            return self._parent.defined_value_node

    @property
    def default_value_node(self) -> ExpressionNode:
        if self._parent is None:
            return self._default_value
        else:
            return self._parent.default_value_node

    @property
    def super_set_node(self) -> BaseSetNode:
        if self._parent is None:
            return self._super_set_node
        else:
            return self._parent.super_set_node

    @property
    def relational_constraints(self) -> Dict[str, ExpressionNode]:
        if self._parent is None:
            return self._relational_constraints
        else:
            return self._parent.relational_constraints

    @property
    def expression_nodes(self) -> List[ExpressionNode]:

        expr_nodes = []

        if self.idx_set_node is not None:
            expr_nodes.append(self.idx_set_node)

        nodes = (self.defined_value_node, self.default_value_node, self.super_set_node)

        for node in nodes:
            if node is not None:
                expr_nodes.append(node)

        for expr_node in self.relational_constraints.values():
            expr_nodes.append(expr_node)

        return expr_nodes

    # Sub-Entity
    # ------------------------------------------------------------------------------------------------------------------

    def build_sub_entity(self, idx_set_node: CompoundSetNode = None) -> "MetaParameter":
        return MetaParameter(parent=self, idx_set_node=idx_set_node)

    # Writing
    # ------------------------------------------------------------------------------------------------------------------

    def generate_declaration(self) -> str:

        declaration = "param {0}".format(self.symbol)

        alias = self.alias
        if alias is not None:
            declaration += " {0} ".format(alias)

        if self.idx_set_node is not None:
            declaration += str(self.idx_set_node)

        attributes = []

        if self.is_binary:
            attributes.append("binary")
        if self.is_integer:
            attributes.append("integer")
        if self.is_symbolic:
            attributes.append("symbolic")

        super_set_node = self.super_set_node
        if super_set_node is not None:
            attributes.append("in {0}".format(super_set_node))

        defined_value_node = self.defined_value_node
        if defined_value_node is not None:
            attributes.append("= {0}".format(defined_value_node))

        default_value_node = self.default_value_node
        if default_value_node is not None:
            attributes.append("default {0}".format(default_value_node))

        for rel_opr, node in self.relational_constraints.items():
            attributes.append("{0} {1}".format(rel_opr, node))

        if len(attributes) > 0:
            declaration += " " + ", ".join(attributes)

        declaration += ";"

        return declaration


class MetaVariable(MetaEntity):
    def __init__(
        self,
        symbol: str = None,
        alias: str = None,
        idx_meta_sets: Union[List[MetaSet], Dict[str, MetaSet]] = None,
        idx_set_node: CompoundSetNode = None,
        is_binary: bool = False,
        is_integer: bool = False,
        is_symbolic: bool = False,
        default_value: ExpressionNode = None,
        defined_value: ExpressionNode = None,
        lower_bound: ExpressionNode = None,
        upper_bound: ExpressionNode = None,
        parent: "MetaVariable" = None,
    ):

        super(MetaVariable, self).__init__(
            symbol=symbol,
            alias=alias,
            idx_meta_sets=idx_meta_sets,
            idx_set_node=idx_set_node,
            parent=parent,
        )

        self._is_binary: bool = is_binary
        self._is_integer: bool = is_integer
        self._is_symbolic: bool = is_symbolic
        self._default_value: ExpressionNode = default_value
        self._defined_value: ExpressionNode = defined_value
        self._lower_bound: ExpressionNode = lower_bound
        self._upper_bound: ExpressionNode = upper_bound

        self._parent: "MetaVariable" = parent

    def __str__(self):
        declaration = "var {0}".format(self.symbol)
        if self.idx_set_dim > 0 and self.idx_set_node is not None:
            declaration += str(self.idx_set_node)
        return declaration

    # Type
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def type(self) -> str:
        return VAR_TYPE

    # Properties
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def is_binary(self) -> bool:
        if self._parent is None:
            return self._is_binary
        else:
            return self._parent.is_binary

    @property
    def is_integer(self) -> bool:
        if self._parent is None:
            return self._is_integer
        else:
            return self._parent.is_integer

    @property
    def is_symbolic(self) -> bool:
        if self._parent is None:
            return self._is_symbolic
        else:
            return self._parent.is_symbolic

    @property
    def has_default(self):
        if self._parent is None:
            return self._default_value is not None
        else:
            return self._parent.has_default

    @property
    def default_value_node(self) -> ExpressionNode:
        if self._parent is None:
            return self._default_value
        else:
            return self._parent.default_value_node

    @property
    def is_defined(self) -> bool:
        if self._parent is None:
            return self._defined_value is not None
        else:
            return self._parent.is_defined

    @property
    def defined_value_node(self) -> ExpressionNode:
        if self._parent is None:
            return self._defined_value
        else:
            return self._parent.defined_value_node

    @property
    def lower_bound_node(self) -> ExpressionNode:
        if self._parent is None:
            return self._lower_bound
        else:
            return self._parent.lower_bound_node

    @property
    def upper_bound_node(self) -> ExpressionNode:
        if self._parent is None:
            return self._upper_bound
        else:
            return self._parent.upper_bound_node

    @property
    def expression_nodes(self) -> List[ExpressionNode]:

        expr_nodes = []

        if self.idx_set_node is not None:
            expr_nodes.append(self.idx_set_node)

        nodes = (
            self.default_value_node,
            self.defined_value_node,
            self.lower_bound_node,
            self.upper_bound_node,
        )

        for node in nodes:
            if node is not None:
                expr_nodes.append(node)

        return expr_nodes

    # Sub-Entity
    # ------------------------------------------------------------------------------------------------------------------

    def build_sub_entity(self, idx_set_node: CompoundSetNode = None) -> "MetaVariable":
        return MetaVariable(parent=self, idx_set_node=idx_set_node)

    # Writing
    # ------------------------------------------------------------------------------------------------------------------

    def generate_declaration(self) -> str:

        declaration = "var {0}".format(self.symbol)

        alias = self.alias
        if alias is not None:
            declaration += " {0} ".format(alias)

        if self.idx_set_node is not None:
            declaration += str(self.idx_set_node)

        attributes = []

        if self.is_binary:
            attributes.append("binary")
        if self.is_integer:
            attributes.append("integer")
        if self.is_symbolic:
            attributes.append("symbolic")

        default_value_node = self.default_value_node
        if default_value_node is not None:
            attributes.append(":= {0}".format(default_value_node))

        defined_value_node = self.defined_value_node
        if defined_value_node is not None:
            attributes.append("= {0}".format(defined_value_node))

        lower_bound_node = self.lower_bound_node
        if lower_bound_node is not None:
            attributes.append(">= {0}".format(lower_bound_node))

        upper_bound_node = self.upper_bound_node
        if upper_bound_node is not None:
            attributes.append("<= {0}".format(upper_bound_node))

        if len(attributes) > 0:
            declaration += " " + ", ".join(attributes)

        declaration += ";"

        return declaration


class MetaObjective(MetaEntity):

    MINIMIZE_DIRECTION = "minimize"
    MAXIMIZE_DIRECTION = "maximize"

    def __init__(
        self,
        symbol: str = None,
        alias: str = None,
        idx_meta_sets: Union[List[MetaSet], Dict[str, MetaSet]] = None,
        idx_set_node: CompoundSetNode = None,
        direction: str = "minimize",
        expression: Expression = None,
        parent: "MetaObjective" = None,
    ):

        super(MetaObjective, self).__init__(
            symbol=symbol,
            alias=alias,
            idx_meta_sets=idx_meta_sets,
            idx_set_node=idx_set_node,
        )

        if direction not in ["minimize", "maximize"]:
            direction = "minimize"

        self._direction: str = direction
        self._expression: Expression = expression

        self._parent: "MetaObjective" = parent

    def __str__(self):
        declaration = "{0} {1}".format(self.direction, self.symbol)
        if self.idx_set_dim > 0 and self.idx_set_node is not None:
            declaration += str(self.idx_set_node)
        return declaration

    # Properties
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def type(self) -> str:
        return OBJ_TYPE

    @property
    def direction(self) -> str:
        if self._parent is None:
            return self._direction
        else:
            return self._parent.direction

    @direction.setter
    def direction(self, direction: str):
        if self._parent is None:
            self._direction = direction
        else:
            self._parent.direction = direction

    @property
    def expression(self) -> Expression:
        if self._parent is None:
            return self._expression
        else:
            return self._parent.expression

    @expression.setter
    def expression(self, expr: Expression):
        if self._parent is None:
            self._expression = expr
        else:
            self._parent.expression = expr

    @property
    def expression_nodes(self) -> List[ExpressionNode]:
        expr_nodes = [self.expression.root_node]
        if self.idx_set_node is not None:
            expr_nodes.append(self.idx_set_node)
        return expr_nodes

    @property
    def variable_symbols(self):
        nodes = get_var_nodes(self.expression.root_node)
        return [n.symbol for n in nodes]

    @property
    def parameter_symbols(self):
        nodes = get_param_nodes(self.expression.root_node)
        return [n.symbol for n in nodes]

    # Sub-Entity
    # ------------------------------------------------------------------------------------------------------------------

    def build_sub_entity(self, idx_set_node: CompoundSetNode = None) -> "MetaObjective":
        return MetaObjective(parent=self, idx_set_node=idx_set_node)

    # Writing
    # ------------------------------------------------------------------------------------------------------------------

    def generate_declaration(self) -> str:

        declaration = "{0} {1}".format(self._direction, self.symbol)

        alias = self.alias
        if alias is not None:
            declaration += " {0} ".format(alias)

        if self.idx_set_node is not None:
            declaration += str(self.idx_set_node)

        declaration += ": {0}".format(self.expression)

        declaration += ";"
        return declaration


class MetaConstraint(MetaEntity):

    EQUALITY_TYPE = "eq"
    INEQUALITY_TYPE = "ineq"
    DOUBLE_INEQUALITY_TYPE = "dbl_ineq"

    def __init__(
        self,
        symbol: str = None,
        alias: str = None,
        idx_meta_sets: Union[List[MetaSet], Dict[str, MetaSet]] = None,
        idx_set_node: CompoundSetNode = None,
        expression: Expression = None,
        ctype: str = None,
        parent: "MetaConstraint" = None,
    ):

        super(MetaConstraint, self).__init__(
            symbol=symbol,
            alias=alias,
            idx_meta_sets=idx_meta_sets,
            idx_set_node=idx_set_node,
        )
        self._expression: Expression = expression
        self._ctype: str = ctype

        self._parent: "MetaConstraint" = parent

    def __str__(self):
        declaration = "subject to {0}".format(self.symbol)
        if self.idx_set_dim > 0 and self.idx_set_node is not None:
            declaration += str(self.idx_set_node)
        return declaration

    # Properties
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def type(self) -> str:
        return CON_TYPE

    @property
    def constraint_type(self) -> str:
        if self._parent is None:
            return self._ctype
        else:
            return self._parent.constraint_type

    def elicit_constraint_type(self):

        if self._parent is not None:
            return self._parent.elicit_constraint_type()

        expr_node = self._expression.root_node

        if isinstance(expr_node, RelationalOperationNode):
            if expr_node.operator == EQUALITY_OPERATOR:
                self._ctype = self.EQUALITY_TYPE
            elif expr_node.operator in [
                LESS_INEQUALITY_OPERATOR,
                LESS_EQUAL_INEQUALITY_OPERATOR,
                GREATER_INEQUALITY_OPERATOR,
                GREATER_EQUAL_INEQUALITY_OPERATOR,
            ]:
                if isinstance(expr_node.lhs_operand, RelationalOperationNode):
                    self._ctype = self.DOUBLE_INEQUALITY_TYPE
                elif isinstance(expr_node.rhs_operand, RelationalOperationNode):
                    self._ctype = self.DOUBLE_INEQUALITY_TYPE
                else:
                    self._ctype = self.INEQUALITY_TYPE
            return self._ctype

        raise ValueError(
            "Meta-constraint '{0}' expected an equality or an inequality expression".format(
                self._symbol
            )
            + " while eliciting the constraint type from the expression node '{0}'".format(
                expr_node
            )
        )

    @property
    def expression(self) -> Expression:
        if self._parent is None:
            return self._expression
        else:
            return self._parent.expression

    @expression.setter
    def expression(self, expr: Expression):
        if self._parent is None:
            self._expression = expr
        else:
            self._parent.expression = expr

    @property
    def expression_nodes(self) -> List[ExpressionNode]:
        expr_nodes = [self.expression.root_node]
        if self.idx_set_node is not None:
            expr_nodes.append(self.idx_set_node)
        return expr_nodes

    @property
    def variable_symbols(self):
        nodes = get_var_nodes(self.expression.root_node)
        return [n.symbol for n in nodes]

    @property
    def parameter_symbols(self):
        nodes = get_param_nodes(self.expression.root_node)
        return [n.symbol for n in nodes]

    # Sub-Entity
    # ------------------------------------------------------------------------------------------------------------------

    def build_sub_entity(
        self, idx_set_node: CompoundSetNode = None
    ) -> "MetaConstraint":
        return MetaConstraint(parent=self, idx_set_node=idx_set_node)

    # Writing
    # ------------------------------------------------------------------------------------------------------------------

    def generate_declaration(self) -> str:

        declaration = "{0}".format(self.symbol)

        alias = self.alias
        if alias is not None:
            declaration += " {0} ".format(alias)

        if self.idx_set_node is not None:
            declaration += str(self.idx_set_node)

        declaration += ": {0}".format(self.expression)

        declaration += ";"
        return declaration
