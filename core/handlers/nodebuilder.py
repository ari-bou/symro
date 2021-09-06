from queue import Queue
from typing import Dict, Iterable, List, Optional, Set, Tuple, Union

import symro.core.mat as mat
from symro.core.prob.problem import Problem
from symro.core.parsing.amplparser import AMPLParser


class NodeBuilder:

    def __init__(self, problem: Problem = None):
        self.problem: Problem = problem
        self.dummy_sym_mapping: Optional[Dict[str, str]] = None  # mapping of dummy symbols created by certain methods
        self.__free_node_id: int = 0

    # Symbol Generation
    # ------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def generate_unique_symbol(base_symbol: str = None,
                               scope_symbols: Iterable[str] = None):

        if base_symbol is None:
            base_symbol = "ENTITY"
        if scope_symbols is None:
            return base_symbol

        i = 1
        sym = base_symbol
        while sym in scope_symbols:
            sym = base_symbol + str(i)
            i += 1

        return sym

    # Unbound Symbols
    # ------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def retrieve_unbound_symbols(node: mat.ExpressionNode,
                                 in_filter: Iterable[str] = None) -> Set[str]:
        """
        Retrieve the unbound symbols that are present in an expression tree. Includes unbound symbols defined in scope
        and in the outer scope.
        :param node: root of the expression tree
        :param in_filter: inclusive filter set of symbols to retrieve if present in the expression tree
        :return: set of unbound symbols present in the expression tree
        """

        dummy_syms = set()

        queue = Queue()
        queue.put(node)

        while not queue.empty():

            node = queue.get()

            if isinstance(node, mat.DummyNode):
                if in_filter is None:
                    dummy_syms.add(node.symbol)
                elif node.symbol in in_filter:
                    dummy_syms.add(node.symbol)
            else:
                children = node.get_children()
                for child in children:
                    queue.put(child)

        return dummy_syms

    @staticmethod
    def replace_unbound_symbols(node: mat.ExpressionNode, mapping: Dict[str, str]):
        """
        Replace a selection of dummy symbols in an expression tree. Modifies the dummy nodes rather than replacing
        them with new objects.
        :param node: root of the expression tree
        :param mapping: dictionary of original dummy symbols mapped to replacement dummy symbols.
        :return: None
        """

        if mapping is None:
            return
        if len(mapping) == 0:
            return

        queue = Queue()
        queue.put(node)

        while not queue.empty():

            node = queue.get()

            if isinstance(node, mat.DummyNode):
                if node.symbol in mapping:
                    node.symbol = mapping[node.symbol]
            else:
                children = node.get_children()
                for child in children:
                    queue.put(child)

    @staticmethod
    def map_unbound_symbols_to_controlling_sets(set_nodes: List[mat.SetExpressionNode],
                                                outer_scope_unbound_syms: Iterable[str] = None
                                                ) -> Dict[str, mat.IndexingSetNode]:

        if outer_scope_unbound_syms is None:
            outer_scope_unbound_syms = set()

        mapping = {}

        def assign_set_node_to_unbound_symbol(unbound_sym: str, sn: mat.IndexingSetNode):
            if unbound_sym not in mapping and unbound_sym not in outer_scope_unbound_syms:
                mapping[unbound_sym] = sn

        for set_node in set_nodes:

            if isinstance(set_node, mat.IndexingSetNode):

                dummy_node = set_node.dummy_node

                if isinstance(dummy_node, mat.DummyNode):
                    assign_set_node_to_unbound_symbol(dummy_node.symbol, set_node)

                elif isinstance(dummy_node, mat.CompoundDummyNode):
                    for component_dummy_node in dummy_node.component_nodes:
                        if isinstance(component_dummy_node, mat.DummyNode):
                            assign_set_node_to_unbound_symbol(component_dummy_node.symbol, set_node)

        return mapping

    def generate_unbound_symbol_clash_replacement_map(self,
                                                      node: mat.ExpressionNode,
                                                      outer_scope_unbound_syms: Iterable[str] = None,
                                                      blacklisted_unbound_syms: Iterable[str] = None):

        mapping = {}
        current_scope_unbound_syms = set()
        if outer_scope_unbound_syms is None:
            outer_scope_unbound_syms = set()
        if blacklisted_unbound_syms is None:
            blacklisted_unbound_syms = set()

        queue = Queue()
        queue.put(node)

        while not queue.empty():

            node = queue.get()

            if isinstance(node, mat.DummyNode):
                if node.symbol not in outer_scope_unbound_syms:
                    # add the unbound symbol to the set of unique unbound symbols defined in the current scope
                    current_scope_unbound_syms.add(node.symbol)

            else:

                # unbound symbols may only be declared in the current scope through an indexing set node
                if isinstance(node, mat.IndexingSetNode):

                    dummy_node = node.dummy_node

                    if isinstance(dummy_node, mat.DummyNode):  # scalar dummy node
                        if dummy_node.symbol not in outer_scope_unbound_syms:
                            if dummy_node.symbol in current_scope_unbound_syms \
                                    or dummy_node.symbol in blacklisted_unbound_syms:
                                mapping[dummy_node.symbol] = ""
                        else:
                            raise ValueError("Node builder encountered an indexing set node '{0}'".format(node)
                                             + " without an uncontrolled dummy")

                    elif isinstance(dummy_node, mat.CompoundDummyNode):  # compound dummy node
                        for component_node in dummy_node.component_nodes:
                            if isinstance(component_node, mat.DummyNode):
                                if component_node.symbol not in outer_scope_unbound_syms:
                                    if component_node.symbol in current_scope_unbound_syms \
                                            or component_node.symbol in blacklisted_unbound_syms:
                                        mapping[component_node.symbol] = ""

                children = node.get_children()
                for child in children:
                    queue.put(child)

        defined_symbols = set(outer_scope_unbound_syms) | current_scope_unbound_syms  # union
        defined_symbols = defined_symbols | set(blacklisted_unbound_syms)  # union
        for non_unique_unbound_sym in mapping:
            unique_unbound_sym = self.generate_unique_symbol(non_unique_unbound_sym, defined_symbols)
            mapping[non_unique_unbound_sym] = unique_unbound_sym

        return mapping

    # Entity Node Construction
    # ------------------------------------------------------------------------------------------------------------------

    def build_default_entity_node(self, meta_entity: mat.MetaEntity) -> mat.DeclaredEntityNode:
        entity_index_node = self.build_default_entity_index_node(meta_entity)
        return mat.DeclaredEntityNode(symbol=meta_entity.symbol,
                                      entity_index_node=entity_index_node,
                                      type=meta_entity.get_type())

    def build_default_entity_index_node(self,
                                        meta_entity: mat.MetaEntity) -> Optional[mat.CompoundDummyNode]:

        if meta_entity.get_reduced_dimension() == 0:
            return None

        component_nodes = []
        dummy_symbols = meta_entity.get_dummy_symbols()
        for ds in dummy_symbols:
            component_nodes.append(mat.DummyNode(id=self.generate_free_node_id(),
                                                 symbol=ds))

        return mat.CompoundDummyNode(id=self.generate_free_node_id(),
                                     component_nodes=component_nodes)

    # Indexing Set Node Construction
    # ------------------------------------------------------------------------------------------------------------------

    def build_entity_idx_set_node(self,
                                  meta_entity: mat.MetaEntity,
                                  remove_sets: Union[List[Union[str, mat.MetaSet]],
                                                     Dict[str, mat.MetaSet]] = None,
                                  custom_dummy_syms: Dict[str, Union[str, Tuple[str, ...]]] = None
                                  ) -> Optional[mat.CompoundSetNode]:
        """
        Build an indexing set node for a meta-entity. By default, the dummy indices are based on the default dummy
        symbols of the indexing meta-sets.
        :param meta_entity: meta-entity for which an indexing set node is built.
        :param remove_sets: List of indexing meta-sets (or their symbols) to be excluded from the indexing set node.
        :param custom_dummy_syms: Mapping of meta-set symbols to custom dummy symbols. A dummy symbol in this dict will
        override the default dummy symbol of the corresponding meta-set. Use a tuple for multi-dimensional sets.
        :return: indexing set node
        """

        def get_set_sym(s) -> str:
            if isinstance(s, str):
                return s
            elif isinstance(s, mat.MetaSet):
                return s.symbol

        if meta_entity.get_dimension() == 0:
            return None

        indexing_meta_sets = [ms for name, ms in meta_entity.idx_meta_sets.items()]

        # Remove controlled sets
        if remove_sets is not None:
            if isinstance(remove_sets, dict):
                remove_sets = [ms for ms in remove_sets.values()]
            remove_sets = [get_set_sym(s) for s in remove_sets]
            indexing_meta_sets = [ms for ms in indexing_meta_sets if ms.symbol not in remove_sets]

        return self.build_idx_set_node(indexing_meta_sets,
                                       meta_entity.idx_set_con_literal,
                                       custom_dummy_syms)

    def build_idx_set_node(self,
                           idx_meta_sets: Optional[Union[List[mat.MetaSet], Dict[str, mat.MetaSet]]],
                           idx_set_con_literal: str = None,
                           custom_dummy_syms: Dict[str, Union[str, Tuple[str, ...]]] = None
                           ) -> Optional[mat.CompoundSetNode]:
        """
        Build an indexing set node from a collection of a indexing meta-sets and an optional set constraint.
        :param idx_meta_sets: Collection of indexing meta-sets.
        :param idx_set_con_literal: String literal of the set constraint.
        :param custom_dummy_syms: Mapping of meta-set symbols to custom dummy symbols. A dummy symbol in this dict will
        override the default dummy symbol of the corresponding meta-set. Use a tuple for multi-dimensional sets.
        :return: indexing set node
        """

        if idx_meta_sets is None:
            return None
        if len(idx_meta_sets) == 0:
            return None
        if isinstance(idx_meta_sets, dict):
            idx_meta_sets = [ms for _, ms in idx_meta_sets.items()]

        if custom_dummy_syms is None:
            custom_dummy_syms = {}

        # Generate mapping of original and replacement dummy symbols
        all_dummy_syms = []
        dummy_sym_mapping = {}
        if len(custom_dummy_syms) > 0:

            # add all custom dummy symbols to the mapping
            for idx_meta_set in idx_meta_sets:

                if idx_meta_set.symbol in custom_dummy_syms:

                    # retrieve the default dummy symbols of the component set
                    default_dummy_syms_i = idx_meta_set.dummy_symbols

                    # retrieve the custom dummy symbols of the indexing meta-set
                    custom_dummy_syms_i = custom_dummy_syms[idx_meta_set.symbol]

                    # convert the custom dummy symbols to a list
                    if isinstance(custom_dummy_syms_i, int) or isinstance(custom_dummy_syms_i, float) \
                            or isinstance(custom_dummy_syms_i, str):
                        custom_dummy_syms_i = [custom_dummy_syms_i]

                    # append the custom dummy symbols to the mapping
                    for d, c in zip(default_dummy_syms_i, custom_dummy_syms_i):
                        dummy_sym_mapping[d] = c
                        all_dummy_syms.append(c)

            # the custom dummy symbols included above may overlap with one or more default dummy symbols...
            # identify non-unique default dummy symbols and generate unique replacement symbols
            for idx_meta_set in idx_meta_sets:

                if idx_meta_set.symbol not in custom_dummy_syms:

                    # retrieve the default dummy symbols of the component set
                    default_dummy_syms_i = idx_meta_set.dummy_symbols

                    for dummy_sym in default_dummy_syms_i:

                        if dummy_sym in all_dummy_syms:  # check whether the dummy symbol is unique

                            k = 0
                            while True:
                                k += 1
                                modified_dummy_sym = dummy_sym + str(k)
                                if modified_dummy_sym not in all_dummy_syms:
                                    break  # terminate the loop once a unique dummy symbol has been found

                            dummy_sym_mapping[dummy_sym] = modified_dummy_sym
                            all_dummy_syms.append(modified_dummy_sym)

        ampl_parser = AMPLParser(self.problem)

        # instantiate a list to contain the component set nodes
        component_set_nodes = []

        # build component set nodes
        for idx_meta_set in idx_meta_sets:

            dummy_syms = idx_meta_set.dummy_symbols  # retrieve the dummy symbols of the component set

            if len(dummy_sym_mapping) > 0:
                dummy_syms = [dummy_sym_mapping.get(d, d) for d in dummy_syms]  # replace selected dummy symbols

            # build an element node for each dummy
            dummy_element_nodes = [ampl_parser.parse_entity(d) for d in dummy_syms]

            if len(dummy_element_nodes) == 1:
                dummy_node = dummy_element_nodes[0]
            else:  # build a compound dummy node if there are multiple elements
                dummy_node = mat.CompoundDummyNode(id=self.generate_free_node_id(),
                                                   component_nodes=dummy_element_nodes)

            set_node = ampl_parser.parse_set_expression(idx_meta_set.symbol)

            # replace selected dummy nodes belonging to the set node
            if len(dummy_sym_mapping) > 0:
                self.replace_unbound_symbols(set_node, mapping=dummy_sym_mapping)

            component_set_node = mat.IndexingSetNode(id=self.generate_free_node_id(),
                                                     dummy_node=dummy_node,
                                                     set_node=set_node)
            component_set_nodes.append(component_set_node)

        # build constraint node
        con_node = None
        if idx_set_con_literal is not None:
            con_node = ampl_parser.parse_logical_expression(idx_set_con_literal)

        # build compound set node
        idx_set_node = mat.CompoundSetNode(id=self.generate_free_node_id(),
                                           set_nodes=component_set_nodes,
                                           constraint_node=con_node)

        self.dummy_sym_mapping = dummy_sym_mapping
        return idx_set_node

    def combine_idx_set_nodes(self, idx_set_nodes: Iterable[Optional[mat.CompoundSetNode]]):

        component_set_nodes = []
        con_operands = []

        for idx_set_node in idx_set_nodes:
            if idx_set_node is not None:
                component_set_nodes.extend(idx_set_node.set_nodes)
                if idx_set_node.constraint_node is not None:
                    con_operands.append(idx_set_node.constraint_node)

        con_node = self.build_conjunction_node(con_operands)

        if len(component_set_nodes) == 0:
            return None

        return mat.CompoundSetNode(id=self.generate_free_node_id(),
                                   set_nodes=component_set_nodes,
                                   constraint_node=con_node)

    # Boolean Operation Node Construction
    # ------------------------------------------------------------------------------------------------------------------

    def build_conjunction_node(self, operands: List[mat.LogicalExpressionNode]):

        if len(operands) == 0:
            return None

        elif len(operands) == 1:
            return operands[0]

        else:
            for op in operands:
                op.is_prioritized = True
            return mat.MultiLogicalOperationNode(id=self.generate_free_node_id(),
                                                 operator="&&",
                                                 operands=operands)

    def build_disjunction_node(self, operands: List[mat.LogicalExpressionNode]):

        if len(operands) == 0:
            return None

        elif len(operands) == 1:
            return operands[0]

        else:
            for op in operands:
                op.is_prioritized = True
            return mat.MultiLogicalOperationNode(id=self.generate_free_node_id(),
                                                 operator="||",
                                                 operands=operands)

    # Arithmetic Operation Node Construction
    # ------------------------------------------------------------------------------------------------------------------

    def build_addition_node(self, terms: List[mat.ArithmeticExpressionNode]):
        if len(terms) == 1:
            return terms[0]
        else:
            all_terms = []
            for term in terms:
                if isinstance(term, mat.MultiArithmeticOperationNode):
                    if term.operator == '+':
                        all_terms.extend(term.operands)
                    else:
                        all_terms.append(term)
                elif isinstance(term, mat.BinaryArithmeticOperationNode):
                    if term.operator == '+':
                        all_terms.append(term.lhs_operand)
                        all_terms.append(term.rhs_operand)
                    else:
                        all_terms.append(term)
                else:
                    all_terms.append(term)
            sum_node = mat.MultiArithmeticOperationNode(id=self.generate_free_node_id(),
                                                        operator='+',
                                                        operands=all_terms)
            return sum_node

    def build_multiplication_node(self, factors: List[mat.ArithmeticExpressionNode]):
        if len(factors) == 1:
            return factors[0]
        else:
            all_factors = []
            for factor in factors:
                if isinstance(factor, mat.MultiArithmeticOperationNode):
                    if factor.operator == '*':
                        all_factors.extend(factor.operands)
                    else:
                        all_factors.append(factor)
                elif isinstance(factor, mat.BinaryArithmeticOperationNode):
                    if factor.operator == '*':
                        all_factors.append(factor.lhs_operand)
                        all_factors.append(factor.rhs_operand)
                    else:
                        all_factors.append(factor)
                else:
                    all_factors.append(factor)
            mult_node = mat.MultiArithmeticOperationNode(id=self.generate_free_node_id(),
                                                         operator='*',
                                                         operands=all_factors)
            return mult_node

    def build_fractional_node(self,
                              numerator: mat.ArithmeticExpressionNode,
                              denominator: mat.ArithmeticExpressionNode):

        num_1 = numerator
        num_2 = None
        den_1 = None
        den_2 = denominator

        if isinstance(numerator, mat.BinaryArithmeticOperationNode):
            if numerator.operator == '/':
                num_1 = numerator.lhs_operand
                den_1 = numerator.rhs_operand
        if isinstance(denominator, mat.BinaryArithmeticOperationNode):
            if denominator.operator == '/':
                den_2 = denominator.lhs_operand
                num_2 = denominator.rhs_operand

        num = None
        if num_1 is not None and num_2 is not None:
            num = self.build_multiplication_node([num_1, num_2])
        elif num_1 is not None:
            num = num_1
        elif num_2 is not None:
            num = num_2

        den = None
        if den_1 is not None and den_2 is not None:
            den = self.build_multiplication_node([den_1, den_2])
        elif den_1 is not None:
            den = den_1
        elif den_2 is not None:
            den = den_2

        return mat.BinaryArithmeticOperationNode(id=self.generate_free_node_id(),
                                                 operator='/',
                                                 lhs_operand=num,
                                                 rhs_operand=den)

    def add_negative_unity_coefficient(self, node: mat.ArithmeticExpressionNode):
        if isinstance(node, mat.NumericNode):
            node.value *= -1
            return node
        else:
            coeff_node = mat.NumericNode(id=self.generate_free_node_id(), value=-1)
            mult_node = mat.BinaryArithmeticOperationNode(id=self.generate_free_node_id(),
                                                          operator='*',
                                                          lhs_operand=coeff_node,
                                                          rhs_operand=node)
            return mult_node

    # Utility
    # ------------------------------------------------------------------------------------------------------------------

    def generate_free_node_id(self) -> int:
        if self.problem is not None:
            return self.problem.generate_free_node_id()
        else:
            free_node_id = self.__free_node_id
            self.__free_node_id += 1
            return free_node_id

    def seed_free_node_id(self, node: mat.ExpressionNode) -> int:
        if self.problem is not None:
            self.problem.seed_free_node_id(node.get_free_id())
            return self.problem.generate_free_node_id()
        else:
            self.__free_node_id = max(self.__free_node_id, node.get_free_id())
            return self.generate_free_node_id()
