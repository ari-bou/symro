from copy import deepcopy
from queue import Queue
from typing import Dict, List, Optional, Set, Tuple, Union

import symro.core.mat as mat
from symro.core.prob.problem import BaseProblem, Problem
from symro.core.handlers.nodebuilder import NodeBuilder


class EntityBuilder:

    def __init__(self, problem: Problem):
        self.problem: Problem = problem
        self.node_builder: NodeBuilder = NodeBuilder(problem)

    # Meta-Entity Construction
    # ------------------------------------------------------------------------------------------------------------------

    def build_meta_set(self,
                       symbol: str,
                       alias: str = None,
                       idx_meta_sets: Union[List[mat.MetaSet], Dict[str, mat.MetaSet]] = None,
                       idx_set_con_literal: str = None,
                       idx_set_node: mat.CompoundSetNode = None,
                       dimension: int = None,
                       reduced_dimension: int = None,
                       dummy_symbols: List[str] = None,
                       reduced_dummy_symbols: List[str] = None,
                       is_dim_fixed: List[bool] = None,
                       super_set_node: mat.BaseSetNode = None,
                       set_node: mat.BaseSetNode = None):
        if idx_meta_sets is None and idx_set_node is not None:
            idx_meta_sets, idx_set_con_literal = self.build_idx_meta_sets_from_idx_set_node(idx_set_node)
        if idx_set_node is None:
            idx_set_node = self.node_builder.build_idx_set_node(idx_meta_sets, idx_set_con_literal)
        return mat.MetaSet(symbol=symbol,
                           alias=alias,
                           idx_meta_sets=idx_meta_sets,
                           idx_set_con_literal=idx_set_con_literal,
                           idx_set_node=idx_set_node,
                           dimension=dimension,
                           reduced_dimension=reduced_dimension,
                           dummy_symbols=dummy_symbols,
                           reduced_dummy_symbols=reduced_dummy_symbols,
                           is_dim_fixed=is_dim_fixed,
                           super_set_node=super_set_node,
                           set_node=set_node)

    def build_meta_param(self,
                         symbol: str,
                         alias: str = None,
                         idx_meta_sets: Union[List[mat.MetaSet], Dict[str, mat.MetaSet]] = None,
                         idx_set_con_literal: str = None,
                         idx_set_node: mat.CompoundSetNode = None,
                         is_binary: bool = False,
                         is_integer: bool = False,
                         is_symbolic: bool = False,
                         default_value: Union[int, float, str, mat.ExpressionNode] = None,
                         super_set_node: mat.BaseSetNode = None,
                         relational_constraints: Dict[str, mat.ExpressionNode] = None):
        if idx_meta_sets is None and idx_set_node is not None:
            idx_meta_sets, idx_set_con_literal = self.build_idx_meta_sets_from_idx_set_node(idx_set_node)
        if idx_set_node is None:
            idx_set_node = self.node_builder.build_idx_set_node(idx_meta_sets, idx_set_con_literal)
        meta_param = mat.MetaParameter(symbol=symbol,
                                       alias=alias,
                                       idx_meta_sets=idx_meta_sets,
                                       idx_set_con_literal=idx_set_con_literal,
                                       idx_set_node=idx_set_node,
                                       is_binary=is_binary,
                                       is_integer=is_integer,
                                       is_symbolic=is_symbolic,
                                       default_value=default_value,
                                       super_set_node=super_set_node,
                                       relational_constraints=relational_constraints)
        return meta_param

    def build_meta_var(self,
                       symbol: str,
                       alias: str = None,
                       idx_meta_sets: Union[List[mat.MetaSet], Dict[str, mat.MetaSet]] = None,
                       idx_set_con_literal: str = None,
                       idx_set_node: mat.CompoundSetNode = None,
                       is_binary: bool = False,
                       is_integer: bool = False,
                       is_symbolic: bool = False,
                       default_value: Union[int, float, str, mat.ExpressionNode] = None,
                       defined_value: Union[int, float, str, mat.ExpressionNode] = None,
                       lower_bound: Union[int, float, str, mat.ExpressionNode] = None,
                       upper_bound: Union[int, float, str, mat.ExpressionNode] = None):
        if idx_meta_sets is None and idx_set_node is not None:
            idx_meta_sets, idx_set_con_literal = self.build_idx_meta_sets_from_idx_set_node(idx_set_node)
        if idx_set_node is None:
            idx_set_node = self.node_builder.build_idx_set_node(idx_meta_sets, idx_set_con_literal)
        meta_var = mat.MetaVariable(symbol=symbol,
                                    alias=alias,
                                    idx_meta_sets=idx_meta_sets,
                                    idx_set_con_literal=idx_set_con_literal,
                                    idx_set_node=idx_set_node,
                                    is_binary=is_binary,
                                    is_integer=is_integer,
                                    is_symbolic=is_symbolic,
                                    default_value=default_value,
                                    defined_value=defined_value,
                                    lower_bound=lower_bound,
                                    upper_bound=upper_bound)
        return meta_var

    def build_meta_obj(self,
                       symbol: str,
                       alias: str = None,
                       idx_meta_sets: Union[List[mat.MetaSet], Dict[str, mat.MetaSet]] = None,
                       idx_set_con_literal: str = None,
                       idx_set_node: mat.CompoundSetNode = None,
                       direction: str = None,
                       expression: mat.Expression = None):
        if idx_meta_sets is None and idx_set_node is not None:
            idx_meta_sets, idx_set_con_literal = self.build_idx_meta_sets_from_idx_set_node(idx_set_node)
        if idx_set_node is None:
            idx_set_node = self.node_builder.build_idx_set_node(idx_meta_sets, idx_set_con_literal)
        meta_obj = mat.MetaObjective(symbol=symbol,
                                     alias=alias,
                                     idx_meta_sets=idx_meta_sets,
                                     idx_set_con_literal=idx_set_con_literal,
                                     idx_set_node=idx_set_node,
                                     direction=direction,
                                     expression=expression)
        return meta_obj

    def build_meta_con(self,
                       symbol: str,
                       alias: str = None,
                       idx_meta_sets: Union[List[mat.MetaSet], Dict[str, mat.MetaSet]] = None,
                       idx_set_con_literal: str = None,
                       idx_set_node: mat.CompoundSetNode = None,
                       expression: mat.Expression = None):
        if idx_meta_sets is None and idx_set_node is not None:
            idx_meta_sets, idx_set_con_literal = self.build_idx_meta_sets_from_idx_set_node(idx_set_node)
        if idx_set_node is None:
            idx_set_node = self.node_builder.build_idx_set_node(idx_meta_sets, idx_set_con_literal)
        meta_con = mat.MetaConstraint(symbol=symbol,
                                      alias=alias,
                                      idx_meta_sets=idx_meta_sets,
                                      idx_set_con_literal=idx_set_con_literal,
                                      idx_set_node=idx_set_node,
                                      expression=expression)
        return meta_con

    # Indexing Set Construction
    # ------------------------------------------------------------------------------------------------------------------

    def build_idx_meta_sets_of_model(self):

        def build(me: mat.MetaEntity):
            idx_meta_sets, idx_set_con_literal = self.build_idx_meta_sets_from_idx_set_node(me.idx_set_node)
            me.idx_meta_sets = idx_meta_sets
            me.idx_set_con_literal = idx_set_con_literal

        for meta_entity in self.problem.model_meta_sets_params:
            build(meta_entity)

        for meta_entity in self.problem.model_meta_vars:
            build(meta_entity)

        for meta_entity in self.problem.model_meta_objs:
            build(meta_entity)

        for meta_entity in self.problem.model_meta_cons:
            build(meta_entity)

    def build_idx_meta_sets_from_idx_set_node(self, idx_set_node: mat.CompoundSetNode):

        idx_meta_sets = {}
        idx_set_con_literal = None

        if idx_set_node is None:
            return idx_meta_sets, idx_set_con_literal

        # Retrieve indexing set constraint
        if idx_set_node.constraint_node is not None:
            idx_set_con_literal = idx_set_node.constraint_node.get_literal()

        component_set_syms = []
        component_dims = []
        raw_component_dummies = []
        all_defined_indexing_dummies = set()

        # Pass 1:
        # - retrieve component set symbols, dimensions, and dummies
        # - store declared dummy symbols
        for component_set_node in idx_set_node.set_nodes:

            component_dim = component_set_node.get_dim(self.problem.state)

            # Component set is indexed
            if isinstance(component_set_node, mat.IndexingSetNode):
                component_set_sym = component_set_node.set_node.get_literal()
                dummy_node = component_set_node.dummy_node

                if isinstance(dummy_node, mat.CompoundDummyNode):
                    dummy_elements = list(dummy_node.component_nodes)
                elif isinstance(dummy_node, mat.DummyNode):
                    dummy_elements = [dummy_node.dummy]
                else:
                    raise ValueError("EntityBuilder expected a dummy node while resolving an indexing set")

                for de in dummy_elements:
                    if isinstance(de, mat.DummyNode):
                        all_defined_indexing_dummies.add(de.dummy)

            else:
                component_set_sym = component_set_node.get_literal()
                dummy_elements = [None] * component_dim

            component_set_syms.append(component_set_sym)
            component_dims.append(component_dim)
            raw_component_dummies.append(dummy_elements)

        # Pass 2: retrieve default dummy symbols of declared component sets
        for i, component_set_node in enumerate(idx_set_node.set_nodes):

            # Component set is declared
            if isinstance(component_set_node, mat.SetNode):

                dummy_elements = list(self.problem.meta_sets[component_set_node.symbol].dummy_symbols)
                for j, de in enumerate(dummy_elements):
                    if de not in all_defined_indexing_dummies:
                        all_defined_indexing_dummies.add(de)
                    else:
                        dummy_elements[j] = None
                raw_component_dummies[i] = dummy_elements

        # Pass 3: process dummy symbols
        k = 0
        prev_defined_indexing_dummies = []
        for i, component_set_node in enumerate(idx_set_node.set_nodes):

            component_set_sym = component_set_syms[i]
            component_dummy = []
            reduced_component_dummy = []
            is_dim_fixed = []

            for j, de in enumerate(raw_component_dummies[i]):

                if isinstance(de, mat.DummyNode):
                    component_dummy.append(de.dummy)
                    if de.dummy not in prev_defined_indexing_dummies:
                        reduced_component_dummy.append(de.dummy)
                        prev_defined_indexing_dummies.append(de.dummy)
                        is_dim_fixed.append(False)
                    else:
                        is_dim_fixed.append(True)

                elif isinstance(de, mat.StringNode):  # always fixed
                    component_dummy.append(de.get_literal())
                    is_dim_fixed.append(True)

                elif isinstance(de, mat.StringExpressionNode):  # always fixed
                    component_dummy.append(de.get_literal())
                    is_dim_fixed.append(True)

                elif isinstance(de, mat.ArithmeticExpressionNode):  # always fixed
                    component_dummy.append(de.get_literal())
                    is_dim_fixed.append(True)

                elif isinstance(de, str):  # indexing dummy of defined set node; will never be fixed
                    component_dummy.append(de)
                    reduced_component_dummy.append(de)
                    is_dim_fixed.append(False)
                    prev_defined_indexing_dummies.append(de)

                else:  # dummy element is undeclared (None); will never be fixed
                    dummy_element_base = ''
                    for c in component_set_syms[i]:
                        if c.isalpha():
                            dummy_element_base = c[0].lower()
                    if dummy_element_base == '':
                        dummy_element_base = 'i'
                    dummy_element = dummy_element_base
                    m = 1
                    while dummy_element in prev_defined_indexing_dummies:
                        dummy_element = dummy_element_base + str(m)
                        m += 1
                    component_dummy.append(dummy_element)
                    reduced_component_dummy.append(dummy_element)
                    is_dim_fixed.append(False)
                    prev_defined_indexing_dummies.append(dummy_element)

                k += 1

            reduced_component_dim = len(reduced_component_dummy)

            # Generate indexing meta-set
            indexing_meta_set = mat.MetaSet(symbol=component_set_sym,
                                            dimension=component_dims[i],
                                            reduced_dimension=reduced_component_dim,
                                            dummy_symbols=component_dummy,
                                            reduced_dummy_symbols=reduced_component_dummy,
                                            is_dim_fixed=is_dim_fixed)
            idx_meta_sets[component_set_sym] = indexing_meta_set

        return idx_meta_sets, idx_set_con_literal

    # Subproblem Construction
    # ------------------------------------------------------------------------------------------------------------------

    def build_subproblem(self,
                         prob_sym: str,
                         prob_idx_set_node: mat.CompoundSetNode,
                         entity_nodes: List[Tuple[Optional[mat.CompoundSetNode], mat.DeclaredEntityNode]]):

        sp = BaseProblem(symbol=prob_sym,
                         idx_set_node=prob_idx_set_node)

        for me_idx_set_node, e_node in entity_nodes:

            meta_entity = self.problem.get_meta_entity(e_node.symbol)

            if prob_idx_set_node is None and me_idx_set_node is None and e_node.idx_node is None:
                sp.add_meta_entity_to_model(meta_entity)

            else:
                idx_subset_node = self.node_builder.combine_idx_set_nodes([prob_idx_set_node, me_idx_set_node])
                sub_meta_entity = self.build_sub_meta_entity(idx_subset_node=deepcopy(idx_subset_node),
                                                             meta_entity=meta_entity,
                                                             entity_idx_node=deepcopy(e_node.idx_node))
                sp.add_meta_entity_to_model(sub_meta_entity)

        return sp

    # Sub-Meta-Entity Construction
    # ------------------------------------------------------------------------------------------------------------------

    def build_sub_meta_entity(self,
                              idx_subset_node: mat.CompoundSetNode,
                              meta_entity: mat.MetaEntity,
                              entity_idx_node: mat.CompoundDummyNode):
        sub_meta_entity_builder = self.SubMetaEntityBuilder(problem=self.problem,
                                                            node_builder=self.node_builder)
        sub_meta_entity = sub_meta_entity_builder.build_sub_meta_entity(idx_subset_node=idx_subset_node,
                                                                        meta_entity=meta_entity,
                                                                        entity_idx_node=entity_idx_node)
        return sub_meta_entity

    class SubMetaEntityBuilder:

        def __init__(self,
                     problem: Problem,
                     node_builder: NodeBuilder):
            self.problem: Problem = problem
            self.node_builder: NodeBuilder = node_builder

        def build_sub_meta_entity(self,
                                  idx_subset_node: mat.CompoundSetNode,
                                  meta_entity: mat.MetaEntity,
                                  entity_idx_node: mat.CompoundDummyNode):

            # return the parent meta-entity if it is scalar
            # return the parent meta-entity if no indexing subset is provided
            if idx_subset_node is None or meta_entity.get_reduced_dimension() == 0:
                return meta_entity

            idx_subset_node = self.__build_sub_meta_entity_idx_set_node(idx_subset_node,
                                                                        meta_entity,
                                                                        entity_idx_node)

            sub_meta_entity = deepcopy(meta_entity)
            sub_meta_entity.idx_set_node = idx_subset_node
            if idx_subset_node.constraint_node is not None:
                sub_meta_entity.idx_set_con_literal = str(idx_subset_node.constraint_node)
            else:
                sub_meta_entity.idx_set_con_literal = None

            return sub_meta_entity

        def __build_sub_meta_entity_idx_set_node(self,
                                                 idx_subset_node: mat.CompoundSetNode,
                                                 meta_entity: mat.MetaEntity,
                                                 entity_idx_node: mat.CompoundDummyNode):

            # check if the dimension of the entity index node matches that of the parent meta-entity's indexing set
            if entity_idx_node.get_dim() != meta_entity.get_reduced_dimension():
                sub_entity_decl = "{0} {1}{2}".format(idx_subset_node, meta_entity.symbol, entity_idx_node)
                raise ValueError("Entity builder encountered an incorrect entity declaration"
                                 + " '{0}' while building a sub-meta-entity:".format(sub_entity_decl)
                                 + " the dimension of the entity index does not match that of the parent entity")

            # deep copy the indexing set node of the parent meta-entity
            idx_set_node = deepcopy(meta_entity.idx_set_node)

            # instantiate a list of logical conjunctive operands for the indexing set constraint node
            idx_set_con_operands = []

            # add the original constraint node as the first operand
            if idx_set_node.constraint_node is not None:
                idx_set_con_operands.append(idx_set_node.constraint_node)

            # check whether a subset membership operation is necessary
            if self.__verify_subset_membership_node_requirement(idx_set_node=meta_entity.idx_set_node,
                                                                idx_subset_node=idx_subset_node,
                                                                entity_idx_node=entity_idx_node):

                # retrieve standard dummy symbols of the indexing set
                outer_unb_syms = meta_entity.get_dummy_symbols()

                # build the subset membership node
                subset_membership_node = self.__build_subset_membership_node(idx_subset_node=idx_subset_node,
                                                                             outer_unb_syms=outer_unb_syms,
                                                                             entity_idx_node=entity_idx_node)

                # add the set membership node to the list of operands
                if subset_membership_node is not None:
                    idx_set_con_operands.append(subset_membership_node)

            # combine the operands of the set constraint into a single node
            idx_set_con_node = self.node_builder.build_conjunction_node(idx_set_con_operands)
            idx_set_node.constraint_node = idx_set_con_node  # assign the constraint node to the indexing set node

            return idx_set_node

        def __verify_subset_membership_node_requirement(self,
                                                        idx_set_node: mat.CompoundSetNode,
                                                        idx_subset_node: mat.CompoundSetNode,
                                                        entity_idx_node: mat.CompoundDummyNode):
            """
            Verify whether a subset membership operation node is necessary.
            :param idx_set_node: indexing set of the parent meta-entity
            :param idx_subset_node: indexing set of the sub-meta-entity
            :param entity_idx_node: index node of the sub-meta-entity
            :return: True if a subset membership operation is necessary, False otherwise
            """

            idx_superset = idx_set_node.evaluate(state=self.problem.state)[0]
            idx_subset = idx_subset_node.evaluate(state=self.problem.state)[0]

            # step 1: check if the indexing subset is different from the indexing superset
            if len(idx_superset.symmetric_difference(idx_subset)) > 0:  # symmetric difference is not the empty set
                return True  # indexing superset and subset are different

            else:  # symmetric difference is the empty set

                # step 2: check if there are any component nodes in the entity index that are not uniquely controlled
                # by a dummy
                unbound_syms = set()
                for component_node in entity_idx_node.component_nodes:
                    if not isinstance(component_node, mat.DummyNode):  # arithmetic/string expression
                        return True  # component node is dependent
                    else:
                        if component_node.dummy in unbound_syms:  # dummy controls another component node
                            return True  # component node is dependent
                        unbound_syms.add(component_node.dummy)

                return False  # all component nodes are independently controlled by unique dummies

        def __build_subset_membership_node(self,
                                           idx_subset_node: mat.CompoundSetNode,
                                           outer_unb_syms: List[str],
                                           entity_idx_node: mat.CompoundDummyNode):

            set_node_map: Dict[int, mat.SetExpressionNode] = {node.id: node for node in idx_subset_node.set_nodes}
            set_node_positions = {node.id: pos for pos, node in enumerate(idx_subset_node.set_nodes)}

            # process the index node of the sub-meta-entity
            inner_to_outer_map, inner_idx_set_con_ops = self.__process_entity_idx_node_of_sub_meta_entity_declaration(
                outer_unb_syms,
                entity_idx_node)

            (def_unb_syms,
             id_to_def_unb_sym_map,
             def_unb_sym_to_sn_id_map,
             middle_set_node_ids,
             inner_set_node_ids,
             middle_unb_syms,
             inner_unb_syms) = self.__assign_component_set_nodes_to_inner_and_middle_scopes(idx_subset_node,
                                                                                            entity_idx_node)

            r = self.__build_middle_scope_idx_set_node(set_node_map=set_node_map,
                                                       set_node_positions=set_node_positions,
                                                       def_unb_syms=def_unb_syms,
                                                       sn_id_to_def_unb_syms_map=id_to_def_unb_sym_map,
                                                       def_unb_sym_to_sn_id_map=def_unb_sym_to_sn_id_map,
                                                       middle_set_node_ids=middle_set_node_ids,
                                                       outer_unb_syms=outer_unb_syms,
                                                       middle_unb_syms=middle_unb_syms,
                                                       inner_scope_unbound_syms=inner_unb_syms)
            (middle_rep_map, tfm_middle_unb_syms, tfm_inner_unb_syms, middle_idx_set_node) = r

            inner_idx_set_node = self.__build_inner_scope_idx_set_node(idx_subset_node=idx_subset_node,
                                                                       inner_idx_set_con_ops=inner_idx_set_con_ops,
                                                                       set_node_map=set_node_map,
                                                                       set_node_positions=set_node_positions,
                                                                       sn_id_to_def_unb_syms_map=id_to_def_unb_sym_map,
                                                                       middle_set_node_ids=middle_set_node_ids,
                                                                       inner_set_node_ids=inner_set_node_ids,
                                                                       middle_rep_map=middle_rep_map,
                                                                       outer_unb_syms=outer_unb_syms,
                                                                       middle_unb_syms=middle_unb_syms,
                                                                       tfm_middle_scope_unb_syms=tfm_middle_unb_syms,
                                                                       tfm_inner_unb_syms=tfm_inner_unb_syms)

            set_node = self.__build_set_node(middle_idx_set_node,
                                             inner_idx_set_node)

            set_member_node = self.__build_set_member_node(inner_to_outer_map,
                                                           inner_unb_syms)

            set_membership_node = mat.SetMembershipOperationNode(id=self.problem.generate_free_node_id(),
                                                                 operator="in",
                                                                 member_node=set_member_node,
                                                                 set_node=set_node)

            return set_membership_node

        def __process_entity_idx_node_of_sub_meta_entity_declaration(self,
                                                                     outer_scope_unbound_syms: List[str],
                                                                     entity_idx_node: mat.CompoundDummyNode
                                                                     ) -> Tuple[Dict[str, str],
                                                                                List[mat.LogicalExpressionNode]]:

            # instantiate dummy symbol replacement mapping
            dummy_sym_mapping = {}  # key: old symbol; value: new symbol

            # instantiate list of equality nodes to be included in the inner indexing set constraint node
            eq_nodes = []

            # compare standard dummy symbols used in the parent indexing set
            # to the dummy symbols used in the entity index node
            for outer_scope_unbound_sym, entity_idx_cmpt_node in zip(outer_scope_unbound_syms,
                                                                     entity_idx_node.component_nodes):

                if isinstance(entity_idx_cmpt_node, mat.DummyNode):  # component node is a dummy
                    if entity_idx_cmpt_node.dummy != outer_scope_unbound_sym:
                        dummy_sym_mapping[entity_idx_cmpt_node.dummy] = outer_scope_unbound_sym

                else:  # component node is an arithmetic/string expression
                    std_dummy_node = mat.DummyNode(id=self.problem.generate_free_node_id(),
                                                   dummy=outer_scope_unbound_sym)
                    eq_node = mat.RelationalOperationNode(id=self.problem.generate_free_node_id(),
                                                          operator="==",
                                                          lhs_operand=std_dummy_node,
                                                          rhs_operand=entity_idx_cmpt_node)
                    eq_nodes.append(eq_node)

            return dummy_sym_mapping, eq_nodes

        @staticmethod
        def __assign_component_set_nodes_to_inner_and_middle_scopes(idx_subset_node: mat.CompoundSetNode,
                                                                    entity_idx_node: mat.CompoundDummyNode):

            defined_unbound_syms = set()
            set_node_id_to_defined_unbound_syms_map: Dict[int, Set[str]] = {}
            defined_unbound_sym_to_set_node_id_map: Dict[str, int] = {}

            middle_scope_unbound_syms = set()
            inner_scope_unbound_syms = set()
            entity_idx_unbound_syms = set()  # instantiate set of unbound symbols that control the entity index

            middle_set_node_ids = set()
            inner_set_node_ids = set()

            # retrieve unbound symbols that control the entity index
            for component_node in entity_idx_node.component_nodes:
                if isinstance(component_node, mat.DummyNode):
                    entity_idx_unbound_syms.add(component_node.dummy)

            def handle_dummy_node(sn_id: int, dn: mat.DummyNode):
                if dn.dummy not in defined_unbound_syms:
                    defined_unbound_syms.add(dn.dummy)
                    set_node_id_to_defined_unbound_syms_map[sn_id].add(dn.dummy)
                    defined_unbound_sym_to_set_node_id_map[dn.dummy] = sn_id
                    if dn.dummy in entity_idx_unbound_syms:
                        inner_scope_unbound_syms.add(dn.dummy)
                        return 0  # unbound symbol is defined in inner scope
                    else:
                        middle_scope_unbound_syms.add(dn.dummy)
                        return 1  # unbound symbol is defined in middle scope
                return -1  # unbound symbol has already been defined

            # iterate over the component nodes of the indexing subset
            for component_set_node in idx_subset_node.set_nodes:

                # component set node is an indexing set with a controlled dummy member
                if isinstance(component_set_node, mat.IndexingSetNode):

                    dummy_node = component_set_node.dummy_node
                    set_node_id_to_defined_unbound_syms_map[component_set_node.id] = set()
                    is_middle = False
                    is_inner = False

                    # scalar dummy node
                    if isinstance(dummy_node, mat.DummyNode):
                        scope = handle_dummy_node(component_set_node.id, dummy_node)
                        if scope == 1:
                            is_middle = True
                        elif scope == 0:
                            is_inner = True

                    # compound dummy node
                    elif isinstance(dummy_node, mat.CompoundDummyNode):
                        for component_dummy_node in dummy_node.component_nodes:
                            scope = handle_dummy_node(component_set_node.id, component_dummy_node)
                            if scope == 1:
                                is_middle = True
                            elif scope == 0:
                                is_inner = True

                    if is_inner:
                        inner_set_node_ids.add(component_set_node.id)
                    if is_middle:
                        middle_set_node_ids.add(component_set_node.id)

            return (defined_unbound_syms,
                    set_node_id_to_defined_unbound_syms_map,
                    defined_unbound_sym_to_set_node_id_map,
                    middle_set_node_ids,
                    inner_set_node_ids,
                    middle_scope_unbound_syms,
                    inner_scope_unbound_syms)

        def __build_middle_scope_idx_set_node(self,
                                              set_node_map: Dict[int, mat.SetExpressionNode],
                                              set_node_positions: Dict[int, int],
                                              def_unb_syms: Set[str],
                                              sn_id_to_def_unb_syms_map: Dict[int, Set[str]],
                                              def_unb_sym_to_sn_id_map: Dict[str, int],
                                              middle_set_node_ids: Set[int],
                                              outer_unb_syms: List[str],
                                              middle_unb_syms: Set[str],
                                              inner_scope_unbound_syms: Set[str]):

            if len(middle_set_node_ids) > 0:

                queue = Queue()

                for node_id in middle_set_node_ids:
                    queue.put(node_id)

                while not queue.empty():

                    node_id: int = queue.get()
                    unbound_syms = self.node_builder.retrieve_unbound_symbols(set_node_map[node_id],
                                                                              unbound_sym_subset=def_unb_syms)
                    outer_unbound_syms = unbound_syms - sn_id_to_def_unb_syms_map[node_id]
                    for unbound_sym in outer_unbound_syms:
                        node_id = def_unb_sym_to_sn_id_map[unbound_sym]
                        if node_id not in middle_set_node_ids:
                            middle_set_node_ids.add(node_id)
                            queue.put(set_node_map[node_id])

                middle_set_nodes = [set_node_map[id] for id in middle_set_node_ids]
                middle_set_nodes.sort(key=lambda sn: set_node_positions[sn.id])

                middle_idx_set_node = mat.CompoundSetNode(id=self.problem.generate_free_node_id(),
                                                          set_nodes=middle_set_nodes)

                middle_rep_map = self.node_builder.generate_unbound_symbol_clash_replacement_map(
                    middle_idx_set_node,
                    blacklisted_unbound_syms=outer_unb_syms)

                tfm_middle_scope_unbound_syms = set()
                for unbound_sym in middle_unb_syms:
                    tfm_middle_scope_unbound_syms.add(middle_rep_map.get(unbound_sym, unbound_sym))

                tfm_inner_scope_unbound_syms = set()
                for unbound_sym in inner_scope_unbound_syms:
                    tfm_inner_scope_unbound_syms.add(middle_rep_map.get(unbound_sym, unbound_sym))

                self.node_builder.replace_unbound_symbols(middle_idx_set_node, middle_rep_map)

            else:
                middle_rep_map = {}
                tfm_middle_scope_unbound_syms = middle_unb_syms
                tfm_inner_scope_unbound_syms = inner_scope_unbound_syms
                middle_idx_set_node = None

            return middle_rep_map, tfm_middle_scope_unbound_syms, tfm_inner_scope_unbound_syms, middle_idx_set_node

        def __build_inner_scope_idx_set_node(self,
                                             idx_subset_node: mat.CompoundSetNode,
                                             inner_idx_set_con_ops: List[mat.LogicalExpressionNode],
                                             set_node_map: Dict[int, mat.SetExpressionNode],
                                             set_node_positions: Dict[int, int],
                                             sn_id_to_def_unb_syms_map: Dict[int, Set[str]],
                                             middle_set_node_ids: Set[int],
                                             inner_set_node_ids: Set[int],
                                             middle_rep_map: Dict[str, str],
                                             outer_unb_syms: List[str],
                                             middle_unb_syms: Set[str],
                                             tfm_middle_scope_unb_syms: Set[str],
                                             tfm_inner_unb_syms: Set[str]):

            if len(inner_set_node_ids) == 0:
                one_node = mat.NumericNode(id=self.problem.generate_free_node_id(), value=1)
                inner_set_nodes = [mat.OrderedSetNode(id=self.problem.generate_free_node_id(),
                                                      start_node=one_node,
                                                      end_node=one_node)]
            else:
                inner_set_nodes = [set_node_map[id] for id in inner_set_node_ids]
                inner_set_nodes.sort(key=lambda sn: set_node_positions[sn.id])

            if idx_subset_node.constraint_node is not None:
                inner_idx_set_con_ops.insert(0, idx_subset_node.constraint_node)

            if len(inner_idx_set_con_ops) == 0:
                inner_idx_set_con_node = None

            elif len(inner_idx_set_con_ops) == 1:
                inner_idx_set_con_node = inner_idx_set_con_ops[0]

            else:
                inner_idx_set_con_node = self.node_builder.build_conjunction_node(inner_idx_set_con_ops)

            inner_idx_set_node = mat.CompoundSetNode(id=self.problem.generate_free_node_id(),
                                                     set_nodes=inner_set_nodes,
                                                     constraint_node=inner_idx_set_con_node)

            inner_rep_map = self.node_builder.generate_unbound_symbol_clash_replacement_map(
                inner_idx_set_node,
                outer_scope_unbound_syms=tfm_middle_scope_unb_syms,
                blacklisted_unbound_syms=tfm_inner_unb_syms | set(outer_unb_syms))

            self.node_builder.replace_unbound_symbols(inner_idx_set_node, inner_rep_map)

            inner_idx_set_con_ops = []
            if idx_subset_node.constraint_node is not None:
                inner_idx_set_con_ops.append(inner_idx_set_node.constraint_node)

            for node_id in middle_set_node_ids:
                unbound_syms = sn_id_to_def_unb_syms_map[node_id]
                duplicated_unbound_syms = unbound_syms - middle_unb_syms
                for unbound_sym in duplicated_unbound_syms:
                    lhs_node = mat.DummyNode(id=self.problem.generate_free_node_id(),
                                             dummy=middle_rep_map.get(unbound_sym, unbound_sym))
                    rhs_node = mat.DummyNode(id=self.problem.generate_free_node_id(),
                                             dummy=inner_rep_map.get(unbound_sym, unbound_sym))
                    eq_node = mat.RelationalOperationNode(id=self.problem.generate_free_node_id(),
                                                          operator="==",
                                                          lhs_operand=lhs_node,
                                                          rhs_operand=rhs_node)
                    inner_idx_set_con_ops.append(eq_node)

            if len(inner_idx_set_con_ops) == 1:
                inner_idx_set_node.constraint_node = inner_idx_set_con_ops[0]
            elif len(inner_idx_set_con_ops) > 1:
                inner_idx_set_node.constraint_node = self.node_builder.build_conjunction_node(inner_idx_set_con_ops)

            return inner_idx_set_node

        def __build_set_node(self,
                             middle_idx_set_node: mat.CompoundSetNode,
                             inner_idx_set_node: mat.CompoundSetNode):

            if middle_idx_set_node is None:
                set_node = inner_idx_set_node

            else:
                set_node = mat.SetReductionOperationNode(id=self.problem.generate_free_node_id(),
                                                         symbol="union",
                                                         idx_set_node=middle_idx_set_node,
                                                         operand=inner_idx_set_node)

            return set_node

        def __build_set_member_node(self,
                                    inner_to_outer_mapping: Dict[str, str],
                                    inner_unb_syms: Set[str]):

            set_member_unbound_syms = [inner_to_outer_mapping.get(sym, sym) for sym in inner_unb_syms]

            if len(set_member_unbound_syms) == 0:
                set_member_node = mat.NumericNode(id=self.problem.generate_free_node_id(),
                                                  value=1)

            elif len(set_member_unbound_syms) == 1:
                set_member_node = mat.DummyNode(id=self.problem.generate_free_node_id(),
                                                dummy=set_member_unbound_syms[0])

            else:
                set_member_dummy_nodes = []
                for unbound_sym in set_member_unbound_syms:
                    set_member_dummy_nodes.append(mat.DummyNode(id=self.problem.generate_free_node_id(),
                                                                dummy=unbound_sym))
                set_member_node = mat.CompoundDummyNode(id=self.problem.generate_free_node_id(),
                                                        component_nodes=set_member_dummy_nodes)

            return set_member_node
