from copy import deepcopy
from queue import Queue
from typing import Dict, Iterable, List, Optional, Set, Union

import symro.mat as mat
from symro.prob.problem import BaseProblem, Problem
import symro.handlers.nodebuilder as nb


# Meta-Entity Construction
# ------------------------------------------------------------------------------------------------------------------


def build_meta_set(
    problem: Problem,
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
    defined_value_node: mat.BaseSetNode = None,
    default_value_node: mat.BaseSetNode = None,
):

    if idx_meta_sets is None and idx_set_node is not None:
        _build_idx_meta_sets_of_meta_entity(
            problem=problem,
            idx_set_node=idx_set_node,
            expr_nodes=[super_set_node, defined_value_node],
        )

    if idx_set_node is None:
        idx_set_node = nb.build_idx_set_node(
            problem, idx_meta_sets, idx_set_con_literal
        )

    return mat.MetaSet(
        symbol=symbol,
        alias=alias,
        idx_meta_sets=idx_meta_sets,
        idx_set_node=idx_set_node,
        dimension=dimension,
        reduced_dimension=reduced_dimension,
        dummy_symbols=dummy_symbols,
        reduced_dummy_symbols=reduced_dummy_symbols,
        is_dim_fixed=is_dim_fixed,
        super_set_node=super_set_node,
        defined_value_node=defined_value_node,
        default_value_node=default_value_node,
    )


def build_meta_param(
    problem: Problem,
    symbol: str,
    alias: str = None,
    idx_meta_sets: Union[List[mat.MetaSet], Dict[str, mat.MetaSet]] = None,
    idx_set_con_literal: str = None,
    idx_set_node: mat.CompoundSetNode = None,
    is_binary: bool = False,
    is_integer: bool = False,
    is_symbolic: bool = False,
    defined_value: Union[int, float, str, mat.ExpressionNode] = None,
    default_value: Union[int, float, str, mat.ExpressionNode] = None,
    super_set_node: mat.BaseSetNode = None,
    relational_constraints: Dict[str, mat.ExpressionNode] = None,
):

    defined_value_node = _build_value_node(defined_value)
    default_value_node = _build_value_node(default_value)

    if relational_constraints is None:
        relational_constraints = {}

    if idx_meta_sets is None and idx_set_node is not None:
        idx_meta_sets = _build_idx_meta_sets_of_meta_entity(
            problem=problem,
            idx_set_node=idx_set_node,
            expr_nodes=[defined_value_node, default_value_node, super_set_node]
            + list(relational_constraints.values()),
        )

    if idx_set_node is None:
        idx_set_node = nb.build_idx_set_node(
            problem, idx_meta_sets, idx_set_con_literal
        )

    meta_param = mat.MetaParameter(
        symbol=symbol,
        alias=alias,
        idx_meta_sets=idx_meta_sets,
        idx_set_node=idx_set_node,
        is_binary=is_binary,
        is_integer=is_integer,
        is_symbolic=is_symbolic,
        defined_value=defined_value_node,
        default_value=default_value_node,
        super_set_node=super_set_node,
        relational_constraints=relational_constraints,
    )
    return meta_param


def build_meta_var(
    problem: Problem,
    symbol: str,
    alias: str = None,
    idx_meta_sets: Union[List[mat.MetaSet], Dict[str, mat.MetaSet]] = None,
    idx_set_con_literal: str = None,
    idx_set_node: mat.CompoundSetNode = None,
    is_binary: bool = False,
    is_integer: bool = False,
    is_symbolic: bool = False,
    defined_value: Union[int, float, str, mat.ExpressionNode] = None,
    default_value: Union[int, float, str, mat.ExpressionNode] = None,
    lower_bound: Union[int, float, str, mat.ExpressionNode] = None,
    upper_bound: Union[int, float, str, mat.ExpressionNode] = None,
):

    defined_value_node = _build_value_node(defined_value)
    default_value_node = _build_value_node(default_value)
    lower_bound_node = _build_value_node(lower_bound)
    upper_bound_node = _build_value_node(upper_bound)

    if idx_meta_sets is None and idx_set_node is not None:
        idx_meta_sets = _build_idx_meta_sets_of_meta_entity(
            problem=problem,
            idx_set_node=idx_set_node,
            expr_nodes=[
                defined_value_node,
                default_value_node,
                lower_bound,
                upper_bound,
            ],
        )

    if idx_set_node is None:
        idx_set_node = nb.build_idx_set_node(
            problem, idx_meta_sets, idx_set_con_literal
        )

    meta_var = mat.MetaVariable(
        symbol=symbol,
        alias=alias,
        idx_meta_sets=idx_meta_sets,
        idx_set_node=idx_set_node,
        is_binary=is_binary,
        is_integer=is_integer,
        is_symbolic=is_symbolic,
        defined_value=defined_value_node,
        default_value=default_value_node,
        lower_bound=lower_bound_node,
        upper_bound=upper_bound_node,
    )
    return meta_var


def build_meta_obj(
    problem: Problem,
    symbol: str,
    alias: str = None,
    idx_meta_sets: Union[List[mat.MetaSet], Dict[str, mat.MetaSet]] = None,
    idx_set_con_literal: str = None,
    idx_set_node: mat.CompoundSetNode = None,
    direction: str = None,
    expression: mat.Expression = None,
):

    if idx_meta_sets is None and idx_set_node is not None:
        idx_meta_sets = _build_idx_meta_sets_of_meta_entity(
            problem=problem,
            idx_set_node=idx_set_node,
            expr_nodes=[expression.root_node],
        )

    if idx_set_node is None:
        idx_set_node = nb.build_idx_set_node(
            problem, idx_meta_sets, idx_set_con_literal
        )

    meta_obj = mat.MetaObjective(
        symbol=symbol,
        alias=alias,
        idx_meta_sets=idx_meta_sets,
        idx_set_node=idx_set_node,
        direction=direction,
        expression=expression,
    )
    return meta_obj


def build_meta_con(
    problem: Problem,
    symbol: str,
    alias: str = None,
    idx_meta_sets: Union[List[mat.MetaSet], Dict[str, mat.MetaSet]] = None,
    idx_set_con_literal: str = None,
    idx_set_node: mat.CompoundSetNode = None,
    expression: mat.Expression = None,
):

    if idx_meta_sets is None and idx_set_node is not None:
        idx_meta_sets = _build_idx_meta_sets_of_meta_entity(
            problem=problem,
            idx_set_node=idx_set_node,
            expr_nodes=[expression.root_node],
        )

    if idx_set_node is None:
        idx_set_node = nb.build_idx_set_node(
            problem, idx_meta_sets, idx_set_con_literal
        )

    meta_con = mat.MetaConstraint(
        symbol=symbol,
        alias=alias,
        idx_meta_sets=idx_meta_sets,
        idx_set_node=idx_set_node,
        expression=expression,
    )
    return meta_con


def _build_value_node(
    value: Union[None, int, float, str, mat.ExpressionNode]
) -> Optional[mat.ExpressionNode]:

    node = value

    if value is not None:

        if isinstance(value, int) or isinstance(value, float):
            node = mat.NumericNode(value)
        elif isinstance(value, str):
            node = mat.StringNode(literal=value, delimiter='"')
    return node


# Indexing Meta-Set Construction
# ------------------------------------------------------------------------------------------------------------------


def build_all_idx_meta_sets(problem: Problem):
    _build_idx_meta_sets_of_problem(problem)
    for sp in problem.subproblems.values():
        _build_idx_meta_sets_of_problem(problem, sp)


def _build_idx_meta_sets_of_problem(problem: Problem, subproblem: BaseProblem = None):

    if subproblem is None:
        subproblem = problem

    for me_iterable in (
        subproblem.model_meta_sets_params,
        subproblem.model_meta_vars,
        subproblem.model_meta_objs,
        subproblem.model_meta_cons,
    ):
        for me in me_iterable:
            me.idx_meta_sets = _build_idx_meta_sets_of_meta_entity(
                problem=problem,
                idx_set_node=me.idx_set_node,
                expr_nodes=me.expression_nodes,
            )


def _build_idx_meta_sets_of_meta_entity(
    problem: Problem,
    idx_set_node: mat.CompoundSetNode,
    expr_nodes: Iterable[mat.ExpressionNode],
):
    blacklist = set()

    for root_node in expr_nodes:
        if root_node is not None:
            blacklist = blacklist.union(nb.retrieve_unbound_symbols(root_node))

    return build_idx_meta_sets(
        problem=problem, idx_set_node=idx_set_node, unb_syms_blacklist=blacklist
    )


def build_idx_meta_sets(
    problem: Problem,
    idx_set_node: mat.CompoundSetNode,
    unb_syms_blacklist: Iterable[str] = None,
) -> List[mat.MetaSet]:

    idx_meta_sets = []

    if idx_set_node is None:
        return idx_meta_sets

    if unb_syms_blacklist is None:
        unb_syms_blacklist = set()

    def_unb_syms = set()

    # iterate over each component set node
    for i, cmpt_set_node in enumerate(idx_set_node.set_nodes):

        cmpt_dim = cmpt_set_node.get_dim(problem.state)
        cmpt_dummy_element = []
        reduced_cmpt_dummy_element = []
        is_dim_fixed = [False] * cmpt_dim

        # indexing set node
        if isinstance(cmpt_set_node, mat.IndexingSetNode):
            cmpt_set_sym = cmpt_set_node.set_node.get_literal()
            dummy_node = cmpt_set_node.dummy_node

            if isinstance(dummy_node, mat.CompoundDummyNode):
                dummy_nodes = list(dummy_node.component_nodes)
            elif isinstance(dummy_node, mat.DummyNode):
                dummy_nodes = [dummy_node]
            else:
                raise ValueError(
                    "Meta-entity builder expected a dummy node while resolving a component indexing set node"
                )

            # iterate over dummy nodes
            for j, dummy_node in enumerate(dummy_nodes):

                cmpt_dummy_sub_element = (
                    dummy_node.get_literal()
                )  # retrieve the dummy sub-element

                cmpt_dummy_element.append(
                    cmpt_dummy_sub_element
                )  # add the sub-element to the dummy element

                # dummy
                if isinstance(dummy_node, mat.DummyNode):

                    # already-defined unbound symbol
                    if dummy_node.symbol in def_unb_syms:
                        is_dim_fixed[j] = True

                    # newly-defined unbound symbol
                    else:
                        # add the sub-element to the reduced dummy element
                        reduced_cmpt_dummy_element.append(cmpt_dummy_sub_element)
                        def_unb_syms.add(
                            dummy_node.symbol
                        )  # store new unbound symbol to set

                # arithmetic or string expression
                else:
                    is_dim_fixed[j] = True

        # non-indexing set node
        else:

            # declared set node
            if isinstance(cmpt_set_node, mat.DeclaredSetNode):

                cmpt_set_sym = cmpt_set_node.get_literal()

                ms = problem.meta_sets[cmpt_set_node.symbol]
                dummy_nodes = []

                # retrieve default unbound symbols of the meta-set
                for d in ms.reduced_dummy_element:

                    # unbound symbol clashes with previously-defined symbol
                    if d in def_unb_syms:

                        # generate new unique symbol
                        d = problem.generate_unique_symbol(
                            d, symbol_blacklist=def_unb_syms
                        )

                        # add unbound symbol to problem
                        problem.unbound_symbols.add(d)

                    def_unb_syms.add(d)

                    cmpt_dummy_element.append(d)  # add sub-element to the dummy element
                    reduced_cmpt_dummy_element.append(
                        d
                    )  # add sub-element to the reduced dummy element

                    dummy_nodes.append(mat.DummyNode(d))  # build new dummy node

            # other
            else:

                cmpt_set_sym = cmpt_set_node.get_literal()
                dummy_nodes = []

                # generate unique unbound symbols for each dimension of the component set
                for j in range(cmpt_dim):

                    # elicit a base symbol
                    unb_sym_base = ""
                    for c in cmpt_set_sym:
                        if c.isalpha():  # base symbol must be a letter
                            unb_sym_base = c[0].lower()
                    if unb_sym_base == "":
                        unb_sym_base = "i"  # default base symbol is 'i'

                    # generate unique unbound symbol
                    unb_sym = problem.generate_unique_symbol(
                        base_symbol=unb_sym_base, symbol_blacklist=unb_syms_blacklist
                    )

                    # add unbound symbol to problem
                    problem.unbound_symbols.add(unb_sym)
                    def_unb_syms.add(unb_sym)

                    cmpt_dummy_element.append(
                        unb_sym
                    )  # add sub-element to the dummy element
                    reduced_cmpt_dummy_element.append(
                        unb_sym
                    )  # add sub-element to the reduced dummy element

                    # build new dummy node
                    dummy_nodes.append(mat.DummyNode(unb_sym))

            # replace component set node with an indexing set node

            if len(dummy_nodes) == 1:
                dummy_node = dummy_nodes[0]

            # build compound dummy node
            else:
                dummy_node = mat.CompoundDummyNode(dummy_nodes)

            # build indexing set node
            idx_set_node.set_nodes[i] = mat.IndexingSetNode(
                dummy_node=dummy_node, set_node=cmpt_set_node
            )

        # compute reduced dimension of the meta-set
        reduced_cmpt_dim = len(reduced_cmpt_dummy_element)

        # build indexing meta-set
        indexing_meta_set = mat.MetaSet(
            symbol=cmpt_set_sym,
            dimension=cmpt_dim,
            reduced_dimension=reduced_cmpt_dim,
            dummy_symbols=tuple(cmpt_dummy_element),
            reduced_dummy_symbols=tuple(reduced_cmpt_dummy_element),
            is_dim_fixed=is_dim_fixed,
        )
        idx_meta_sets.append(indexing_meta_set)

    return idx_meta_sets


# Sub-Meta-Entity Construction
# ------------------------------------------------------------------------------------------------------------------


def build_sub_meta_entity(
    problem: Problem,
    meta_entity: mat.MetaEntity,
    idx_subset_node: mat.CompoundSetNode,
    entity_idx_node: mat.CompoundDummyNode,
) -> mat.MetaEntity:
    """
    Builds a sub-meta-entity object. A sub-meta-entity is defined over a subset of the indexing set over which the
    parent meta-entity is defined.

    :param problem: current problem
    :param meta_entity: parent meta-entity
    :param idx_subset_node: compound set node over which the sub-meta-entity is defined
    :param entity_idx_node: index node describing how the idx_subset_node controls the sub-meta-entity
    :return: sub-meta-entity
    """

    # return the parent meta-entity if it is scalar
    # return the parent meta-entity if no indexing subset is provided
    if idx_subset_node is None or meta_entity.idx_set_reduced_dim == 0:
        return meta_entity

    idx_subset_node = __build_sub_meta_entity_idx_set_node(
        problem=problem,
        idx_subset_node=idx_subset_node,
        meta_entity=meta_entity,
        entity_idx_node=entity_idx_node,
    )

    sub_meta_entity = meta_entity.build_sub_entity(idx_set_node=idx_subset_node)

    return sub_meta_entity


def __build_sub_meta_entity_idx_set_node(
    problem: Problem,
    idx_subset_node: mat.CompoundSetNode,
    meta_entity: mat.MetaEntity,
    entity_idx_node: mat.CompoundDummyNode,
):

    # check if the dimension of the entity index node matches that of the parent meta-entity's indexing set
    if entity_idx_node.get_dim() != meta_entity.idx_set_reduced_dim:
        sub_entity_decl = "{0} {1}{2}".format(
            idx_subset_node, meta_entity.symbol, entity_idx_node
        )
        raise ValueError(
            "Meta-entity builder encountered an incorrect entity declaration"
            + " {0} while building a sub-meta-entity:".format(sub_entity_decl)
            + " the dimension of the entity index does not match that of the parent entity"
        )

    # deep copy the indexing set node of the parent meta-entity
    idx_set_node = deepcopy(meta_entity.idx_set_node)

    # instantiate a list of logical conjunctive operands for the indexing set constraint node
    idx_set_con_operands = []

    # add the original constraint node as the first operand
    if idx_set_node.constraint_node is not None:
        idx_set_con_operands.append(idx_set_node.constraint_node)

    # retrieve standard dummy symbols of the indexing set
    outer_unb_syms = meta_entity.idx_set_reduced_dummy_element

    # compare the indexing superset and the indexing subset
    are_super_and_subset_equal = __compare_super_and_sub_idx_sets(
        problem=problem,
        idx_set_node=meta_entity.idx_set_node,
        idx_subset_node=idx_subset_node,
    )

    # process the index node of the sub-meta-entity
    (
        inner_to_outer_map,
        eq_idx_set_con_ops,
        has_dependent_cmpt,
    ) = __map_inner_to_outer_scope(
        problem=problem,
        idx_subset_node=idx_subset_node,
        outer_unb_syms=outer_unb_syms,
        entity_idx_node=entity_idx_node,
    )

    # check whether a subset membership operation is necessary
    if not are_super_and_subset_equal or has_dependent_cmpt:

        # build the subset membership node
        subset_membership_node = __build_subset_membership_node(
            problem=problem,
            idx_subset_node=idx_subset_node,
            outer_unb_syms=outer_unb_syms,
            entity_idx_node=entity_idx_node,
            inner_to_outer_map=inner_to_outer_map,
            inner_idx_set_con_ops=eq_idx_set_con_ops,
        )

        # add the set membership node to the list of operands
        if subset_membership_node is not None:
            idx_set_con_operands.append(subset_membership_node)

    else:
        if len(eq_idx_set_con_ops) > 0:
            idx_set_con_operands.extend(eq_idx_set_con_ops)

    # combine the operands of the set constraint into a single node
    idx_set_con_node = nb.build_conjunction_node(idx_set_con_operands)
    idx_set_node.constraint_node = (
        idx_set_con_node  # assign the constraint node to the indexing set node
    )

    return idx_set_node


def __compare_super_and_sub_idx_sets(
    problem: Problem,
    idx_set_node: mat.CompoundSetNode,
    idx_subset_node: mat.CompoundSetNode,
):
    """
    Compares the indexing superset of the parent meta-entity to the indexing subset of the sub-meta-entity.

    :param problem: current problem
    :param idx_set_node: indexing set of the parent meta-entity
    :param idx_subset_node: indexing set of the sub-meta-entity
    :return: True if a the superset and the subset are identical, False otherwise
    """

    # retrieve the indexing superset and subset
    idx_superset = idx_set_node.evaluate(state=problem.state)[0]
    idx_subset = idx_subset_node.evaluate(state=problem.state)[0]

    # compute the symmetric difference of both sets
    if (
        len(idx_superset.symmetric_difference(idx_subset)) > 0
    ):  # symmetric difference is not the empty set
        return False  # indexing superset and subset are different
    else:
        return True  # indexing superset and subset are identical


def __map_inner_to_outer_scope(
    problem: Problem,
    idx_subset_node: mat.CompoundSetNode,
    outer_unb_syms: List[str],
    entity_idx_node: mat.CompoundDummyNode,
):
    """
    Maps each component of the index of the sub-meta-entity to a component of the indexing set node of the parent
    meta-entity.

    Returns a tuple of length 3.
        1. dictionary mapping the unbound symbols of the inner scope their corresponding unbound symbols in the outer scope
        2. list of equality nodes to be included in the constraint node of the final indexing set node
        3. True if the inner scope has one or more dependent index component nodes, False otherwise

    :param problem: current problem
    :param idx_subset_node: indexing set of the parent meta-entity
    :param outer_unb_syms: ordered list of unbound symbols defined in the indexing set node of the parent meta-entity
    :param entity_idx_node: index node of the sub-meta-entity
    :return: 3-tuple containing results
    """

    # initialize return values
    has_dependent_cmpt = False
    dummy_sym_mapping = (
        {}
    )  # dummy symbol replacement mapping; key: old symbol; value: new symbol
    eq_nodes = (
        []
    )  # list of equality nodes to be included in the inner indexing set constraint node

    inner_unb_syms = (
        set()
    )  # set of all unbound symbols appearing in the index node of the sub-meta-entity

    # set of all unbound symbols appearing in the indexing set node of the sub-meta-entity
    inner_middle_unb_syms = set()

    # iterate over all component dummy nodes of the indexing set node of the sub-meta-entity
    for cmpt_node in idx_subset_node.get_dummy_component_nodes(state=problem.state):
        if isinstance(cmpt_node, mat.DummyNode):
            inner_middle_unb_syms.add(cmpt_node.symbol)

    # - check if there are any component nodes in the entity index that are not uniquely controlled
    #   by a dummy
    # - build a map of inner to outer scope unbound symbols
    # - build equality nodes to be included in indexing set constraint node

    # iterate over each pair of index sub-elements in the outer and inner scopes
    for outer_scope_unb_sym, cmpt_node in zip(
        outer_unb_syms, entity_idx_node.component_nodes
    ):

        # dummy
        if isinstance(cmpt_node, mat.DummyNode):

            if (
                cmpt_node.symbol in inner_unb_syms
            ):  # dummy controls another component node
                has_dependent_cmpt = True  # component node is dependent

            inner_unb_syms.add(cmpt_node.symbol)

            if cmpt_node.symbol != outer_scope_unb_sym:
                dummy_sym_mapping[cmpt_node.symbol] = outer_scope_unb_sym

        # arithmetic/string expression
        else:

            # check if the component node is controlled
            ctrl_unb_syms = nb.retrieve_unbound_symbols(
                root_node=cmpt_node, in_filter=inner_middle_unb_syms
            )
            if len(ctrl_unb_syms):
                # if the set of controlling unbound symbols is not empty, then the component is controlled
                has_dependent_cmpt = True

            # build equality node to be included as an indexing set constraint
            outer_dummy_node = mat.DummyNode(symbol=outer_scope_unb_sym)
            eq_node = mat.RelationalOperationNode(
                operator=mat.EQUALITY_OPERATOR,
                lhs_operand=outer_dummy_node,
                rhs_operand=deepcopy(cmpt_node),
            )
            eq_nodes.append(eq_node)

    return dummy_sym_mapping, eq_nodes, has_dependent_cmpt


def __build_subset_membership_node(
    problem: Problem,
    idx_subset_node: mat.CompoundSetNode,
    outer_unb_syms: List[str],
    entity_idx_node: mat.CompoundDummyNode,
    inner_to_outer_map: Dict[str, str],
    inner_idx_set_con_ops: List[mat.LogicalExpressionNode],
):

    set_node_map: Dict[int, mat.SetExpressionNode] = {
        id(node): node for node in idx_subset_node.set_nodes
    }
    set_node_positions = {
        id(node): pos for pos, node in enumerate(idx_subset_node.set_nodes)
    }

    (
        def_unb_syms,
        id_to_def_unb_sym_map,
        def_unb_sym_to_sn_id_map,
        middle_set_node_ids,
        inner_set_node_ids,
        middle_unb_syms,
        inner_unb_syms,
    ) = __assign_component_set_nodes_to_inner_and_middle_scopes(
        idx_subset_node, entity_idx_node
    )

    r = __build_middle_scope_idx_set_node(
        problem=problem,
        set_node_map=set_node_map,
        set_node_positions=set_node_positions,
        def_unb_syms=def_unb_syms,
        sn_id_to_def_unb_syms_map=id_to_def_unb_sym_map,
        def_unb_sym_to_sn_id_map=def_unb_sym_to_sn_id_map,
        middle_set_node_ids=middle_set_node_ids,
        outer_unb_syms=outer_unb_syms,
        middle_unb_syms=middle_unb_syms,
        inner_unb_syms=inner_unb_syms,
    )
    (middle_rep_map, tfm_middle_unb_syms, tfm_inner_unb_syms, middle_idx_set_node) = r

    inner_idx_set_node = __build_inner_scope_idx_set_node(
        problem=problem,
        idx_subset_node=idx_subset_node,
        inner_idx_set_con_ops=inner_idx_set_con_ops,
        set_node_map=set_node_map,
        set_node_positions=set_node_positions,
        sn_id_to_def_unb_syms_map=id_to_def_unb_sym_map,
        middle_set_node_ids=middle_set_node_ids,
        inner_set_node_ids=inner_set_node_ids,
        middle_rep_map=middle_rep_map,
        outer_unb_syms=outer_unb_syms,
        middle_unb_syms=middle_unb_syms,
        tfm_middle_unb_syms=tfm_middle_unb_syms,
        tfm_inner_unb_syms=tfm_inner_unb_syms,
    )

    set_node = __build_set_node(middle_idx_set_node, inner_idx_set_node)

    set_member_node = __build_set_member_node(inner_to_outer_map, inner_unb_syms)

    set_membership_node = mat.SetMembershipOperationNode(
        operator="in", member_node=set_member_node, set_node=set_node
    )

    return set_membership_node


def __assign_component_set_nodes_to_inner_and_middle_scopes(
    idx_subset_node: mat.CompoundSetNode, entity_idx_node: mat.CompoundDummyNode
):

    def_unb_syms = set()
    set_node_id_to_defined_unbound_syms_map: Dict[int, Set[str]] = {}
    def_unb_sym_to_sn_id_map: Dict[str, int] = {}

    middle_unb_syms = mat.OrderedSet()
    inner_unb_syms = mat.OrderedSet()
    entity_idx_unbound_syms = (
        set()
    )  # instantiate set of unbound symbols that control the entity index

    middle_set_node_ids = set()
    inner_set_node_ids = set()

    # retrieve unbound symbols that control the entity index
    for component_node in entity_idx_node.component_nodes:
        if isinstance(component_node, mat.DummyNode):
            entity_idx_unbound_syms.add(component_node.symbol)

    def handle_dummy_node(sn_id: int, dn: mat.DummyNode):
        if dn.symbol not in def_unb_syms:
            def_unb_syms.add(dn.symbol)
            set_node_id_to_defined_unbound_syms_map[sn_id].add(dn.symbol)
            def_unb_sym_to_sn_id_map[dn.symbol] = sn_id
            if dn.symbol in entity_idx_unbound_syms:
                inner_unb_syms.add(dn.symbol)
                return 0  # unbound symbol is defined in inner scope
            else:
                middle_unb_syms.add(dn.symbol)
                return 1  # unbound symbol is defined in middle scope
        return -1  # unbound symbol has already been defined

    # iterate over the component nodes of the indexing subset
    for component_set_node in idx_subset_node.set_nodes:

        # component set node is an indexing set with a controlled dummy member
        if isinstance(component_set_node, mat.IndexingSetNode):

            dummy_node = component_set_node.dummy_node
            set_node_id_to_defined_unbound_syms_map[id(component_set_node)] = set()
            is_middle = False
            is_inner = False

            # scalar dummy node
            if isinstance(dummy_node, mat.DummyNode):
                scope = handle_dummy_node(id(component_set_node), dummy_node)
                if scope == 1:
                    is_middle = True
                elif scope == 0:
                    is_inner = True

            # compound dummy node
            elif isinstance(dummy_node, mat.CompoundDummyNode):
                for component_dummy_node in dummy_node.component_nodes:
                    if isinstance(component_dummy_node, mat.DummyNode):
                        scope = handle_dummy_node(
                            id(component_set_node), component_dummy_node
                        )
                        if scope == 1:
                            is_middle = True
                        elif scope == 0:
                            is_inner = True

            if is_inner:
                inner_set_node_ids.add(id(component_set_node))
            if is_middle:
                middle_set_node_ids.add(id(component_set_node))

    return (
        def_unb_syms,
        set_node_id_to_defined_unbound_syms_map,
        def_unb_sym_to_sn_id_map,
        middle_set_node_ids,
        inner_set_node_ids,
        middle_unb_syms,
        inner_unb_syms,
    )


def __build_middle_scope_idx_set_node(
    problem: Problem,
    set_node_map: Dict[int, mat.SetExpressionNode],
    set_node_positions: Dict[int, int],
    def_unb_syms: Set[str],
    sn_id_to_def_unb_syms_map: Dict[int, Set[str]],
    def_unb_sym_to_sn_id_map: Dict[str, int],
    middle_set_node_ids: Set[int],
    outer_unb_syms: List[str],
    middle_unb_syms: mat.OrderedSet[str],
    inner_unb_syms: mat.OrderedSet[str],
):

    if len(middle_set_node_ids) > 0:

        queue = Queue()

        for node_id in middle_set_node_ids:
            queue.put(node_id)

        while not queue.empty():

            node_id: int = queue.get()
            unbound_syms = nb.retrieve_unbound_symbols(
                set_node_map[node_id], in_filter=def_unb_syms
            )
            outer_unbound_syms = unbound_syms - sn_id_to_def_unb_syms_map[node_id]
            for unbound_sym in outer_unbound_syms:
                node_id = def_unb_sym_to_sn_id_map[unbound_sym]
                if node_id not in middle_set_node_ids:
                    middle_set_node_ids.add(node_id)
                    queue.put(set_node_map[node_id])

        middle_set_nodes = [set_node_map[id] for id in middle_set_node_ids]
        middle_set_nodes.sort(key=lambda sn: set_node_positions[id(sn)])

        middle_idx_set_node = mat.CompoundSetNode(set_nodes=middle_set_nodes)

        middle_rep_map = nb.generate_unbound_symbol_mapping(
            problem=problem,
            root_node=middle_idx_set_node,
            blacklisted_unb_syms=outer_unb_syms,
        )

        tfm_middle_scope_unbound_syms = mat.OrderedSet()
        for unbound_sym in middle_unb_syms:
            tfm_middle_scope_unbound_syms.add(
                middle_rep_map.get(unbound_sym, unbound_sym)
            )

        tfm_inner_scope_unbound_syms = mat.OrderedSet()
        for unbound_sym in inner_unb_syms:
            tfm_inner_scope_unbound_syms.add(
                middle_rep_map.get(unbound_sym, unbound_sym)
            )

        nb.replace_unbound_symbols(middle_idx_set_node, middle_rep_map)

    else:
        middle_rep_map = {}
        tfm_middle_scope_unbound_syms = middle_unb_syms
        tfm_inner_scope_unbound_syms = inner_unb_syms
        middle_idx_set_node = None

    return (
        middle_rep_map,
        tfm_middle_scope_unbound_syms,
        tfm_inner_scope_unbound_syms,
        middle_idx_set_node,
    )


def __build_inner_scope_idx_set_node(
    problem: Problem,
    idx_subset_node: mat.CompoundSetNode,
    inner_idx_set_con_ops: List[mat.LogicalExpressionNode],
    set_node_map: Dict[int, mat.SetExpressionNode],
    set_node_positions: Dict[int, int],
    sn_id_to_def_unb_syms_map: Dict[int, Set[str]],
    middle_set_node_ids: Set[int],
    inner_set_node_ids: Set[int],
    middle_rep_map: Dict[str, str],
    outer_unb_syms: List[str],
    middle_unb_syms: mat.OrderedSet[str],
    tfm_middle_unb_syms: mat.OrderedSet[str],
    tfm_inner_unb_syms: mat.OrderedSet[str],
):

    if len(inner_set_node_ids) == 0:
        one_node = mat.NumericNode(1)
        inner_set_nodes = [mat.OrderedSetNode(start_node=one_node, end_node=one_node)]
    else:
        inner_set_nodes = [set_node_map[id] for id in inner_set_node_ids]
        inner_set_nodes.sort(key=lambda sn: set_node_positions[id(sn)])

    if idx_subset_node.constraint_node is not None:
        inner_idx_set_con_ops.insert(0, idx_subset_node.constraint_node)

    if len(inner_idx_set_con_ops) == 0:
        inner_idx_set_con_node = None

    elif len(inner_idx_set_con_ops) == 1:
        inner_idx_set_con_node = inner_idx_set_con_ops[0]

    else:
        inner_idx_set_con_node = nb.build_conjunction_node(inner_idx_set_con_ops)

    inner_idx_set_node = mat.CompoundSetNode(
        set_nodes=inner_set_nodes, constraint_node=inner_idx_set_con_node
    )

    inner_rep_map = nb.generate_unbound_symbol_mapping(
        problem=problem,
        root_node=inner_idx_set_node,
        outer_unb_syms=tfm_middle_unb_syms,
        blacklisted_unb_syms=tfm_inner_unb_syms | set(outer_unb_syms),
    )

    nb.replace_unbound_symbols(inner_idx_set_node, inner_rep_map)

    inner_idx_set_con_ops = []
    if idx_subset_node.constraint_node is not None:
        inner_idx_set_con_ops.append(inner_idx_set_node.constraint_node)

    for node_id in middle_set_node_ids:
        unbound_syms = sn_id_to_def_unb_syms_map[node_id]
        duplicated_unbound_syms = unbound_syms - middle_unb_syms
        for unbound_sym in duplicated_unbound_syms:
            lhs_node = mat.DummyNode(
                symbol=middle_rep_map.get(unbound_sym, unbound_sym)
            )
            rhs_node = mat.DummyNode(symbol=inner_rep_map.get(unbound_sym, unbound_sym))
            eq_node = mat.RelationalOperationNode(
                operator=mat.EQUALITY_OPERATOR,
                lhs_operand=lhs_node,
                rhs_operand=rhs_node,
            )
            inner_idx_set_con_ops.append(eq_node)

    if len(inner_idx_set_con_ops) == 1:
        inner_idx_set_node.constraint_node = inner_idx_set_con_ops[0]
    elif len(inner_idx_set_con_ops) > 1:
        inner_idx_set_node.constraint_node = nb.build_conjunction_node(
            inner_idx_set_con_ops
        )

    return inner_idx_set_node


def __build_set_node(
    middle_idx_set_node: mat.CompoundSetNode, inner_idx_set_node: mat.CompoundSetNode
):

    if middle_idx_set_node is None:
        set_node = inner_idx_set_node

    else:
        set_node = mat.SetReductionNode(
            operator=mat.UNION_OPERATOR,
            idx_set_node=middle_idx_set_node,
            operand=inner_idx_set_node,
        )

    return set_node


def __build_set_member_node(
    inner_to_outer_mapping: Dict[str, str], inner_unb_syms: mat.OrderedSet[str]
):

    set_member_unbound_syms = [
        inner_to_outer_mapping.get(sym, sym) for sym in inner_unb_syms
    ]

    if len(set_member_unbound_syms) == 0:
        set_member_node = mat.NumericNode(1)

    elif len(set_member_unbound_syms) == 1:
        set_member_node = mat.DummyNode(symbol=set_member_unbound_syms[0])

    else:
        set_member_dummy_nodes = []
        for unbound_sym in set_member_unbound_syms:
            set_member_dummy_nodes.append(mat.DummyNode(symbol=unbound_sym))
        set_member_node = mat.CompoundDummyNode(component_nodes=set_member_dummy_nodes)

    return set_member_node
