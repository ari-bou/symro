from copy import deepcopy
from numbers import Number
from queue import Queue
from typing import Dict, Iterable, List, Optional, Set, Tuple, Union

import numpy as np
import symro.mat as mat
from symro.prob.problem import Problem, BaseProblem
import symro.handlers.nodebuilder as nb
import symro.handlers.metaentitybuilder as eb


# Model Standardization
# ------------------------------------------------------------------------------------------------------------------


def standardize_model(problem: Problem) -> Dict[str, List[mat.MetaConstraint]]:

    # standardize objective functions
    for meta_obj in problem.model_meta_objs:
        __standardize_objective(meta_obj)

    # standardize constraints

    std_meta_cons = []  # list of standardized meta-constraints

    for meta_con in problem.model_meta_cons:

        problem.meta_cons.pop(meta_con.symbol)  # remove original meta-constraint

        std_meta_con_list = __standardize_constraint(problem, meta_con)
        problem.origin_to_std_con_map[meta_con.symbol] = std_meta_con_list

        std_meta_cons.extend(std_meta_con_list)

    # add standardized constraints to problem
    problem.model_meta_cons.clear()
    for std_meta_con in std_meta_cons:
        problem.add_meta_constraint(std_meta_con)

    # add standardized constraints to subproblems
    for sp in problem.subproblems.values():

        std_sp_meta_cons = (
            []
        )  # instantiate list of standardized meta-constraints for the subproblems

        for (
            meta_con
        ) in sp.model_meta_cons:  # iterate over all meta-constraints in the subproblem

            # retrieve the standardized parent meta-constraint
            std_meta_cons_c = problem.origin_to_std_con_map[meta_con.symbol]

            if not meta_con.is_sub:  # original meta-constraint
                # add the standardized parent meta-constraints to the list
                std_sp_meta_cons.extend(std_meta_cons_c)

            else:  # meta-constraint subset
                for i, std_meta_con in enumerate(std_meta_cons_c):

                    std_sub_meta_con = deepcopy(
                        std_meta_con
                    )  # build a sub-meta-constraint

                    # retrieve indexing subset and assign it to sub-meta-constraint
                    idx_subset_node = (
                        std_meta_con.idx_set_node
                        if i == 0
                        else deepcopy(std_meta_con.idx_set_node)
                    )
                    std_sub_meta_con.idx_set_node = idx_subset_node

                    std_sp_meta_cons.append(std_sub_meta_con)

        sp.model_meta_cons = std_sp_meta_cons  # assign list of standardized meta-constraints to the subproblem

    return problem.origin_to_std_con_map


def __standardize_objective(meta_obj: mat.MetaObjective):
    if meta_obj.direction == mat.MetaObjective.MAXIMIZE_DIRECTION:

        meta_obj.direction = mat.MetaObjective.MINIMIZE_DIRECTION

        expression = meta_obj.expression
        operand = expression.root_node
        if not isinstance(operand, mat.ArithmeticExpressionNode):
            raise ValueError(
                "Formulator expected an arithmetic expression node"
                " while reformulating an objective function"
            )

        operand.is_prioritized = True
        neg_op = nb.append_negative_unity_coefficient(operand)

        expression.root_node = neg_op
        expression.link_nodes()


def __standardize_constraint(
    problem: Problem, meta_con: mat.MetaConstraint
) -> List[mat.MetaConstraint]:

    ctype = meta_con.elicit_constraint_type()  # elicit constraint type

    if __is_constraint_standardized(meta_con):
        return [
            meta_con
        ]  # return the original meta-constraint if it is already in standard form

    else:
        if ctype == mat.MetaConstraint.EQUALITY_TYPE:
            ref_meta_cons = [__standardize_equality_constraint(meta_con)]
        elif ctype == mat.MetaConstraint.INEQUALITY_TYPE:
            ref_meta_cons = [__standardize_inequality_constraint(meta_con)]
        elif ctype == mat.MetaConstraint.DOUBLE_INEQUALITY_TYPE:
            ref_meta_cons = __standardize_double_inequality_constraint(
                problem, meta_con
            )
        else:
            raise ValueError(
                "Formulator unable to resolve the constraint type of '{0}'".format(
                    meta_con
                )
            )

    return ref_meta_cons


def __is_constraint_standardized(meta_con: mat.MetaConstraint):

    # double inequality
    if meta_con.constraint_type == mat.MetaConstraint.DOUBLE_INEQUALITY_TYPE:
        return False

    # single inequality or equality
    else:

        rel_node = meta_con.expression.root_node
        if not isinstance(rel_node, mat.RelationalOperationNode):
            raise ValueError(
                "Formulator expected a relational operation node"
                " while verifying whether the constraint '{0}' is in standard form".format(
                    meta_con
                )
            )

        if rel_node.operator == mat.GREATER_EQUAL_INEQUALITY_OPERATOR:
            return False  # inequality is reversed

        rhs_node = rel_node.rhs_operand

        if not isinstance(rhs_node, mat.NumericNode):
            return False  # rhs operand is non-zero

        else:
            if rhs_node.value != 0:
                return False  # rhs operand is non-zero
            else:
                return True


def __standardize_equality_constraint(
    meta_con: mat.MetaConstraint,
) -> mat.MetaConstraint:

    eq_op_node = meta_con.expression.root_node
    if not isinstance(eq_op_node, mat.RelationalOperationNode):
        raise ValueError(
            "Formulator encountered unexpected expression node"
            + " while standardizing equality constraint '{0}'".format(meta_con)
        )

    __move_relational_expression_operands_to_lhs(eq_op_node)

    meta_con.expression.link_nodes()

    return meta_con


def __standardize_inequality_constraint(
    meta_con: mat.MetaConstraint,
) -> mat.MetaConstraint:

    ineq_op_node = meta_con.expression.root_node
    if not isinstance(ineq_op_node, mat.RelationalOperationNode):
        raise ValueError(
            "Formulator encountered unexpected expression node"
            + " while standardizing inequality constraint '{0}'".format(meta_con)
        )

    __move_relational_expression_operands_to_lhs(ineq_op_node)

    meta_con.expression.link_nodes()

    return meta_con


def __standardize_double_inequality_constraint(
    problem: Problem, meta_con: mat.MetaConstraint
) -> List[mat.MetaConstraint]:

    ref_meta_cons = []

    lb_operand, mid_operand, ub_operand = __extract_operands_from_double_inequality(
        meta_con
    )

    for i in range(2):

        if i == 0:
            lhs_operand = lb_operand
            rhs_operand = mid_operand

        else:
            lhs_operand = deepcopy(mid_operand)
            rhs_operand = ub_operand

        rhs_operand = nb.append_negative_unity_coefficient(rhs_operand)

        sub_node = nb.build_addition_node([lhs_operand, rhs_operand])

        ref_ineq_op_node = mat.RelationalOperationNode(
            operator=mat.LESS_EQUAL_INEQUALITY_OPERATOR,
            lhs_operand=sub_node,
            rhs_operand=mat.NumericNode(0),
        )

        mc_clone = deepcopy(meta_con)

        new_sym = problem.generate_unique_symbol(
            "{0}_I{1}".format(meta_con.symbol, i + 1)
        )
        mc_clone.symbol = new_sym
        mc_clone.non_std_symbol = meta_con.symbol

        expr_clone = mc_clone.expression
        expr_clone.root_node = ref_ineq_op_node
        expr_clone.link_nodes()

        mc_clone.elicit_constraint_type()

        ref_meta_cons.append(mc_clone)

    problem.origin_to_std_con_map[meta_con.symbol] = ref_meta_cons

    return ref_meta_cons


def __extract_operands_from_double_inequality(meta_con: mat.MetaConstraint):

    ineq_op_node = meta_con.expression.root_node

    if not isinstance(ineq_op_node, mat.RelationalOperationNode):
        raise ValueError(
            "Formulator encountered unexpected expression node"
            + " while standardizing double inequality constraint '{0}'".format(meta_con)
        )

    # (L --- M) --- R
    if isinstance(ineq_op_node.lhs_operand, mat.RelationalOperationNode):

        child_ineq_op_node = ineq_op_node.lhs_operand

        # (L <= M) <= R
        if (
            ineq_op_node.operator == mat.LESS_EQUAL_INEQUALITY_OPERATOR
            and child_ineq_op_node.operator == mat.LESS_EQUAL_INEQUALITY_OPERATOR
        ):
            lb_operand = child_ineq_op_node.lhs_operand  # L
            mid_operand = child_ineq_op_node.rhs_operand  # M
            ub_operand = ineq_op_node.rhs_operand  # R

        # (L >= M) >= R
        elif (
            ineq_op_node.operator == mat.GREATER_EQUAL_INEQUALITY_OPERATOR
            and child_ineq_op_node.operator == mat.GREATER_EQUAL_INEQUALITY_OPERATOR
        ):
            lb_operand = ineq_op_node.rhs_operand  # R
            mid_operand = child_ineq_op_node.rhs_operand  # M
            ub_operand = child_ineq_op_node.lhs_operand  # L

        else:
            raise ValueError(
                "Formulator encountered unexpected expression structure"
                + " while standardizing double inequality constraint '{0}'".format(
                    meta_con
                )
            )

    # L --- (M --- R)
    elif isinstance(ineq_op_node.rhs_operand, mat.RelationalOperationNode):

        child_ineq_op_node = ineq_op_node.rhs_operand

        # L <= (M <= R)
        if (
            ineq_op_node.operator == mat.LESS_EQUAL_INEQUALITY_OPERATOR
            and child_ineq_op_node.operator == mat.LESS_EQUAL_INEQUALITY_OPERATOR
        ):
            lb_operand = ineq_op_node.lhs_operand  # L
            mid_operand = child_ineq_op_node.lhs_operand  # M
            ub_operand = child_ineq_op_node.rhs_operand  # R

        # L >= (M >= R)
        elif (
            ineq_op_node.operator == mat.GREATER_EQUAL_INEQUALITY_OPERATOR
            and child_ineq_op_node.operator == mat.GREATER_EQUAL_INEQUALITY_OPERATOR
        ):
            lb_operand = child_ineq_op_node.rhs_operand  # R
            mid_operand = child_ineq_op_node.lhs_operand  # M
            ub_operand = ineq_op_node.lhs_operand  # L

        else:
            raise ValueError(
                "Formulator encountered unexpected expression structure"
                + " while standardizing double inequality constraint '{0}'".format(
                    meta_con
                )
            )

    else:
        raise ValueError(
            "Formulator encountered unexpected expression node"
            + " while standardizing double inequality constraint '{0}'".format(meta_con)
        )

    return lb_operand, mid_operand, ub_operand


# Constraint Reformulation
# ------------------------------------------------------------------------------------------------------------------


def __move_relational_expression_operands_to_lhs(
    rel_op_node: mat.RelationalOperationNode,
):

    operator = rel_op_node.operator

    if operator in (mat.EQUALITY_OPERATOR, mat.LESS_EQUAL_INEQUALITY_OPERATOR):
        lhs_operand = rel_op_node.lhs_operand
        rhs_operand = rel_op_node.rhs_operand
    else:  # >=
        rel_op_node.operator = mat.LESS_EQUAL_INEQUALITY_OPERATOR
        lhs_operand = rel_op_node.rhs_operand
        rhs_operand = rel_op_node.lhs_operand

    rhs_operand = nb.append_negative_unity_coefficient(rhs_operand)

    sub_node = nb.build_addition_node([lhs_operand, rhs_operand])

    rel_op_node.lhs_operand = sub_node
    rel_op_node.rhs_operand = mat.NumericNode(0)


def convert_equality_to_inequality_constraints(
    problem: Problem, meta_con: mat.MetaConstraint
):

    ref_meta_cons = []
    old_sym = meta_con.symbol

    for i in range(2):

        ref_meta_con = deepcopy(meta_con)

        new_sym = problem.generate_unique_symbol("{0}_E{1}".format(old_sym, i + 1))
        ref_meta_con.symbol = new_sym
        ref_meta_con.non_std_symbol = old_sym

        eq_op_node = ref_meta_con.expression.root_node
        if not isinstance(eq_op_node, mat.RelationalOperationNode):
            raise ValueError(
                "Formulator encountered an unexpected expression node"
                + " while converting equality constraint {0}".format(old_sym)
                + " into two inequality constraints"
            )

        eq_op_node.operator = mat.LESS_EQUAL_INEQUALITY_OPERATOR

        if i == 1:
            eq_op_node.lhs_operand = nb.append_negative_unity_coefficient(
                eq_op_node.lhs_operand
            )
            eq_op_node.rhs_operand = nb.append_negative_unity_coefficient(
                eq_op_node.rhs_operand
            )

        ref_meta_con.elicit_constraint_type()

        ref_meta_cons.append(ref_meta_con)

    problem.replace_model_meta_constraint(
        old_symbol=old_sym, new_meta_cons=ref_meta_cons
    )

    return ref_meta_cons


# Slack Variables
# ------------------------------------------------------------------------------------------------------------------


def formulate_slackened_constraint(
    problem: Problem, meta_con: mat.MetaConstraint
) -> Tuple[List[mat.MetaVariable], mat.MetaConstraint]:

    ctype = meta_con.elicit_constraint_type()

    # generate slack variables for an equality constraint
    if ctype == mat.MetaConstraint.EQUALITY_TYPE:
        """
        pos_sl_meta_var, pos_sl_slack_var_node = self.__generate_slack_var(meta_con, symbol_suffix="P")
        neg_sl_meta_var, neg_sl_slack_var_node = self.__generate_slack_var(meta_con, symbol_suffix="N")
        sl_meta_vars = [pos_sl_meta_var, neg_sl_meta_var]

        slack_node = mat.BinaryArithmeticOperationNode(id=self.generate_free_node_id(),
                                                       operator=mat.SUBTRACTION_OPERATOR,
                                                       lhs_operand=pos_sl_slack_var_node,
                                                       rhs_operand=neg_sl_slack_var_node)
        """
        return [], meta_con

    # generate slack variable for an inequality constraint
    elif ctype == mat.MetaConstraint.INEQUALITY_TYPE:

        sl_meta_var, sl_slack_var_node = __generate_slack_var(problem, meta_con)
        sl_meta_vars = [sl_meta_var]

        slack_node = sl_slack_var_node

    else:
        raise ValueError(
            "Formulator encountered an unexpected constraint type"
            + " while building a slackened constraint for '{0}'".format(meta_con)
        )

    sl_meta_con = deepcopy(meta_con)
    expr_clone = deepcopy(meta_con.expression)

    con_sym = problem.generate_unique_symbol("{0}_F".format(meta_con.symbol))
    sl_meta_con.symbol = con_sym

    rel_op_node = expr_clone.root_node
    if not isinstance(rel_op_node, mat.RelationalOperationNode):
        raise ValueError(
            "Formulator encountered unexpected expression node"
            + " while building a slackened constraint for '{0}'".format(meta_con)
        )

    rhs_node = nb.append_negative_unity_coefficient(slack_node)
    rel_op_node.lhs_operand = nb.build_addition_node(
        [rel_op_node.lhs_operand, rhs_node]
    )

    expr_clone.link_nodes()
    sl_meta_con.expression = expr_clone

    return sl_meta_vars, sl_meta_con


def __generate_slack_var(
    problem: Problem, meta_con: mat.MetaConstraint, symbol_suffix: str = ""
):

    sym = problem.generate_unique_symbol(
        "{0}_SL{1}".format(meta_con.symbol, symbol_suffix)
    )
    sl_meta_var = mat.MetaVariable(
        symbol=sym,
        idx_meta_sets=deepcopy(meta_con.idx_meta_sets),
        idx_set_node=meta_con.idx_set_node,
        default_value=mat.NumericNode(0),
        lower_bound=mat.NumericNode(0),
    )

    entity_index_node = nb.build_default_entity_index_node(sl_meta_var)
    sl_var_node = mat.DeclaredEntityNode(
        symbol=sym, idx_node=entity_index_node, type=mat.VAR_TYPE
    )

    return sl_meta_var, sl_var_node


def formulate_slack_min_objective(
    problem: Problem,
    idx_meta_sets: Union[List[mat.MetaSet], Dict[str, mat.MetaSet]],
    sl_meta_vars: Union[List[mat.MetaVariable], Dict[str, mat.MetaVariable]],
    obj_sym: str,
) -> mat.MetaObjective:

    operands = []

    if isinstance(sl_meta_vars, dict):
        sl_meta_vars = list(sl_meta_vars.values())

    for sl_meta_var in sl_meta_vars:

        entity_index_node = nb.build_default_entity_index_node(sl_meta_var)
        slack_node = mat.DeclaredEntityNode(
            symbol=sl_meta_var.symbol, idx_node=entity_index_node
        )

        if sl_meta_var.idx_set_reduced_dim == 0:
            operand = slack_node
        else:
            idx_set_node = nb.build_entity_idx_set_node(
                problem=problem, meta_entity=sl_meta_var, remove_sets=idx_meta_sets
            )
            if idx_set_node is None:
                operand = slack_node
            else:
                operand = mat.ArithmeticTransformationNode(
                    fcn=mat.SUMMATION_FUNCTION,
                    idx_set_node=idx_set_node,
                    operands=slack_node,
                )

        operands.append(operand)

    if len(operands) > 1:
        expr_node = nb.build_addition_node(operands)

    elif len(operands) == 1:
        expr_node = operands[0]

    else:
        expr_node = mat.NumericNode(0)

    expression = mat.Expression(expr_node)

    meta_obj = eb.build_meta_obj(
        problem=problem,
        symbol=obj_sym,
        idx_meta_sets=idx_meta_sets,
        direction="minimize",
        expression=expression,
    )

    return meta_obj


# Substitution
# ------------------------------------------------------------------------------------------------------------------


def substitute(
    root_node: mat.ExpressionNode,
    sub_map: Dict[str, Tuple[mat.ExpressionNode, Iterable[str]]],
) -> mat.ExpressionNode:
    """
    Substitute select declared entity nodes with corresponding expression nodes.

    :param root_node: root node of the expression in which substitution(s) are to take place
    :param sub_map: mapping of the symbols of the original declared entities to the corresponding substitutes
    :return: root node
    """

    if sub_map is None:
        return root_node

    if len(sub_map) == 0:
        return root_node
    else:
        for sub_node, unb_syms in sub_map.values():
            sub_node.is_prioritized = True

    if isinstance(root_node, mat.DeclaredEntityNode):
        if root_node.symbol in sub_map:
            sub_node, _ = sub_map[root_node.symbol]
            return deepcopy(sub_node)
        else:
            return root_node

    queue = Queue()
    queue.put(root_node)

    while not queue.empty():

        node: mat.ExpressionNode = queue.get()
        modified_children = []

        for child in node.get_children():

            if isinstance(child, mat.DeclaredEntityNode):

                if child.symbol in sub_map:

                    sub_node, unb_syms = sub_map[child.symbol]
                    sub_node = deepcopy(sub_node)

                    if child.idx_node is not None:
                        dummy_map = {
                            unb_sym: cmpt_node
                            for unb_sym, cmpt_node in zip(
                                unb_syms, child.idx_node.component_nodes
                            )
                        }
                        sub_node = nb.replace_dummy_nodes(sub_node, mapping=dummy_map)

                    modified_children.append(sub_node)

                else:
                    modified_children.append(child)
                    queue.put(child)

            else:
                modified_children.append(child)
                queue.put(child)

        node.set_children(modified_children)

    return root_node


def substitute_defined_variables(problem: BaseProblem):

    # identify all defined variables

    sub_map = {}  # map of defined variable symbols to their defined values
    for mv in problem.model_meta_vars:
        if mv.is_defined:
            sub_map[mv.symbol] = (
                mv.defined_value_node,
                mv.idx_set_reduced_dummy_element,
            )

    if len(sub_map) > 0:

        # modify meta-objective expressions
        for mo in problem.model_meta_objs:
            mo.expression.root_node = substitute(
                root_node=mo.expression.root_node, sub_map=sub_map
            )

        # modify meta-constraint expressions
        for mc in problem.model_meta_cons:
            mc.expression.root_node = substitute(
                root_node=mc.expression.root_node, sub_map=sub_map
            )


# Simplification
# ------------------------------------------------------------------------------------------------------------------


def simplify(problem: Problem, node: mat.ExpressionNode):

    # dummy
    if isinstance(node, mat.DummyNode):
        return node

    # compound dummy
    elif isinstance(node, mat.CompoundDummyNode):
        for i, cmpt_node in enumerate(node.component_nodes):
            node.component_nodes[i] = simplify(problem=problem, node=cmpt_node)
        return node

    # conditional
    elif isinstance(node, mat.ArithmeticConditionalNode) or isinstance(
        node, mat.SetConditionalNode
    ):
        return __simplify_conditional_expression(problem=problem, node=node)

    # arithmetic
    elif isinstance(node, mat.ArithmeticExpressionNode):
        return __simplify_arithmetic_expression(problem=problem, node=node)

    # logical
    elif isinstance(node, mat.LogicalExpressionNode):
        return __simplify_logical_expression(problem=problem, node=node)

    # set
    elif isinstance(node, mat.SetExpressionNode):
        return __simplify_set_expression(problem=problem, node=node)

    # string
    elif isinstance(node, mat.StringExpressionNode):
        return __simplify_string_expression(problem=problem, node=node)


def __simplify_conditional_expression(
    problem: Problem, node: Union[mat.ArithmeticConditionalNode, mat.SetConditionalNode]
):
    spl_operands = []
    spl_conditions = []

    # simplify operand nodes
    for o in node.operands:
        spl_operands.append(simplify(problem=problem, node=o))

    # simplify conditional nodes
    for o in node.conditions:
        spl_conditions.append(__simplify_logical_expression(problem=problem, node=o))

    red_spl_operands = []
    red_spl_conditions = []

    # eliminate operands whose conditions always evaluate to false
    i = 0
    while i < len(spl_operands):

        if i < len(spl_conditions):
            spl_condition = spl_conditions[i]
        else:
            spl_condition = None

        if spl_condition is not None:

            # condition cannot be simplified to an explicit boolean value
            if not isinstance(spl_condition, mat.BooleanNode):
                red_spl_operands.append(spl_operands[i])
                red_spl_conditions.append(spl_condition)

            # condition simplifies to explicit True
            elif spl_condition.value:
                red_spl_operands.append(spl_operands[i])
                # discard the corresponding condition
                break  # discard remaining conditions

            # if the condition simplifies to explicit False, then the entire operand is discarded

        else:
            red_spl_operands.append(spl_operands[i])

        i += 1  # increment index

    # check if a trailing else clause is the sole remaining clause
    if len(red_spl_operands) == 1 and len(red_spl_conditions) == 0:
        return spl_operands[0]  # return the expression of the trailing else clause

    node.operands = red_spl_operands
    node.conditions = red_spl_conditions

    return node  # return the simplified conditional node


def __simplify_arithmetic_expression(
    problem: Problem, node: mat.ArithmeticExpressionNode
):

    # numeric
    if isinstance(node, mat.NumericNode):
        return node

    # declared entity
    elif isinstance(node, mat.DeclaredEntityNode):
        return node

    # transformation
    elif isinstance(node, mat.ArithmeticTransformationNode):
        return __simplify_arithmetic_transformation(problem=problem, node=node)

    # operation
    elif isinstance(node, mat.ArithmeticOperationNode):
        return __simplify_arithmetic_operation(problem=problem, node=node)

    # conditional
    elif isinstance(node, mat.ArithmeticConditionalNode):
        return __simplify_conditional_expression(problem=problem, node=node)

    # other
    else:
        raise ValueError(
            "Formulator encountered an unexpected node '{0}'".format(node)
            + " while simplifying an arithmetic expression"
        )


def __simplify_arithmetic_transformation(
    problem: Problem, node: mat.ArithmeticTransformationNode
):

    if node.is_reductive():

        # check number of component indexing set nodes
        if len(node.idx_set_node.set_nodes) == 0:
            # return 0 if the indexing set is empty
            return mat.NumericNode(0)

    spl_operands = []

    for o in node.operands:
        spl_operands.append(__simplify_arithmetic_expression(problem=problem, node=o))

    node.operands = spl_operands

    # special case: all operands are numeric nodes
    if all([isinstance(spl_operand, mat.NumericNode) for spl_operand in spl_operands]):
        value = node.evaluate(problem.state)[0]  # evaluate the transformation
        return mat.NumericNode(value)  # return a numeric node

    # one or more non-numeric operand nodes
    else:
        return node  # return transformation node


def __simplify_arithmetic_operation(
    problem: Problem, node: mat.ArithmeticOperationNode
):

    spl_operands = []  # list of simplified operand nodes

    # simplify operand nodes
    for o in node.operands:
        spl_operands.append(__simplify_arithmetic_expression(problem=problem, node=o))

    # unary positive
    if node.operator == mat.UNARY_POSITIVE_OPERATOR:
        return spl_operands[0]

    # unary negative
    elif node.operator == mat.UNARY_NEGATION_OPERATOR:

        spl_operand = spl_operands[0]

        # special case: operand is a numeric node
        if isinstance(spl_operand, mat.NumericNode):
            spl_operand.value *= -1
            return spl_operand

        # operand is non-numeric
        else:
            node.operands[0] = spl_operand
            return node

    # addition and multiplication
    elif node.operator in (mat.ADDITION_OPERATOR, mat.MULTIPLICATION_OPERATOR):

        flat_spl_operands = []  # flattened list of simplified operands

        # flatten embedded arithmetic operation nodes with same operator as the root node
        for spl_operand in spl_operands:

            # operation with identical operator
            if (
                isinstance(spl_operand, mat.ArithmeticOperationNode)
                and spl_operand.operator == node.operator
            ):
                flat_spl_operands.extend(spl_operand.operands)

            else:  # other
                flat_spl_operands.append(spl_operand)

        # initialize the default constant value
        if node.operator == mat.ADDITION_OPERATOR:
            const_val = 0  # default term
        else:
            const_val = 1  # default factor

        non_num_spl_operands = []

        # combine numeric nodes together into a single constant term or coefficient
        for spl_operand in flat_spl_operands:

            # numeric
            if isinstance(spl_operand, mat.NumericNode):

                # update constant value
                if node.operator == mat.ADDITION_OPERATOR:  # addition
                    const_val += spl_operand.value
                else:  # multiplication
                    const_val *= spl_operand.value

            else:  # other
                non_num_spl_operands.append(spl_operand)

        # all operands were simplified to numeric constants
        if len(non_num_spl_operands) == 0:
            return mat.NumericNode(const_val)  # return single operand

        # at least one non-numeric node remaining after simplification
        else:

            # insert constant node into the list of simplified operands
            if (node.operator == mat.ADDITION_OPERATOR and const_val != 0) or (
                node.operator == mat.MULTIPLICATION_OPERATOR and const_val != 1
            ):
                non_num_spl_operands.insert(0, mat.NumericNode(const_val))

            node.set_children(
                non_num_spl_operands
            )  # update operands of the operation node

            return node  # return simplified operation node

    # subtraction, division, exponentiation
    else:

        lhs_operand = spl_operands[0]
        rhs_operand = spl_operands[1]

        node.set_children(spl_operands)

        # both operand nodes are numeric
        if isinstance(lhs_operand, mat.NumericNode) and isinstance(
            rhs_operand, mat.NumericNode
        ):
            value = node.evaluate(problem.state)[0]  # evaluate the operation node
            return mat.NumericNode(value)  # return numeric node with simplified value

        # at least one of the operand nodes is non-numeric
        else:
            return node  # return operation node with simplified operands


def __simplify_logical_expression(problem: Problem, node: mat.LogicalExpressionNode):

    # logical reduction
    if isinstance(node, mat.LogicalReductionNode):
        return __simplify_logical_reduction(problem=problem, node=node)

    # logical operation
    elif isinstance(node, mat.LogicalOperationNode):
        return __simplify_logical_operation(problem=problem, node=node)

    # relational operation
    elif isinstance(node, mat.RelationalOperationNode):
        return __simplify_relational_operation(problem=problem, node=node)

    # other logical expression node
    else:
        return node


def __simplify_logical_reduction(problem: Problem, node: mat.LogicalReductionNode):

    node.idx_set_node = __simplify_set_expression(
        problem=problem, node=node.idx_set_node
    )

    # check number of component indexing set nodes
    if len(node.idx_set_node.set_nodes) == 0:

        if node.operator == mat.EXISTS_OPERATOR:  # exists
            return mat.BooleanNode(False)

        else:  # for all
            return mat.BooleanNode(True)

    node.operand = __simplify_logical_expression(problem=problem, node=node.operand)

    # special case: operand is a boolean node
    if isinstance(node.operand, mat.BooleanNode):
        value = node.evaluate(problem.state)[0]  # evaluate the reduction
        return mat.BooleanNode(value)  # return a boolean node

    # operand is a more complex logical expression
    else:
        return node  # return logical reduction node


def __simplify_logical_operation(problem: Problem, node: mat.LogicalOperationNode):

    spl_operands = []  # list of simplified operand nodes

    # simplify operand nodes
    for o in node.operands:
        spl_operands.append(simplify(problem=problem, node=o))

    # inversion
    if node.operator == mat.UNARY_INVERSION_OPERATOR:

        spl_operand = spl_operands[0]

        # special case: operand is boolean
        if isinstance(spl_operand, mat.BooleanNode):
            spl_operand.value = not spl_operand.value
            return spl_operand

        # operand is non-numeric
        else:
            node.operands[0] = spl_operand
            return node

    # binary logical operation
    else:

        node.set_children(spl_operands)

        # check whether the operands are elementary
        all_elementary = True
        for spl_operand in spl_operands:
            if not (
                isinstance(spl_operand, mat.BooleanNode)
                or isinstance(spl_operand, mat.NumericNode)
                or isinstance(spl_operand, mat.StringNode)
            ):
                all_elementary = False

        if all_elementary:
            return mat.BooleanNode(
                node.evaluate(problem.state)[0]
            )  # return boolean node with simplified value

        # at least one of the operand nodes is non-elementary
        else:
            return node  # return operation node with simplified operands


def __simplify_relational_operation(
    problem: Problem, node: mat.RelationalOperationNode
):

    spl_operands = []  # list of simplified operand nodes

    # simplify operand nodes
    for o in [node.lhs_operand, node.rhs_operand]:
        spl_operands.append(simplify(problem=problem, node=o))

    lhs_operand = spl_operands[0]
    rhs_operand = spl_operands[1]

    node.set_children(spl_operands)

    # both operand nodes are elementary
    if (
        isinstance(lhs_operand, mat.BooleanNode)
        or isinstance(lhs_operand, mat.NumericNode)
        or isinstance(lhs_operand, mat.StringNode)
    ) and (
        isinstance(rhs_operand, mat.BooleanNode)
        or isinstance(rhs_operand, mat.NumericNode)
        or isinstance(rhs_operand, mat.StringNode)
    ):
        value = node.evaluate(problem.state)[0]  # evaluate the operation node
        return mat.BooleanNode(value)  # return numeric node with simplified value

    # at least one of the operand nodes is non-numeric
    else:
        return node  # return operation node with simplified operands


def __simplify_set_expression(problem: Problem, node: mat.SetExpressionNode):

    # compound set
    if isinstance(node, mat.CompoundSetNode):

        if node.constraint_node is not None:
            node.constraint_node = __simplify_logical_expression(
                problem=problem, node=node.constraint_node
            )

        return node

    # reduction
    elif isinstance(node, mat.SetReductionNode):

        # check number of component indexing sets
        if len(node.idx_set_node.set_nodes) == 0:
            # return empty set if the indexing set is empty
            return mat.EnumeratedSetNode()

        return node

    # conditional
    elif isinstance(node, mat.SetConditionalNode):
        return __simplify_conditional_expression(problem=problem, node=node)

    # other set expression node
    else:
        return node


def __simplify_string_expression(problem: Problem, node: mat.StringExpressionNode):

    # string operation
    if isinstance(node, mat.StringOperationNode):

        spl_operands = []  # list of simplified operand nodes

        # simplify operand nodes
        for o in node.operands:
            spl_operands.append(simplify(problem=problem, node=o))

        const_val = ""  # initialize the default constant value
        non_lit_spl_operands = []

        for spl_operand in spl_operands:

            # operand is a string literal
            if isinstance(spl_operand, mat.StringNode):

                # update constant value
                if node.operator == mat.CONCATENATION_OPERATOR:  # concatenation
                    const_val += spl_operand.literal

            else:
                non_lit_spl_operands.append(spl_operand)

        # all operands were simplified to string literals
        if len(non_lit_spl_operands) == 0:
            return mat.StringNode(const_val)  # return single operand

        # at least one non-literal node remaining after simplification
        else:

            # insert constant node into the list of simplified operands
            if node.operator == mat.CONCATENATION_OPERATOR and const_val != "":
                non_lit_spl_operands.insert(0, mat.StringNode(const_val))

            node.set_children(
                non_lit_spl_operands
            )  # update operands of the operation node

            return node  # return simplified operation node

    # string literal
    else:
        return node


def simplify_node_to_scalar_value(
    problem: Problem,
    node: mat.ExpressionNode,
    idx_set: mat.IndexingSet,
    dummy_element: mat.Element,
) -> Optional[Union[bool, Number, str, mat.IndexingSet]]:

    declared_entity_nodes = mat.get_param_and_var_nodes(node)

    if len(declared_entity_nodes) == 0:

        values = node.evaluate(
            state=problem.state, idx_set=idx_set, dummy_element=dummy_element
        )

        if len(values) > 1:
            for i in range(1, len(values)):
                if values[0] != values[i]:
                    return None

        if len(values) == 0:
            return 0
        else:
            return values[0]

    return None


# Subtraction and Arithmetic Negation
# ------------------------------------------------------------------------------------------------------------------


def reformulate_subtraction_and_unary_negation(root_node: mat.ExpressionNode):

    root_node = __reformulate_subtraction_or_unary_negation_node(root_node)

    queue = Queue()
    queue.put(root_node)

    while not queue.empty():

        node = queue.get()

        ref_children = []
        for child in node.get_children():

            if isinstance(child, mat.RelationalOperationNode) or isinstance(
                child, mat.ArithmeticExpressionNode
            ):
                child = __reformulate_subtraction_or_unary_negation_node(child)
                queue.put(child)

            ref_children.append(child)

        node.set_children(ref_children)

    return root_node


def __reformulate_subtraction_or_unary_negation_node(node: mat.ExpressionNode):

    if isinstance(node, mat.ArithmeticOperationNode):

        if node.operator == mat.SUBTRACTION_OPERATOR:
            rhs_operand = nb.append_negative_unity_coefficient(node.get_rhs_operand())
            node = mat.AdditionNode(operands=[node.get_lhs_operand(), rhs_operand])

        elif node.operator == mat.UNARY_NEGATION_OPERATOR:
            node = nb.append_negative_unity_coefficient(node.operands[0])

    return node


# Exponentiation
# ------------------------------------------------------------------------------------------------------------------


def __factorize_exponentiation(
    problem: Problem,
    exp_op_node: mat.ExponentiationNode,
    exponents: Iterable[int],
    idx_set: mat.IndexingSet = None,
    dummy_element: mat.Element = None,
):

    factors = []

    if isinstance(exp_op_node.get_rhs_operand(), mat.NumericNode):
        exp_val = exp_op_node.get_rhs_operand().value

    else:
        exp_val = simplify_node_to_scalar_value(
            problem=problem,
            node=exp_op_node.get_rhs_operand(),
            idx_set=idx_set,
            dummy_element=dummy_element,
        )

    if exp_val is not None:

        if exp_val == 0:
            factors.append(mat.NumericNode(1))

        elif exp_val == 1:
            factors.append(exp_op_node.get_lhs_operand())

        else:

            # TODO: account for negative exponents by moving the base node to the denominator of a division node

            if exp_val in exponents:

                exp_op_node.get_lhs_operand().is_prioritized = True
                factors = [exp_op_node.get_lhs_operand()]

                for i in range(2, exp_val + 1):
                    factors.append(deepcopy(exp_op_node.get_lhs_operand()))

            else:
                factors.append(exp_op_node)

    else:
        factors.append(exp_op_node)

    return factors


def __distribute_exponent(
    base_node: mat.ArithmeticExpressionNode, exponent_node: mat.ArithmeticExpressionNode
):

    if (
        isinstance(base_node, mat.ArithmeticOperationNode)
        and base_node.operator == mat.MULTIPLICATION_OPERATOR
    ):

        factors = []
        for factor in base_node.operands:
            factor.is_prioritized = True
            factors.append(
                mat.ExponentiationNode(
                    lhs_operand=factor,
                    rhs_operand=deepcopy(exponent_node),
                    is_prioritized=True,
                )
            )

    else:
        exp_op_node = mat.ExponentiationNode(
            lhs_operand=base_node, rhs_operand=exponent_node, is_prioritized=True
        )
        factors = [exp_op_node]

    return factors


# Expansion
# ----------------------------------------------------------------------------------------------------------------------


def expand_multiplication(
    problem: Problem,
    node: mat.ArithmeticExpressionNode,
    idx_set: mat.IndexingSet = None,
    dummy_element: mat.Element = None,
) -> List[mat.ArithmeticExpressionNode]:
    terms = __expand_multiplication(
        problem=problem, node=node, idx_set=idx_set, dummy_element=dummy_element
    )
    for term in terms:
        term.is_prioritized = True
    return terms


def __expand_multiplication(
    problem: Problem,
    node: mat.ArithmeticExpressionNode,
    idx_set: mat.IndexingSet = None,
    dummy_element: mat.Element = None,
) -> List[mat.ArithmeticExpressionNode]:

    # arithmetic operation
    if isinstance(node, mat.ArithmeticExpressionNode):

        # constant or declared entity
        if isinstance(node, mat.NumericNode) or isinstance(
            node, mat.DeclaredEntityNode
        ):
            return [node]

        # transformation
        elif isinstance(node, mat.ArithmeticTransformationNode):

            # reductive summation
            if node.is_reductive() and node.fcn == mat.SUMMATION_FUNCTION:

                idx_sets = node.idx_set_node.generate_combined_idx_sets(
                    state=problem.state,
                    idx_set=idx_set,
                    dummy_element=dummy_element,
                    can_reduce=False,
                )
                idx_set = mat.OrderedSet().union(*idx_sets)
                dummy_element = node.idx_set_node.combined_dummy_element

                terms = __expand_multiplication(
                    problem, node.operands[0], idx_set, dummy_element
                )

                # single term
                if len(terms) == 1:  # return the summation node
                    node.operands[0] = terms[0]
                    return [node]

                # multiple terms
                else:  # distribute the summation to each term

                    # retrieve the unbound symbols of the indexing set node of the summation node
                    unb_syms = node.idx_set_node.get_defined_unbound_symbols()

                    node.operands.clear()  # clear the list of operands of the summation node

                    dist_terms = (
                        []
                    )  # list of terms onto which the summation node was distributed

                    for term in terms:

                        # retrieve the set of unbound symbols present in the term
                        term_unb_syms = nb.retrieve_unbound_symbols(
                            root_node=term, in_filter=unb_syms
                        )

                        # if any defined unbound symbols are present in the term, then the term is controlled by
                        # the indexing set of the summation node
                        if (
                            len(term_unb_syms) > 0
                        ):  # the term is controlled by the indexing set node
                            sum_node = deepcopy(node)
                            sum_node.operands.append(term)
                            dist_terms.append(sum_node)

                        else:  # the term is not controlled by the indexing set node
                            dist_terms.append(
                                term
                            )  # append the term without a summation transformation

                    return dist_terms

            # other
            else:

                for i, operand in enumerate(node.operands):
                    terms = __expand_multiplication(
                        problem, operand, idx_set, dummy_element
                    )
                    node.operands[i] = nb.build_addition_node(terms)

                return [node]

        # operation
        elif isinstance(node, mat.ArithmeticOperationNode):

            # unary positive
            if node.operator == mat.UNARY_POSITIVE_OPERATOR:
                return __expand_multiplication(
                    problem=problem,
                    node=node.operands[0],
                    idx_set=idx_set,
                    dummy_element=dummy_element,
                )

            # unary negation
            elif node.operator == mat.UNARY_POSITIVE_OPERATOR:
                return __expand_multiplication(
                    problem=problem,
                    node=nb.append_negative_unity_coefficient(node.operands[0]),
                    idx_set=idx_set,
                    dummy_element=dummy_element,
                )

            # addition
            elif node.operator == mat.ADDITION_OPERATOR:
                terms = []
                for operand in node.operands:
                    terms.extend(
                        __expand_multiplication(
                            problem, operand, idx_set, dummy_element
                        )
                    )
                return terms

            # multiplication
            elif node.operator == mat.MULTIPLICATION_OPERATOR:
                term_lists = [
                    __expand_multiplication(problem, o, idx_set, dummy_element)
                    for o in node.operands
                ]
                return expand_factors_n(term_lists)

            # binary
            else:

                lhs_terms = __expand_multiplication(
                    problem, node.get_lhs_operand(), idx_set, dummy_element
                )
                rhs_terms = __expand_multiplication(
                    problem, node.get_rhs_operand(), idx_set, dummy_element
                )

                # subtraction
                if node.operator == mat.SUBTRACTION_OPERATOR:
                    rhs_terms = [
                        nb.append_negative_unity_coefficient(t) for t in rhs_terms
                    ]
                    return lhs_terms + rhs_terms

                # division
                elif node.operator == mat.DIVISION_OPERATOR:

                    if len(lhs_terms) == 1:

                        numerator = lhs_terms[0]

                        # special case: numerator is 0
                        if (
                            isinstance(numerator, mat.NumericNode)
                            and numerator.value == 0
                        ):
                            return [numerator]

                        # special case: numerator is 1
                        if (
                            isinstance(numerator, mat.NumericNode)
                            and numerator.value == 1
                        ):
                            if len(rhs_terms) == 1:
                                node.set_rhs_operand(rhs_terms[0])
                            else:
                                node.set_rhs_operand(
                                    nb.build_addition_node(
                                        rhs_terms, is_prioritized=True
                                    )
                                )
                            return [node]

                    node.set_lhs_operand(mat.NumericNode(1))
                    node.is_prioritized = True

                    if len(rhs_terms) == 1:
                        node.set_rhs_operand(rhs_terms[0])
                    else:
                        node.set_rhs_operand(
                            nb.build_multiplication_node(rhs_terms, is_prioritized=True)
                        )

                    rhs_terms = [node]

                    return expand_factors(lhs_terms, rhs_terms)

                # exponentiation
                else:

                    exponent_node = nb.build_addition_node(
                        rhs_terms, is_prioritized=True
                    )

                    if len(lhs_terms) == 1:
                        # distribute exponent to each LHS factor
                        exp_op_factor_nodes = __distribute_exponent(
                            lhs_terms[0], exponent_node
                        )
                    else:
                        exp_op_factor_nodes = [node]

                    expanded_factors = []

                    for i, factor in enumerate(exp_op_factor_nodes):
                        expanded_factors.extend(
                            __factorize_exponentiation(
                                problem=problem,
                                exp_op_node=factor,
                                exponents=(2, 3),
                                idx_set=idx_set,
                                dummy_element=dummy_element,
                            )
                        )

                    term_lists = []

                    for expanded_factor in expanded_factors:
                        if (
                            isinstance(expanded_factor, mat.ArithmeticOperationNode)
                            and node.operator == mat.ADDITION_OPERATOR
                        ):
                            term_lists.append(
                                __expand_multiplication(
                                    problem, expanded_factor, idx_set, dummy_element
                                )
                            )
                        else:
                            term_lists.append([expanded_factor])

                    return expand_factors_n(term_lists)

        # conditional operation
        elif isinstance(node, mat.ArithmeticConditionalNode):

            # uni-conditional
            if len(node.operands) == 1:

                terms = __expand_multiplication(
                    problem, node.operands[0], idx_set, dummy_element
                )

                dist_terms = (
                    []
                )  # list of terms onto which the conditional is distributed
                for term in terms:
                    dist_terms.append(
                        mat.ArithmeticConditionalNode(
                            operands=[term],
                            conditions=[deepcopy(node.conditions[0])],
                            is_prioritized=True,
                        )
                    )

                return dist_terms

            # n-conditional
            else:

                for i, operand in enumerate(node.operands):
                    terms = __expand_multiplication(
                        problem, operand, idx_set, dummy_element
                    )
                    node.operands[i] = nb.build_addition_node(terms)

                return [node]

    else:
        raise ValueError(
            "Formulator encountered an unexpected node while expanding factors"
        )


def expand_factors(
    lhs_terms: List[mat.ArithmeticExpressionNode],
    rhs_terms: List[mat.ArithmeticExpressionNode],
):
    terms = []

    for lhs_term in lhs_terms:
        for rhs_term in rhs_terms:

            factors = [deepcopy(lhs_term), deepcopy(rhs_term)]
            term = nb.build_multiplication_node(factors)
            terms.append(term)

    return terms


def expand_factors_n(factors: List[List[mat.ArithmeticExpressionNode]]):

    terms = []

    factor_count = len(factors)
    terms_per_factor_count = [len(f) for f in factors]  # length factor_count
    expanded_term_count = np.prod(
        terms_per_factor_count
    )  # total number of terms after expansion

    term_indices = np.zeros(shape=(factor_count,), dtype=int)  # length factor_count

    for _ in range(expanded_term_count):

        expanded_factors = []  # unique combination of terms

        # retrieve the expanded factors of a term from each supplied factor according to the current indices
        for i, j in enumerate(term_indices):
            factor = deepcopy(factors[i][j])  # deep copy original factor node
            if (
                isinstance(factor, mat.ArithmeticOperationNode)
                and factor.operator == mat.MULTIPLICATION_OPERATOR
            ):
                # if multiplication operand, add its operands to the list of factors
                expanded_factors.extend(factor.operands)
            else:
                # otherwise, add the factor to the list
                expanded_factors.append(factor)

        # increment indices
        for i in range(factor_count - 1, -1, -1):
            term_indices[i] += 1  # increment term index of current factor
            if (
                term_indices[i] == terms_per_factor_count[i]
            ):  # processed last term in factor
                term_indices[
                    i
                ] = 0  # reset term index to 0 and proceed to previous factor index
            else:  # otherwise, break loop
                break

        term = nb.build_multiplication_node(expanded_factors)  # build term node
        terms.append(term)  # add node to term list

    return terms


# Arithmetic Reduction Combination
# ------------------------------------------------------------------------------------------------------------------


def combine_arithmetic_reduction_nodes(
    problem: Problem,
    node: mat.ArithmeticExpressionNode,
    outer_unb_syms: Iterable[str] = None,
):

    if outer_unb_syms is None:
        outer_unb_syms = set()
    elif isinstance(outer_unb_syms, set):
        outer_unb_syms = set(outer_unb_syms)

    (
        ref_factors,
        cmpt_set_nodes,
        con_nodes,
    ) = __extract_idx_set_nodes_and_constraint_nodes(
        problem=problem, node=node, outer_unb_syms=outer_unb_syms
    )

    # build multiplication node with collected factors
    prod_node = nb.build_multiplication_node(ref_factors, is_prioritized=True)

    # no component indexing set nodes and no constraint nodes
    if len(cmpt_set_nodes) == 0 and len(con_nodes) == 0:
        return prod_node

    # one or more constraint nodes
    else:

        # build a conjunctive constraint node for the combined indexing set node
        constraint_node = None
        if len(con_nodes) > 0:
            constraint_node = nb.build_conjunction_node(con_nodes)

        # no component indexing set nodes
        if len(cmpt_set_nodes) == 0:
            return mat.ArithmeticConditionalNode(
                operands=[prod_node], conditions=[constraint_node], is_prioritized=True
            )

        # one or more component indexing set nodes
        else:

            # build the combined indexing set node
            idx_set_node = mat.CompoundSetNode(
                set_nodes=cmpt_set_nodes, constraint_node=constraint_node
            )

            # build the combined summation node
            return mat.ArithmeticTransformationNode(
                fcn=mat.SUMMATION_FUNCTION,
                idx_set_node=idx_set_node,
                operands=prod_node,
            )


def __extract_idx_set_nodes_and_constraint_nodes(
    problem: Problem, node: mat.ArithmeticExpressionNode, outer_unb_syms: Set[str]
) -> Tuple[
    List[mat.ArithmeticExpressionNode],
    List[mat.SetExpressionNode],
    List[mat.LogicalExpressionNode],
]:

    # multiplication
    if (
        isinstance(node, mat.ArithmeticOperationNode)
        and node.operator == mat.MULTIPLICATION_OPERATOR
    ):

        ref_factors = []
        cmpt_set_nodes = []
        con_nodes = []
        def_unb_syms = set()

        for i, factor in enumerate(node.operands):

            # reformulate factor
            (
                ref_inner_factors,
                inner_cmpt_set_nodes,
                inner_con_nodes,
            ) = __extract_idx_set_nodes_and_constraint_nodes(
                problem=problem,
                node=factor,
                outer_unb_syms=outer_unb_syms | def_unb_syms,
            )

            ref_factors.extend(ref_inner_factors)
            cmpt_set_nodes.extend(inner_cmpt_set_nodes)
            con_nodes.extend(inner_con_nodes)

            unb_syms = nb.retrieve_unbound_symbols(factor)
            unb_syms = (
                unb_syms - outer_unb_syms
            )  # remove unbound symbols defined in the outer scope
            clashing_unb_syms = (
                def_unb_syms & unb_syms
            )  # retrieve previously-defined unbound symbols
            def_unb_syms = (
                def_unb_syms | unb_syms
            )  # update set of defined unbound symbols

            if len(clashing_unb_syms) > 0:  # one or more unbound symbol conflicts

                # generate mapping of old conflicting symbols to new unique symbols
                mapping = {}
                for unb_sym in clashing_unb_syms:
                    mapping[unb_sym] = problem.generate_unique_symbol(unb_sym)

                # replace the conflicting symbols with unique symbols
                nb.replace_unbound_symbols(node=factor, mapping=mapping)

                # update set of defined unbound symbols with newly-generated symbols
                def_unb_syms = def_unb_syms | set(mapping.values())

        return ref_factors, cmpt_set_nodes, con_nodes

    # summation
    elif (
        isinstance(node, mat.ArithmeticTransformationNode)
        and node.fcn == mat.SUMMATION_FUNCTION
    ):

        middle_unb_syms = nb.retrieve_unbound_symbols(node.idx_set_node)

        # reformulate node
        (
            inner_nodes,
            cmpt_set_nodes,
            con_nodes,
        ) = __extract_idx_set_nodes_and_constraint_nodes(
            problem=problem,
            node=node.operands[0],
            outer_unb_syms=outer_unb_syms | middle_unb_syms,
        )

        cmpt_set_nodes = node.idx_set_node.set_nodes + cmpt_set_nodes

        if node.idx_set_node.constraint_node is not None:
            con_nodes.insert(0, node.idx_set_node.constraint_node)

        return inner_nodes, cmpt_set_nodes, con_nodes

    # uni-conditional
    elif isinstance(node, mat.ArithmeticConditionalNode) and len(node.operands) == 1:

        # reformulate node
        (
            ref_factors,
            cmpt_set_nodes,
            con_nodes,
        ) = __extract_idx_set_nodes_and_constraint_nodes(
            problem=problem, node=node.operands[0], outer_unb_syms=outer_unb_syms
        )

        if node.conditions[0] is not None:
            con_nodes.insert(0, node.conditions[0])

        return ref_factors, cmpt_set_nodes, con_nodes

    # other
    else:
        return [node], [], []


# Conditional Expression Reformulation
# ------------------------------------------------------------------------------------------------------------------


def reformulate_arithmetic_conditional_expressions(root_node: mat.ExpressionNode):

    if isinstance(root_node, mat.ArithmeticConditionalNode):
        root_node = __reformulate_n_conditional_expression(
            root_node.operands, root_node.conditions
        )

    queue = Queue()
    queue.put(root_node)

    while not queue.empty():

        node = queue.get()

        ref_children = []
        for child in node.get_children():

            if isinstance(child, mat.ArithmeticConditionalNode):
                if len(child.operands) > 1:
                    child = __reformulate_n_conditional_expression(
                        child.operands, child.conditions
                    )

            if isinstance(child, mat.RelationalOperationNode) or isinstance(
                child, mat.ArithmeticExpressionNode
            ):
                queue.put(child)

            ref_children.append(child)

        node.set_children(ref_children)

    return root_node


def __reformulate_n_conditional_expression(
    operands: List[mat.ArithmeticExpressionNode],
    conditions: List[mat.LogicalExpressionNode],
):

    sum_operands = []
    conj_node: Optional[mat.LogicalOperationNode] = None

    # assign operation priority to all condition nodes
    for condition in conditions:
        condition.is_prioritized = True

    for i, operand in enumerate(operands):

        if i == 0:
            mod_cond_node = conditions[i]

        else:

            if i == 1:

                prev_condition = deepcopy(conditions[i - 1])

                neg_prev_condition = mat.LogicalOperationNode(
                    operator=mat.UNARY_INVERSION_OPERATOR, operands=[prev_condition]
                )
                neg_prev_condition.is_prioritized = True

                conj_node = mat.LogicalOperationNode(
                    operator=mat.CONJUNCTION_OPERATOR, operands=[neg_prev_condition]
                )
                mod_cond_node = conj_node

            else:

                conj_node = deepcopy(conj_node)

                prev_condition = conj_node.operands[i - 1]

                neg_prev_condition = mat.LogicalOperationNode(
                    operator=mat.UNARY_INVERSION_OPERATOR, operands=[prev_condition]
                )
                neg_prev_condition.is_prioritized = True

                conj_node.operands[i - 1] = neg_prev_condition

                mod_cond_node = conj_node

            if i < len(conditions):
                conj_node.operands.append(conditions[i])

        operand.is_prioritized = True
        mono_cond_expr_node = mat.ArithmeticConditionalNode(
            operands=[operand], conditions=[mod_cond_node], is_prioritized=True
        )

        sum_operands.append(mono_cond_expr_node)

    if len(sum_operands) == 1:
        return sum_operands[0]

    else:
        return mat.AdditionNode(operands=sum_operands)
