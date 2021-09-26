from copy import deepcopy
from ordered_set import OrderedSet
from queue import Queue
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import symro.core.constants as const
import symro.core.mat as mat
from symro.core.prob.problem import Problem
from symro.core.handlers.nodebuilder import NodeBuilder
from symro.core.handlers.entitybuilder import EntityBuilder


class Formulator:

    def __init__(self, problem: Problem):
        self._problem: Optional[Problem] = problem
        self._node_builder: NodeBuilder = NodeBuilder(self._problem)

    # Model Standardization
    # ------------------------------------------------------------------------------------------------------------------

    def standardize_model(self) -> Dict[str, List[mat.MetaConstraint]]:

        # standardize objective functions
        for meta_obj in self._problem.model_meta_objs:
            self.__standardize_objective(meta_obj)

        # standardize constraints

        original_to_standard_con_map = {}  # map of original to standard meta-constraints
        std_meta_cons = []  # list of standardized meta-constraints

        for meta_con in self._problem.model_meta_cons:

            self._problem.meta_cons.pop(meta_con.get_symbol())  # remove original meta-constraint

            std_meta_con_list = self.__standardize_constraint(meta_con)
            original_to_standard_con_map[meta_con.get_symbol()] = std_meta_con_list

            std_meta_cons.extend(std_meta_con_list)

        # add standardized constraints to problem
        self._problem.model_meta_cons.clear()
        for std_meta_con in std_meta_cons:
            self._problem.add_meta_constraint(std_meta_con)

        # add standardized constraints to subproblems
        for sp in self._problem.subproblems.values():

            std_sp_meta_cons = []  # instantiate list of standardized meta-constraints for the subproblems

            for meta_con in sp.model_meta_cons:  # iterate over all meta-constraints in the subproblem

                # retrieve the standardized parent meta-constraint
                std_meta_cons_c = original_to_standard_con_map[meta_con.get_symbol()]

                if not meta_con.is_sub():  # original meta-constraint
                    # add the standardized parent meta-constraints to the list
                    std_sp_meta_cons.extend(std_meta_cons_c)

                else:  # meta-constraint subset
                    for i, std_meta_con in enumerate(std_meta_cons_c):

                        std_sub_meta_con = deepcopy(std_meta_con)  # build a sub-meta-constraint

                        # retrieve indexing subset and assign it to sub-meta-constraint
                        idx_subset_node = std_meta_con.idx_set_node if i == 0 else deepcopy(std_meta_con.idx_set_node)
                        std_sub_meta_con.idx_set_node = idx_subset_node

                        std_sp_meta_cons.append(std_sub_meta_con)

            sp.model_meta_cons = std_sp_meta_cons  # assign list of standardized meta-constraints to the subproblem

        return original_to_standard_con_map

    def __standardize_objective(self, meta_obj: mat.MetaObjective):
        if meta_obj.get_direction() == mat.MetaObjective.MAXIMIZE_DIRECTION:

            meta_obj.set_direction(mat.MetaObjective.MINIMIZE_DIRECTION)

            expression = meta_obj.get_expression()
            operand = expression.expression_node
            if not isinstance(operand, mat.ArithmeticExpressionNode):
                raise ValueError("Formulator expected an arithmetic expression node"
                                 " while reformulating an objective function")

            operand.is_prioritized = True
            neg_op = self._node_builder.append_negative_unity_coefficient(operand)

            expression.expression_node = neg_op
            expression.link_nodes()

    def __standardize_constraint(self, meta_con: mat.MetaConstraint) -> List[mat.MetaConstraint]:

        ctype = meta_con.elicit_constraint_type()  # elicit constraint type

        if self.__is_constraint_standardized(meta_con):
            return [meta_con]  # return the original meta-constraint if it is already in standard form

        else:
            if ctype == mat.MetaConstraint.EQUALITY_TYPE:
                ref_meta_cons = [self.__standardize_equality_constraint(meta_con)]
            elif ctype == mat.MetaConstraint.INEQUALITY_TYPE:
                ref_meta_cons = [self.__standardize_inequality_constraint(meta_con)]
            elif ctype == mat.MetaConstraint.DOUBLE_INEQUALITY_TYPE:
                ref_meta_cons = self.__standardize_double_inequality_constraint(meta_con)
            else:
                raise ValueError("Formulator unable to resolve the constraint type of '{0}'".format(meta_con))

        return ref_meta_cons

    @staticmethod
    def __is_constraint_standardized(meta_con: mat.MetaConstraint):

        # double inequality
        if meta_con.get_constraint_type() == mat.MetaConstraint.DOUBLE_INEQUALITY_TYPE:
            return False

        # single inequality or equality
        else:

            rel_node = meta_con.get_expression().expression_node
            if not isinstance(rel_node, mat.RelationalOperationNode):
                raise ValueError("Formulator expected a relational operation node"
                                 " while verifying whether the constraint '{0}' is in standard form".format(meta_con))

            if rel_node.operator == ">=":
                return False  # inequality is reversed

            rhs_node = rel_node.rhs_operand

            if not isinstance(rhs_node, mat.NumericNode):
                return False  # rhs operand is non-zero

            else:
                if rhs_node.value != 0:
                    return False  # rhs operand is non-zero
                else:
                    return True

    def __standardize_equality_constraint(self, meta_con: mat.MetaConstraint) -> mat.MetaConstraint:

        eq_op_node = meta_con.get_expression().expression_node
        if not isinstance(eq_op_node, mat.RelationalOperationNode):
            raise ValueError("Formulator encountered unexpected expression node"
                             + " while standardizing equality constraint '{0}'".format(meta_con))

        self.__move_relational_expression_operands_to_lhs(eq_op_node)

        meta_con.get_expression().link_nodes()

        return meta_con

    def __standardize_inequality_constraint(self, meta_con: mat.MetaConstraint) -> mat.MetaConstraint:

        ineq_op_node = meta_con.get_expression().expression_node
        if not isinstance(ineq_op_node, mat.RelationalOperationNode):
            raise ValueError("Formulator encountered unexpected expression node"
                             + " while standardizing inequality constraint '{0}'".format(meta_con))

        self.__move_relational_expression_operands_to_lhs(ineq_op_node)

        meta_con.get_expression().link_nodes()

        return meta_con

    def __standardize_double_inequality_constraint(self,
                                                   meta_con: mat.MetaConstraint) -> List[mat.MetaConstraint]:

        ref_meta_cons = []

        lb_operand, mid_operand, ub_operand = self.__extract_operands_from_double_inequality(meta_con)

        for i in range(2):

            if i == 0:
                lhs_operand = lb_operand
                rhs_operand = mid_operand

            else:
                lhs_operand = deepcopy(mid_operand)
                rhs_operand = ub_operand

            rhs_operand = self._node_builder.append_negative_unity_coefficient(rhs_operand)

            sub_node = self._node_builder.build_addition_node([lhs_operand, rhs_operand])

            ref_ineq_op_node = mat.RelationalOperationNode(id=self.generate_free_node_id(),
                                                           operator="<=",
                                                           lhs_operand=sub_node,
                                                           rhs_operand=mat.NumericNode(id=self.generate_free_node_id(),
                                                                                       value=0))

            mc_clone = deepcopy(meta_con)

            new_sym = self._problem.generate_unique_symbol("{0}_I{1}".format(meta_con.get_symbol(), i + 1))
            mc_clone.set_symbol(new_sym)

            expr_clone = mc_clone.get_expression()
            expr_clone.expression_node = ref_ineq_op_node
            expr_clone.link_nodes()

            mc_clone.elicit_constraint_type()

            ref_meta_cons.append(mc_clone)

        return ref_meta_cons

    @staticmethod
    def __extract_operands_from_double_inequality(meta_con: mat.MetaConstraint):

        ineq_op_node = meta_con.get_expression().expression_node

        if not isinstance(ineq_op_node, mat.RelationalOperationNode):
            raise ValueError("Formulator encountered unexpected expression node"
                             + " while standardizing double inequality constraint '{0}'".format(meta_con))

        # (L --- M) --- R
        if isinstance(ineq_op_node.lhs_operand, mat.RelationalOperationNode):

            child_ineq_op_node = ineq_op_node.lhs_operand

            # (L <= M) <= R
            if ineq_op_node.operator == "<=" and child_ineq_op_node.operator == "<=":
                lb_operand = child_ineq_op_node.lhs_operand  # L
                mid_operand = child_ineq_op_node.rhs_operand  # M
                ub_operand = ineq_op_node.rhs_operand  # R

            # (L >= M) >= R
            elif ineq_op_node.operator == ">=" and child_ineq_op_node.operator == ">=":
                lb_operand = ineq_op_node.rhs_operand  # R
                mid_operand = child_ineq_op_node.rhs_operand  # M
                ub_operand = child_ineq_op_node.lhs_operand  # L

            else:
                raise ValueError("Formulator encountered unexpected expression structure"
                                 + " while standardizing double inequality constraint '{0}'".format(meta_con))

        # L --- (M --- R)
        elif isinstance(ineq_op_node.rhs_operand, mat.RelationalOperationNode):

            child_ineq_op_node = ineq_op_node.rhs_operand

            # L <= (M <= R)
            if ineq_op_node.operator == "<=" and child_ineq_op_node.operator == "<=":
                lb_operand = ineq_op_node.lhs_operand  # L
                mid_operand = child_ineq_op_node.lhs_operand  # M
                ub_operand = child_ineq_op_node.rhs_operand  # R

            # L >= (M >= R)
            elif ineq_op_node.operator == ">=" and child_ineq_op_node.operator == ">=":
                lb_operand = child_ineq_op_node.rhs_operand  # R
                mid_operand = child_ineq_op_node.lhs_operand  # M
                ub_operand = ineq_op_node.lhs_operand  # L

            else:
                raise ValueError("Formulator encountered unexpected expression structure"
                                 + " while standardizing double inequality constraint '{0}'".format(meta_con))

        else:
            raise ValueError("Formulator encountered unexpected expression node"
                             + " while standardizing double inequality constraint '{0}'".format(meta_con))

        return lb_operand, mid_operand, ub_operand

    # Constraint Reformulation
    # ------------------------------------------------------------------------------------------------------------------

    def __move_relational_expression_operands_to_lhs(self, rel_op_node: mat.RelationalOperationNode):

        operator = rel_op_node.operator

        if operator in ('=', "==", "<="):
            lhs_operand = rel_op_node.lhs_operand
            rhs_operand = rel_op_node.rhs_operand
        else:  # >=
            rel_op_node.operator = "<="
            lhs_operand = rel_op_node.rhs_operand
            rhs_operand = rel_op_node.lhs_operand

        rhs_operand = self._node_builder.append_negative_unity_coefficient(rhs_operand)

        sub_node = self._node_builder.build_addition_node([lhs_operand, rhs_operand])

        rel_op_node.lhs_operand = sub_node
        rel_op_node.rhs_operand = mat.NumericNode(id=self.generate_free_node_id(), value=0)

    def convert_equality_to_inequality_constraint(self, meta_con: mat.MetaConstraint):

        ref_meta_cons = []
        old_sym = meta_con.get_symbol()

        for i in range(2):

            if i == 0:
                ref_meta_con = meta_con
            else:
                ref_meta_con = deepcopy(meta_con)

            new_sym = self._problem.generate_unique_symbol("{0}_E{1}".format(old_sym, i + 1))
            ref_meta_con.set_symbol(new_sym)

            eq_op_node = ref_meta_con.get_expression().expression_node
            if not isinstance(eq_op_node, mat.RelationalOperationNode):
                raise ValueError("Formulator encountered unexpected expression node"
                                 + " while converting equality constraint '{0}'".format(meta_con)
                                 + " into inequality constraints")

            eq_op_node.operator = "<="

            if i == 1:
                eq_op_node.lhs_operand = self._node_builder.append_negative_unity_coefficient(eq_op_node.lhs_operand)
                eq_op_node.rhs_operand = self._node_builder.append_negative_unity_coefficient(eq_op_node.rhs_operand)

            ref_meta_cons.append(ref_meta_con)

        self._problem.replace_model_meta_constraint(old_symbol=old_sym,
                                                    new_meta_cons=ref_meta_cons)

        return ref_meta_cons

    # Slack Variables
    # ------------------------------------------------------------------------------------------------------------------

    def formulate_slackened_constraint(self,
                                       meta_con: mat.MetaConstraint) -> Tuple[List[mat.MetaVariable],
                                                                              mat.MetaConstraint]:

        ctype = meta_con.elicit_constraint_type()

        # generate slack variables for an equality constraint
        if ctype == mat.MetaConstraint.EQUALITY_TYPE:
            """
            pos_sl_meta_var, pos_sl_slack_var_node = self.__generate_slack_var(meta_con, symbol_suffix="P")
            neg_sl_meta_var, neg_sl_slack_var_node = self.__generate_slack_var(meta_con, symbol_suffix="N")
            sl_meta_vars = [pos_sl_meta_var, neg_sl_meta_var]

            slack_node = mat.BinaryArithmeticOperationNode(id=self.generate_free_node_id(),
                                                           operator='-',
                                                           lhs_operand=pos_sl_slack_var_node,
                                                           rhs_operand=neg_sl_slack_var_node)
            """
            return [], meta_con

        # generate slack variable for an inequality constraint
        elif ctype == mat.MetaConstraint.INEQUALITY_TYPE:

            sl_meta_var, sl_slack_var_node = self.__generate_slack_var(meta_con)
            sl_meta_vars = [sl_meta_var]

            slack_node = sl_slack_var_node

        else:
            raise ValueError("Formulator encountered an unexpected constraint type"
                             + " while building a slackened constraint for '{0}'".format(meta_con))

        sl_meta_con = deepcopy(meta_con)
        expr_clone = deepcopy(meta_con.get_expression())

        con_sym = self._problem.generate_unique_symbol("{0}_F".format(meta_con.get_symbol()))
        sl_meta_con.set_symbol(con_sym)

        rel_op_node = expr_clone.expression_node
        if not isinstance(rel_op_node, mat.RelationalOperationNode):
            raise ValueError("Formulator encountered unexpected expression node"
                             + " while building a slackened constraint for '{0}'".format(meta_con))

        rhs_node = self._node_builder.append_negative_unity_coefficient(slack_node)
        rel_op_node.lhs_operand = mat.MultiArithmeticOperationNode(id=self.generate_free_node_id(),
                                                                   operator='+',
                                                                   operands=[rel_op_node.lhs_operand, rhs_node])

        expr_clone.link_nodes()
        sl_meta_con.set_expression(expr_clone)

        return sl_meta_vars, sl_meta_con

    def __generate_slack_var(self,
                             meta_con: mat.MetaConstraint,
                             symbol_suffix: str = ""):

        sym = self._problem.generate_unique_symbol("{0}_SL{1}".format(meta_con.get_symbol(), symbol_suffix))
        sl_meta_var = mat.MetaVariable(symbol=sym,
                                       idx_meta_sets=deepcopy(meta_con.get_idx_meta_sets()),
                                       idx_set_node=meta_con.idx_set_node,
                                       default_value=mat.NumericNode(id=self.generate_free_node_id(), value=0),
                                       lower_bound=mat.NumericNode(id=self.generate_free_node_id(), value=0))

        entity_index_node = self._node_builder.build_default_entity_index_node(sl_meta_var)
        sl_var_node = mat.DeclaredEntityNode(id=self.generate_free_node_id(),
                                             symbol=sym,
                                             entity_index_node=entity_index_node,
                                             type=const.VAR_TYPE)

        return sl_meta_var, sl_var_node

    def formulate_slack_min_objective(self,
                                      idx_meta_sets: Union[List[mat.MetaSet], Dict[str, mat.MetaSet]],
                                      sl_meta_vars: Union[List[mat.MetaVariable], Dict[str, mat.MetaVariable]],
                                      obj_sym: str) -> mat.MetaObjective:

        operands = []
        entity_builder = EntityBuilder(self._problem)

        if isinstance(sl_meta_vars, dict):
            sl_meta_vars = list(sl_meta_vars.values())

        for sl_meta_var in sl_meta_vars:

            entity_index_node = self._node_builder.build_default_entity_index_node(sl_meta_var)
            slack_node = mat.DeclaredEntityNode(id=self.generate_free_node_id(),
                                                symbol=sl_meta_var.get_symbol(),
                                                entity_index_node=entity_index_node)

            if sl_meta_var.get_idx_set_reduced_dim() == 0:
                operand = slack_node
            else:
                idx_set_node = self._node_builder.build_entity_idx_set_node(sl_meta_var,
                                                                            remove_sets=idx_meta_sets)
                if idx_set_node is None:
                    operand = slack_node
                else:
                    operand = mat.ArithmeticTransformationNode(id=self.generate_free_node_id(),
                                                               symbol="sum",
                                                               idx_set_node=idx_set_node,
                                                               operands=slack_node)

            operands.append(operand)

        if len(operands) > 1:
            expression_node = mat.MultiArithmeticOperationNode(id=self.generate_free_node_id(),
                                                               operator='+',
                                                               operands=operands)
        elif len(operands) == 1:
            expression_node = operands[0]

        else:
            expression_node = mat.NumericNode(id=self.generate_free_node_id(), value=0)

        expression = mat.Expression(expression_node)

        meta_obj = entity_builder.build_meta_obj(symbol=obj_sym,
                                                 idx_meta_sets=idx_meta_sets,
                                                 direction="minimize",
                                                 expression=expression)

        return meta_obj

    # Substitution
    # ------------------------------------------------------------------------------------------------------------------

    def substitute(self,
                   root_node: mat.ExpressionNode,
                   sub_map: Dict[str, Tuple[mat.ExpressionNode, Iterable[str]]]) -> mat.ExpressionNode:
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
                            dummy_map = {unb_sym: cmpt_node for unb_sym, cmpt_node in
                                         zip(unb_syms, child.idx_node.component_nodes)}
                            sub_node = self._node_builder.replace_dummy_nodes(sub_node,
                                                                              mapping=dummy_map)

                        modified_children.append(sub_node)

                    else:
                        modified_children.append(child)
                        queue.put(child)

                else:
                    modified_children.append(child)
                    queue.put(child)

            node.set_children(modified_children)

        return root_node

    def substitute_defined_variables(self):

        # identify all defined variables

        sub_map = {}  # map of defined variable symbols to their defined values
        for mv in self._problem.model_meta_vars:
            if mv.is_defined():
                sub_map[mv.get_symbol()] = (mv.get_defined_value_node(), mv.get_idx_set_dummy_element())

        if len(sub_map) > 0:

            # modify meta-objective expressions
            for mo in self._problem.model_meta_objs:
                mo.get_expression().expression_node = self.substitute(root_node=mo.get_expression().expression_node,
                                                                      sub_map=sub_map)

            # modify meta-constraint expressions
            for mc in self._problem.model_meta_cons:
                mc.get_expression().expression_node = self.substitute(root_node=mc.get_expression().expression_node,
                                                                      sub_map=sub_map)

    # Subtraction and Arithmetic Negation
    # ------------------------------------------------------------------------------------------------------------------

    def reformulate_subtraction_and_unary_negation(self, root_node: mat.ExpressionNode):

        root_node = self.__reformulate_subtraction_or_unary_negation_node(root_node)

        queue = Queue()
        queue.put(root_node)

        while not queue.empty():

            node = queue.get()

            ref_children = []
            for child in node.get_children():

                if isinstance(child, mat.RelationalOperationNode) or isinstance(child, mat.ArithmeticExpressionNode):
                    child = self.__reformulate_subtraction_or_unary_negation_node(child)
                    queue.put(child)

                ref_children.append(child)

            node.set_children(ref_children)

        return root_node

    def __reformulate_subtraction_or_unary_negation_node(self, node: mat.ExpressionNode):

        if isinstance(node, mat.BinaryArithmeticOperationNode) and node.operator == '-':
            rhs_operand = self._node_builder.append_negative_unity_coefficient(node.rhs_operand)
            node = mat.AdditionNode(id=self.generate_free_node_id(),
                                    operands=[node.lhs_operand, rhs_operand])

        elif isinstance(node, mat.UnaryArithmeticOperationNode) and node.operator == '-':
            node = self._node_builder.append_negative_unity_coefficient(node.operand)

        return node

    # Exponentiation
    # ------------------------------------------------------------------------------------------------------------------

    def __factorize_exponentiation(self,
                                   exp_op_node: mat.ExponentiationNode,
                                   exponents: Iterable[int],
                                   idx_set: mat.IndexingSet = None,
                                   dummy_element: mat.Element = None):

        factors = []

        if isinstance(exp_op_node.rhs_operand, mat.NumericNode):
            is_exp_const = True
            exp_val = exp_op_node.rhs_operand.value

        else:
            is_exp_const, exp_vals = self.__evaluate_exponent_node(
                exponent_node=exp_op_node.rhs_operand,
                idx_set=idx_set,
                dummy_element=dummy_element)

            if is_exp_const:
                exp_val = exp_vals[0]
            else:
                exp_val = None

        if is_exp_const:

            if exp_val == 0:
                factors.append(mat.NumericNode(id=self.generate_free_node_id(), value=1))

            elif exp_val == 1:
                factors.append(exp_op_node.lhs_operand)

            else:

                # TODO: account for negative exponents by moving the base node to the denominator of a division node

                if exp_val in exponents:

                    exp_op_node.lhs_operand.is_prioritized = True
                    factors = [exp_op_node.lhs_operand]

                    for i in range(2, exp_val + 1):
                        factors.append(deepcopy(exp_op_node.lhs_operand))

                else:
                    factors.append(exp_op_node)

        else:
            factors.append(exp_op_node)

        return factors

    def __distribute_exponent(self,
                              base_node: mat.ArithmeticExpressionNode,
                              exponent_node: mat.ArithmeticExpressionNode):

        if isinstance(base_node, mat.MultiplicationNode):

            factors = []
            for factor in base_node.operands:
                factor.is_prioritized = True
                factors.append(mat.ExponentiationNode(id=self.generate_free_node_id(),
                                                      lhs_operand=factor,
                                                      rhs_operand=deepcopy(exponent_node),
                                                      is_prioritized=True))

        else:
            exp_op_node = mat.ExponentiationNode(id=self.generate_free_node_id(),
                                                 lhs_operand=base_node,
                                                 rhs_operand=exponent_node,
                                                 is_prioritized=True)
            factors = [exp_op_node]

        return factors

    def __evaluate_exponent_node(self,
                                 exponent_node: mat.ArithmeticExpressionNode,
                                 idx_set: mat.IndexingSet,
                                 dummy_element: mat.Element):

        exponent_var_nodes = mat.get_var_nodes(exponent_node)

        is_exponent_const = True
        exponent_values = None

        if len(exponent_var_nodes) == 0:

            exponent_values = exponent_node.evaluate(state=self._problem.state,
                                                     idx_set=idx_set,
                                                     dummy_element=dummy_element)

            for i in range(1, len(exponent_values)):
                if exponent_values[0] != exponent_values[i]:
                    is_exponent_const = False
                    break

        return is_exponent_const, exponent_values

    # Expansion
    # ------------------------------------------------------------------------------------------------------------------

    def expand_multiplication(self,
                              node: mat.ArithmeticExpressionNode,
                              idx_set: mat.IndexingSet,
                              dummy_element: mat.Element) -> List[mat.ArithmeticExpressionNode]:
        terms = self.__expand_multiplication(node=node,
                                             idx_set=idx_set,
                                             dummy_element=dummy_element)
        for term in terms:
            term.is_prioritized = True
        return terms

    def __expand_multiplication(self,
                                node: mat.ArithmeticExpressionNode,
                                idx_set: mat.IndexingSet,
                                dummy_element: mat.Element) -> List[mat.ArithmeticExpressionNode]:

        # arithmetic operation
        if isinstance(node, mat.ArithmeticExpressionNode):

            # constant or declared entity
            if isinstance(node, mat.NumericNode) or isinstance(node, mat.DeclaredEntityNode):
                return [node]

            # transformation
            elif isinstance(node, mat.ArithmeticTransformationNode):

                # reductive summation
                if node.is_reductive() and node.symbol == "sum":

                    inner_idx_sets = node.idx_set_node.evaluate(self._problem.state, idx_set, dummy_element)
                    inner_idx_set = OrderedSet().union(*inner_idx_sets)
                    idx_set = mat.cartesian_product([idx_set, inner_idx_set])
                    dummy_element = node.idx_set_node.combined_dummy_element

                    terms = self.__expand_multiplication(node.operands[0], idx_set, dummy_element)

                    # single term
                    if len(terms) == 1:  # return the summation node
                        node.operands[0] = terms[0]
                        return [node]

                    # multiple terms
                    else:  # distribute the summation to each term

                        # retrieve the unbound symbols of the indexing set node of the summation node
                        unb_syms = node.idx_set_node.get_defined_unbound_symbols()

                        node.operands.clear()  # clear the list of operands of the summation node

                        dist_terms = []  # list of terms onto which the summation node was distributed

                        for term in terms:

                            # retrieve the set of unbound symbols present in the term
                            term_unb_syms = self._node_builder.retrieve_unbound_symbols(root_node=term,
                                                                                        in_filter=unb_syms)

                            # if any defined unbound symbols are present in the term, then the term is controlled by
                            # the indexing set of the summation node
                            if len(term_unb_syms) > 0:  # the term is controlled by the indexing set node
                                sum_node = deepcopy(node)
                                sum_node.operands.append(term)
                                dist_terms.append(sum_node)

                            else:  # the term is not controlled by the indexing set node
                                dist_terms.append(term)  # append the term without a summation transformation

                        return dist_terms

                # other
                else:

                    for i, operand in enumerate(node.operands):
                        terms = self.__expand_multiplication(operand, idx_set, dummy_element)
                        node.operands[i] = self._node_builder.build_addition_node(terms)

                    return [node]

            # unary operation
            elif isinstance(node, mat.UnaryArithmeticOperationNode):
                if node.operator == '-':
                    node = self.reformulate_subtraction_and_unary_negation(node)
                return self.__expand_multiplication(node, idx_set, dummy_element)

            # binary operation
            elif isinstance(node, mat.BinaryArithmeticOperationNode):

                lhs_terms = self.__expand_multiplication(node.lhs_operand, idx_set, dummy_element)
                rhs_terms = self.__expand_multiplication(node.rhs_operand, idx_set, dummy_element)

                # addition
                if node.operator == '+':
                    return lhs_terms + rhs_terms

                # subtraction
                elif node.operator == '-':
                    rhs_terms = [self._node_builder.append_negative_unity_coefficient(t) for t in rhs_terms]
                    return lhs_terms + rhs_terms

                # multiplication
                elif node.operator == '*':
                    return self.expand_factors(lhs_terms, rhs_terms)

                # division
                elif node.operator == '/':

                    if len(lhs_terms) == 1:

                        numerator = lhs_terms[0]

                        # special case: numerator is 0
                        if isinstance(numerator, mat.NumericNode) and numerator.value == 0:
                            return [numerator]

                        # special case: numerator is 1
                        if isinstance(numerator, mat.NumericNode) and numerator.value == 1:
                            node.rhs_operand = self._node_builder.build_addition_node(rhs_terms)
                            return [node]

                    node.lhs_operand = mat.NumericNode(id=self.generate_free_node_id(), value=1)
                    node.rhs_operand = rhs_terms
                    rhs_terms = [node]

                    return self.expand_factors(lhs_terms, rhs_terms)

                # exponentiation
                else:

                    exponent_node = self._node_builder.build_addition_node(rhs_terms)
                    if len(rhs_terms) > 1:
                        exponent_node.is_prioritized = True

                    if len(lhs_terms) == 1:
                        # distribute exponent to each LHS factor
                        exp_op_factor_nodes = self.__distribute_exponent(lhs_terms[0], exponent_node)
                    else:
                        exp_op_factor_nodes = [node]

                    expanded_factors = []

                    for i, factor in enumerate(exp_op_factor_nodes):
                        expanded_factors.extend(self.__factorize_exponentiation(
                            exp_op_node=factor,
                            exponents=(2, 3),
                            idx_set=idx_set,
                            dummy_element=dummy_element))

                    term_lists = []

                    for expanded_factor in expanded_factors:
                        if isinstance(expanded_factor, mat.AdditionNode):
                            term_lists.append(self.__expand_multiplication(expanded_factor, idx_set, dummy_element))
                        else:
                            term_lists.append([expanded_factor])

                    return self.expand_factors_n(term_lists)

            # multi-operand operation
            elif isinstance(node, mat.MultiArithmeticOperationNode):

                # addition
                if node.operator == '+':
                    terms = []
                    for operand in node.operands:
                        terms.extend(self.__expand_multiplication(operand, idx_set, dummy_element))
                    return terms

                # multiplication
                elif node.operator == '*':
                    term_lists = [self.__expand_multiplication(o, idx_set, dummy_element) for o in node.operands]
                    return self.expand_factors_n(term_lists)

            # conditional operation
            elif isinstance(node, mat.ConditionalArithmeticExpressionNode):
                for i, operand in enumerate(node.operands):
                    terms = self.__expand_multiplication(operand, idx_set, dummy_element)
                    node.operands[i] = self._node_builder.build_addition_node(terms)
                return [node]

        else:
            raise ValueError("Formulator encountered an unexpected node while expanding factors")

    def expand_factors(self,
                       lhs_terms: List[mat.ArithmeticExpressionNode],
                       rhs_terms: List[mat.ArithmeticExpressionNode]):
        terms = []

        for lhs_term in lhs_terms:
            for rhs_term in rhs_terms:

                factors = [deepcopy(lhs_term), deepcopy(rhs_term)]
                term = self._node_builder.build_multiplication_node(factors)
                terms.append(term)

        return terms

    def expand_factors_n(self, factors: List[List[mat.ArithmeticExpressionNode]]):

        terms = []

        factor_count = len(factors)
        terms_per_factor_count = [len(f) for f in factors]  # length factor_count
        expanded_term_count = np.prod(terms_per_factor_count)  # total number of terms after expansion

        term_indices = np.zeros(shape=(factor_count,))  # length factor_count

        for _ in range(expanded_term_count):

            # retrieve a term from each supplied factor according to the current indices
            factor_terms = [factors[i][j] for i, j in enumerate(term_indices)]  # unique combination of terms

            # increment indices
            for i in range(factor_count, -1, -1):
                term_indices[i] += 1  # increment term index of current factor
                if term_indices[i] == terms_per_factor_count[i]:  # processed last term in factor
                    term_indices[i] = 0  # reset term index to 0 and proceed to previous factor index
                else:  # otherwise, break loop
                    break

            term = self._node_builder.build_multiplication_node(factor_terms)  # build term node
            terms.append(term)  # add node to term list

        return terms

    # Summation Combination
    # ------------------------------------------------------------------------------------------------------------------

    def combine_summation_factor_nodes(self,
                                       factors: Iterable[mat.ArithmeticExpressionNode],
                                       outer_unb_syms: Iterable[str]):

        if outer_unb_syms is None:
            outer_unb_syms = set()
        elif isinstance(outer_unb_syms, set):
            outer_unb_syms = set(outer_unb_syms)

        cmpt_set_nodes = []  # component set nodes of the combined indexing set node
        conj_operands = []  # conjunctive operands of the constraint node of the combined indexing set node
        ref_factors = []  # factors subject to the combined summation node

        def_unb_syms = set()  # previously defined unbound symbols

        for factor in factors:

            unb_syms = self._node_builder.retrieve_unbound_symbols(factor)
            unb_syms = unb_syms - outer_unb_syms  # remove unbound symbols defined in the outer scope
            clashing_unb_syms = def_unb_syms & unb_syms  # retrieve previously-defined unbound symbols
            def_unb_syms = def_unb_syms | unb_syms  # update set of defined unbound symbols

            if len(clashing_unb_syms) > 0:  # one or more unbound symbol conflicts

                # generate mapping of old conflicting symbols to new unique symbols
                mapping = {}
                for unb_sym in clashing_unb_syms:
                    mapping[unb_sym] = self._problem.generate_unique_symbol(unb_sym)

                # replace the conflicting symbols with unique symbols
                self._node_builder.replace_unbound_symbols(node=factor, mapping=mapping)

            # reductive summation node
            if isinstance(factor, mat.ArithmeticTransformationNode) and factor.symbol == "sum":

                # collect set nodes of the current indexing set node
                cmpt_set_nodes.extend(factor.idx_set_node.set_nodes)

                # collect constraint node of the current indexing set node
                if factor.idx_set_node.constraint_node is not None:
                    conj_operands.append(factor.idx_set_node.constraint_node)

                # designate the operand of the current summation node as the factorable node
                factorable_node = factor.operands[0]

            # other
            else:
                # designate the current node as the factorable node
                factorable_node = factor

            # retrieve underlying factors of the factorable node

            # multiple factors
            if isinstance(factorable_node, mat.MultiArithmeticOperationNode) and factorable_node.operator == '*':
                ref_factors.extend(factorable_node.operands)

            # single factor
            else:
                ref_factors.append(factorable_node)

        # build multiplication node with collected factors
        multiplication_node = self._node_builder.build_multiplication_node(ref_factors)

        # indexing set of the combined summation node is empty
        if len(cmpt_set_nodes) == 0:
            return multiplication_node

        # indexing set of the combined summation node is not empty
        else:

            # build a conjunctive constraint node for the combined indexing set node
            constraint_node = None
            if len(conj_operands) > 0:
                constraint_node = self._node_builder.build_conjunction_node(conj_operands)

            # build the combined indexing set node
            idx_set_node = mat.CompoundSetNode(id=self._problem.generate_free_node_id(),
                                               set_nodes=cmpt_set_nodes,
                                               constraint_node=constraint_node)

            # build the combined summation node
            return mat.ArithmeticTransformationNode(id=self._problem.generate_free_node_id(),
                                                    symbol="sum",
                                                    idx_set_node=idx_set_node,
                                                    operands=multiplication_node)

    # Utility
    # ------------------------------------------------------------------------------------------------------------------

    def get_node_builder(self) -> NodeBuilder:
        return self._node_builder

    def generate_free_node_id(self):
        if self._problem is not None:
            return self._problem.generate_free_node_id()
        else:
            return self._node_builder.generate_free_node_id()

    def seed_free_node_id(self, node: mat.ExpressionNode) -> int:
        return self._node_builder.seed_free_node_id(node)
