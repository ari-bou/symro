import warnings
from copy import deepcopy
from queue import Queue
from typing import Dict, List, Optional, Tuple, Union

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

            self._problem.meta_cons.pop(meta_con.symbol)  # remove original meta-constraint

            std_meta_con_list = self.__standardize_constraint(meta_con)
            original_to_standard_con_map[meta_con.symbol] = std_meta_con_list

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
                std_meta_cons_c = original_to_standard_con_map[meta_con.symbol]

                if not meta_con.is_sub:  # original meta-constraint
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
        if meta_obj.direction == mat.MetaObjective.MAXIMIZE_DIRECTION:

            meta_obj.direction = mat.MetaObjective.MINIMIZE_DIRECTION

            expression = meta_obj.expression
            operand = expression.expression_node
            if not isinstance(operand, mat.ArithmeticExpressionNode):
                raise ValueError("Model formulator expected an arithmetic expression node"
                                 " while reformulating an objective function")

            operand.is_prioritized = True
            unary_op = mat.UnaryArithmeticOperationNode(id=self.generate_free_node_id(),
                                                        operator='-',
                                                        operand=operand)
            expression.expression_node = unary_op
            expression.link_nodes()

    def __standardize_constraint(self, meta_con: mat.MetaConstraint):

        ctype = meta_con.elicit_constraint_type()  # elicit constraint type

        if self.__is_constraint_standardized(meta_con):
            return [meta_con]  # return the original meta-constraint if it is already in standard form

        else:
            if ctype == mat.MetaConstraint.EQUALITY_TYPE:
                ref_meta_cons = self.__standardize_equality_constraint(meta_con)
            elif ctype == mat.MetaConstraint.INEQUALITY_TYPE:
                ref_meta_cons = self.__standardize_inequality_constraint(meta_con)
            elif ctype == mat.MetaConstraint.DOUBLE_INEQUALITY_TYPE:
                ref_meta_cons = self.__standardize_double_inequality_constraint(meta_con)
            else:
                raise ValueError("Formulator unable to resolve the constraint type of '{0}'".format(meta_con))

        return ref_meta_cons

    @staticmethod
    def __is_constraint_standardized(meta_con: mat.MetaConstraint):

        # double inequality
        if meta_con.ctype == mat.MetaConstraint.DOUBLE_INEQUALITY_TYPE:
            return False

        # single inequality or equality
        else:

            rel_node = meta_con.expression.expression_node
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

    def __standardize_equality_constraint(self, meta_con: mat.MetaConstraint) -> List[mat.MetaConstraint]:

        eq_op_node = meta_con.expression.expression_node
        if not isinstance(eq_op_node, mat.RelationalOperationNode):
            raise ValueError("Encountered unexpected expression node.")

        bin_ari_op_node = mat.BinaryArithmeticOperationNode(id=self.generate_free_node_id(),
                                                            operator='-')
        bin_ari_op_node.lhs_operand = eq_op_node.lhs_operand
        bin_ari_op_node.rhs_operand = eq_op_node.rhs_operand
        bin_ari_op_node.rhs_operand.is_prioritized = True

        eq_op_node.lhs_operand = bin_ari_op_node
        eq_op_node.rhs_operand = mat.NumericNode(id=self.generate_free_node_id(), value=0)

        meta_con.expression.link_nodes()

        return [meta_con]

    def __standardize_inequality_constraint(self, meta_con: mat.MetaConstraint) -> List[mat.MetaConstraint]:

        ineq_op_node = meta_con.expression.expression_node
        if not isinstance(ineq_op_node, mat.RelationalOperationNode):
            raise ValueError("Encountered unexpected expression node.")

        operator = ineq_op_node.operator

        bin_ari_op_node = mat.BinaryArithmeticOperationNode(id=self.generate_free_node_id(), operator='-')
        if operator == "<=":
            bin_ari_op_node.lhs_operand = ineq_op_node.lhs_operand
            bin_ari_op_node.rhs_operand = ineq_op_node.rhs_operand
        else:
            bin_ari_op_node.lhs_operand = ineq_op_node.rhs_operand
            bin_ari_op_node.rhs_operand = ineq_op_node.lhs_operand
        bin_ari_op_node.rhs_operand.is_prioritized = True

        ineq_op_node.lhs_operand = bin_ari_op_node
        ineq_op_node.rhs_operand = mat.NumericNode(id=self.generate_free_node_id(), value='0')
        ineq_op_node.operator = "<="

        meta_con.expression.link_nodes()

        return [meta_con]

    def __standardize_double_inequality_constraint(self, meta_con: mat.MetaConstraint) -> List[mat.MetaConstraint]:

        ref_meta_cons = []

        for i in range(2):

            mc_clone = deepcopy(meta_con)
            expr_clone = deepcopy(meta_con.expression)

            mc_clone.symbol = "{0}_I{1}".format(meta_con.symbol, i + 1)

            ineq_op_node = expr_clone.expression_node
            self.seed_free_node_id(ineq_op_node)

            if not isinstance(ineq_op_node, mat.RelationalOperationNode):
                raise ValueError("Encountered unexpected expression node.")

            child_ineq_op_node = ineq_op_node.rhs_operand
            if not isinstance(child_ineq_op_node, mat.RelationalOperationNode):
                raise ValueError("Encountered unexpected expression node.")

            ref_ineq_op_node = mat.RelationalOperationNode(id=ineq_op_node.id, operator="<=")

            bin_ari_op_node = mat.BinaryArithmeticOperationNode(id=self.generate_free_node_id(),
                                                                operator='-')
            ref_ineq_op_node.add_operand(bin_ari_op_node)
            ref_ineq_op_node.add_operand(mat.NumericNode(id=self.generate_free_node_id(), value='0'))

            if i == 0:
                body_node = child_ineq_op_node.lhs_operand
                if ineq_op_node.operator == "<=":
                    lb_node = ineq_op_node.lhs_operand
                else:
                    lb_node = child_ineq_op_node.rhs_operand
                bin_ari_op_node.lhs_operand = lb_node
                bin_ari_op_node.rhs_operand = body_node
            else:
                body_node = child_ineq_op_node.lhs_operand
                if ineq_op_node.operator == "<=":
                    ub_node = child_ineq_op_node.rhs_operand
                else:
                    ub_node = ineq_op_node.lhs_operand
                bin_ari_op_node.lhs_operand = body_node
                bin_ari_op_node.rhs_operand = ub_node

            bin_ari_op_node.rhs_operand.is_prioritized = True

            expr_clone.expression_node = ref_ineq_op_node
            expr_clone.link_nodes()

            mc_clone.expression = expr_clone
            mc_clone.elicit_constraint_type()

            ref_meta_cons.append(mc_clone)

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
                             + " while building a slackened constraint")

        sl_meta_con = deepcopy(meta_con)
        expr_clone = deepcopy(meta_con.expression)

        con_sym = "{0}_F".format(meta_con.symbol)
        sl_meta_con.symbol = con_sym

        rel_op_node = expr_clone.expression_node
        if not isinstance(rel_op_node, mat.RelationalOperationNode):
            raise ValueError("Formulator encountered unexpected expression node.")

        rel_op_node.lhs_operand = mat.BinaryArithmeticOperationNode(id=self.generate_free_node_id(),
                                                                    operator='-',
                                                                    lhs_operand=rel_op_node.lhs_operand,
                                                                    rhs_operand=slack_node)

        expr_clone.link_nodes()
        sl_meta_con.expression = expr_clone

        return sl_meta_vars, sl_meta_con

    def __generate_slack_var(self,
                             meta_con: mat.MetaConstraint,
                             symbol_suffix: str = ""):

        sym = "{0}_SL{1}".format(meta_con.symbol, symbol_suffix)
        sl_meta_var = mat.MetaVariable(symbol=sym,
                                       idx_meta_sets=deepcopy(meta_con.idx_meta_sets),
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
                                                symbol=sl_meta_var.symbol,
                                                entity_index_node=entity_index_node)

            if sl_meta_var.get_reduced_dimension() == 0:
                operand = slack_node
            else:
                idx_set_node = self._node_builder.build_entity_idx_set_node(sl_meta_var,
                                                                            remove_sets=idx_meta_sets)
                if idx_set_node is None:
                    operand = slack_node
                else:
                    operand = mat.FunctionNode(id=self.generate_free_node_id(),
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

    @staticmethod
    def substitute(root_node: mat.ExpressionNode,
                   sub_map: Dict[str, mat.ExpressionNode]):
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

        if isinstance(root_node, mat.DeclaredEntityNode):
            if root_node.symbol in sub_map:
                return deepcopy(sub_map[root_node.symbol])
            else:
                return root_node

        queue = Queue()
        queue.put(root_node)

        while not queue.empty():

            node = queue.get()

            modified_children = []
            for child in node.get_children():

                if isinstance(child, mat.DeclaredEntityNode):
                    if child.symbol in sub_map:
                        modified_children.append(deepcopy(sub_map[child.symbol]))
                    else:
                        modified_children.append(child)
                        queue.put(child)

                else:
                    modified_children.append(child)
                    queue.put(child)

        return root_node

    def substitute_defined_variables(self):

        # identify all defined variables

        sub_map = {}  # map of defined variable symbols to their defined values
        for mv in self._problem.model_meta_vars:
            if mv.is_defined():
                sub_map[mv.symbol] = mv.defined_value

        if len(sub_map) > 0:

            # modify meta-objective expressions
            for mo in self._problem.model_meta_objs:
                mo.expression.expression_node = self.substitute(root_node=mo.expression.expression_node,
                                                                sub_map=sub_map)

            # modify meta-constraint expressions
            for mc in self._problem.model_meta_cons:
                mc.expression.expression_node = self.substitute(root_node=mc.expression.expression_node,
                                                                sub_map=sub_map)

    # Miscellaneous Reformulation
    # ------------------------------------------------------------------------------------------------------------------

    def reformulate_subtraction_ops(self,
                                    expr: Union[mat.Expression, mat.ExpressionNode]):
        if isinstance(expr, mat.Expression):
            node = expr.expression_node
            expr.expression_node = self.__reformulate_subtraction_ops(node)
            expr.link_nodes()
            return expr
        else:
            return self.__reformulate_subtraction_ops(expr)

    def __reformulate_subtraction_ops(self,
                                      node: mat.ExpressionNode
                                      ) -> Union[mat.ArithmeticExpressionNode,
                                                 mat.RelationalOperationNode]:

        if isinstance(node, mat.RelationalOperationNode):
            node.lhs_operand = self.__reformulate_subtraction_ops(node.lhs_operand)
            node.rhs_operand = self.__reformulate_subtraction_ops(node.rhs_operand)
            return node

        elif isinstance(node, mat.ArithmeticExpressionNode):

            if isinstance(node, mat.NumericNode) or isinstance(node, mat.DeclaredEntityNode):
                return node

            elif isinstance(node, mat.FunctionNode):
                for i, operand in enumerate(node.operands):
                    node.operands[i] = self.__reformulate_subtraction_ops(operand)

            elif isinstance(node, mat.UnaryArithmeticOperationNode):
                return self.__reformulate_subtraction_ops(node.operand)

            elif isinstance(node, mat.BinaryArithmeticOperationNode):

                node.lhs_operand = self.__reformulate_subtraction_ops(node.lhs_operand)
                node.rhs_operand = self.__reformulate_subtraction_ops(node.rhs_operand)

                if node.operator == '-':
                    node.operator = '+'
                    node.rhs_operand = self._node_builder.add_negative_unity_coefficient(node.rhs_operand)
                elif node.operator == "less":
                    warnings.warn("Problem Formulator unable to reformulate a 'less' operation node")

                return node

            elif isinstance(node, mat.MultiArithmeticOperationNode):

                for i, operand in enumerate(node.operands):
                    node.operands[i] = self.__reformulate_subtraction_ops(operand)

                if node.operator == '-':
                    node.operator = '+'
                    for i in range(1, len(node.operands)):
                        operand = node.operands[i]
                        node.operands[i] = self._node_builder.add_negative_unity_coefficient(operand)
                elif node.operator == "less":
                    warnings.warn("Problem Formulator unable to reformulate a 'less' operation node")

                return node

            elif isinstance(node, mat.ConditionalArithmeticExpressionNode):
                for i, operand in enumerate(node.operands):
                    node.operands[i] = self.__reformulate_subtraction_ops(operand)
                return node

        else:
            raise ValueError("Problem Formulator encountered an unexpected node"
                             + " while reformulating subtraction operations")

    def reformulate_unary_ops(self,
                              expr: Union[mat.Expression, mat.ExpressionNode]):
        if isinstance(expr, mat.Expression):
            node = expr.expression_node
            expr.expression_node = self.__reformulate_unary_ops(node)
            expr.link_nodes()
            return expr
        else:
            return self.__reformulate_unary_ops(expr)

    def __reformulate_unary_ops(self,
                                node: mat.ExpressionNode
                                ) -> Union[mat.ArithmeticExpressionNode,
                                           mat.RelationalOperationNode]:

        if isinstance(node, mat.RelationalOperationNode):
            node.lhs_operand = self.__reformulate_unary_ops(node.lhs_operand)
            node.rhs_operand = self.__reformulate_unary_ops(node.rhs_operand)
            return node

        elif isinstance(node, mat.ArithmeticExpressionNode):

            if isinstance(node, mat.NumericNode) or isinstance(node, mat.DeclaredEntityNode):
                return node

            elif isinstance(node, mat.FunctionNode):
                for i, operand in enumerate(node.operands):
                    node.operands[i] = self.__reformulate_unary_ops(operand)

            elif isinstance(node, mat.UnaryArithmeticOperationNode):
                operand = self.__reformulate_unary_ops(node.operand)
                if node.operator == '+':
                    return operand
                else:
                    return self._node_builder.add_negative_unity_coefficient(operand)

            elif isinstance(node, mat.BinaryArithmeticOperationNode):
                node.lhs_operand = self.__reformulate_unary_ops(node.lhs_operand)
                node.rhs_operand = self.__reformulate_unary_ops(node.rhs_operand)
                return node

            elif isinstance(node, mat.MultiArithmeticOperationNode):
                for i, operand in enumerate(node.operands):
                    node.operands[i] = self.__reformulate_unary_ops(operand)
                return node

            elif isinstance(node, mat.ConditionalArithmeticExpressionNode):
                for i, operand in enumerate(node.operands):
                    node.operands[i] = self.__reformulate_unary_ops(operand)
                return node

        else:
            raise ValueError("Problem Formulator encountered an unexpected node"
                             + " while reformulating unary arithmetic operations")

    def expand_factors(self,
                       expr: Union[mat.Expression, mat.ExpressionNode]):
        if isinstance(expr, mat.Expression):
            node = expr.expression_node
            terms = self.__expand_factors(node)
            expr.expression_node = self._node_builder.build_addition_node(terms)
            expr.link_nodes()
            return expr
        else:
            terms = self.__expand_factors(expr)
            return self._node_builder.build_addition_node(terms)

    def __expand_factors(self,
                         node: mat.ExpressionNode
                         ) -> List[Union[mat.ArithmeticExpressionNode,
                                         mat.RelationalOperationNode]]:

        # Relational Operation
        if isinstance(node, mat.RelationalOperationNode):
            ref_operands = []
            for operand in [node.lhs_operand, node.rhs_operand]:
                terms = self.__expand_factors(operand)
                ref_operands.append(self._node_builder.build_addition_node(terms))
            node.lhs_operand = ref_operands[0]
            node.rhs_operand = ref_operands[1]
            return [node]

        # Arithmetic Operation
        elif isinstance(node, mat.ArithmeticExpressionNode):

            # Constant or Declared Entity
            if isinstance(node, mat.NumericNode) or isinstance(node, mat.DeclaredEntityNode):
                return [node]

            # Function
            elif isinstance(node, mat.FunctionNode):
                for i, operand in enumerate(node.operands):
                    terms = self.__expand_factors(operand)
                    node.operands[i] = self._node_builder.build_addition_node(terms)

            # Unary Operation
            elif isinstance(node, mat.UnaryArithmeticOperationNode):
                node = self.__reformulate_unary_ops(node)
                return self.__expand_factors(node)

            # Binary Operation
            elif isinstance(node, mat.BinaryArithmeticOperationNode):

                lhs_terms = self.__expand_factors(node.lhs_operand)
                rhs_terms = self.__expand_factors(node.rhs_operand)

                # Addition
                if node.operator == '+':
                    return lhs_terms + rhs_terms

                # Subtraction
                elif node.operator == '-':
                    rhs_terms = [self._node_builder.add_negative_unity_coefficient(t) for t in rhs_terms]
                    return lhs_terms + rhs_terms

                # Multiplication
                elif node.operator == '*':
                    terms = []
                    for lhs_term in lhs_terms:
                        for rhs_term in rhs_terms:
                            factors = [deepcopy(lhs_term), deepcopy(rhs_term)]
                            term = self._node_builder.build_multiplication_node(factors)
                            terms.append(term)
                    return terms

                # Division
                elif node.operator == '/':
                    if len(lhs_terms) == 1:
                        node.lhs_operand = self._node_builder.build_addition_node(lhs_terms)
                        node.rhs_operand = self._node_builder.build_addition_node(rhs_terms)
                        return [node]
                    else:
                        terms = []
                        den = self._node_builder.build_addition_node(rhs_terms)
                        for num_term in lhs_terms:
                            term = self._node_builder.build_fractional_node(num_term, deepcopy(den))
                            terms.append(term)
                        return terms

                # Other
                else:
                    if node.operator == "less":
                        warnings.warn("Problem Formulator unable to reformulate a 'less' operation node")
                    node.lhs_operand = self._node_builder.build_addition_node(lhs_terms)
                    node.rhs_operand = self._node_builder.build_addition_node(rhs_terms)
                    return [node]

            # Multi Operation
            elif isinstance(node, mat.MultiArithmeticOperationNode):

                # Addition
                if node.operator == '+':
                    terms = []
                    for operand in node.operands:
                        terms.extend(self.__expand_factors(operand))
                    return terms

                # Multiplication
                elif node.operator == '*':
                    operand_count = len(node.operands)
                    term_lists = [self.__expand_factors(o) for o in node.operands]

                    def assemble_factor_lists(f: list = None, j: int = 0):
                        f_list = []
                        if f is None:
                            f = []
                        for t in term_lists[j]:
                            f_new = deepcopy(f)
                            f_new.append(t)
                            if j == operand_count - 1:
                                f_list.append(f_new)
                            else:
                                f_list.extend(assemble_factor_lists(f_new, j + 1))
                        return f_list

                    factor_lists = assemble_factor_lists()

                    terms = []
                    for factors in factor_lists:
                        term = self._node_builder.build_multiplication_node(factors)
                        terms.append(term)

                    return terms

            # Conditional Operation
            elif isinstance(node, mat.ConditionalArithmeticExpressionNode):
                for i, operand in enumerate(node.operands):
                    terms = self.__expand_factors(operand)
                    node.operands[i] = self._node_builder.build_addition_node(terms)
                return [node]

        else:
            raise ValueError("Problem formulator encountered an unexpected node"
                             + " while expanding factors")

    # Utility
    # ------------------------------------------------------------------------------------------------------------------

    def generate_free_node_id(self):
        if self._problem is not None:
            return self._problem.generate_free_node_id()
        else:
            return self._node_builder.generate_free_node_id()

    def seed_free_node_id(self, node: mat.ExpressionNode) -> int:
        return self._node_builder.seed_free_node_id(node)
