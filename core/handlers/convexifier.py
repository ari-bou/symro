from copy import deepcopy
from ordered_set import OrderedSet
from typing import Dict, Iterable, List, Optional, Tuple, Union
import warnings

import symro.core.constants as const
import symro.core.mat as mat
from symro.core.prob.problem import Problem
import symro.core.handlers.nodebuilder as nb
import symro.core.handlers.formulator as fmr
import symro.core.handlers.entitybuilder as eb

"""
References:

-   Maranas, C.D., Floudas, C.A. Finding all solutions of nonlinearly constrained systems of equations. J Glob Optim 7, 
    143–182 (1995). https://doi.org/10.1007/BF01097059
-   C.S. Adjiman, S. Dallwig, C.A. Floudas, A. Neumaier, A global optimization method, αBB, for general 
    twice-differentiable constrained NLPs — I. Theoretical advances, Computers & Chemical Engineering, Volume 22, Issue 
    9, 1998, Pages 1137-1158, ISSN 0098-1354, https://doi.org/10.1016/S0098-1354(98)00027-1.
"""


def convexify(problem: Problem,
              problem_symbol: str = None,
              description: str = None,
              working_dir_path: str = None):
    convexifier = Convexifier()
    return convexifier.convexify(problem=problem,
                                 problem_symbol=problem_symbol,
                                 description=description,
                                 working_dir_path=working_dir_path)


class Convexifier:

    def __init__(self):

        self.convex_relaxation: Optional[Problem] = None

        self.lb_params: Dict[str, mat.MetaParameter] = {}
        self.ub_params: Dict[str, mat.MetaParameter] = {}

        self.ue_meta_vars: Dict[tuple, mat.MetaVariable] = {}

    def convexify(self,
                  problem: Problem,
                  problem_symbol: str = None,
                  description: str = None,
                  working_dir_path: str = None):

        if problem_symbol is None:
            problem_symbol = problem.symbol
        problem_symbol = problem.generate_unique_symbol(base_symbol=problem_symbol)

        if description is None:
            description = "Convex relaxation of problem {0}".format(problem.symbol)

        if working_dir_path is None:
            working_dir_path = problem.working_dir_path

        self.convex_relaxation = Problem(symbol=problem_symbol,
                                         description=description,
                                         working_dir_path=working_dir_path)
        Problem.deepcopy(problem, self.convex_relaxation)

        fmr.substitute_defined_variables(self.convex_relaxation)
        fmr.standardize_model(self.convex_relaxation)
        self.__reformulate_nonlinear_equality_constraints()

        self.__convexify_objectives()
        self.__convexify_constraints()

        return self.convex_relaxation

    def __reformulate_nonlinear_equality_constraints(self):
        for mc in self.convex_relaxation.model_meta_cons:
            if mc.get_constraint_type() == mat.MetaConstraint.EQUALITY_TYPE \
                    and not mat.is_linear(mc.get_expression().root_node):
                fmr.convert_equality_to_inequality_constraints(self.convex_relaxation, mc)

    def __convexify_objectives(self):

        for mo in self.convex_relaxation.model_meta_objs:

            expr = mo.get_expression()

            expr_node = expr.root_node
            if not isinstance(expr_node, mat.ArithmeticExpressionNode):
                raise ValueError("Convexifier encountered unexpected expression node"
                                 + " while convexifying objective function '{0}'".format(mo))

            terms = self.__standardize_expression(root_node=expr_node,
                                                  idx_set_node=mo.idx_set_node,
                                                  dummy_element=tuple(mo.get_idx_set_dummy_element()))

            convex_terms = []
            for term in terms:
                convex_terms.append(self.__convexify_node(term))

            convex_root_node = nb.build_addition_node(convex_terms)

            expr.root_node = convex_root_node

    def __convexify_constraints(self):

        for mc in self.convex_relaxation.model_meta_cons:

            if mc.get_constraint_type() == mat.MetaConstraint.INEQUALITY_TYPE:

                expr = mc.get_expression()

                root_node = expr.root_node
                if not isinstance(root_node, mat.RelationalOperationNode):
                    raise ValueError("Convexifier encountered unexpected expression node"
                                     + " while convexifying objective function '{0}'".format(mc))

                lhs_operand = root_node.lhs_operand

                terms = self.__standardize_expression(root_node=lhs_operand,
                                                      idx_set_node=mc.idx_set_node,
                                                      dummy_element=tuple(mc.get_idx_set_dummy_element()))

                convex_terms = []
                for term in terms:
                    convex_terms.append(self.__convexify_node(term))

                convex_root_node = nb.build_addition_node(convex_terms)

                root_node.lhs_operand = convex_root_node

    # Expression Standardization
    # ------------------------------------------------------------------------------------------------------------------

    def __standardize_expression(self,
                                 root_node: mat.ArithmeticExpressionNode,
                                 idx_set_node: mat.CompoundSetNode = None,
                                 dummy_element: mat.Element = None):

        root_node = fmr.reformulate_subtraction_and_unary_negation(root_node)

        if idx_set_node is None:
            idx_set = None
            outer_unb_syms = None
        else:
            idx_set = idx_set_node.evaluate(state=self.convex_relaxation.state)[0]
            outer_unb_syms = idx_set_node.get_defined_unbound_symbols()

        terms = fmr.expand_multiplication(
            problem=self.convex_relaxation,
            node=root_node,
            idx_set=idx_set,
            dummy_element=dummy_element)

        ref_terms = []
        for term in terms:
            if isinstance(term, mat.MultiplicationNode):
                term = fmr.combine_summation_factor_nodes(
                    problem=self.convex_relaxation,
                    factors=term.operands,
                    outer_unb_syms=outer_unb_syms)
                ref_terms.append(term)
            else:
                ref_terms.append(term)

        return ref_terms

    # Expression Convexification
    # ------------------------------------------------------------------------------------------------------------------

    def __convexify_node(self, node: mat.ArithmeticExpressionNode) -> mat.ArithmeticExpressionNode:

        if isinstance(node, mat.ArithmeticTransformationNode) and node.symbol == "sum":
            convexified_node = self.__convexify_node(node.operands[0])
            node.operands[0] = convexified_node

        elif isinstance(node, mat.MultiplicationNode):

            factors = node.operands

            is_const = [mat.is_constant(factor) for factor in factors]

            var_factors = [f for f, is_f_const in zip(factors, is_const) if not is_f_const]
            var_factor_types = [self.__identify_node(f) for f in var_factors]
            var_factor_count = len(var_factors)

            const_factors = [f for f, is_f_const in zip(factors, is_const) if is_f_const]

            # constant
            if var_factor_count == 0:
                return node

            # general nonconvexity
            elif const.GENERAL_NONCONVEX in var_factor_types:
                warnings.warn("Convexifier was unable to convexify general nonconvex term '{0}'".format(node))
                return node

            # univariate concave
            elif var_factor_count == 1 and var_factor_types[0] == const.UNIVARIATE_CONCAVE:
                return self.__build_sign_conditional_convex_underestimator(ue_type=const.UNIVARIATE_CONCAVE,
                                                                           coefficient_nodes=const_factors,
                                                                           operands=var_factors)

            # linear (x)
            elif var_factor_count == 1 and var_factor_types.count(const.LINEAR) == 1:
                return node

            # bilinear (xy)
            elif var_factor_count == 2 and var_factor_types.count(const.LINEAR) == 2:
                return self.__build_sign_conditional_convex_underestimator(ue_type=const.BILINEAR,
                                                                           coefficient_nodes=const_factors,
                                                                           operands=var_factors)

            # trilinear (xyz)
            elif var_factor_count == 3 and var_factor_types.count(const.LINEAR) == 3:
                return self.__build_sign_conditional_convex_underestimator(ue_type=const.TRILINEAR,
                                                                           coefficient_nodes=const_factors,
                                                                           operands=var_factors)

            # fraction with nonlinear denominator (1/xy)
            elif var_factor_types.count(const.FRACTIONAL) > 1:
                warnings.warn("Convexifier was unable to convexify term '{0}'".format(node)
                              + " with more than 1 fractional factor ")
                return node

            # fractional (1/x)
            elif var_factor_count == 1 and var_factor_types.count(const.FRACTIONAL) == 1:
                return self.__build_sign_conditional_convex_underestimator(ue_type=const.FRACTIONAL,
                                                                           coefficient_nodes=const_factors,
                                                                           operands=var_factors)

            # bilinear fractional (x/y)
            elif var_factor_count == 2 and var_factor_types.count(const.LINEAR) == 1 \
                    and var_factor_types.count(const.FRACTIONAL) == 1:
                return self.__build_sign_conditional_convex_underestimator(ue_type=const.FRACTIONAL_BILINEAR,
                                                                           coefficient_nodes=const_factors,
                                                                           operands=var_factors)

            # trilinear fractional (xy/z)
            elif var_factor_count == 3 and var_factor_types.count(const.LINEAR) == 2 \
                    and var_factor_types.count(const.FRACTIONAL) == 1:
                return self.__build_sign_conditional_convex_underestimator(ue_type=const.FRACTIONAL_TRILINEAR,
                                                                           coefficient_nodes=const_factors,
                                                                           operands=var_factors)

        else:

            # linear
            if mat.is_linear(node):
                return node

            type = self.__identify_node(node)

            # univariate concave
            if type == const.UNIVARIATE_CONCAVE:
                return self.__build_convex_envelope_for_general_univariate_concave_function(node)

            # general nonconvexity
            else:
                warnings.warn("Convexifier was unable to convexify term '{0}'".format(node))
                return node

    # Term Identification
    # ------------------------------------------------------------------------------------------------------------------

    def __identify_node(self, node: mat.ArithmeticExpressionNode) -> int:

        if mat.is_constant(node):
            return const.CONSTANT

        # declared entity
        elif isinstance(node, mat.DeclaredEntityNode):
            return const.LINEAR

        # division
        elif isinstance(node, mat.DivisionNode):

            # by default, numerator is a numeric node with value 1

            # linear denominator
            if mat.is_linear(node.rhs_operand):
                return const.FRACTIONAL

            # nonlinear denominator
            else:
                return const.GENERAL_NONCONVEX

        # exponentiation
        elif isinstance(node, mat.ExponentiationNode):

            # univariate exponential with constant base: b^x
            if mat.is_constant(node.lhs_operand) and mat.is_linear(node.rhs_operand):
                return const.UNIVARIATE_CONCAVE

            else:
                # all other non-factorizable exponentiation operations assumed to be general nonconvexities
                return const.GENERAL_NONCONVEX

        # transformation
        elif isinstance(node, mat.ArithmeticTransformationNode):

            # reductive transformation
            if node.is_reductive():

                # summation
                if node.symbol == "sum":
                    return self.__identify_node(node.operands[0])

                # product, maximum, or minimum
                else:
                    return const.GENERAL_NONCONVEX

            # non-reductive transformation
            else:

                if node.symbol in ("log", "log10", "exp", "cos", "sin", "tan"):

                    if mat.is_linear(node.operands[0]):  # operand is linear
                        return const.UNIVARIATE_CONCAVE

                    else:  # operand is nonlinear
                        return const.GENERAL_NONCONVEX

                else:  # all other transformations are assumed to be nonconvex
                    return const.GENERAL_NONCONVEX

        else:

            # addition nodes should be expanded
            if isinstance(node, mat.AdditionNode):
                raise ValueError("Convexifier encountered an illegal addition node"
                                 + " while identify the type of a term '{0}'".format(node))

            # subtraction nodes should be converted to addition nodes
            if isinstance(node, mat.SubtractionNode):
                raise ValueError("Convexifier encountered an illegal subtraction node"
                                 + " while identify the type of a term '{0}'".format(node))

            # multiplication nodes should be handled in a preceding method
            if isinstance(node, mat.MultiplicationNode):
                raise ValueError("Convexifier encountered an illegal multiplication node"
                                 + " while identify the type of a term '{0}'".format(node))

            raise ValueError("Convexifier encountered an unexpected term '{0}'".format(node)
                             + " while trying to identify its type")

    @staticmethod
    def __is_negative_unity_node(node: mat.ArithmeticExpressionNode):
        if isinstance(node, mat.NumericNode) and node.value == -1:
            return True
        else:
            return False

    # Underestimator Construction
    # ------------------------------------------------------------------------------------------------------------------

    def __build_sign_conditional_convex_underestimator(self,
                                                       ue_type: int,
                                                       coefficient_nodes: List[mat.ArithmeticExpressionNode],
                                                       operands: List[mat.ArithmeticExpressionNode]):

        operand_count = len(operands)

        pos_ue_node = self.__build_and_retrieve_convex_underestimator(
            ue_type=ue_type,
            sign="POS",
            operands=operands,
            is_negative=False
        )

        if len(coefficient_nodes) == 0:
            return pos_ue_node

        else:

            # single operand
            if operand_count == 1:
                neg_ue_node = operands[0]  # assume convex by default

            # multiple operands
            else:
                neg_ue_node = self.__build_and_retrieve_convex_underestimator(
                    ue_type=ue_type,
                    sign="NEG",
                    operands=operands,
                    is_negative=True
                )

            coefficient_node = nb.build_multiplication_node(coefficient_nodes, is_prioritized=True)

            cond_node = mat.ConditionalArithmeticExpressionNode(
                operands=[
                    pos_ue_node,
                    nb.append_negative_unity_coefficient(neg_ue_node)
                ],
                conditions=[
                    mat.RelationalOperationNode(operator=">=",
                                                lhs_operand=deepcopy(coefficient_node),
                                                rhs_operand=nb.build_numeric_node(0))
                ],
                is_prioritized=True
            )

        return nb.build_multiplication_node([coefficient_node, cond_node])

    def __build_and_retrieve_convex_underestimator(self,
                                                   ue_type: int,
                                                   sign: str,
                                                   operands: List[mat.ArithmeticExpressionNode],
                                                   is_negative: bool):

        operand_count = len(operands)  # count number of supplied operands

        # no operands
        if operand_count == 0:
            raise ValueError("Convexifier expected at least one arithmetic expression node"
                             + " while building and retrieving a convex underestimator")

        # single operand
        elif operand_count == 1:
            return self.__build_univariate_convex_envelope(operand=operands[0], is_negative=False)

        # multiple operands
        else:

            # - build lower and upper bound meta-parameters for each variable node
            # - standardize order of operands
            # - retrieve an identifying symbol and any associated dummy nodes for each operand
            operands, operand_syms, is_var, dummy_nodes = self.__process_operands_of_nonconvex_term(operands)

            # deterministically generate a corresponding underestimator id for the supplied operands
            ue_id = self.__generate_constrained_underestimator_id(ue_type=ue_type,
                                                                  sign=sign,
                                                                  operand_syms=operand_syms)

            # underestimator has already been constructed
            if ue_id in self.ue_meta_vars:
                ue_meta_var = self.ue_meta_vars[ue_id]

            # underestimator does not exist
            else:
                idx_set_node, var_unb_sym_map = self.__build_constrained_underestimator_idx_set_node(
                    operand_syms=operand_syms,
                    is_var=is_var)
                ue_meta_var = self.__build_constrained_underestimator_meta_variable(ue_id=ue_id,
                                                                                    idx_set_node=idx_set_node)
                mod_operands = self.__modify_idx_nodes_of_operands(operand_syms=operand_syms,
                                                                   operands=operands,
                                                                   var_unb_sym_map=var_unb_sym_map)
                self.__build_convex_envelope_constraints(ue_id=ue_id,
                                                         ue_meta_var=ue_meta_var,
                                                         operands=mod_operands,
                                                         is_negative=is_negative)

            ue_node = self.__build_constrained_underestimator_node(ue_meta_var=ue_meta_var,
                                                                   dummy_nodes=dummy_nodes)

            return ue_node

    def __process_operands_of_nonconvex_term(self, operands: Iterable[mat.ArithmeticExpressionNode]):

        operand_syms = []  # operand symbols
        sym_to_operand_map = {}
        is_var = {}  # flags designating whether the corresponding operands vary
        dummy_nodes = []  # combined list of dummy nodes controlling the operands

        for operand in operands:

            if isinstance(operand, mat.NumericNode):
                sym = str(abs(int(operand.value)))
                is_var[sym] = False

            elif isinstance(operand, mat.DeclaredEntityNode):

                sym = operand.symbol

                self.__build_bound_meta_entities(sym)

                is_var[sym] = True
                if operand.idx_node is not None:
                    dummy_nodes.extend(operand.idx_node.component_nodes)

            elif isinstance(operand, mat.DivisionNode):

                den_node = operand.rhs_operand
                if not isinstance(den_node, mat.DeclaredEntityNode):
                    raise ValueError("Convexifier encountered an unexpected operand '{0}'".format(operand)
                                     + " while building a constrained underestimator")

                sym = den_node.symbol

                self.__build_bound_meta_entities(sym)

                is_var[sym] = True
                if den_node.idx_node is not None:
                    dummy_nodes.extend(den_node.idx_node.component_nodes)

            else:
                raise ValueError("Convexifier encountered an unexpected operand '{0}'".format(operand)
                                 + " while building a constrained underestimator")

            sym_to_operand_map[sym] = operand
            operand_syms.append(sym)

        operand_syms = sorted(operand_syms)  # sort operand symbols alphabetically
        operands = [sym_to_operand_map[s] for s in operand_syms]

        return operands, operand_syms, is_var, dummy_nodes

    @staticmethod
    def __generate_constrained_underestimator_id(ue_type: int, sign: str, operand_syms: Iterable[str]):
        return (ue_type, sign) + tuple(operand_syms)

    def __build_constrained_underestimator_idx_set_node(self,
                                                        operand_syms: List[str],
                                                        is_var: Dict[str, bool]):

        cmpt_set_nodes = []
        conj_operands = []

        var_unb_sym_map = {}
        def_unb_syms = set()

        for operand_sym in operand_syms:

            if is_var[operand_sym]:
                mv = self.convex_relaxation.meta_vars[operand_sym]

                if mv.is_indexed():

                    var_idx_set_node = deepcopy(mv.idx_set_node)

                    unb_syms = nb.retrieve_unbound_symbols(var_idx_set_node)  # retrieve unbound symbols
                    clashing_unb_syms = unb_syms & def_unb_syms  # retrieving clashing unbound symbols
                    def_unb_syms = def_unb_syms | unb_syms  # update set of defined unbound symbols

                    # identify and resolve unbound symbol conflicts
                    if len(clashing_unb_syms) > 0:

                        # generate map of clashing unbound symbols to unique unbound symbols
                        mapping = {clashing_sym: self.convex_relaxation.generate_unique_symbol(clashing_sym)
                                   for clashing_sym in clashing_unb_syms}

                        # replace the conflicting symbols with unique symbols
                        nb.replace_unbound_symbols(node=var_idx_set_node, mapping=mapping)

                        # update set of defined unbound symbols with newly-generated symbols
                        def_unb_syms = def_unb_syms | set(mapping.values())

                    # add component set nodes of the variable indexing set node to the list of component set nodes
                    cmpt_set_nodes.extend(var_idx_set_node.set_nodes)

                    var_unb_sym_map[operand_sym] = var_idx_set_node.get_defined_unbound_symbols()

                    # add the constraint node of the variable indexing set node to the list of constraint operands
                    if var_idx_set_node.constraint_node is not None:
                        conj_operands.append(var_idx_set_node.constraint_node)

        if len(cmpt_set_nodes) == 0:
            return None, var_unb_sym_map

        else:

            if len(conj_operands) == 0:
                con_node = None
            else:
                con_node = nb.build_conjunction_node(conj_operands)

            ue_idx_set_node = mat.CompoundSetNode(set_nodes=cmpt_set_nodes,
                                                  constraint_node=con_node)

            return ue_idx_set_node, var_unb_sym_map

    def __build_constrained_underestimator_meta_variable(self,
                                                         ue_id: Tuple[Union[int, str], ...],
                                                         idx_set_node: mat.CompoundSetNode = None):

        # generate unique symbol for underestimator meta-variable
        operand_syms = ue_id[2:]
        base_ue_sym = "UE_{0}_{1}_{2}".format(ue_id[0],  # underestimator type
                                              ue_id[1],  # sign
                                              ''.join([s[0].upper() for s in operand_syms]))
        ue_sym = self.convex_relaxation.generate_unique_symbol(base_ue_sym)

        # build meta-variable for underestimator
        ue_meta_var = eb.build_meta_var(
            problem=self.convex_relaxation,
            symbol=ue_sym,
            idx_set_node=idx_set_node)

        self.ue_meta_vars[ue_id] = ue_meta_var
        self.convex_relaxation.add_meta_variable(ue_meta_var, is_in_model=True)

        return ue_meta_var

    @staticmethod
    def __build_constrained_underestimator_node(ue_meta_var: mat.MetaVariable,
                                                dummy_nodes: List[Union[mat.DummyNode,
                                                                        mat.ArithmeticExpressionNode,
                                                                        mat.StringExpressionNode]]):

        # scalar underestimator
        if len(dummy_nodes) == 0:
            idx_node = None

        # indexed underestimator
        else:
            # build an index node using the supplied dummy nodes
            # deep copy the original dummy nodes in case they need to be used in the original nonconvex expression node
            idx_node = mat.CompoundDummyNode(component_nodes=deepcopy(dummy_nodes))

        # build a declared entity node for the underestimating variable
        return mat.DeclaredEntityNode(symbol=ue_meta_var.get_symbol(),
                                      idx_node=idx_node)

    @staticmethod
    def __modify_idx_nodes_of_operands(operand_syms: Iterable[str],
                                       operands: Iterable[Union[mat.NumericNode,
                                                                mat.DeclaredEntityNode,
                                                                mat.DivisionNode]],
                                       var_unb_sym_map: Dict[str, OrderedSet[str]]):

        mod_operands = []

        for operand_sym, operand in zip(operand_syms, operands):

            if operand_sym in var_unb_sym_map:

                dummy_nodes = [nb.build_dummy_node(unb_sym)
                               for unb_sym in var_unb_sym_map[operand_sym]]
                idx_node = mat.CompoundDummyNode(component_nodes=dummy_nodes)

                if isinstance(operand, mat.DeclaredEntityNode):
                    operand = deepcopy(operand)
                    operand.idx_node = idx_node

                elif isinstance(operand, mat.DivisionNode):

                    den_node = operand.rhs_operand
                    if not isinstance(den_node, mat.DeclaredEntityNode):
                        raise ValueError("Convexifier encountered an unexpected operand '{0}'".format(operand)
                                         + " while building a constrained underestimator")

                    operand = deepcopy(den_node)
                    operand.idx_node = idx_node

            mod_operands.append(operand)

        return mod_operands

    def __build_convex_envelope_constraints(self,
                                            ue_id: Tuple[Union[int, str], ...],
                                            ue_meta_var: mat.MetaVariable,
                                            operands: List[mat.ArithmeticExpressionNode],
                                            is_negative: bool):

        operand_syms = ue_id[2:]
        base_ue_bound_sym = "UE_BOUND_{0}_{1}_{2}".format(ue_id[0],  # underestimator type
                                                          ue_id[1],  # sign
                                                          ''.join([s[0].upper() for s in operand_syms]))

        ue_node = nb.build_default_entity_node(ue_meta_var)

        ce_expr_nodes = self.__build_multivariate_convex_envelope(operands, is_negative=is_negative)

        for i, ce_expr_node in enumerate(ce_expr_nodes, start=1):

            ue_bound_sym = self.convex_relaxation.generate_unique_symbol("{0}_{1}".format(base_ue_bound_sym, i))

            rel_op_node = mat.RelationalOperationNode(
                operator="<=",
                lhs_operand=nb.build_subtraction_node(ce_expr_node, deepcopy(ue_node)),
                rhs_operand=nb.build_numeric_node(0)
            )

            meta_con = eb.build_meta_con(
                problem=self.convex_relaxation,
                symbol=ue_bound_sym,
                idx_set_node=deepcopy(ue_meta_var.idx_set_node),
                expression=mat.Expression(rel_op_node)
            )

            self.convex_relaxation.add_meta_constraint(meta_con, is_in_model=True)

    def __build_multivariate_convex_envelope(self,
                                             operands: List[mat.ArithmeticExpressionNode],
                                             is_negative: bool
                                             ) -> List[mat.ArithmeticExpressionNode]:

        if len(operands) > 2:

            convex_envelopes = []

            for i in range(len(operands) - 1):
                for j in range(i + 1, len(operands)):

                    x_node = operands[i]
                    xL_node = self.__build_lower_bound_node(operand=x_node, is_negative=False)
                    xU_node = self.__build_upper_bound_node(operand=x_node, is_negative=False)

                    if i == 0:
                        sub_operands = operands[1:]
                    else:
                        sub_operands = operands[:i] + operands[i + 1:]

                    s_node = nb.build_multiplication_node(sub_operands)
                    sL_node = self.__build_lower_bound_node(s_node, is_negative=is_negative)
                    sU_node = self.__build_lower_bound_node(s_node, is_negative=is_negative)

                    ce_s_nodes = self.__build_multivariate_convex_envelope(sub_operands, is_negative=is_negative)

                    for ce_s_node in ce_s_nodes:

                        ce_xL_s_node = nb.build_multiplication_node([xL_node, ce_s_node])

                        ce_x_sL_node = self.__build_univariate_convex_envelope(
                            operand=x_node, coefficient=sL_node, is_negative=False)

                        ce_xU_s_node = nb.build_multiplication_node([xU_node, ce_s_node])

                        ce_x_sU_node = self.__build_univariate_convex_envelope(
                            operand=x_node, coefficient=sU_node, is_negative=False)

                        convex_envelopes.extend(self.__build_bivariate_convex_envelope(
                            x_node=x_node, xL_node=xL_node, xU_node=xU_node,
                            y_node=ce_s_node, yL_node=sL_node, yU_node=sU_node,
                            ce_xL_y_node=ce_xL_s_node, ce_x_yL_node=ce_x_sL_node,
                            ce_xU_y_node=ce_xU_s_node, ce_x_yU_node=ce_x_sU_node
                        ))

            return convex_envelopes

        else:

            x_node = operands[0]
            xL_node = self.__build_lower_bound_node(operand=x_node, is_negative=is_negative)
            xU_node = self.__build_upper_bound_node(operand=x_node, is_negative=is_negative)

            y_node = operands[1]
            yL_node = self.__build_lower_bound_node(operand=y_node, is_negative=False)
            yU_node = self.__build_upper_bound_node(operand=y_node, is_negative=False)

            ce_xL_y_node = self.__build_univariate_convex_envelope(
                operand=y_node, coefficient=xL_node, is_negative=False)

            ce_x_yL_node = self.__build_univariate_convex_envelope(
                operand=x_node, coefficient=yL_node, is_negative=is_negative)

            ce_xU_y_node = self.__build_univariate_convex_envelope(
                operand=y_node, coefficient=xU_node, is_negative=False)

            ce_x_yU_node = self.__build_univariate_convex_envelope(
                operand=x_node, coefficient=yU_node, is_negative=is_negative)

            return self.__build_bivariate_convex_envelope(
                x_node=x_node, xL_node=xL_node, xU_node=xU_node,
                y_node=y_node, yL_node=yL_node, yU_node=yU_node,
                ce_xL_y_node=ce_xL_y_node, ce_x_yL_node=ce_x_yL_node,
                ce_xU_y_node=ce_xU_y_node, ce_x_yU_node=ce_x_yU_node
            )

    @staticmethod
    def __build_bivariate_convex_envelope(x_node: mat.ArithmeticExpressionNode,
                                          xL_node: mat.ArithmeticExpressionNode,
                                          xU_node: mat.ArithmeticExpressionNode,
                                          y_node: mat.ArithmeticExpressionNode,
                                          yL_node: mat.ArithmeticExpressionNode,
                                          yU_node: mat.ArithmeticExpressionNode,
                                          ce_xL_y_node: mat.ArithmeticExpressionNode,
                                          ce_x_yL_node: mat.ArithmeticExpressionNode,
                                          ce_xU_y_node: mat.ArithmeticExpressionNode,
                                          ce_x_yU_node: mat.ArithmeticExpressionNode
                                          ) -> List[mat.ArithmeticExpressionNode]:

        if mat.is_constant(x_node):
            return [ce_xL_y_node]
        elif mat.is_constant(y_node):
            return [ce_x_yL_node]

        expr_node_1 = nb.build_addition_node(
            [
                ce_xL_y_node,
                ce_x_yL_node,
                nb.build_multiplication_node(
                    [
                        nb.build_numeric_node(-1),
                        deepcopy(xL_node),
                        deepcopy(yL_node)
                    ]
                )
            ],
            is_prioritized=True
        )

        expr_node_2 = nb.build_addition_node(
            [
                ce_xU_y_node,
                ce_x_yU_node,
                nb.build_multiplication_node(
                    [
                        nb.build_numeric_node(-1),
                        deepcopy(xU_node),
                        deepcopy(yU_node)
                    ]
                )
            ],
            is_prioritized=True
        )

        return [expr_node_1, expr_node_2]

    def __build_univariate_convex_envelope(self,
                                           operand: mat.ArithmeticExpressionNode,
                                           is_negative: bool,
                                           coefficient: mat.ArithmeticExpressionNode = None):

        if is_negative:
            if coefficient is None:
                coefficient = nb.build_numeric_node(-1)
            else:
                coefficient = nb.build_multiplication_node(
                    [
                        nb.build_numeric_node(-1),
                        coefficient
                    ]
                )

        # constant (c) or linear (x)
        if isinstance(operand, mat.NumericNode) or isinstance(operand, mat.DeclaredEntityNode):
            if coefficient is None:
                return deepcopy(operand)
            else:
                return nb.build_multiplication_node([deepcopy(coefficient), deepcopy(operand)])

        # fractional (1/x)
        elif isinstance(operand, mat.DivisionNode):

            operand_lb_node = self.__build_lower_bound_node(operand=operand.rhs_operand, is_negative=is_negative)
            operand_ub_node = self.__build_upper_bound_node(operand=operand.rhs_operand, is_negative=is_negative)

            if coefficient is None:
                coefficient = nb.build_numeric_node(1)

            cond_operands = [
                nb.build_multiplication_node(
                    [
                        deepcopy(coefficient),
                        deepcopy(operand)
                    ]),
                nb.build_multiplication_node(
                    [
                        deepcopy(coefficient),
                        nb.build_addition_node(
                            [
                                deepcopy(operand_lb_node),
                                deepcopy(operand_ub_node),
                                nb.append_negative_unity_coefficient(deepcopy(operand))
                            ],
                            is_prioritized=True
                        ),
                        nb.build_fractional_node_with_unity_numerator(deepcopy(operand_lb_node),
                                                                      is_prioritized=True),
                        nb.build_fractional_node_with_unity_numerator(deepcopy(operand_ub_node),
                                                                      is_prioritized=True)
                    ])
            ]

            conditions = [
                mat.RelationalOperationNode(operator=">=",
                                            lhs_operand=deepcopy(coefficient),
                                            rhs_operand=nb.build_numeric_node(0)),
                mat.RelationalOperationNode(operator="<",
                                            lhs_operand=deepcopy(coefficient),
                                            rhs_operand=nb.build_numeric_node(0))
            ]

            return mat.ConditionalArithmeticExpressionNode(
                operands=cond_operands,
                conditions=conditions,
                is_prioritized=True
            )

        # general univariate concave term
        else:
            ue_node = self.__build_convex_envelope_for_general_univariate_concave_function(operand,
                                                                                           is_negative=is_negative)
            if coefficient is None:
                return ue_node
            else:
                return nb.build_multiplication_node([deepcopy(coefficient), ue_node])

    def __build_convex_envelope_for_general_univariate_concave_function(self,
                                                                        uc_node: mat.ArithmeticExpressionNode,
                                                                        is_negative: bool = False):

        # function is mirrored vertically due to a negative coefficient
        if is_negative:
            # function is convex
            return deepcopy(uc_node)

        if isinstance(uc_node, mat.ArithmeticTransformationNode):
            x_node = uc_node.operands[0]

        elif isinstance(uc_node, mat.ExponentiationNode):

            # base is constant and exponent is univariate: b^x
            if mat.is_constant(uc_node.lhs_operand):
                x_node = uc_node.rhs_operand

            # base is univariate and exponent is constant: x^c
            else:
                x_node = uc_node.lhs_operand

        else:
            raise ValueError("Convexifier encountered an unexpected node '{0}'".format(uc_node)
                             + " while constructing a convex envelope for a general univariate concave term")

        var_node = mat.get_var_nodes(x_node)[0]
        var_sym = var_node.symbol

        self.__build_bound_meta_entities(var_sym)  # build bounding meta-parameters

        # generate mappings of replacement symbols
        lb_mapping = {var_sym: self.lb_params[var_sym].get_symbol()}
        ub_mapping = {var_sym: self.ub_params[var_sym].get_symbol()}

        # build elemental nodes

        uc_lb_node = deepcopy(uc_node)
        nb.replace_declared_symbols(uc_lb_node, lb_mapping)

        uc_ub_node = deepcopy(uc_node)
        nb.replace_declared_symbols(uc_ub_node, ub_mapping)

        x_lb_node = deepcopy(x_node)
        x_lb_node.is_prioritized = True
        nb.replace_declared_symbols(x_lb_node, lb_mapping)

        x_ub_node = deepcopy(x_node)
        nb.replace_declared_symbols(x_ub_node, ub_mapping)

        # build underestimator node

        num_node = nb.build_subtraction_node(uc_ub_node, deepcopy(uc_lb_node), is_prioritized=True)
        num_node.rhs_operand.is_prioritized = True
        den_node = nb.build_subtraction_node(x_ub_node, x_lb_node, is_prioritized=True)

        return nb.build_addition_node(
            [
                uc_lb_node,
                nb.build_multiplication_node(
                    [
                        nb.build_fractional_node(num_node, den_node, is_prioritized=True),
                        nb.build_subtraction_node(deepcopy(x_node), deepcopy(x_lb_node), is_prioritized=True)
                    ]
                )
            ],
            is_prioritized=True
        )

    # Bound Construction
    # ------------------------------------------------------------------------------------------------------------------

    def __build_bound_meta_entities(self, var_sym: str):

        if var_sym not in self.lb_params:

            meta_var = self.convex_relaxation.meta_vars[var_sym]

            param_syms = (self.convex_relaxation.generate_unique_symbol("{0}_LB".format(var_sym)),
                          self.convex_relaxation.generate_unique_symbol("{0}_UB".format(var_sym)))

            default_value_nodes = (deepcopy(meta_var.get_lower_bound_node()),
                                   deepcopy(meta_var.get_upper_bound_node()))

            meta_params = []

            for param_sym, default_value_node in zip(param_syms, default_value_nodes):

                idx_set_node = None
                if meta_var.idx_set_node is not None:
                    idx_set_node = deepcopy(meta_var.idx_set_node)

                meta_param = eb.build_meta_param(
                    problem=self.convex_relaxation,
                    symbol=param_sym,
                    idx_set_node=idx_set_node,
                    default_value=default_value_node)

                meta_params.append(meta_param)
                self.convex_relaxation.add_meta_parameter(meta_param, is_in_model=True)

            self.lb_params[var_sym] = meta_params[0]
            self.ub_params[var_sym] = meta_params[1]

    def __build_lower_bound_node(self,
                                 operand: mat.ArithmeticExpressionNode,
                                 is_negative: bool):

        # constant (c)
        if isinstance(operand, mat.NumericNode):
            operand = deepcopy(operand)  # (c)
            if is_negative:
                operand.value *= -1
            return operand

        # linear (x)
        elif isinstance(operand, mat.DeclaredEntityNode):
            if not is_negative:  # x_L
                return self.__build_linear_lower_bound_node(operand)
            else:  # -x_U
                return nb.append_negative_unity_coefficient(
                    self.__build_linear_upper_bound_node(operand)
                )

        # factors (x1*x2*...xn)
        elif isinstance(operand, mat.MultiplicationNode):

            lb_nodes = []
            ub_nodes = []

            for i, o in enumerate(operand.operands):
                if i == 0:
                    lb_nodes.append(self.__build_lower_bound_node(o, is_negative=is_negative))
                    ub_nodes.append(self.__build_upper_bound_node(o, is_negative=is_negative))
                else:
                    lb_nodes.append(self.__build_lower_bound_node(o, is_negative=False))
                    ub_nodes.append(self.__build_upper_bound_node(o, is_negative=False))

            bound_nodes = [[lb_node, ub_node] for lb_node, ub_node in zip(lb_nodes, ub_nodes)]

            return mat.ArithmeticTransformationNode(
                symbol="min",
                operands=fmr.expand_factors_n(bound_nodes)
            )

        # fractional (1/x)
        elif isinstance(operand, mat.DivisionNode):
            if not is_negative:  # 1/x_U
                return self.__build_fractional_lower_bound_node(operand)
            else:  # -1/x_L
                return nb.append_negative_unity_coefficient(
                    self.__build_fractional_upper_bound_node(operand)
                )

        else:
            raise ValueError("Convexifier encountered an unexpected operand '{0}'".format(operand)
                             + " while building a lower bound node")

    def __build_upper_bound_node(self,
                                 operand: mat.ArithmeticExpressionNode,
                                 is_negative: bool):

        # constant (c)
        if isinstance(operand, mat.NumericNode):
            operand = deepcopy(operand)  # (c)
            if is_negative:
                operand.value *= -1
            return operand

        # linear (x)
        elif isinstance(operand, mat.DeclaredEntityNode):
            self.__build_linear_upper_bound_node(operand)
            if not is_negative:  # x_U
                return self.__build_linear_upper_bound_node(operand)
            else:  # -x_L
                return nb.append_negative_unity_coefficient(
                    self.__build_linear_lower_bound_node(operand)
                )

        # factors (x1*x2*...xn)
        elif isinstance(operand, mat.MultiplicationNode):

            lb_nodes = []
            ub_nodes = []

            for i, o in enumerate(operand.operands):
                if i == 0:
                    lb_nodes.append(self.__build_lower_bound_node(o, is_negative=is_negative))
                    ub_nodes.append(self.__build_upper_bound_node(o, is_negative=is_negative))
                else:
                    lb_nodes.append(self.__build_lower_bound_node(o, is_negative=False))
                    ub_nodes.append(self.__build_upper_bound_node(o, is_negative=False))

            bound_nodes = [[lb_node, ub_node] for lb_node, ub_node in zip(lb_nodes, ub_nodes)]

            return mat.ArithmeticTransformationNode(
                symbol="max",
                operands=fmr.expand_factors_n(bound_nodes)
            )

        # fractional (1/x)
        elif isinstance(operand, mat.DivisionNode):
            if not is_negative:  # 1/x_L
                return self.__build_fractional_upper_bound_node(operand)
            else:  # -1/x_U
                return nb.append_negative_unity_coefficient(
                    self.__build_fractional_lower_bound_node(operand)
                )

        else:
            raise ValueError("Convexifier encountered an unexpected operand '{0}'".format(operand)
                             + " while building an upper bound node")

    def __build_linear_lower_bound_node(self, operand: mat.DeclaredEntityNode):
        lb_param = self.lb_params[operand.symbol]
        return self.__build_declared_bound_node(lb_param, operand.idx_node)  # x_L

    def __build_linear_upper_bound_node(self, operand: mat.DeclaredEntityNode):
        ub_param = self.ub_params[operand.symbol]
        return self.__build_declared_bound_node(ub_param, operand.idx_node)  # x_U

    def __build_fractional_lower_bound_node(self, operand: mat.DivisionNode):

        den_node = operand.rhs_operand
        if not isinstance(den_node, mat.DeclaredEntityNode):
            raise ValueError("Convexifier encountered an unexpected operand '{0}'".format(operand)
                             + " while building a lower bound node")

        ub_param = self.ub_params[den_node.symbol]
        den_ub_node = self.__build_declared_bound_node(ub_param, den_node.idx_node)  # x_U

        # 1/x_U
        return nb.build_fractional_node_with_unity_numerator(denominator=den_ub_node)

    def __build_fractional_upper_bound_node(self, operand: mat.DivisionNode):

        den_node = operand.rhs_operand
        if not isinstance(den_node, mat.DeclaredEntityNode):
            raise ValueError("Convexifier encountered an unexpected operand '{0}'".format(operand)
                             + " while building an upper bound node")

        lb_param = self.lb_params[den_node.symbol]
        den_lb_node = self.__build_declared_bound_node(lb_param, den_node.idx_node)  # x_L

        # 1/x_L
        return nb.build_fractional_node_with_unity_numerator(denominator=den_lb_node)

    @staticmethod
    def __build_declared_bound_node(meta_param: mat.MetaParameter, idx_node: Optional[mat.CompoundDummyNode]):
        return mat.DeclaredEntityNode(symbol=meta_param.get_symbol(),
                                      idx_node=deepcopy(idx_node))
