from copy import deepcopy
from ordered_set import OrderedSet
from typing import Dict, Iterable, List, Optional, Tuple, Union
import warnings

import symro.src.mat as mat
from symro.src.prob.problem import Problem
import symro.src.handlers.nodebuilder as nb
import symro.src.handlers.formulator as fmr
import symro.src.handlers.entitybuilder as eb

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

        self.n_linear_bound_params: Dict[tuple, mat.MetaParameter] = {}

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
        for mc in list(self.convex_relaxation.model_meta_cons):
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

            # reformulate the objective expression into a list of standardized terms
            terms = self.__standardize_expression(root_node=expr_node,
                                                  idx_set_node=mo.idx_set_node,
                                                  dummy_element=tuple(mo.get_idx_set_dummy_element()))

            # retrieve indexing set of the constraint
            idx_set = None
            dummy_element = None
            if mo.is_indexed():
                idx_set = mo.idx_set_node.evaluate(state=self.convex_relaxation.state)[0]
                dummy_element = mo.idx_set_node.get_dummy_element(state=self.convex_relaxation.state)

            convex_terms = []  # list of convexified terms

            # convexify each term in the expression
            for term in terms:
                convex_terms.append(self.__convexify_node(
                    node=term,
                    idx_set=idx_set,
                    dummy_element=dummy_element
                ))

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

                # retrieve indexing set of the constraint
                idx_set = None
                dummy_element = None
                if mc.is_indexed():
                    idx_set = mc.idx_set_node.evaluate(state=self.convex_relaxation.state)[0]
                    dummy_element = mc.idx_set_node.get_dummy_element(state=self.convex_relaxation.state)

                convex_terms = []  # list of convexified terms

                # convexify each term of the constraint expression
                for term in terms:
                    convex_terms.append(self.__convexify_node(
                        node=term,
                        idx_set=idx_set,
                        dummy_element=dummy_element
                    ))

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
            if isinstance(term, mat.ArithmeticOperationNode) and term.operator == mat.MULTIPLICATION_OPERATOR:
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

    def __convexify_node(self,
                         node: mat.ArithmeticExpressionNode,
                         idx_set: mat.IndexingSet,
                         dummy_element: mat.Element) -> mat.ArithmeticExpressionNode:

        if isinstance(node, mat.ArithmeticTransformationNode) and node.symbol == "sum":

            # retrieve the combined indexing set
            inner_idx_sets = node.idx_set_node.evaluate(self.convex_relaxation.state, idx_set, dummy_element)
            inner_idx_set = OrderedSet().union(*inner_idx_sets)
            idx_set = mat.cartesian_product([idx_set, inner_idx_set])
            dummy_element = node.idx_set_node.combined_dummy_element

            # convexify operand of summation
            convexified_node = self.__convexify_node(
                node=node.operands[0],
                idx_set=idx_set,
                dummy_element=dummy_element
            )

            node.operands[0] = convexified_node  # replace original operand with convexified node

            return node

        elif isinstance(node, mat.ArithmeticOperationNode) and node.operator == mat.MULTIPLICATION_OPERATOR:

            factors = node.operands

            is_const = [mat.is_constant(factor) for factor in factors]

            var_factors = [f for f, is_f_const in zip(factors, is_const) if not is_f_const]
            var_factor_types = [self.__identify_node(f, idx_set, dummy_element) for f in var_factors]
            var_factor_count = len(var_factors)

            const_factors = [f for f, is_f_const in zip(factors, is_const) if is_f_const]

            # constant
            if var_factor_count == 0:
                return node

            # general nonconvexity
            elif mat.GENERAL_NONCONVEX in var_factor_types:
                warnings.warn("Convexifier was unable to convexify general nonconvex term '{0}'".format(node))
                return node

            # general univariate nonlinear
            elif var_factor_count == 1 and var_factor_types[0] == mat.UNIVARIATE_NONLINEAR:
                return self.__build_sign_conditional_convex_underestimator(ue_type=mat.UNIVARIATE_NONLINEAR,
                                                                           coefficient_nodes=const_factors,
                                                                           factors=var_factors,
                                                                           idx_set=idx_set,
                                                                           dummy_element=dummy_element)

            # linear (x)
            elif var_factor_count == 1 and var_factor_types.count(mat.LINEAR) == 1:
                return node

            # bilinear (xy)
            elif var_factor_count == 2 and var_factor_types.count(mat.LINEAR) == 2:

                bilinear_node = mat.MultiplicationNode(operands=var_factors)

                if mat.is_univariate(
                    node=bilinear_node,
                    state=self.convex_relaxation.state,
                    idx_set=idx_set,
                    dummy_element=dummy_element
                ):
                    quadratic_node = mat.ExponentiationNode(lhs_operand=var_factors[0],
                                                            rhs_operand=mat.NumericNode(2),
                                                            is_prioritized=True)
                    return self.__build_sign_conditional_convex_underestimator(ue_type=mat.BILINEAR,
                                                                               coefficient_nodes=const_factors,
                                                                               factors=[quadratic_node],
                                                                               idx_set=idx_set,
                                                                               dummy_element=dummy_element)

                else:
                    return self.__build_sign_conditional_convex_underestimator(ue_type=mat.BILINEAR,
                                                                               coefficient_nodes=const_factors,
                                                                               factors=var_factors,
                                                                               idx_set=idx_set,
                                                                               dummy_element=dummy_element)

            # trilinear (xyz)
            elif var_factor_count == 3 and var_factor_types.count(mat.LINEAR) == 3:
                return self.__build_sign_conditional_convex_underestimator(ue_type=mat.TRILINEAR,
                                                                           coefficient_nodes=const_factors,
                                                                           factors=var_factors,
                                                                           idx_set=idx_set,
                                                                           dummy_element=dummy_element)

            # fraction with nonlinear denominator (1/xy)
            elif var_factor_types.count(mat.FRACTIONAL) > 1:
                warnings.warn("Convexifier was unable to convexify term '{0}'".format(node)
                              + " with more than 1 fractional factor ")
                return node

            # fractional (1/x)
            elif var_factor_count == 1 and var_factor_types.count(mat.FRACTIONAL) == 1:
                return self.__build_sign_conditional_convex_underestimator(ue_type=mat.FRACTIONAL,
                                                                           coefficient_nodes=const_factors,
                                                                           factors=var_factors,
                                                                           idx_set=idx_set,
                                                                           dummy_element=dummy_element)

            # bilinear fractional (x/y)
            elif var_factor_count == 2 and var_factor_types.count(mat.LINEAR) == 1 \
                    and var_factor_types.count(mat.FRACTIONAL) == 1:
                return self.__build_sign_conditional_convex_underestimator(ue_type=mat.FRACTIONAL_BILINEAR,
                                                                           coefficient_nodes=const_factors,
                                                                           factors=var_factors,
                                                                           idx_set=idx_set,
                                                                           dummy_element=dummy_element)

            # trilinear fractional (xy/z)
            elif var_factor_count == 3 and var_factor_types.count(mat.LINEAR) == 2 \
                    and var_factor_types.count(mat.FRACTIONAL) == 1:
                return self.__build_sign_conditional_convex_underestimator(ue_type=mat.FRACTIONAL_TRILINEAR,
                                                                           coefficient_nodes=const_factors,
                                                                           factors=var_factors,
                                                                           idx_set=idx_set,
                                                                           dummy_element=dummy_element)

        else:

            # linear
            if mat.is_linear(node):
                return node

            type = self.__identify_node(node,
                                        idx_set=idx_set,
                                        dummy_element=dummy_element)

            # univariate nonlinear
            if type == mat.UNIVARIATE_NONLINEAR:
                return self.__build_sign_conditional_convex_underestimator(ue_type=mat.UNIVARIATE_NONLINEAR,
                                                                           coefficient_nodes=[],
                                                                           factors=[node],
                                                                           idx_set=idx_set,
                                                                           dummy_element=dummy_element)

            # general nonconvexity
            else:
                warnings.warn("Convexifier was unable to convexify term '{0}'".format(node))
                return node

    # Function Identification
    # ------------------------------------------------------------------------------------------------------------------

    def __identify_node(self,
                        node: mat.ArithmeticExpressionNode,
                        idx_set: mat.IndexingSet,
                        dummy_element: mat.Element) -> int:

        # numeric constant or parameter
        if mat.is_constant(node):
            return mat.CONSTANT

        # declared entity
        elif isinstance(node, mat.DeclaredEntityNode):
            return mat.LINEAR

        # division
        elif isinstance(node, mat.ArithmeticOperationNode) and node.operator == mat.DIVISION_OPERATOR:

            # by default, numerator is a numeric node with value 1

            # linear denominator
            if mat.is_linear(node.get_rhs_operand()):
                return mat.FRACTIONAL

            # nonlinear denominator
            else:
                return mat.GENERAL_NONCONVEX

        # exponentiation
        elif isinstance(node, mat.ArithmeticOperationNode) and node.operator == mat.EXPONENTIATION_OPERATOR:

            # univariate exponential with constant base: b^x
            if mat.is_constant(node.get_lhs_operand()):

                exponent_node = node.get_rhs_operand()

                # exponent is linear and univariate
                if mat.is_linear(exponent_node) and mat.is_univariate(
                        node=exponent_node,
                        state=self.convex_relaxation.state,
                        idx_set=idx_set,
                        dummy_element=dummy_element):
                    return mat.UNIVARIATE_NONLINEAR

                # exponent is nonlinear and/or multivariate
                else:
                    return mat.GENERAL_NONCONVEX

            # univariate exponential with constant exponent: x^c
            if mat.is_constant(node.get_rhs_operand()) and isinstance(node.get_lhs_operand(), mat.DeclaredEntityNode):

                exponent_node = node.get_rhs_operand()

                # simplify exponent node to a scalar value
                exp_val = fmr.simplify_node_to_scalar_value(
                    problem=self.convex_relaxation,
                    node=exponent_node,
                    idx_set=idx_set,
                    dummy_element=dummy_element
                )

                if exp_val is None:  # exponent value cannot be resolved as a scalar
                    return mat.GENERAL_NONCONVEX
                elif exp_val % 2 != 0:  # scalar exponent value is not even
                    return mat.GENERAL_NONCONVEX
                else:  # scalar exponent value is an even number
                    return mat.UNIVARIATE_NONLINEAR

            else:
                # all other non-factorizable exponentiation operations assumed to be general nonconvexities
                return mat.GENERAL_NONCONVEX

        # transformation
        elif isinstance(node, mat.ArithmeticTransformationNode):

            # reductive transformation
            if node.is_reductive():

                # retrieve the combined indexing set
                inner_idx_sets = node.idx_set_node.evaluate(self.convex_relaxation.state, idx_set, dummy_element)
                inner_idx_set = OrderedSet().union(*inner_idx_sets)
                idx_set = mat.cartesian_product([idx_set, inner_idx_set])
                dummy_element = node.idx_set_node.combined_dummy_element

                # summation
                if node.symbol == "sum":
                    return self.__identify_node(node.operands[0],
                                                idx_set=idx_set,
                                                dummy_element=dummy_element)

                # product, maximum, or minimum
                else:
                    return mat.GENERAL_NONCONVEX

            # non-reductive transformation
            else:

                if node.symbol in ("log", "log10", "exp", "cos", "sin", "tan"):

                    operand = node.operands[0]

                    # operand is linear and univariate
                    if mat.is_linear(operand) and mat.is_univariate(
                            node=operand,
                            state=self.convex_relaxation.state,
                            idx_set=idx_set,
                            dummy_element=dummy_element):
                        return mat.UNIVARIATE_NONLINEAR

                    # operand is nonlinear and/or multivariate
                    else:
                        return mat.GENERAL_NONCONVEX

                else:  # all other transformations are assumed to be nonconvex
                    return mat.GENERAL_NONCONVEX

        else:

            if isinstance(node, mat.ArithmeticOperationNode):

                # addition nodes should be expanded
                if node.operator == mat.ADDITION_OPERATOR:
                    raise ValueError("Convexifier encountered an illegal addition node"
                                     + " while identify the type of a term '{0}'".format(node))

                # subtraction nodes should be converted to addition nodes
                if node.operator == mat.SUBTRACTION_OPERATOR:
                    raise ValueError("Convexifier encountered an illegal subtraction node"
                                     + " while identify the type of a term '{0}'".format(node))

                # multiplication nodes should be handled in a preceding method
                if node.operator == mat.MULTIPLICATION_OPERATOR:
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

    # Factor Processing
    # ------------------------------------------------------------------------------------------------------------------

    def __process_factors(self, factors: Iterable[mat.ArithmeticExpressionNode]):

        id_sym_type_tuples = []
        id_to_factor_map = {}
        id_to_var_node_map = {}

        for factor in factors:

            # constant
            if isinstance(factor, mat.NumericNode):
                sym = str(abs(int(factor.value)))
                fcn_type = mat.CONSTANT
                id_to_var_node_map[id(factor)] = None

            # linear
            elif isinstance(factor, mat.DeclaredEntityNode):

                sym = factor.symbol
                fcn_type = mat.LINEAR
                id_to_var_node_map[id(factor)] = factor

                self.__build_bound_meta_entities(sym)

            # fractional
            elif isinstance(factor, mat.ArithmeticOperationNode) and factor.operator == mat.DIVISION_OPERATOR:

                den_node = factor.get_rhs_operand()
                if not isinstance(den_node, mat.DeclaredEntityNode):
                    raise ValueError("Convexifier encountered an unexpected operand '{0}'".format(factor)
                                     + " while building a constrained underestimator")

                sym = den_node.symbol
                fcn_type = mat.FRACTIONAL
                id_to_var_node_map[id(factor)] = den_node

                self.__build_bound_meta_entities(sym)

            else:
                raise ValueError("Convexifier encountered an unexpected operand '{0}'".format(factor)
                                 + " while building a constrained underestimator")

            id_sym_type_tuples.append((id(factor), sym, fcn_type))
            id_to_factor_map[id(factor)] = factor

        # sort operand symbols alphabetically
        id_sym_type_tuples = sorted(id_sym_type_tuples, key=lambda t: t[1] + str(t[2]))

        syms = [sym for factor_id, sym, fcn_type in id_sym_type_tuples]
        types = [fcn_type for factor_id, sym, fcn_type in id_sym_type_tuples]

        factors = [id_to_factor_map[factor_id] for factor_id, sym, fcn_type in id_sym_type_tuples]
        var_nodes = [id_to_var_node_map.get(factor_id, None) for factor_id, sym, fcn_type in id_sym_type_tuples]

        idx_nodes = []

        for var_node in var_nodes:
            if var_node is not None and var_node.idx_node is not None:
                idx_nodes.append(var_node.idx_node)
            else:
                idx_nodes.append(None)

        return factors, var_nodes, idx_nodes, syms, types

    # General Entity Construction
    # ------------------------------------------------------------------------------------------------------------------

    def __build_combined_idx_set_nodes_of_meta_entities(self,
                                                        entity_nodes: List[Optional[mat.DeclaredEntityNode]]
                                                        ) -> Tuple[Optional[mat.CompoundSetNode],
                                                                   List[Optional[OrderedSet[str]]]]:

        cmpt_set_nodes = []
        conj_operands = []

        var_unb_syms = []
        def_unb_syms = set()

        i = 0
        for entity_node in entity_nodes:

            var_unb_syms.append(None)

            if entity_node is not None:
                mv = self.convex_relaxation.meta_vars[entity_node.symbol]

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

                    var_unb_syms[i] = var_idx_set_node.get_defined_unbound_symbols()

                    # add the constraint node of the variable indexing set node to the list of constraint operands
                    if var_idx_set_node.constraint_node is not None:
                        conj_operands.append(var_idx_set_node.constraint_node)

            i += 1

        if len(cmpt_set_nodes) == 0:
            return None, var_unb_syms

        else:

            if len(conj_operands) == 0:
                con_node = None
            else:
                con_node = nb.build_conjunction_node(conj_operands)

            combined_idx_set_node = mat.CompoundSetNode(set_nodes=cmpt_set_nodes,
                                                        constraint_node=con_node)

            return combined_idx_set_node, var_unb_syms

    @staticmethod
    def __build_declared_entity_node(meta_entity: mat.MetaEntity,
                                     idx_nodes: List[Optional[mat.CompoundDummyNode]]):

        dummy_nodes = []

        # retrieve the component dummy nodes of each index node
        # deep copy the original dummy nodes in case they need to be used in the original nonconvex expression node
        for idx_node in idx_nodes:
            if idx_node is not None:
                dummy_nodes.extend(deepcopy(idx_node.component_nodes))

        # scalar underestimator
        if len(dummy_nodes) == 0:
            idx_node = None

        # indexed underestimator
        else:
            # build an index node using the supplied dummy nodes
            idx_node = mat.CompoundDummyNode(component_nodes=dummy_nodes)

        # build a declared entity node for the underestimating variable
        return mat.DeclaredEntityNode(symbol=meta_entity.get_symbol(),
                                      idx_node=idx_node)

    @staticmethod
    def __modify_idx_nodes(operands: Iterable[mat.ArithmeticExpressionNode],
                           entity_unb_syms: List[Optional[OrderedSet[str]]]):

        mod_operands = []

        for operand, unb_syms in zip(operands, entity_unb_syms):

            if unb_syms is not None:

                dummy_nodes = [mat.DummyNode(unb_sym) for unb_sym in unb_syms]
                idx_node = mat.CompoundDummyNode(component_nodes=dummy_nodes)

                # linear variable or constant
                if isinstance(operand, mat.DeclaredEntityNode):
                    operand = deepcopy(operand)
                    operand.idx_node = idx_node

                # fractional
                elif isinstance(operand, mat.ArithmeticOperationNode) and operand.operator == mat.DIVISION_OPERATOR:

                    den_node = operand.get_rhs_operand()
                    if not isinstance(den_node, mat.DeclaredEntityNode):
                        raise ValueError("Convexifier encountered an unexpected operand '{0}'".format(operand)
                                         + " while modifying the index nodes of declared entity nodes")

                    operand = deepcopy(den_node)
                    operand.idx_node = idx_node

            mod_operands.append(operand)

        return mod_operands

    # Underestimator Construction
    # ------------------------------------------------------------------------------------------------------------------

    def __build_sign_conditional_convex_underestimator(self,
                                                       ue_type: int,
                                                       coefficient_nodes: List[mat.ArithmeticExpressionNode],
                                                       factors: List[mat.ArithmeticExpressionNode],
                                                       idx_set: mat.IndexingSet,
                                                       dummy_element: mat.Element):

        if len(coefficient_nodes) == 0:
            return self.__build_and_retrieve_convex_underestimator(
                ue_type=ue_type,
                sign="P",
                factors=factors,
                is_negative=False
            )

        else:

            coefficient_node = nb.build_multiplication_node(coefficient_nodes, is_prioritized=True)

            vals = coefficient_node.evaluate(
                state=self.convex_relaxation.state,
                idx_set=idx_set,
                dummy_element=dummy_element)

            is_scalar = True
            if len(vals) > 1:
                for i in range(1, len(vals)):
                    if vals[0] != vals[i]:
                        is_scalar = False

            if is_scalar and vals[0] == 0:
                return mat.NumericNode(0)

            all_pos = all(vals > 0)
            all_neg = all(vals < 0)
            mixed_sign = not (all_pos or all_neg)

            pos_ue_node = None
            neg_ue_node = None

            if all_pos or mixed_sign:
                pos_ue_node = self.__build_and_retrieve_convex_underestimator(
                    ue_type=ue_type,
                    sign="P",
                    factors=factors,
                    is_negative=False
                )

            if all_neg or mixed_sign:
                neg_ue_node = self.__build_and_retrieve_convex_underestimator(
                    ue_type=ue_type,
                    sign="N",
                    factors=factors,
                    is_negative=True
                )

            if all_pos:
                ue_node = nb.build_multiplication_node([coefficient_node, pos_ue_node])

            elif all_neg:
                ue_node = nb.build_multiplication_node([mat.NumericNode(-1), coefficient_node, neg_ue_node])

            else:
                ue_node = nb.build_multiplication_node(
                    [
                        coefficient_node,
                        mat.ConditionalArithmeticExpressionNode(
                            operands=[
                                pos_ue_node,
                                nb.append_negative_unity_coefficient(neg_ue_node)
                            ],
                            conditions=[
                                mat.RelationalOperationNode(operator=mat.GREATER_EQUAL_INEQUALITY_OPERATOR,
                                                            lhs_operand=deepcopy(coefficient_node),
                                                            rhs_operand=mat.NumericNode(0))
                            ],
                            is_prioritized=True
                        )
                    ]
                )

            return ue_node

    def __build_and_retrieve_convex_underestimator(self,
                                                   ue_type: int,
                                                   sign: str,
                                                   factors: List[mat.ArithmeticExpressionNode],
                                                   is_negative: bool):

        factor_count = len(factors)  # count number of supplied operands

        # no operands
        if factor_count == 0:
            raise ValueError("Convexifier expected at least one arithmetic expression node"
                             + " while building and retrieving a convex underestimator")

        # single operand
        elif factor_count == 1:
            return self.__build_univariate_convex_envelope(operand=factors[0], is_negative=is_negative)

        # multiple operands
        else:

            # deterministically sort the factors and retrieve any identifying information
            (
                factors,  # sorted list of factor nodes
                var_nodes,  # sorted list of variable nodes embedded within each factor
                idx_nodes,  # sorted list of indexing nodes of the embedded variable nodes
                syms,  # sorted list of characteristic symbols for each factor
                types  # sorted list of function types for each factor
            ) = self.__process_factors(factors)

            # deterministically generate a corresponding underestimator id for the supplied operands
            ue_id = self.__generate_constrained_underestimator_id(ue_type=ue_type,
                                                                  sign=sign,
                                                                  syms=syms,
                                                                  types=types)

            # underestimator has already been constructed
            if ue_id in self.ue_meta_vars:
                ue_meta_var = self.ue_meta_vars[ue_id]

            # underestimator does not exist
            else:
                idx_set_node, var_unb_syms = self.__build_combined_idx_set_nodes_of_meta_entities(
                    entity_nodes=var_nodes)
                ue_meta_var = self.__build_constrained_underestimator_meta_variable(ue_id=ue_id,
                                                                                    idx_set_node=idx_set_node)
                mod_factors = self.__modify_idx_nodes(operands=factors,
                                                      entity_unb_syms=var_unb_syms)
                self.__build_convex_envelope_constraints(ue_id=ue_id,
                                                         ue_meta_var=ue_meta_var,
                                                         factors=mod_factors,
                                                         is_negative=is_negative)

            ue_node = self.__build_declared_entity_node(meta_entity=ue_meta_var,
                                                        idx_nodes=idx_nodes)

            return ue_node

    @staticmethod
    def __generate_constrained_underestimator_id(ue_type: int,
                                                 sign: str,
                                                 syms: Iterable[str],
                                                 types: Iterable[int]):
        return (ue_type, sign) + tuple(syms) + tuple(types)

    def __build_constrained_underestimator_meta_variable(self,
                                                         ue_id: Tuple[Union[int, str], ...],
                                                         idx_set_node: mat.CompoundSetNode = None):

        # generate unique symbol for underestimator meta-variable
        base_ue_sym = "UE_{0}_{1}_{2}".format(ue_id[0],  # underestimator type
                                              ue_id[1],  # sign
                                              ''.join([str(s)[0].upper() for s in ue_id[2:]]))
        ue_sym = self.convex_relaxation.generate_unique_symbol(base_ue_sym)

        # build meta-variable for underestimator
        ue_meta_var = eb.build_meta_var(
            problem=self.convex_relaxation,
            symbol=ue_sym,
            idx_set_node=idx_set_node)

        self.ue_meta_vars[ue_id] = ue_meta_var
        self.convex_relaxation.add_meta_variable(ue_meta_var, is_in_model=True)

        return ue_meta_var

    def __build_convex_envelope_constraints(self,
                                            ue_id: Tuple[Union[int, str], ...],
                                            ue_meta_var: mat.MetaVariable,
                                            factors: List[mat.ArithmeticExpressionNode],
                                            is_negative: bool):

        base_ue_bound_sym = "UE_ENV_{0}_{1}_{2}".format(ue_id[0],  # underestimator type
                                                        ue_id[1],  # sign
                                                        ''.join([str(s)[0].upper() for s in ue_id[2:]]))

        ue_node = nb.build_default_entity_node(ue_meta_var)

        ce_expr_nodes = self.__build_multivariate_convex_envelope(factors, is_negative=is_negative)

        for i, ce_expr_node in enumerate(ce_expr_nodes, start=1):

            ue_bound_sym = self.convex_relaxation.generate_unique_symbol("{0}_{1}".format(base_ue_bound_sym, i))

            rel_op_node = mat.RelationalOperationNode(
                operator=mat.LESS_EQUAL_INEQUALITY_OPERATOR,
                lhs_operand=nb.build_subtraction_node(ce_expr_node, deepcopy(ue_node)),
                rhs_operand=mat.NumericNode(0)
            )

            meta_con = eb.build_meta_con(
                problem=self.convex_relaxation,
                symbol=ue_bound_sym,
                idx_set_node=deepcopy(ue_meta_var.idx_set_node),
                expression=mat.Expression(rel_op_node)
            )

            self.convex_relaxation.add_meta_constraint(meta_con, is_in_model=True)

    def __build_multivariate_convex_envelope(self,
                                             factors: List[mat.ArithmeticExpressionNode],
                                             is_negative: bool
                                             ) -> List[mat.ArithmeticExpressionNode]:

        if len(factors) > 2:

            convex_envelopes = []

            for i in range(len(factors)):

                x_node = factors[i]
                xL_node = self.__build_lower_bound_node(operand=x_node, is_negative=False)
                xU_node = self.__build_upper_bound_node(operand=x_node, is_negative=False)

                if i == 0:
                    sub_factors = factors[1:]
                else:
                    sub_factors = factors[:i] + factors[i + 1:]

                s_node = nb.build_multiplication_node(sub_factors)
                sL_node = self.__build_lower_bound_node(s_node, is_negative=is_negative)
                sU_node = self.__build_upper_bound_node(s_node, is_negative=is_negative)

                ce_s_nodes = self.__build_multivariate_convex_envelope(sub_factors, is_negative=is_negative)

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

            x_node = factors[0]
            xL_node = self.__build_lower_bound_node(operand=x_node, is_negative=is_negative)
            xU_node = self.__build_upper_bound_node(operand=x_node, is_negative=is_negative)

            y_node = factors[1]
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
                        mat.NumericNode(-1),
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
                        mat.NumericNode(-1),
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
                coefficient = mat.NumericNode(-1)
            else:
                coefficient = nb.build_multiplication_node(
                    [
                        mat.NumericNode(-1),
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
        elif isinstance(operand, mat.ArithmeticOperationNode) and operand.operator == mat.DIVISION_OPERATOR:

            operand_lb_node = self.__build_lower_bound_node(operand=operand.get_rhs_operand(), is_negative=is_negative)
            operand_ub_node = self.__build_upper_bound_node(operand=operand.get_rhs_operand(), is_negative=is_negative)

            if coefficient is None:
                coefficient = mat.NumericNode(1)

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
                                nb.append_negative_unity_coefficient(deepcopy(operand.get_rhs_operand()))
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
                mat.RelationalOperationNode(operator=mat.GREATER_EQUAL_INEQUALITY_OPERATOR,
                                            lhs_operand=deepcopy(coefficient),
                                            rhs_operand=mat.NumericNode(0)),
            ]

            return mat.ConditionalArithmeticExpressionNode(
                operands=cond_operands,
                conditions=conditions,
                is_prioritized=True
            )

        # general univariate concave term
        else:
            ue_node = self.__build_convex_envelope_for_general_univariate_nonlinear_function(operand,
                                                                                             is_negative=is_negative)
            if coefficient is None:
                return ue_node
            else:
                return nb.build_multiplication_node([deepcopy(coefficient), ue_node])

    def __build_convex_envelope_for_general_univariate_nonlinear_function(self,
                                                                          unl_node: mat.ArithmeticExpressionNode,
                                                                          is_negative: bool = False):

        if isinstance(unl_node, mat.ArithmeticTransformationNode):

            # special case: -log(x) and -log10(x)
            if unl_node.symbol in ('log', 'log10') and is_negative:
                return deepcopy(unl_node)  # function is convex

            # special case: +exp(x)
            elif unl_node.symbol == "exp" and not is_negative:
                return deepcopy(unl_node)  # function is convex

            # special case: sin(x), cos(x), and tan(x)
            elif unl_node.symbol in ("sin", "cos", "tan"):
                raise NotImplementedError("Convexification logic for trigonometric functions not yet implemented")

            x_node = unl_node.operands[0]

        elif isinstance(unl_node, mat.ArithmeticOperationNode) and unl_node.operator == mat.EXPONENTIATION_OPERATOR:

            # base is constant and exponent is univariate: +/- b^x
            if mat.is_constant(unl_node.get_lhs_operand()):

                # special case: +b^x
                if not is_negative:
                    return deepcopy(unl_node)  # function is convex

                x_node = unl_node.get_rhs_operand()

            # base is univariate and exponent evaluates to an even number: +/- x^c
            elif mat.is_constant(unl_node.get_rhs_operand()):

                # special case: +x^c
                if not is_negative:
                    return deepcopy(unl_node)  # function is convex

                x_node = unl_node.get_lhs_operand()

            else:
                raise ValueError("Convexifier encountered an unexpected node '{0}'".format(unl_node)
                                 + " while constructing a convex envelope for a general univariate nonlinear term")

        else:
            raise ValueError("Convexifier encountered an unexpected node '{0}'".format(unl_node)
                             + " while constructing a convex envelope for a general univariate nonlinear term")

        var_node = mat.get_var_nodes(x_node)[0]
        var_sym = var_node.symbol

        self.__build_bound_meta_entities(var_sym)  # build bounding meta-parameters

        # generate mappings of replacement symbols
        lb_mapping = {var_sym: self.lb_params[var_sym].get_symbol()}
        ub_mapping = {var_sym: self.ub_params[var_sym].get_symbol()}

        # build elemental nodes

        uc_lb_node = deepcopy(unl_node)
        nb.replace_declared_symbols(uc_lb_node, lb_mapping)

        uc_ub_node = deepcopy(unl_node)
        nb.replace_declared_symbols(uc_ub_node, ub_mapping)

        x_lb_node = deepcopy(x_node)
        x_lb_node.is_prioritized = True
        nb.replace_declared_symbols(x_lb_node, lb_mapping)

        x_ub_node = deepcopy(x_node)
        nb.replace_declared_symbols(x_ub_node, ub_mapping)

        # build underestimator node

        num_node = nb.build_subtraction_node(uc_ub_node, deepcopy(uc_lb_node), is_prioritized=True)
        num_node.get_rhs_operand().is_prioritized = True
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

    # Bound Entity Construction
    # ------------------------------------------------------------------------------------------------------------------

    def __build_bound_meta_entities(self, var_sym: str):

        if var_sym not in self.lb_params:

            meta_var = self.convex_relaxation.meta_vars[var_sym]

            param_syms = (self.convex_relaxation.generate_unique_symbol("{0}_L".format(var_sym)),
                          self.convex_relaxation.generate_unique_symbol("{0}_U".format(var_sym)))

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

    def __build_n_linear_bound_meta_parameter(self,
                                              bound_id: tuple,
                                              factors: List[mat.ArithmeticExpressionNode],
                                              var_nodes: List[mat.DeclaredEntityNode],
                                              is_negative: bool,
                                              is_lower: bool) -> mat.MetaParameter:

        # generate unique symbol for underestimator meta-variable
        base_ue_sym = "{0}_{1}_{2}".format(''.join([str(s)[0].upper() for s in bound_id[2:]]),
                                           "N" if is_negative else "P",  # sign
                                           "L" if is_lower else "U")  # bound type
        bound_sym = self.convex_relaxation.generate_unique_symbol(base_ue_sym)

        idx_set_node, var_unb_syms = self.__build_combined_idx_set_nodes_of_meta_entities(entity_nodes=var_nodes)
        factors = self.__modify_idx_nodes(operands=factors,
                                          entity_unb_syms=var_unb_syms)

        bound_node = self.__build_n_linear_bound_node(factors=factors,
                                                      is_negative=is_negative,
                                                      is_lower=is_lower)

        mp = eb.build_meta_param(
            problem=self.convex_relaxation,
            symbol=bound_sym,
            idx_set_node=idx_set_node,
            defined_value=bound_node
        )

        self.convex_relaxation.add_meta_parameter(mp, is_in_model=True)
        self.n_linear_bound_params[bound_id] = mp

        return mp

    # Bound Node Construction
    # ------------------------------------------------------------------------------------------------------------------

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
        elif isinstance(operand, mat.ArithmeticOperationNode) and operand.operator == mat.MULTIPLICATION_OPERATOR:
            return self.__build_and_retrieve_n_linear_bound_node(operand.operands,
                                                                 is_negative=is_negative,
                                                                 is_lower=True)

        # fractional (1/x)
        elif isinstance(operand, mat.ArithmeticOperationNode) and operand.operator == mat.DIVISION_OPERATOR:
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
        elif isinstance(operand, mat.ArithmeticOperationNode) and operand.operator == mat.MULTIPLICATION_OPERATOR:
            return self.__build_and_retrieve_n_linear_bound_node(operand.operands,
                                                                 is_negative=is_negative,
                                                                 is_lower=False)

        # fractional (1/x)
        elif isinstance(operand, mat.ArithmeticOperationNode) and operand.operator == mat.DIVISION_OPERATOR:
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

    def __build_and_retrieve_n_linear_bound_node(self,
                                                 factors: List[mat.ArithmeticExpressionNode],
                                                 is_negative: bool,
                                                 is_lower: bool):

        # deterministically sort the factors and retrieve any identifying information
        (
            factors,  # sorted list of factor nodes
            var_nodes,  # sorted list of variable nodes embedded within each factor
            idx_nodes,  # sorted list of indexing nodes of the embedded variable nodes
            syms,  # sorted list of characteristic symbols for each factor
            types  # sorted list of function types for each factor
        ) = self.__process_factors(factors)

        bound_id = tuple([is_lower, is_negative]) + tuple(syms) + tuple(types)

        if bound_id in self.n_linear_bound_params:
            bound_mp = self.n_linear_bound_params[bound_id]

        else:
            bound_mp = self.__build_n_linear_bound_meta_parameter(
                bound_id=bound_id,
                factors=factors,
                var_nodes=var_nodes,
                is_negative=is_negative,
                is_lower=is_lower
            )

        return self.__build_declared_entity_node(bound_mp, idx_nodes)

    def __build_n_linear_bound_node(self,
                                    factors: List[mat.ArithmeticExpressionNode],
                                    is_negative: bool,
                                    is_lower: bool):

        lb_nodes = []
        ub_nodes = []

        for i, factor in enumerate(factors):
            if i == 0:
                lb_nodes.append(self.__build_lower_bound_node(factor, is_negative=is_negative))
                ub_nodes.append(self.__build_upper_bound_node(factor, is_negative=is_negative))
            else:
                lb_nodes.append(self.__build_lower_bound_node(factor, is_negative=False))
                ub_nodes.append(self.__build_upper_bound_node(factor, is_negative=False))

        bound_nodes = [[lb_node, ub_node] for lb_node, ub_node in zip(lb_nodes, ub_nodes)]

        return mat.ArithmeticTransformationNode(
            symbol="min" if is_lower else "max",
            operands=fmr.expand_factors_n(bound_nodes)
        )

    def __build_fractional_lower_bound_node(self, operand: mat.ArithmeticOperationNode):

        den_node = operand.get_rhs_operand()
        if not isinstance(den_node, mat.DeclaredEntityNode):
            raise ValueError("Convexifier encountered an unexpected operand '{0}'".format(operand)
                             + " while building a lower bound node")

        ub_param = self.ub_params[den_node.symbol]
        den_ub_node = self.__build_declared_bound_node(ub_param, den_node.idx_node)  # x_U

        # 1/x_U
        return nb.build_fractional_node_with_unity_numerator(denominator=den_ub_node)

    def __build_fractional_upper_bound_node(self, operand: mat.ArithmeticOperationNode):

        den_node = operand.get_rhs_operand()
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
