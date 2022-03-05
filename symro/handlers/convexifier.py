from copy import deepcopy
from numbers import Number
from typing import Dict, Iterable, List, Optional, Set, Tuple, Union
import warnings

import symro.mat as mat
from symro.prob.problem import Problem
import symro.handlers.nodebuilder as nb
import symro.handlers.formulator as fmr
import symro.handlers.metaentitybuilder as eb

"""
References:
-   Maranas, C.D., Floudas, C.A. Finding all solutions of nonlinearly constrained systems of equations. J Glob Optim 7, 
    143–182 (1995). https://doi.org/10.1007/BF01097059
-   C.S. Adjiman, S. Dallwig, C.A. Floudas, A. Neumaier, A global optimization method, αBB, for general 
    twice-differentiable constrained NLPs — I. Theoretical advances, Computers & Chemical Engineering, Volume 22, Issue 
    9, 1998, Pages 1137-1158, ISSN 0098-1354, https://doi.org/10.1016/S0098-1354(98)00027-1.
"""


def convexify_problem(
    problem: Problem,
    problem_symbol: str = None,
    description: str = None,
    working_dir_path: str = None,
    auxiliary: bool = False,
):
    convexifier = Convexifier()
    return convexifier.convexify_problem(
        problem=problem,
        problem_symbol=problem_symbol,
        description=description,
        working_dir_path=working_dir_path,
        auxiliary=auxiliary,
    )


def convexify_expression(
    problem: Problem,
    root_node: mat.ArithmeticExpressionNode,
    idx_set_node: mat.CompoundSetNode = None,
    auxiliary: bool = True,
) -> mat.ArithmeticExpressionNode:
    convexifier = Convexifier()
    convex_root_node = convexifier.convexify_expression(
        problem=problem,
        root_node=root_node,
        idx_set_node=idx_set_node,
        auxiliary=auxiliary,
    )
    return convex_root_node


class Convexifier:
    def __init__(self):

        self.__problem: Optional[Problem] = None

        self.lb_params: Dict[str, mat.MetaParameter] = {}
        self.ub_params: Dict[str, mat.MetaParameter] = {}

        self.ue_meta_vars: Dict[str, mat.MetaVariable] = {}
        self.ue_env_meta_cons: Dict[str, mat.MetaConstraint] = {}
        self.var_to_env_map: Dict[str, List[str]] = {}
        self.sym_to_aux_map: Dict[
            str, List[Union[mat.MetaVariable, mat.MetaConstraint]]
        ] = {}

        self.__auxiliary: bool = False

    # Core
    # ------------------------------------------------------------------------------------------------------------------

    def reset(self):
        self.__problem = None
        self.lb_params.clear()
        self.ub_params.clear()
        self.ue_meta_vars.clear()
        self.ue_env_meta_cons.clear()
        self.var_to_env_map.clear()

    def convexify_problem(
        self,
        problem: Problem,
        problem_symbol: str = None,
        description: str = None,
        working_dir_path: str = None,
        auxiliary: bool = False,
    ):

        self.__auxiliary = auxiliary

        if problem_symbol is None:
            problem_symbol = problem.symbol
        problem_symbol = problem.generate_unique_symbol(base_symbol=problem_symbol)

        if description is None:
            description = "Convex relaxation of problem {0}".format(problem.symbol)

        if working_dir_path is None:
            working_dir_path = problem.working_dir_path

        self.__problem = Problem(
            symbol=problem_symbol,
            description=description,
            working_dir_path=working_dir_path,
        )
        Problem.deepcopy(problem, self.__problem)

        fmr.substitute_defined_variables(self.__problem)
        fmr.standardize_model(self.__problem)
        self.__reformulate_nonlinear_equality_constraints()

        self.__convexify_objectives()
        self.__convexify_constraints()

        return self.__problem

    def convexify_expression(
        self,
        problem: Problem,
        root_node: mat.ArithmeticExpressionNode,
        idx_set_node: mat.CompoundSetNode = None,
        auxiliary: bool = True,
    ) -> mat.ArithmeticExpressionNode:

        self.__problem = problem
        self.__auxiliary = auxiliary

        convex_root_node, ue_meta_vars, ue_env_meta_cons = self.__convexify_expression(
            root_node=deepcopy(root_node), idx_set_node=idx_set_node
        )

        return convex_root_node

    # Problem Convexification
    # ------------------------------------------------------------------------------------------------------------------

    def __reformulate_nonlinear_equality_constraints(self):
        for mc in list(self.__problem.model_meta_cons):
            if (
                mc.constraint_type == mat.MetaConstraint.EQUALITY_TYPE
                and not mat.is_linear(mc.expression.root_node)
            ):
                fmr.convert_equality_to_inequality_constraints(self.__problem, mc)

    def __convexify_objectives(self):

        sp_meta_obj_syms = {
            sp_sym: sp.objective_symbols
            for sp_sym, sp in self.__problem.subproblems.items()
        }

        for mo in self.__problem.model_meta_objs:

            expr = mo.expression

            expr_node = expr.root_node
            if not isinstance(expr_node, mat.ArithmeticExpressionNode):
                raise ValueError(
                    "Convexifier encountered unexpected expression node"
                    + " while convexifying objective function '{0}'".format(mo)
                )

            # convexify the objective function
            convex_node, ue_meta_vars, ue_env_meta_cons = self.__convexify_meta_entity(
                mo, expr_node
            )

            expr.root_node = (
                convex_node  # swap original expression node with convexified node
            )

            # add auxiliary meta-entities to the appropriate subproblems
            self.__add_auxiliary_meta_entities_to_subproblems(
                meta_entity=mo,
                sp_meta_entity_syms=sp_meta_obj_syms,
                ue_meta_vars=ue_meta_vars,
                ue_env_meta_cons=ue_env_meta_cons,
            )

    def __convexify_constraints(self):

        sp_meta_con_syms = {
            sp_sym: sp.constraint_symbols
            for sp_sym, sp in self.__problem.subproblems.items()
        }

        for mc in self.__problem.model_meta_cons:

            if mc.constraint_type == mat.MetaConstraint.INEQUALITY_TYPE:

                expr = mc.expression

                root_node = expr.root_node
                if not isinstance(root_node, mat.RelationalOperationNode):
                    raise ValueError(
                        "Convexifier encountered unexpected expression node"
                        + " while convexifying constraint '{0}'".format(mc)
                    )

                # convexify the LHS expression of the constraint
                (
                    convex_node,
                    ue_meta_vars,
                    ue_env_meta_cons,
                ) = self.__convexify_meta_entity(mc, root_node.lhs_operand)

                root_node.lhs_operand = (
                    convex_node  # swap original expression node with convexified node
                )

                # add auxiliary meta-entities to the appropriate subproblems
                self.__add_auxiliary_meta_entities_to_subproblems(
                    meta_entity=mc,
                    sp_meta_entity_syms=sp_meta_con_syms,
                    ue_meta_vars=ue_meta_vars,
                    ue_env_meta_cons=ue_env_meta_cons,
                )

    def __convexify_meta_entity(
        self, me: mat.MetaEntity, root_node: mat.ArithmeticExpressionNode
    ):

        try:

            convex_node, ue_meta_vars, ue_env_meta_cons = self.__convexify_expression(
                root_node=root_node, idx_set_node=me.idx_set_node
            )

            aux_meta_entities: List[Union[mat.MetaVariable, mat.MetaConstraint]] = []
            aux_meta_entities.extend(ue_meta_vars)
            aux_meta_entities.extend(ue_env_meta_cons)

            if len(aux_meta_entities) > 0:
                self.sym_to_aux_map[me.symbol] = aux_meta_entities

            return convex_node, ue_meta_vars, ue_env_meta_cons

        except Exception as e:
            warnings.warn(
                "Convexifier was unable to convexify the expression"
                + " of the meta-entity '{0}':".format(me.symbol)
                + " {0}".format(e)
            )
            return root_node, [], []

    # Expression Standardization
    # ------------------------------------------------------------------------------------------------------------------

    def __standardize_expression(
        self,
        root_node: mat.ArithmeticExpressionNode,
        idx_set_node: mat.CompoundSetNode = None,
        idx_set: mat.IndexingSet = None,
        dummy_element: mat.Element = None,
    ):

        root_node = fmr.reformulate_arithmetic_conditional_expressions(root_node)
        root_node = fmr.reformulate_subtraction_and_unary_negation(root_node)

        if idx_set_node is None:
            outer_unb_syms = None
        else:
            outer_unb_syms = idx_set_node.get_defined_unbound_symbols()

        terms = fmr.expand_multiplication(
            problem=self.__problem,
            node=root_node,
            idx_set=idx_set,
            dummy_element=dummy_element,
        )

        ref_terms = []

        for term in terms:

            ref_term = fmr.combine_arithmetic_reduction_nodes(
                problem=self.__problem, node=term, outer_unb_syms=outer_unb_syms
            )

            ref_terms.append(ref_term)

        return ref_terms

    # Expression Convexification
    # ------------------------------------------------------------------------------------------------------------------

    def __convexify_expression(
        self, root_node: mat.ArithmeticExpressionNode, idx_set_node: mat.CompoundSetNode
    ) -> Tuple[
        mat.ArithmeticExpressionNode, List[mat.MetaVariable], List[mat.MetaConstraint]
    ]:

        cmpt_set_nodes = []
        cmpt_con_nodes = []
        idx_set = None
        dummy_element = None

        if idx_set_node is not None:

            cmpt_set_nodes.extend(idx_set_node.set_nodes)
            if idx_set_node.constraint_node is not None:
                cmpt_con_nodes.append(idx_set_node.constraint_node)

            idx_set = idx_set_node.evaluate(state=self.__problem.state)[0]
            dummy_element = idx_set_node.get_dummy_element(state=self.__problem.state)

        if idx_set is not None and len(idx_set) == 0:
            raise ValueError("indexing set of the supplied expression is empty")

        terms = self.__standardize_expression(
            root_node=root_node,
            idx_set_node=idx_set_node,
            idx_set=idx_set,
            dummy_element=dummy_element,
        )

        convex_terms = []  # list of convexified terms
        ue_meta_vars = []
        ue_env_meta_cons = []

        # convexify each term of the expression
        for term in terms:
            (
                convex_term,
                term_ue_meta_vars,
                term_ue_env_meta_cons,
            ) = self.__convexify_node(
                node=term,
                cmpt_set_nodes=cmpt_set_nodes,
                cmpt_con_nodes=cmpt_con_nodes,
                idx_set=idx_set,
                dummy_element=dummy_element,
            )

            convex_terms.append(convex_term)
            ue_meta_vars.extend(term_ue_meta_vars)
            ue_env_meta_cons.extend(term_ue_env_meta_cons)

        convex_root_node = nb.build_addition_node(convex_terms)

        spl_root_node = fmr.simplify(problem=self.__problem, node=convex_root_node)

        return spl_root_node, ue_meta_vars, ue_env_meta_cons

    def __convexify_node(
        self,
        node: mat.ArithmeticExpressionNode,
        cmpt_set_nodes: List[mat.SetExpressionNode],
        cmpt_con_nodes: List[mat.LogicalExpressionNode],
        idx_set: Optional[mat.IndexingSet],
        dummy_element: Optional[mat.Element],
    ) -> Tuple[
        mat.ArithmeticExpressionNode, List[mat.MetaVariable], List[mat.MetaConstraint]
    ]:

        if (
            isinstance(node, mat.ArithmeticTransformationNode)
            and node.fcn == mat.SUMMATION_FUNCTION
        ):

            # retrieve the component indexing nodes of the current scope
            cmpt_set_nodes = cmpt_set_nodes + node.idx_set_node.set_nodes
            if node.idx_set_node.constraint_node is not None:
                cmpt_con_nodes = cmpt_con_nodes + [node.idx_set_node.constraint_node]

            # retrieve the combined indexing set
            idx_sets = node.idx_set_node.generate_combined_idx_sets(
                state=self.__problem.state,
                idx_set=idx_set,
                dummy_element=dummy_element,
                can_reduce=False,
            )
            idx_set = mat.OrderedSet().union(*idx_sets)
            dummy_element = node.idx_set_node.combined_dummy_element

            # convexify operand of summation
            convexified_node, ue_meta_vars, ue_env_meta_cons = self.__convexify_node(
                node=node.operands[0],
                cmpt_set_nodes=cmpt_set_nodes,
                cmpt_con_nodes=cmpt_con_nodes,
                idx_set=idx_set,
                dummy_element=dummy_element,
            )

            node.operands[
                0
            ] = convexified_node  # replace original operand with convexified node

            return node, ue_meta_vars, ue_env_meta_cons

        elif (
            isinstance(node, mat.ArithmeticConditionalNode) and len(node.operands) == 1
        ):

            # convexify operand of summation
            convexified_node, ue_meta_vars, ue_env_meta_cons = self.__convexify_node(
                node=node.operands[0],
                cmpt_set_nodes=cmpt_set_nodes,
                cmpt_con_nodes=cmpt_con_nodes + [node.conditions[0]],
                idx_set=idx_set,
                dummy_element=dummy_element,
            )

            node.operands[
                0
            ] = convexified_node  # replace original operand with convexified node

            return node, ue_meta_vars, ue_env_meta_cons

        elif (
            isinstance(node, mat.ArithmeticOperationNode)
            and node.operator == mat.MULTIPLICATION_OPERATOR
        ):

            factors = node.operands

            is_const = [mat.is_constant(factor) for factor in factors]

            var_factors = [
                f for f, is_f_const in zip(factors, is_const) if not is_f_const
            ]
            var_factor_types = [
                self.__identify_node(f, idx_set, dummy_element) for f in var_factors
            ]
            var_factor_count = len(var_factors)

            const_factors = [
                f for f, is_f_const in zip(factors, is_const) if is_f_const
            ]

            # constant
            if var_factor_count == 0:
                return node, [], []

            # general nonconvexity
            elif mat.GENERAL_NONCONVEX in var_factor_types:
                warnings.warn(
                    "Convexifier was unable to convexify general nonconvex term '{0}'".format(
                        node
                    )
                )
                return node, [], []

            # general univariate nonlinear
            elif (
                var_factor_count == 1
                and var_factor_types[0] == mat.UNIVARIATE_NONLINEAR
            ):
                return self.__build_sign_conditional_convex_underestimator(
                    ue_type=mat.UNIVARIATE_NONLINEAR,
                    coefficient_nodes=const_factors,
                    factors=var_factors,
                    cmpt_set_nodes=cmpt_set_nodes,
                    cmpt_con_nodes=cmpt_con_nodes,
                    idx_set=idx_set,
                    dummy_element=dummy_element,
                )

            # linear (x)
            elif var_factor_count == 1 and var_factor_types.count(mat.LINEAR) == 1:
                return node, [], []

            # bilinear (xy)
            elif var_factor_count == 2 and var_factor_types.count(mat.LINEAR) == 2:

                bilinear_node = mat.MultiplicationNode(operands=var_factors)

                if mat.is_univariate(
                    root_node=bilinear_node,
                    state=self.__problem.state,
                    idx_set=idx_set,
                    dummy_element=dummy_element,
                ):
                    quadratic_node = mat.ExponentiationNode(
                        lhs_operand=var_factors[0],
                        rhs_operand=mat.NumericNode(2),
                        is_prioritized=True,
                    )
                    return self.__build_sign_conditional_convex_underestimator(
                        ue_type=mat.BILINEAR,
                        coefficient_nodes=const_factors,
                        factors=[quadratic_node],
                        cmpt_set_nodes=cmpt_set_nodes,
                        cmpt_con_nodes=cmpt_con_nodes,
                        idx_set=idx_set,
                        dummy_element=dummy_element,
                    )

                else:
                    return self.__build_sign_conditional_convex_underestimator(
                        ue_type=mat.BILINEAR,
                        coefficient_nodes=const_factors,
                        factors=var_factors,
                        cmpt_set_nodes=cmpt_set_nodes,
                        cmpt_con_nodes=cmpt_con_nodes,
                        idx_set=idx_set,
                        dummy_element=dummy_element,
                    )

            # trilinear (xyz)
            elif var_factor_count == 3 and var_factor_types.count(mat.LINEAR) == 3:
                return self.__build_sign_conditional_convex_underestimator(
                    ue_type=mat.TRILINEAR,
                    coefficient_nodes=const_factors,
                    factors=var_factors,
                    cmpt_set_nodes=cmpt_set_nodes,
                    cmpt_con_nodes=cmpt_con_nodes,
                    idx_set=idx_set,
                    dummy_element=dummy_element,
                )

            # fraction with nonlinear denominator (1/xy)
            elif var_factor_types.count(mat.FRACTIONAL) > 1:
                warnings.warn(
                    "Convexifier was unable to convexify term '{0}'".format(node)
                    + " with more than 1 fractional factor "
                )
                return node, [], []

            # fractional (1/x)
            elif var_factor_count == 1 and var_factor_types.count(mat.FRACTIONAL) == 1:
                return self.__build_sign_conditional_convex_underestimator(
                    ue_type=mat.FRACTIONAL,
                    coefficient_nodes=const_factors,
                    factors=var_factors,
                    cmpt_set_nodes=cmpt_set_nodes,
                    cmpt_con_nodes=cmpt_con_nodes,
                    idx_set=idx_set,
                    dummy_element=dummy_element,
                )

            # bilinear fractional (x/y)
            elif (
                var_factor_count == 2
                and var_factor_types.count(mat.LINEAR) == 1
                and var_factor_types.count(mat.FRACTIONAL) == 1
            ):
                return self.__build_sign_conditional_convex_underestimator(
                    ue_type=mat.FRACTIONAL_BILINEAR,
                    coefficient_nodes=const_factors,
                    factors=var_factors,
                    cmpt_set_nodes=cmpt_set_nodes,
                    cmpt_con_nodes=cmpt_con_nodes,
                    idx_set=idx_set,
                    dummy_element=dummy_element,
                )

            # trilinear fractional (xy/z)
            elif (
                var_factor_count == 3
                and var_factor_types.count(mat.LINEAR) == 2
                and var_factor_types.count(mat.FRACTIONAL) == 1
            ):
                return self.__build_sign_conditional_convex_underestimator(
                    ue_type=mat.FRACTIONAL_TRILINEAR,
                    coefficient_nodes=const_factors,
                    factors=var_factors,
                    cmpt_set_nodes=cmpt_set_nodes,
                    cmpt_con_nodes=cmpt_con_nodes,
                    idx_set=idx_set,
                    dummy_element=dummy_element,
                )

        else:

            # linear
            if mat.is_linear(node):
                return node, [], []

            type = self.__identify_node(
                node, idx_set=idx_set, dummy_element=dummy_element
            )

            # univariate nonlinear
            if type == mat.UNIVARIATE_NONLINEAR:
                return self.__build_sign_conditional_convex_underestimator(
                    ue_type=mat.UNIVARIATE_NONLINEAR,
                    coefficient_nodes=[],
                    factors=[node],
                    cmpt_set_nodes=cmpt_set_nodes,
                    cmpt_con_nodes=cmpt_con_nodes,
                    idx_set=idx_set,
                    dummy_element=dummy_element,
                )

            # general nonconvexity
            else:
                warnings.warn(
                    "Convexifier was unable to convexify term '{0}'".format(node)
                )
                return node, [], []

    # Function Identification
    # ------------------------------------------------------------------------------------------------------------------

    def __identify_node(
        self,
        node: mat.ArithmeticExpressionNode,
        idx_set: mat.IndexingSet,
        dummy_element: mat.Element,
    ) -> int:

        # numeric constant or parameter
        if mat.is_constant(node):
            return mat.CONSTANT

        # declared entity
        elif isinstance(node, mat.DeclaredEntityNode):
            return mat.LINEAR

        # division
        elif (
            isinstance(node, mat.ArithmeticOperationNode)
            and node.operator == mat.DIVISION_OPERATOR
        ):

            # by default, numerator is a numeric node with value 1

            # denominator is a linear univariate function
            if mat.is_linear(node.get_rhs_operand()) and mat.is_univariate(
                root_node=node,
                state=self.__problem.state,
                idx_set=idx_set,
                dummy_element=dummy_element,
            ):
                return mat.FRACTIONAL

            # denominator is a nonlinear and/or multivariate function
            else:
                return mat.GENERAL_NONCONVEX

        # exponentiation
        elif (
            isinstance(node, mat.ArithmeticOperationNode)
            and node.operator == mat.EXPONENTIATION_OPERATOR
        ):

            # univariate exponential with constant base: b^x
            if mat.is_constant(node.get_lhs_operand()):

                exponent_node = node.get_rhs_operand()

                # exponent is linear and univariate
                if mat.is_linear(exponent_node) and mat.is_univariate(
                    root_node=exponent_node,
                    state=self.__problem.state,
                    idx_set=idx_set,
                    dummy_element=dummy_element,
                ):
                    return mat.UNIVARIATE_NONLINEAR

                # exponent is nonlinear and/or multivariate
                else:
                    return mat.GENERAL_NONCONVEX

            # univariate exponential with constant exponent: x^c
            if mat.is_constant(node.get_rhs_operand()) and isinstance(
                node.get_lhs_operand(), mat.DeclaredEntityNode
            ):

                exponent_node = node.get_rhs_operand()

                # simplify exponent node to a scalar value
                exp_val = fmr.simplify_node_to_scalar_value(
                    problem=self.__problem,
                    node=exponent_node,
                    idx_set=idx_set,
                    dummy_element=dummy_element,
                )

                if exp_val is None:  # exponent value cannot be resolved as a scalar
                    return mat.GENERAL_NONCONVEX

                elif isinstance(exp_val, Number):

                    node.set_rhs_operand(mat.NumericNode(exp_val))

                    if isinstance(exp_val, float):
                        if exp_val.is_integer():
                            exp_val = int(exp_val)

                    if isinstance(exp_val, int):  # integer exponent
                        #  assumption: x is defined over R
                        if exp_val % 2 == 0:  # scalar exponent value is even
                            return mat.UNIVARIATE_NONLINEAR
                        else:  # scalar exponent value is odd
                            return mat.GENERAL_NONCONVEX

                    else:  # fractional exponent
                        #  assumption: x is only defined over the positive ray of R
                        return mat.UNIVARIATE_NONLINEAR

                else:
                    return mat.GENERAL_NONCONVEX

            else:
                # all other non-factorizable exponentiation operations assumed to be general nonconvexities
                return mat.GENERAL_NONCONVEX

        # transformation
        elif isinstance(node, mat.ArithmeticTransformationNode):

            # reductive transformation
            if node.is_reductive():

                # retrieve the combined indexing set
                idx_sets = node.idx_set_node.generate_combined_idx_sets(
                    state=self.__problem.state,
                    idx_set=idx_set,
                    dummy_element=dummy_element,
                    can_reduce=False,
                )
                idx_set = mat.OrderedSet().union(*idx_sets)
                dummy_element = node.idx_set_node.combined_dummy_element

                # summation
                if node.fcn == mat.SUMMATION_FUNCTION:
                    return self.__identify_node(
                        node.operands[0], idx_set=idx_set, dummy_element=dummy_element
                    )

                # product, maximum, or minimum
                else:
                    return mat.GENERAL_NONCONVEX

            # non-reductive transformation
            else:

                if node.fcn in (
                    mat.NATURAL_LOGARITHM_FUNCTION,
                    mat.BASE_10_LOGARITHM_FUNCTION,
                    mat.EXPONENTIAL_FUNCTION,
                ):

                    operand = node.operands[0]

                    # operand is linear and univariate
                    if mat.is_linear(operand) and mat.is_univariate(
                        root_node=operand,
                        state=self.__problem.state,
                        idx_set=idx_set,
                        dummy_element=dummy_element,
                    ):
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
                    raise ValueError(
                        "Convexifier encountered an illegal addition node"
                        + " while identifying the type of a term '{0}'".format(node)
                    )

                # subtraction nodes should be converted to addition nodes
                if node.operator == mat.SUBTRACTION_OPERATOR:
                    raise ValueError(
                        "Convexifier encountered an illegal subtraction node"
                        + " while identifying the type of a term '{0}'".format(node)
                    )

                # multiplication nodes should be handled in a preceding method
                if node.operator == mat.MULTIPLICATION_OPERATOR:
                    raise ValueError(
                        "Convexifier encountered an illegal multiplication node"
                        + " while identifying the type of a term '{0}'".format(node)
                    )

            raise ValueError(
                "Convexifier encountered an unexpected term '{0}'".format(node)
                + " while trying to identify its type"
            )

    @staticmethod
    def __is_negative_unity_node(node: mat.ArithmeticExpressionNode):
        if isinstance(node, mat.NumericNode) and node.value == -1:
            return True
        else:
            return False

    # Underestimator Construction
    # ------------------------------------------------------------------------------------------------------------------

    def __build_sign_conditional_convex_underestimator(
        self,
        ue_type: int,
        coefficient_nodes: List[mat.ArithmeticExpressionNode],
        factors: List[mat.ArithmeticExpressionNode],
        cmpt_set_nodes: List[mat.SetExpressionNode],
        cmpt_con_nodes: List[mat.LogicalExpressionNode],
        idx_set: mat.IndexingSet,
        dummy_element: mat.Element,
    ) -> Tuple[
        mat.ArithmeticExpressionNode, List[mat.MetaVariable], List[mat.MetaConstraint]
    ]:

        # coefficient defaults to 1
        if len(coefficient_nodes) == 0:

            (
                ue_node,
                ue_meta_var,
                ue_meta_cons,
            ) = self.__build_and_retrieve_convex_underestimator(
                ue_type=ue_type,
                factors=factors,
                is_negative=False,
                cmpt_set_nodes=cmpt_set_nodes,
                cmpt_con_nodes=cmpt_con_nodes,
            )

            if ue_meta_var is not None:
                ue_meta_vars = [ue_meta_var]
            else:
                ue_meta_vars = []

            return ue_node, ue_meta_vars, ue_meta_cons

        # coefficient may not be 1
        else:

            coefficient_node = nb.build_multiplication_node(
                coefficient_nodes, is_prioritized=True
            )

            values = coefficient_node.evaluate(
                state=self.__problem.state, idx_set=idx_set, dummy_element=dummy_element
            )

            is_scalar = True
            if len(values) > 1:
                for i in range(1, len(values)):
                    if values[0] != values[i]:
                        is_scalar = False

            if is_scalar and values[0] == 0:
                return mat.NumericNode(0), [], []

            all_pos = all(values > 0)
            all_neg = all(values < 0)
            mixed_sign = not (all_pos or all_neg)

            pos_ue_node = None
            neg_ue_node = None

            ue_meta_vars = []
            ue_env_meta_cons = []

            # build convex underestimator for the positive-signed function
            if all_pos or mixed_sign:

                (
                    pos_ue_node,
                    pos_ue_meta_var,
                    pos_ue_env_meta_cons,
                ) = self.__build_and_retrieve_convex_underestimator(
                    ue_type=ue_type,
                    factors=factors,
                    is_negative=False,
                    cmpt_set_nodes=cmpt_set_nodes,
                    cmpt_con_nodes=cmpt_con_nodes,
                )

                if pos_ue_meta_var is not None:
                    ue_meta_vars.append(pos_ue_meta_var)

                ue_env_meta_cons.extend(pos_ue_env_meta_cons)

            # build convex underestimator for the negative-signed function
            if all_neg or mixed_sign:

                (
                    neg_ue_node,
                    neg_ue_meta_var,
                    neg_ue_env_meta_cons,
                ) = self.__build_and_retrieve_convex_underestimator(
                    ue_type=ue_type,
                    factors=factors,
                    is_negative=True,
                    cmpt_set_nodes=cmpt_set_nodes,
                    cmpt_con_nodes=cmpt_con_nodes,
                )

                if neg_ue_meta_var is not None:
                    ue_meta_vars.append(neg_ue_meta_var)

                ue_env_meta_cons.extend(neg_ue_env_meta_cons)

            if all_pos:
                ue_node = nb.build_multiplication_node([coefficient_node, pos_ue_node])

            elif all_neg:
                ue_node = nb.build_multiplication_node(
                    [mat.NumericNode(-1), coefficient_node, neg_ue_node]
                )

            else:
                ue_node = nb.build_multiplication_node(
                    [
                        coefficient_node,
                        mat.ArithmeticConditionalNode(
                            operands=[
                                pos_ue_node,
                                nb.append_negative_unity_coefficient(neg_ue_node),
                            ],
                            conditions=[
                                mat.RelationalOperationNode(
                                    operator=mat.GREATER_EQUAL_INEQUALITY_OPERATOR,
                                    lhs_operand=deepcopy(coefficient_node),
                                    rhs_operand=mat.NumericNode(0),
                                )
                            ],
                            is_prioritized=True,
                        ),
                    ]
                )

            return ue_node, ue_meta_vars, ue_env_meta_cons

    def __build_and_retrieve_convex_underestimator(
        self,
        ue_type: int,
        factors: List[mat.ArithmeticExpressionNode],
        is_negative: bool,
        cmpt_set_nodes: List[mat.SetExpressionNode],
        cmpt_con_nodes: List[mat.LogicalExpressionNode],
    ) -> Tuple[
        mat.ArithmeticExpressionNode,
        Optional[mat.MetaVariable],
        List[mat.MetaConstraint],
    ]:

        factor_count = len(factors)  # count number of supplied operands

        # no operands
        if factor_count == 0:
            raise ValueError(
                "Convexifier expected at least one arithmetic expression node"
                + " while building and retrieving a convex underestimator"
            )

        # single operand
        elif factor_count == 1:
            return (
                self.__build_univariate_convex_envelope(
                    operand=factors[0], is_negative=is_negative
                ),
                None,
                [],
            )

        # multiple operands
        else:

            # deterministically sort the factors and retrieve any identifying information
            (
                factors,  # sorted list of factor nodes
                syms,  # sorted list of characteristic symbols for each factor
                types,  # sorted list of function types for each factor
            ) = self.__process_factors(factors)

            # deterministically generate a corresponding underestimator id for the supplied operands
            ue_id = self.__generate_constrained_underestimator_id(
                ue_type=ue_type, is_negative=is_negative, syms=syms, types=types
            )

            # build constraint node
            con_node = None
            if len(cmpt_con_nodes) > 0:
                con_node = nb.build_conjunction_node(cmpt_con_nodes)

            if len(cmpt_set_nodes) > 0:
                # build indexing set node
                idx_set_node = mat.CompoundSetNode(
                    set_nodes=deepcopy(cmpt_set_nodes), constraint_node=con_node
                )

            elif con_node is not None:
                # build placeholder indexing set node with constraint
                idx_set_node = mat.CompoundSetNode(
                    set_nodes=[
                        mat.OrderedSetNode(
                            start_node=mat.NumericNode(1), end_node=mat.NumericNode(1)
                        )
                    ],
                    constraint_node=con_node,
                )

            else:
                idx_set_node = None

            # build underestimator meta-variable
            ue_meta_var = self.__build_constrained_underestimator_meta_variable(
                ue_id=ue_id, idx_set_node=idx_set_node
            )

            # build envelope meta-constraints
            ue_env_meta_cons = self.__build_convex_envelope_constraints(
                ue_id=ue_id,
                ue_meta_var=ue_meta_var,
                factors=factors,
                is_negative=is_negative,
            )

            # build declared entity node for the underestimator
            ue_node = self.__build_declared_entity_node(meta_entity=ue_meta_var)

            return ue_node, ue_meta_var, ue_env_meta_cons

    def __process_factors(self, factors: Iterable[mat.ArithmeticExpressionNode]):

        id_sym_type_tuples = []
        id_to_factor_map = {}

        for factor in factors:

            # constant
            if isinstance(factor, mat.NumericNode):
                sym = str(abs(int(factor.value)))
                fcn_type = mat.CONSTANT

                factor.is_prioritized = False

            # linear
            elif isinstance(factor, mat.DeclaredEntityNode):

                sym = factor.symbol
                fcn_type = mat.LINEAR

                factor.is_prioritized = False

                self.__build_bound_meta_entities(sym)

            # fractional
            elif (
                isinstance(factor, mat.ArithmeticOperationNode)
                and factor.operator == mat.DIVISION_OPERATOR
            ):

                den_node = factor.get_rhs_operand()
                if not isinstance(den_node, mat.DeclaredEntityNode):
                    raise ValueError(
                        "Convexifier encountered an unexpected operand '{0}'".format(
                            factor
                        )
                        + " while building a constrained underestimator"
                    )

                sym = den_node.symbol
                fcn_type = mat.FRACTIONAL

                factor.is_prioritized = True

                self.__build_bound_meta_entities(sym)

            else:
                raise ValueError(
                    "Convexifier encountered an unexpected operand '{0}'".format(factor)
                    + " while building a constrained underestimator"
                )

            id_sym_type_tuples.append((id(factor), sym, fcn_type))
            id_to_factor_map[id(factor)] = factor

        # sort operand symbols alphabetically
        id_sym_type_tuples = sorted(id_sym_type_tuples, key=lambda t: t[1] + str(t[2]))

        syms = [sym for factor_id, sym, fcn_type in id_sym_type_tuples]
        types = [fcn_type for factor_id, sym, fcn_type in id_sym_type_tuples]

        factors = [
            id_to_factor_map[factor_id]
            for factor_id, sym, fcn_type in id_sym_type_tuples
        ]

        return factors, syms, types

    @staticmethod
    def __generate_constrained_underestimator_id(
        ue_type: int, is_negative: bool, syms: Iterable[str], types: Iterable[int]
    ):
        sign = "N" if is_negative else "P"
        return (ue_type, sign) + tuple(syms) + tuple(types)

    def __build_constrained_underestimator_meta_variable(
        self,
        ue_id: Tuple[Union[int, str], ...],
        idx_set_node: mat.CompoundSetNode = None,
    ):

        # generate unique symbol for underestimator meta-variable
        base_ue_sym = "UE_{0}_{1}_{2}".format(
            ue_id[0],  # underestimator type
            ue_id[1],  # sign
            "".join([str(s)[0].upper() for s in ue_id[2:]]),
        )
        ue_sym = self.__problem.generate_unique_symbol(base_ue_sym)

        # build meta-variable for underestimator
        ue_meta_var = eb.build_meta_var(
            problem=self.__problem, symbol=ue_sym, idx_set_node=idx_set_node
        )

        self.ue_meta_vars[ue_sym] = ue_meta_var
        self.__problem.add_meta_variable(ue_meta_var)

        return ue_meta_var

    def __build_convex_envelope_constraints(
        self,
        ue_id: Tuple[Union[int, str], ...],
        ue_meta_var: mat.MetaVariable,
        factors: List[mat.ArithmeticExpressionNode],
        is_negative: bool,
    ):

        ue_env_meta_cons = []

        base_ue_env_sym = "UE_ENV_{0}_{1}_{2}".format(
            ue_id[0],  # underestimator type
            ue_id[1],  # sign
            "".join([str(s)[0].upper() for s in ue_id[2:]]),
        )

        ue_node = nb.build_default_entity_node(ue_meta_var)

        ce_expr_nodes = self.__build_multivariate_convex_envelope(
            factors, is_negative=is_negative
        )

        for i, ce_expr_node in enumerate(ce_expr_nodes, start=1):

            ue_env_sym = self.__problem.generate_unique_symbol(
                "{0}_{1}".format(base_ue_env_sym, i)
            )

            rel_op_node = mat.RelationalOperationNode(
                operator=mat.LESS_EQUAL_INEQUALITY_OPERATOR,
                lhs_operand=nb.build_subtraction_node(ce_expr_node, deepcopy(ue_node)),
                rhs_operand=mat.NumericNode(0),
            )

            meta_con = eb.build_meta_con(
                problem=self.__problem,
                symbol=ue_env_sym,
                idx_set_node=deepcopy(ue_meta_var.idx_set_node),
                expression=mat.Expression(rel_op_node),
            )

            self.__problem.add_meta_constraint(meta_con)

            ue_env_meta_cons.append(meta_con)
            self.ue_env_meta_cons[ue_env_sym] = meta_con
            self.var_to_env_map.setdefault(ue_meta_var.symbol, []).append(ue_env_sym)

        return ue_env_meta_cons

    def __add_auxiliary_meta_entities_to_subproblems(
        self,
        meta_entity: mat.MetaEntity,
        sp_meta_entity_syms: Dict[str, Set[str]],
        ue_meta_vars: List[mat.MetaVariable],
        ue_env_meta_cons: List[mat.MetaConstraint],
    ):
        for sp_sym, sp in self.__problem.subproblems.items():

            in_model = meta_entity.symbol in sp_meta_entity_syms[sp_sym]
            if in_model:

                for ue_meta_var in ue_meta_vars:
                    sp.add_meta_entity(ue_meta_var, is_auxiliary=False)

                for ue_env_meta_con in ue_env_meta_cons:
                    sp.add_meta_entity(ue_env_meta_con, is_auxiliary=False)

    @staticmethod
    def __build_declared_entity_node(meta_entity: mat.MetaEntity):

        dummy_nodes = []

        # build dummy nodes for each unbound symbol in the reduced dummy element
        for unb_sym in meta_entity.idx_set_reduced_dummy_element:
            dummy_nodes.append(mat.DummyNode(unb_sym))

        # scalar
        if len(dummy_nodes) == 0:
            idx_node = None

        # indexed
        else:
            # build an index node
            idx_node = mat.CompoundDummyNode(component_nodes=dummy_nodes)

        # build a declared entity node for the underestimating variable
        return mat.DeclaredEntityNode(
            symbol=meta_entity.symbol, idx_node=idx_node, type=meta_entity.type
        )

    # Convex Envelope Construction
    # ------------------------------------------------------------------------------------------------------------------

    def __build_multivariate_convex_envelope(
        self, factors: List[mat.ArithmeticExpressionNode], is_negative: bool
    ) -> List[mat.ArithmeticExpressionNode]:

        if len(factors) > 2:

            convex_envelopes = []

            for i in range(len(factors)):

                x_node = factors[i]
                xL_node = self.__build_lower_bound_node(
                    operand=x_node, is_negative=False
                )
                xU_node = self.__build_upper_bound_node(
                    operand=x_node, is_negative=False
                )

                if i == 0:
                    sub_factors = factors[1:]
                else:
                    sub_factors = factors[:i] + factors[i + 1 :]

                s_node = nb.build_multiplication_node(sub_factors)
                sL_node = self.__build_lower_bound_node(s_node, is_negative=is_negative)
                sU_node = self.__build_upper_bound_node(s_node, is_negative=is_negative)

                ce_s_nodes = self.__build_multivariate_convex_envelope(
                    sub_factors, is_negative=is_negative
                )

                for ce_s_node in ce_s_nodes:

                    ce_xL_s_node = nb.build_multiplication_node([xL_node, ce_s_node])

                    ce_x_sL_node = self.__build_univariate_convex_envelope(
                        operand=x_node, coefficient=sL_node, is_negative=False
                    )

                    ce_xU_s_node = nb.build_multiplication_node([xU_node, ce_s_node])

                    ce_x_sU_node = self.__build_univariate_convex_envelope(
                        operand=x_node, coefficient=sU_node, is_negative=False
                    )

                    convex_envelopes.extend(
                        self.__build_bivariate_convex_envelope(
                            x_node=x_node,
                            xL_node=xL_node,
                            xU_node=xU_node,
                            y_node=ce_s_node,
                            yL_node=sL_node,
                            yU_node=sU_node,
                            ce_xL_y_node=ce_xL_s_node,
                            ce_x_yL_node=ce_x_sL_node,
                            ce_xU_y_node=ce_xU_s_node,
                            ce_x_yU_node=ce_x_sU_node,
                        )
                    )

            return convex_envelopes

        else:

            x_node = factors[0]
            xL_node = self.__build_lower_bound_node(
                operand=x_node, is_negative=is_negative
            )
            xU_node = self.__build_upper_bound_node(
                operand=x_node, is_negative=is_negative
            )

            y_node = factors[1]
            yL_node = self.__build_lower_bound_node(operand=y_node, is_negative=False)
            yU_node = self.__build_upper_bound_node(operand=y_node, is_negative=False)

            ce_xL_y_node = self.__build_univariate_convex_envelope(
                operand=y_node, coefficient=xL_node, is_negative=False
            )

            ce_x_yL_node = self.__build_univariate_convex_envelope(
                operand=x_node, coefficient=yL_node, is_negative=is_negative
            )

            ce_xU_y_node = self.__build_univariate_convex_envelope(
                operand=y_node, coefficient=xU_node, is_negative=False
            )

            ce_x_yU_node = self.__build_univariate_convex_envelope(
                operand=x_node, coefficient=yU_node, is_negative=is_negative
            )

            return self.__build_bivariate_convex_envelope(
                x_node=x_node,
                xL_node=xL_node,
                xU_node=xU_node,
                y_node=y_node,
                yL_node=yL_node,
                yU_node=yU_node,
                ce_xL_y_node=ce_xL_y_node,
                ce_x_yL_node=ce_x_yL_node,
                ce_xU_y_node=ce_xU_y_node,
                ce_x_yU_node=ce_x_yU_node,
            )

    @staticmethod
    def __build_bivariate_convex_envelope(
        x_node: mat.ArithmeticExpressionNode,
        xL_node: mat.ArithmeticExpressionNode,
        xU_node: mat.ArithmeticExpressionNode,
        y_node: mat.ArithmeticExpressionNode,
        yL_node: mat.ArithmeticExpressionNode,
        yU_node: mat.ArithmeticExpressionNode,
        ce_xL_y_node: mat.ArithmeticExpressionNode,
        ce_x_yL_node: mat.ArithmeticExpressionNode,
        ce_xU_y_node: mat.ArithmeticExpressionNode,
        ce_x_yU_node: mat.ArithmeticExpressionNode,
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
                    [mat.NumericNode(-1), deepcopy(xL_node), deepcopy(yL_node)]
                ),
            ],
            is_prioritized=True,
        )

        expr_node_2 = nb.build_addition_node(
            [
                ce_xU_y_node,
                ce_x_yU_node,
                nb.build_multiplication_node(
                    [mat.NumericNode(-1), deepcopy(xU_node), deepcopy(yU_node)]
                ),
            ],
            is_prioritized=True,
        )

        return [expr_node_1, expr_node_2]

    def __build_univariate_convex_envelope(
        self,
        operand: mat.ArithmeticExpressionNode,
        is_negative: bool,
        coefficient: mat.ArithmeticExpressionNode = None,
    ):

        can_append_coeff = True

        # constant (c) or linear (x)
        if isinstance(operand, mat.NumericNode) or isinstance(
            operand, mat.DeclaredEntityNode
        ):
            ue_node = deepcopy(operand)

        # fractional (1/x)
        elif (
            isinstance(operand, mat.ArithmeticOperationNode)
            and operand.operator == mat.DIVISION_OPERATOR
        ):
            can_append_coeff = False
            ue_node = self.__build_convex_envelope_for_univariate_fractional_function(
                operand=operand, is_negative=is_negative, coefficient=coefficient
            )

        # general univariate concave term
        else:
            ue_node = (
                self.__build_convex_envelope_for_general_univariate_nonlinear_function(
                    unl_node=operand, is_negative=is_negative
                )
            )

        if is_negative and can_append_coeff:
            if coefficient is None:
                coefficient = mat.NumericNode(-1)
            else:
                coefficient = nb.build_multiplication_node(
                    [mat.NumericNode(-1), coefficient]
                )

        if coefficient is None or not can_append_coeff:
            return ue_node
        else:
            return nb.build_multiplication_node([deepcopy(coefficient), ue_node])

    def __build_convex_envelope_for_univariate_fractional_function(
        self,
        operand: mat.ArithmeticOperationNode,
        is_negative: bool,
        coefficient: mat.ArithmeticExpressionNode = None,
    ):

        den_node = deepcopy(operand.get_rhs_operand())
        operand_lb_node = self.__build_lower_bound_node(
            operand=operand.get_rhs_operand(), is_negative=False
        )
        operand_ub_node = self.__build_upper_bound_node(
            operand=operand.get_rhs_operand(), is_negative=False
        )

        if coefficient is None:
            coefficient = mat.NumericNode(1)

        cdn_operand_1 = mat.MultiplicationNode(
            [deepcopy(coefficient), deepcopy(operand)]
        )

        cdn_operand_2 = mat.MultiplicationNode(
            [
                deepcopy(coefficient),
                nb.build_addition_node(
                    [
                        deepcopy(operand_lb_node),
                        deepcopy(operand_ub_node),
                        den_node
                        if is_negative
                        else nb.append_negative_unity_coefficient(den_node),
                    ],
                    is_prioritized=True,
                ),
                nb.build_fractional_node_with_unity_numerator(
                    deepcopy(operand_lb_node), is_prioritized=True
                ),
                nb.build_fractional_node_with_unity_numerator(
                    deepcopy(operand_ub_node), is_prioritized=True
                ),
            ]
        )

        if is_negative:
            cdn_operand_1.operands.insert(0, mat.NumericNode(-1))
            cdn_operand_2.operands.insert(0, mat.NumericNode(-1))

        if is_negative:
            cdn_rel_operator = mat.LESS_INEQUALITY_OPERATOR
        else:
            cdn_rel_operator = mat.GREATER_INEQUALITY_OPERATOR

        conditions = [
            mat.RelationalOperationNode(
                operator=cdn_rel_operator,
                lhs_operand=deepcopy(coefficient),
                rhs_operand=mat.NumericNode(0),
            ),
        ]

        return mat.ArithmeticConditionalNode(
            operands=[cdn_operand_1, cdn_operand_2],
            conditions=conditions,
            is_prioritized=True,
        )

    def __build_convex_envelope_for_general_univariate_nonlinear_function(
        self, unl_node: mat.ArithmeticExpressionNode, is_negative: bool = False
    ):

        if isinstance(unl_node, mat.ArithmeticTransformationNode):

            # special case: -log(x) and -log10(x)
            if (
                unl_node.fcn
                in (mat.NATURAL_LOGARITHM_FUNCTION, mat.BASE_10_LOGARITHM_FUNCTION)
                and is_negative
            ):
                return deepcopy(unl_node)  # function is convex

            # special case: +exp(x)
            elif unl_node.fcn == mat.EXPONENTIAL_FUNCTION and not is_negative:
                return deepcopy(unl_node)  # function is convex

            # special case: sin(x), cos(x), and tan(x)
            elif unl_node.fcn in (
                mat.SINE_FUNCTION,
                mat.COSINE_FUNCTION,
                mat.TANGENT_FUNCTION,
            ):
                raise NotImplementedError(
                    "Convexification logic for trigonometric functions not yet implemented"
                )

            x_node = unl_node.operands[0]

        elif (
            isinstance(unl_node, mat.ArithmeticOperationNode)
            and unl_node.operator == mat.EXPONENTIATION_OPERATOR
        ):

            unl_lhs_operand = unl_node.get_lhs_operand()
            unl_rhs_operand = unl_node.get_rhs_operand()

            # base is constant and exponent is univariate: +/- b^x
            if mat.is_constant(unl_lhs_operand):

                # special case: +b^x
                if not is_negative:
                    return deepcopy(unl_node)  # function is convex

                x_node = unl_node.get_rhs_operand()

            # base is univariate and exponent is constant: +/- x^c
            elif mat.is_constant(unl_rhs_operand):

                if not isinstance(unl_rhs_operand, mat.NumericNode):
                    raise ValueError(
                        "Convexifier encountered an unexpected node '{0}'".format(
                            unl_node
                        )
                        + " while constructing a convex envelope for a general univariate nonlinear term"
                    )

                exp_val = unl_rhs_operand.value

                # special case: +x^c where c is even
                if not is_negative and exp_val % 2 == 0:
                    return deepcopy(unl_node)  # function is convex

                # special case: -x^c where c is fractional
                elif is_negative and exp_val % 1 != 0:
                    return deepcopy(unl_node)  # function is convex

                x_node = unl_node.get_lhs_operand()

            else:
                raise ValueError(
                    "Convexifier encountered an unexpected node '{0}'".format(unl_node)
                    + " while constructing a convex envelope for a general univariate nonlinear term"
                )

        else:
            raise ValueError(
                "Convexifier encountered an unexpected node '{0}'".format(unl_node)
                + " while constructing a convex envelope for a general univariate nonlinear term"
            )

        var_node = mat.get_var_nodes(x_node)[0]
        var_sym = var_node.symbol

        self.__build_bound_meta_entities(var_sym)  # build bounding meta-parameters

        # generate mappings of replacement symbols
        lb_mapping = {var_sym: self.lb_params[var_sym].symbol}
        ub_mapping = {var_sym: self.ub_params[var_sym].symbol}
        type_mapping = {var_sym: mat.PARAM_TYPE}

        # build elemental nodes

        uc_lb_node = deepcopy(unl_node)
        nb.replace_declared_symbols(uc_lb_node, lb_mapping, type_mapping)

        uc_ub_node = deepcopy(unl_node)
        nb.replace_declared_symbols(uc_ub_node, ub_mapping, type_mapping)

        x_lb_node = deepcopy(x_node)
        x_lb_node.is_prioritized = True
        nb.replace_declared_symbols(x_lb_node, lb_mapping, type_mapping)

        x_ub_node = deepcopy(x_node)
        nb.replace_declared_symbols(x_ub_node, ub_mapping, type_mapping)

        # build underestimator node

        num_node = nb.build_subtraction_node(
            uc_ub_node, deepcopy(uc_lb_node), is_prioritized=True
        )
        num_node.get_rhs_operand().is_prioritized = True
        den_node = nb.build_subtraction_node(x_ub_node, x_lb_node, is_prioritized=True)

        ue_node = nb.build_addition_node(
            [
                uc_lb_node,
                nb.build_multiplication_node(
                    [
                        nb.build_fractional_node(
                            num_node, den_node, is_prioritized=True
                        ),
                        nb.build_subtraction_node(
                            deepcopy(x_node), deepcopy(x_lb_node), is_prioritized=True
                        ),
                    ]
                ),
            ],
            is_prioritized=True,
        )

        x_lb_node.is_prioritized = False

        return mat.ArithmeticConditionalNode(
            operands=[deepcopy(x_node), ue_node],
            conditions=[
                mat.RelationalOperationNode(
                    operator=mat.EQUALITY_OPERATOR,
                    lhs_operand=deepcopy(x_lb_node),
                    rhs_operand=deepcopy(x_ub_node),
                )
            ],
            is_prioritized=True,
        )

    # Bound Entity Construction
    # ------------------------------------------------------------------------------------------------------------------

    def __build_bound_meta_entities(self, var_sym: str):

        if var_sym not in self.lb_params:

            meta_var = self.__problem.meta_vars[var_sym]

            param_syms = (
                self.__problem.generate_unique_symbol("{0}_L".format(var_sym)),
                self.__problem.generate_unique_symbol("{0}_U".format(var_sym)),
            )

            default_value_nodes = (
                deepcopy(meta_var.lower_bound_node),
                deepcopy(meta_var.upper_bound_node),
            )

            meta_params = []

            for param_sym, default_value_node in zip(param_syms, default_value_nodes):

                idx_set_node = None
                if meta_var.idx_set_node is not None:
                    idx_set_node = deepcopy(meta_var.idx_set_node)

                meta_param = eb.build_meta_param(
                    problem=self.__problem,
                    symbol=param_sym,
                    idx_set_node=idx_set_node,
                    default_value=default_value_node,
                )

                meta_params.append(meta_param)
                self.__problem.add_meta_parameter(meta_param)
                self.__build_state_parameter(meta_param)

            self.lb_params[var_sym] = meta_params[0]
            self.ub_params[var_sym] = meta_params[1]

    def __build_state_parameter(self, meta_param: mat.MetaParameter):

        if meta_param.idx_set_node is None:
            value = meta_param.default_value_node.evaluate(state=self.__problem.state)[
                0
            ]
            self.__problem.state.add_parameter(symbol=meta_param.symbol, value=value)

        else:
            idx_set: mat.IndexingSet = meta_param.idx_set_node.evaluate(
                state=self.__problem.state
            )[0]
            values = meta_param.default_value_node.evaluate(
                state=self.__problem.state,
                idx_set=idx_set,
                dummy_element=tuple(meta_param.idx_set_dummy_element),
            )
            for idx, value in zip(idx_set, values):
                self.__problem.state.add_parameter(
                    symbol=meta_param.symbol, idx=idx, value=value
                )

    # Bound Node Construction
    # ------------------------------------------------------------------------------------------------------------------

    def __build_lower_bound_node(
        self, operand: mat.ArithmeticExpressionNode, is_negative: bool
    ):

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
        elif (
            isinstance(operand, mat.ArithmeticOperationNode)
            and operand.operator == mat.MULTIPLICATION_OPERATOR
        ):
            return self.__build_n_linear_bound_node(
                factors=operand.operands, is_negative=is_negative, is_lower=True
            )

        # fractional (1/x)
        elif (
            isinstance(operand, mat.ArithmeticOperationNode)
            and operand.operator == mat.DIVISION_OPERATOR
        ):
            if not is_negative:  # 1/x_U
                return self.__build_fractional_lower_bound_node(operand)
            else:  # -1/x_L
                return nb.append_negative_unity_coefficient(
                    self.__build_fractional_upper_bound_node(operand)
                )

        else:
            raise ValueError(
                "Convexifier encountered an unexpected operand '{0}'".format(operand)
                + " while building a lower bound node"
            )

    def __build_upper_bound_node(
        self, operand: mat.ArithmeticExpressionNode, is_negative: bool
    ):

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
        elif (
            isinstance(operand, mat.ArithmeticOperationNode)
            and operand.operator == mat.MULTIPLICATION_OPERATOR
        ):
            return self.__build_n_linear_bound_node(
                factors=operand.operands, is_negative=is_negative, is_lower=False
            )

        # fractional (1/x)
        elif (
            isinstance(operand, mat.ArithmeticOperationNode)
            and operand.operator == mat.DIVISION_OPERATOR
        ):
            if not is_negative:  # 1/x_L
                return self.__build_fractional_upper_bound_node(operand)
            else:  # -1/x_U
                return nb.append_negative_unity_coefficient(
                    self.__build_fractional_lower_bound_node(operand)
                )

        else:
            raise ValueError(
                "Convexifier encountered an unexpected operand '{0}'".format(operand)
                + " while building an upper bound node"
            )

    def __build_linear_lower_bound_node(self, operand: mat.DeclaredEntityNode):
        lb_param = self.lb_params[operand.symbol]
        return self.__build_declared_bound_node(lb_param, operand.idx_node)  # x_L

    def __build_linear_upper_bound_node(self, operand: mat.DeclaredEntityNode):
        ub_param = self.ub_params[operand.symbol]
        return self.__build_declared_bound_node(ub_param, operand.idx_node)  # x_U

    def __build_n_linear_bound_node(
        self,
        factors: List[mat.ArithmeticExpressionNode],
        is_negative: bool,
        is_lower: bool,
    ):

        lb_nodes = []
        ub_nodes = []

        for i, factor in enumerate(factors):
            if i == 0:
                lb_nodes.append(
                    self.__build_lower_bound_node(factor, is_negative=is_negative)
                )
                ub_nodes.append(
                    self.__build_upper_bound_node(factor, is_negative=is_negative)
                )
            else:
                lb_nodes.append(
                    self.__build_lower_bound_node(factor, is_negative=False)
                )
                ub_nodes.append(
                    self.__build_upper_bound_node(factor, is_negative=False)
                )

        bound_nodes = [
            [lb_node, ub_node] for lb_node, ub_node in zip(lb_nodes, ub_nodes)
        ]

        bound_node = mat.ArithmeticTransformationNode(
            fcn=mat.MINIMUM_FUNCTION if is_lower else mat.MAXIMUM_FUNCTION,
            operands=fmr.expand_factors_n(bound_nodes),
        )

        return bound_node

    def __build_fractional_lower_bound_node(self, operand: mat.ArithmeticOperationNode):

        den_node = operand.get_rhs_operand()
        if not isinstance(den_node, mat.DeclaredEntityNode):
            raise ValueError(
                "Convexifier encountered an unexpected operand '{0}'".format(operand)
                + " while building a lower bound node"
            )

        ub_param = self.ub_params[den_node.symbol]
        den_ub_node = self.__build_declared_bound_node(
            ub_param, den_node.idx_node
        )  # x_U

        # 1/x_U
        return nb.build_fractional_node_with_unity_numerator(denominator=den_ub_node)

    def __build_fractional_upper_bound_node(self, operand: mat.ArithmeticOperationNode):

        den_node = operand.get_rhs_operand()
        if not isinstance(den_node, mat.DeclaredEntityNode):
            raise ValueError(
                "Convexifier encountered an unexpected operand '{0}'".format(operand)
                + " while building an upper bound node"
            )

        lb_param = self.lb_params[den_node.symbol]
        den_lb_node = self.__build_declared_bound_node(
            lb_param, den_node.idx_node
        )  # x_L

        # 1/x_L
        return nb.build_fractional_node_with_unity_numerator(denominator=den_lb_node)

    @staticmethod
    def __build_declared_bound_node(
        meta_param: mat.MetaParameter, idx_node: Optional[mat.CompoundDummyNode]
    ):
        return mat.DeclaredEntityNode(
            symbol=meta_param.symbol, idx_node=deepcopy(idx_node), type=mat.PARAM_TYPE
        )
