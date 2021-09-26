from typing import Optional

import symro.core.mat as mat
from symro.core.prob.problem import Problem
from symro.core.handlers.formulator import Formulator


# See the following paper for convexification techniques
# Adjiman, Dallwig, Floudas, and Neumaier
# A global optimization method, αBB, for general twice-differentiable constrained NLPs-I. Theoretical advances

class Convexifier:

    def __init__(self):
        self.formulator: Optional[Formulator] = None
        self.convex_relaxation: Optional[Problem] = None

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

        self.formulator = Formulator(self.convex_relaxation)

        self.formulator.substitute_defined_variables()
        self.formulator.standardize_model()

        self.__convexify_objectives()

        return self.convex_relaxation

    def __convexify_objectives(self):

        for mo in self.convex_relaxation.model_meta_objs:

            expr_node = mo.get_expression().expression_node
            if not isinstance(expr_node, mat.ArithmeticExpressionNode):
                raise ValueError("Convexifier encountered unexpected expression node"
                                 + " while convexifying objective function '{0}'".format(mo))

            terms = self.__standardize_expression(root_node=expr_node,
                                                  idx_set_node=mo.idx_set_node)

    def __standardize_expression(self,
                                 root_node: mat.ArithmeticExpressionNode,
                                 idx_set_node: mat.CompoundSetNode):

        root_node = self.formulator.reformulate_subtraction_and_unary_negation(root_node)

        terms = self.formulator.expand_multiplication(root_node)

        if idx_set_node is None:
            outer_unb_syms = None
        else:
            outer_unb_syms = idx_set_node.get_defined_unbound_symbols()

        ref_terms = []
        for term in terms:
            if isinstance(term, mat.MultiplicationNode):
                term = self.formulator.combine_summation_factor_nodes(term.operands,
                                                                      outer_unb_syms=outer_unb_syms)
                ref_terms.append(term)
            else:
                ref_terms.append(term)

        return ref_terms

    def __identify_node(self, root_node: mat.ArithmeticExpressionNode):

        if isinstance(root_node, mat.ArithmeticTransformationNode) and root_node.symbol == "sum":
            node = root_node.operands[0]
        else:
            node = root_node

        if isinstance(node, mat.MultiplicationNode):
            factors = node.operands

    def __identify_child_node(self, child_node: mat.ArithmeticExpressionNode):
        pass

