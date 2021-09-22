from copy import deepcopy
import numpy as np
from queue import Queue
from typing import Callable, Dict, List, Optional, Tuple, Union

from ordered_set import OrderedSet

import symro.core.constants as const

import symro.core.mat as mat

from symro.core.prob.problem import Problem, BaseProblem
import symro.core.prob.statement as stm

from symro.core.handlers.formulator import Formulator
from symro.core.handlers.nodebuilder import NodeBuilder
from symro.core.handlers.entitybuilder import EntityBuilder
from symro.core.handlers.scriptbuilder import ScriptBuilder

from symro.core.parsing.amplparser import AMPLParser

from symro.core.algo.gbd import GBDProblem, GBDSubproblemContainer


# TODO: create FormulationError exception class and raise it whenever an incorrect formulation is encountered
class GBDProblemBuilder:

    CONST_NODE = 0
    PURE_X_NODE = 1
    PURE_Y_NODE = 2
    MIXED_NODE = 3

    COMP_STATUSES = [PURE_Y_NODE, MIXED_NODE]

    # Construction
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, problem: Problem):

        # Problems
        self.problem: Problem = problem
        self.gbd_problem: Optional[GBDProblem] = None

        # Expression Nodes
        self.comp_obj_expression: Optional[mat.ArithmeticExpressionNode] = None
        self.comp_con_expressions: Dict[str, mat.ArithmeticExpressionNode] = {}

        # Entities
        self.comp_vars: Dict[str, mat.Variable] = {}

        # Handlers
        self.formulator: Optional[Formulator] = None
        self.node_builder: Optional[NodeBuilder] = None
        self.entity_builder: Optional[EntityBuilder] = None

        # Flags
        self.__is_primal_sp_obj_comp: Dict[str, bool] = {}

        # Parameters
        self.__first_dim_index: int = 0

    # Core
    # ------------------------------------------------------------------------------------------------------------------

    def build_gbd_problem(self,
                          problem: Problem,
                          comp_var_defs: List[str],
                          mp_symbol: str = None,
                          primal_sp_symbol: str = None,
                          fbl_sp_symbol: str = None,
                          primal_sp_obj_symbol: str = None,
                          init_lb: float = -np.inf,
                          working_dir_path: str = None) -> GBDProblem:

        # build GBD problem
        self.gbd_problem = GBDProblem(problem=problem,
                                      mp_symbol=mp_symbol,
                                      default_primal_sp_symbol=primal_sp_symbol,
                                      default_fbl_sp_symbol=fbl_sp_symbol,
                                      primal_sp_obj_sym=primal_sp_obj_symbol,
                                      working_dir_path=working_dir_path)

        # standardize model
        self.formulator = Formulator(self.gbd_problem)
        self.formulator.substitute_defined_variables()
        self.gbd_problem.origin_to_std_con_map = self.formulator.standardize_model()

        # retrieve or construct complicating meta-variables
        self.__collect_complicating_meta_variables(comp_var_defs)

        # build master problem constructs
        self.gbd_problem.build_mp_constructs(init_lb)

        return self.gbd_problem

    def build_gbd_constructs(self) -> GBDProblem:

        self.node_builder = NodeBuilder(self.gbd_problem)
        self.entity_builder = EntityBuilder(self.gbd_problem)

        # --- Primal Problem ---
        self.__categorize_constraints()
        if len(self.gbd_problem.primal_sps) == 0:
            self.__build_default_primal_sp()
        self.__modify_primal_sp_objs()
        self.__categorize_primal_objective()

        # --- Feasibility Problem ---
        self.__build_slack_variables_and_slackened_constraints()
        self.__build_fbl_sps()

        # --- Subproblem Containers ---
        self.__build_subproblem_containers()

        # --- Master Problem ---
        self.__build_complicating_storage_params()
        self.__build_master_objective()
        self.__build_auxiliary_constructs()
        self.__build_cuts()
        self.__build_mp()

        # --- Script ---
        self.__build_and_write_script()

        return self.gbd_problem

    # Complicating Variables
    # ------------------------------------------------------------------------------------------------------------------

    def __collect_complicating_meta_variables(self, comp_var_defs: List[str]):
        for i, comp_var_def in enumerate(comp_var_defs, start=1):
            comp_var_id = "y_{0}".format(i)
            meta_vars = self.__retrieve_meta_entity_from_definition(comp_var_def)
            self.gbd_problem.comp_meta_vars[comp_var_id] = meta_vars[0]

    # Primal Objective Function
    # ------------------------------------------------------------------------------------------------------------------

    def __modify_primal_sp_objs(self):

        for primal_sp in self.gbd_problem.primal_sps:

            primal_sp_obj = primal_sp.model_meta_objs[0]

            idx_meta_sets = primal_sp_obj.idx_meta_sets
            if len(idx_meta_sets) > 0:  # modify the meta-objective if it is indexed

                # generate a symbol for the new meta-objective
                primal_sp_obj_sym = self.gbd_problem.generate_unique_symbol("{0}_{1}".format(primal_sp_obj.symbol,
                                                                                             primal_sp.symbol))

                # retrieve the symbols of the subproblem indexing meta-sets
                sp_idx_meta_set_syms = [s for s in self.gbd_problem.idx_meta_sets.keys()]

                # identify and collect indexing meta-sets of the primal subproblem objective function
                # that are not decomposed
                sum_idx_meta_sets = []
                obj_idx_meta_sets = []
                for ms in idx_meta_sets:
                    if ms.symbol not in sp_idx_meta_set_syms:
                        sum_idx_meta_sets.append(ms)
                    else:
                        obj_idx_meta_sets.append(ms)

                # remove indexing sets that are not decomposed
                if len(sum_idx_meta_sets) > 0:

                    # build the indexing set node of a reductive summation operation
                    sum_idx_set_node = self.node_builder.build_idx_set_node(sum_idx_meta_sets)

                    # retrieve the root expression node of the original objective function
                    obj_exp_node = primal_sp_obj.expression.expression_node
                    if not isinstance(obj_exp_node, mat.ArithmeticExpressionNode):
                        raise ValueError("GBD problem builder expected an arithmetic expression node"
                                         + " while generating a new primal subproblem meta-objective")

                    # embed the original function in a reductive summation operation
                    sum_node = mat.ArithmeticTransformationNode(id=self.gbd_problem.generate_free_node_id(),
                                                                symbol="sum",
                                                                idx_set_node=sum_idx_set_node,
                                                                operands=obj_exp_node)

                    expr_node = sum_node

                else:
                    expr_node = primal_sp_obj.expression.expression_node

                # retrieve default dummy symbols of the subproblem indexing sets
                default_dummy_syms = {sym: tuple(ms.dummy_symbols)
                                      for sym, ms in self.gbd_problem.idx_meta_sets.items()}

                # build an indexing set node for the new meta-objective
                obj_idx_set_node = self.node_builder.build_idx_set_node(obj_idx_meta_sets,
                                                                        custom_dummy_syms=default_dummy_syms)

                # replace dummy symbols in expression with default dummy symbols
                self.node_builder.replace_unbound_symbols(expr_node, mapping=self.node_builder.unb_sym_map)

                # build meta-objective expression
                expression = mat.Expression(expr_node)

                # build a new meta-objective
                primal_sp_obj = mat.MetaObjective(symbol=primal_sp_obj_sym,
                                                  alias=primal_sp_obj.alias,
                                                  idx_meta_sets=obj_idx_meta_sets,
                                                  idx_set_node=obj_idx_set_node,
                                                  direction=primal_sp_obj.direction,
                                                  expression=expression)

                # add the new meta-objective to the problem
                self.gbd_problem.add_meta_objective(primal_sp_obj)
                primal_sp.model_meta_objs.clear()
                primal_sp.model_meta_objs.append(primal_sp_obj)

    # Expression Categorization
    # ------------------------------------------------------------------------------------------------------------------

    def __categorize_primal_objective(self):

        comp_meta_vars = self.gbd_problem.comp_meta_vars

        for primal_sp in self.gbd_problem.primal_sps:

            meta_obj = primal_sp.model_meta_objs[0]

            comp_vars, vars = self.__retrieve_complicating_variable_entities(meta_obj, comp_meta_vars)

            # Categorize constraint
            if len(comp_vars) > 0:
                self.__is_primal_sp_obj_comp[meta_obj.symbol] = True
            else:
                self.__is_primal_sp_obj_comp[meta_obj.symbol] = False

    def __categorize_constraints(self):

        comp_meta_vars = self.gbd_problem.comp_meta_vars

        for meta_con in self.gbd_problem.model_meta_cons:

            con_sym = meta_con.symbol  # retrieve meta-constraint symbol

            # Retrieve symbols of all variables in the constraint (fast)
            var_syms = meta_con.expression.get_var_syms()

            # Check if any of the variables might be complicating
            can_be_complicating = False
            for _, comp_meta_var in comp_meta_vars.items():
                if comp_meta_var.symbol in var_syms:
                    can_be_complicating = True
                    break

            # Constraint is non-complicating
            if not can_be_complicating:
                self.gbd_problem.non_comp_cons[con_sym] = meta_con

            # Constraint might be complicating
            else:

                # Retrieve all variable instances (slow)
                comp_vars, vars = self.__retrieve_complicating_variable_entities(meta_con, comp_meta_vars)

                if len(comp_vars) > 0:
                    if len(vars) == len(comp_vars):
                        self.gbd_problem.pure_comp_cons[con_sym] = meta_con
                    else:
                        self.gbd_problem.mixed_comp_cons[con_sym] = meta_con
                else:
                    self.gbd_problem.non_comp_cons[con_sym] = meta_con

    def __retrieve_complicating_variable_entities(self,
                                                  meta_entity: Union[mat.MetaObjective, mat.MetaConstraint],
                                                  comp_meta_vars: Dict[str, mat.MetaVariable]):

        # Retrieve expression node of the meta-entity
        expr_node = meta_entity.expression.expression_node
        if (not isinstance(expr_node, mat.RelationalOperationNode)
                and not isinstance(expr_node, mat.ArithmeticExpressionNode)):
            raise ValueError("GBD problem builder expected a relational operation node or an arithmetic expression node"
                             + " while categorizing a meta-entity '{0}'".format(meta_entity))

        # Retrieve the indexing set of the constraint
        if meta_entity.idx_set_node is not None:
            idx_set = meta_entity.idx_set_node.evaluate(self.gbd_problem.state)[0]
            dummy_syms = meta_entity.idx_set_node.get_dummy_elements(self.gbd_problem.state)
        else:
            idx_set = None
            dummy_syms = None

        # Retrieve all variables used in the constraint
        entities = expr_node.collect_declared_entities(self.gbd_problem.state, idx_set, dummy_syms)
        vars = {k: v for k, v in entities.items() if isinstance(v, mat.Variable)}  # filter out parameters

        # Identify complicating variables
        vars: Dict[str, mat.Variable]
        comp_vars = {}
        for var_id, var in vars.items():
            for comp_meta_var_id, comp_meta_var in comp_meta_vars.items():
                if comp_meta_var.is_owner(var, self.gbd_problem.state):
                    comp_vars[var_id] = var

        self.comp_vars.update(comp_vars)

        return comp_vars, vars

    # Primal Subproblems
    # ------------------------------------------------------------------------------------------------------------------

    def build_defined_primal_sp(self,
                                sp_sym: str,
                                var_defs: List[str] = None,
                                obj_def: str = None,
                                con_defs: List[str] = None):

        # retrieve existing subproblem
        if obj_def is None and var_defs is None:
            if sp_sym in self.gbd_problem.subproblems:
                self.gbd_problem.primal_sps.append(self.gbd_problem.subproblems[sp_sym])
            else:
                raise ValueError("GBD problem builder encountered an undefined subproblem symbol")

        # build new subproblem
        else:

            meta_vars = []
            for var_def in var_defs:
                meta_vars.extend(self.__retrieve_meta_entity_from_definition(var_def))

            meta_obj = self.__retrieve_meta_entity_from_definition(obj_def)[0]

            meta_cons = []
            for con_def in con_defs:
                meta_cons.extend(self.__retrieve_meta_entity_from_definition(con_def))

            sp = BaseProblem(symbol=sp_sym,
                             description="Primal subproblem")
            sp.model_meta_vars = meta_vars
            sp.model_meta_objs.append(meta_obj)
            sp.model_meta_cons = meta_cons

            self.gbd_problem.add_subproblem(sp)

    def __build_default_primal_sp(self):

        # retrieve meta-variables
        meta_vars = [mv for mv in self.gbd_problem.model_meta_vars]

        # retrieve meta-objective

        primal_sp_obj_sym = self.gbd_problem.primal_sp_obj_sym

        if primal_sp_obj_sym is not None:
            primal_sp_meta_obj = self.gbd_problem.meta_objs[primal_sp_obj_sym]  # retrieve primal meta-objective

        # elicit a suitable primal meta-objective
        else:
            primal_sp_meta_obj = None
            idx_meta_sets = self.gbd_problem.get_idx_meta_sets()
            for meta_obj in self.gbd_problem.model_meta_objs:
                if idx_meta_sets is not None:
                    if all([meta_obj.is_indexed_with(idx_meta_set) for idx_meta_set in idx_meta_sets]):
                        primal_sp_meta_obj = meta_obj
                        break

        if primal_sp_meta_obj is None:
            raise ValueError("GBD problem builder could not identify a suitable objective function"
                             " for the primal subproblem(s)")

        # retrieve meta-constraints
        meta_cons = [mc for _, mc in self.gbd_problem.non_comp_cons.items()]
        meta_cons.extend([mc for _, mc in self.gbd_problem.mixed_comp_cons.items()])

        # build subproblem
        primal_sp = BaseProblem(symbol=self.gbd_problem.default_primal_sp_sym,
                                description="Primal subproblem")
        primal_sp.model_meta_vars = meta_vars
        primal_sp.model_meta_objs.append(primal_sp_meta_obj)
        primal_sp.model_meta_cons = meta_cons

        self.gbd_problem.add_subproblem(primal_sp)
        self.gbd_problem.primal_sps.append(primal_sp)

    # Feasibility Subproblems
    # ------------------------------------------------------------------------------------------------------------------

    def __build_slack_variables_and_slackened_constraints(self):

        meta_cons = list(self.gbd_problem.mixed_comp_cons.values()) + list(self.gbd_problem.non_comp_cons.values())

        for meta_con in meta_cons:

            # formulate modified constraint with slack variables
            sl_result = self.formulator.formulate_slackened_constraint(meta_con)
            sl_meta_var_list, sl_meta_con = sl_result

            # add slack meta-variables to the problem
            for mv in sl_meta_var_list:
                self.gbd_problem.slack_vars[mv.symbol] = mv
                self.gbd_problem.add_meta_variable(mv, is_in_model=False)

            # add slackened meta-constraint to the problem
            if len(sl_meta_var_list) > 0:
                self.gbd_problem.sl_fbl_cons[sl_meta_con.symbol] = sl_meta_con
                self.gbd_problem.add_meta_constraint(sl_meta_con, is_in_model=False)

            # add slack meta-variables and slackened meta-constraint to mapping
            if len(sl_meta_var_list) > 0:
                self.gbd_problem.std_to_sl_map[meta_con.symbol] = sl_result

    def __build_slack_minimization_objective(self,
                                             obj_sym: str,
                                             sl_meta_vars: List[mat.MetaVariable]):

        # formulate infeasibility minimization objective
        fbl_meta_obj = self.formulator.formulate_slack_min_objective(idx_meta_sets=self.gbd_problem.idx_meta_sets,
                                                                     sl_meta_vars=sl_meta_vars,
                                                                     obj_sym=obj_sym)

        # add meta-objective to the problem
        self.gbd_problem.fbl_sp_objs[obj_sym] = fbl_meta_obj
        self.gbd_problem.add_meta_objective(fbl_meta_obj, is_in_model=False)

        return fbl_meta_obj

    def __build_fbl_sps(self):

        for primal_sp in self.gbd_problem.primal_sps:

            meta_vars = [mv for mv in primal_sp.model_meta_vars]
            sl_meta_vars_list = []
            meta_cons = []

            for meta_con in primal_sp.model_meta_cons:
                if meta_con.symbol in self.gbd_problem.std_to_sl_map:
                    sl_meta_vars, sl_con = self.gbd_problem.std_to_sl_map[meta_con.symbol]
                    meta_vars.extend(sl_meta_vars)
                    sl_meta_vars_list.extend(sl_meta_vars)
                    meta_cons.append(sl_con)
                else:
                    meta_cons.append(meta_con)

            fbl_obj_sym = self.gbd_problem.default_fbl_sp_obj_sym + '_' + primal_sp.symbol
            meta_obj = self.__build_slack_minimization_objective(obj_sym=fbl_obj_sym,
                                                                 sl_meta_vars=sl_meta_vars_list)

            fbl_sp_sym = primal_sp.symbol + "_FBL"
            fbl_sp = BaseProblem(symbol=fbl_sp_sym,
                                 description="Feasibility subproblem")
            fbl_sp.model_meta_vars = meta_vars
            fbl_sp.model_meta_objs.append(meta_obj)
            fbl_sp.model_meta_cons = meta_cons

            self.gbd_problem.add_subproblem(fbl_sp)
            self.gbd_problem.fbl_sps.append(fbl_sp)

    # Complicating Storage Parameters
    # ------------------------------------------------------------------------------------------------------------------

    def __build_complicating_storage_params(self):

        # generate storage meta-parameters for the complicating variables

        for id, meta_var in self.gbd_problem.comp_meta_vars.items():
            if meta_var.symbol not in self.gbd_problem.stored_comp_decisions:

                # Generate parameter symbol
                storage_symbol = self.gbd_problem.generate_unique_symbol("{0}_stored".format(meta_var.symbol))

                # Retrieve parent meta-variable
                parent_meta_var = self.gbd_problem.meta_vars[meta_var.symbol]

                # Retrieve indexing meta-sets
                idx_meta_sets = [ms for ms in parent_meta_var.idx_meta_sets]
                idx_meta_sets.append(self.gbd_problem.cuts)

                # Retrieve constraint
                idx_set_con_literal = parent_meta_var.get_idx_set_con_literal()

                # Build storage meta-parameter
                stored_comp_param = self.entity_builder.build_meta_param(symbol=storage_symbol,
                                                                         idx_meta_sets=idx_meta_sets,
                                                                         idx_set_con_literal=idx_set_con_literal,
                                                                         default_value=0)

                # Add storage meta-parameter to problem
                self.gbd_problem.stored_comp_decisions[meta_var.symbol] = stored_comp_param
                self.gbd_problem.add_meta_parameter(stored_comp_param, is_in_model=False)

    # Master Objective
    # ------------------------------------------------------------------------------------------------------------------

    def __build_master_objective(self):

        eta = self.gbd_problem.eta

        idx_node = None
        if eta.get_reduced_dimension() > 0:
            idx_node = self.node_builder.build_default_entity_index_node(eta)

        eta_node = mat.DeclaredEntityNode(id=self.generate_free_node_id(),
                                          symbol=eta.symbol,
                                          entity_index_node=idx_node,
                                          type=const.VAR_TYPE)

        expression = mat.Expression(eta_node)

        # Build meta-objective for the master problem
        meta_obj = self.entity_builder.build_meta_obj(symbol=self.gbd_problem.mp_obj_sym,
                                                      direction="minimize",
                                                      expression=expression)

        # Add meta-objective to problem
        self.gbd_problem.master_obj = meta_obj
        self.gbd_problem.add_meta_objective(meta_obj, is_in_model=False)

    # Master Problem Auxiliary Constructs
    # ------------------------------------------------------------------------------------------------------------------

    def __build_auxiliary_constructs(self):
        self.__build_master_problem_auxiliary_constructs_f()
        self.__build_master_problem_auxiliary_constructs_g()

    def __build_master_problem_auxiliary_constructs_f(self):

        operands = []

        for sp in self.gbd_problem.primal_sps:

            sp_meta_obj = sp.model_meta_objs[0]
            obj_expr = sp_meta_obj.expression
            idx_set_node = sp_meta_obj.idx_set_node

            if idx_set_node is not None:
                idx_set = idx_set_node.evaluate(self.gbd_problem.state)[0]
                dummy_syms = idx_set_node.get_dummy_elements(self.gbd_problem.state)
            else:
                idx_set = None
                dummy_syms = None

            # Build f(y) node
            if self.__is_primal_sp_obj_comp[sp_meta_obj.symbol]:
                if not isinstance(obj_expr.expression_node, mat.ArithmeticExpressionNode):
                    raise ValueError("GBD problem builder expected an arithmetic expression node"
                                     + " while retrieving the expression node of a mixed-complicating"
                                     + " objective function '{0}'".format(sp_meta_obj))
                node = self.__verify_and_modify_complicating_expression(obj_expr.expression_node,
                                                                        idx_set,
                                                                        dummy_syms)
                operands.append(node)

        if len(operands) > 0:
            sum_node = mat.MultiArithmeticOperationNode(id=self.generate_free_node_id(),
                                                        operator='+',
                                                        operands=operands)
        else:
            sum_node = mat.NumericNode(id=self.generate_free_node_id(),
                                       value=0)

        # Build variable indexing set node
        idx_set_node = None
        if len(self.gbd_problem.idx_meta_sets) > 0:
            idx_set_node = self.node_builder.build_idx_set_node(idx_meta_sets=self.gbd_problem.idx_meta_sets)
            dummy_node = mat.DummyNode(id=self.generate_free_node_id(),
                                       symbol=self.gbd_problem.cuts_unb_sym)
            cuts_set_node = mat.SetNode(id=self.generate_free_node_id(),
                                        symbol=self.gbd_problem.cuts_sym)
            idx_set_node.set_nodes.append(mat.IndexingSetNode(id=self.generate_free_node_id(),
                                                              dummy_node=dummy_node,
                                                              set_node=cuts_set_node))

        # Build meta-variable

        idx_meta_sets = {sym: ms for sym, ms in self.gbd_problem.idx_meta_sets.items()}
        idx_meta_sets[self.gbd_problem.cuts_sym] = self.gbd_problem.cuts

        aux_f_sym = self.gbd_problem.generate_unique_symbol("GBD_F")
        aux_f_meta_var = self.entity_builder.build_meta_var(symbol=aux_f_sym,
                                                            idx_meta_sets=idx_meta_sets,
                                                            idx_set_node=idx_set_node,
                                                            defined_value=sum_node)

        # Add meta-variable to the problem
        self.gbd_problem.aux_f_meta_var = aux_f_meta_var
        self.gbd_problem.add_meta_variable(aux_f_meta_var, is_in_model=False)

    def __build_master_problem_auxiliary_constructs_g(self):

        i = 0
        for name, meta_con in self.gbd_problem.mixed_comp_cons.items():

            con_expr = meta_con.expression
            idx_set_node = meta_con.idx_set_node

            if idx_set_node is not None:
                idx_set = idx_set_node.evaluate(self.gbd_problem.state)[0]
                dummy_syms = idx_set_node.get_dummy_elements(self.gbd_problem.state)
            else:
                idx_set = None
                dummy_syms = None

            # Build g(y) node
            if not isinstance(con_expr.expression_node, mat.RelationalOperationNode):
                raise ValueError("GBD problem builder expected a relational operation node"
                                 + " while retrieving the LHS expression of a mixed-complicating constraint"
                                 + " '{0}'".format(meta_con))
            rhs_node = self.__verify_and_modify_complicating_expression(con_expr.expression_node.lhs_operand,
                                                                        idx_set,
                                                                        dummy_syms)

            # Build auxiliary variable indexing set node
            if idx_set_node is not None:
                idx_set_node = deepcopy(idx_set_node)
                dummy_node = mat.DummyNode(id=self.generate_free_node_id(),
                                           symbol=self.gbd_problem.cuts_unb_sym)
                cuts_set_node = mat.SetNode(id=self.generate_free_node_id(),
                                            symbol=self.gbd_problem.cuts_sym)
                idx_set_node.set_nodes.append(mat.IndexingSetNode(id=self.generate_free_node_id(),
                                                                  dummy_node=dummy_node,
                                                                  set_node=cuts_set_node))

            # Build meta-variable

            idx_meta_sets = list(meta_con.idx_meta_sets)
            idx_meta_sets.append(self.gbd_problem.cuts)

            aux_g_meta_var_sym = self.gbd_problem.generate_unique_symbol("GBD_G_{0}".format(i))
            gv = self.entity_builder.build_meta_var(symbol=aux_g_meta_var_sym,
                                                    idx_meta_sets=idx_meta_sets,
                                                    idx_set_con_literal=meta_con.get_idx_set_con_literal(),
                                                    idx_set_node=idx_set_node,
                                                    defined_value=rhs_node)

            # Add meta-variable to the problem
            self.gbd_problem.aux_g_meta_vars[i] = gv
            self.gbd_problem.add_meta_variable(gv, is_in_model=False)

            # Build duality multiplier meta-parameter
            self.__build_duality_multiplier(meta_con, i)

            # Increment id
            i += 1

    def __build_duality_multiplier(self, meta_con: mat.MetaConstraint, id: int):

        sym = self.problem.generate_unique_symbol("lambda_{0}".format(meta_con.symbol))
        idx_meta_sets = [ms for ms in meta_con.idx_meta_sets]
        idx_meta_sets.append(self.gbd_problem.cuts)
        idx_set_con_literal = meta_con.get_idx_set_con_literal()

        # Build duality multiplier meta-parameter
        duality_multiplier = self.entity_builder.build_meta_param(symbol=sym,
                                                                  idx_meta_sets=idx_meta_sets,
                                                                  idx_set_con_literal=idx_set_con_literal,
                                                                  default_value=0)

        # Add duality multiplier meta-parameter to the problem
        self.gbd_problem.duality_multipliers[id] = duality_multiplier
        self.gbd_problem.add_meta_parameter(duality_multiplier, is_in_model=False)

        return duality_multiplier

    def __verify_and_modify_complicating_expression(self,
                                                    node: mat.ArithmeticExpressionNode,
                                                    idx_set: mat.IndexingSet,
                                                    dummy_syms: Tuple[Union[int, float, str, tuple, None], ...]
                                                    ) -> mat.ArithmeticExpressionNode:
        node = deepcopy(node)
        self.__verify_complicating_node(node, idx_set, dummy_syms)  # check whether the problem satisfies property (P)
        comp_node, status = self.__reformulate_node(node, idx_set, dummy_syms)
        sub_node = self.__build_complicating_subtraction_node(comp_node, idx_set, dummy_syms)
        return sub_node

    def __verify_complicating_node(self,
                                   node: mat.ArithmeticExpressionNode,
                                   idx_set: mat.IndexingSet,
                                   dummy_syms: Tuple[Union[int, float, str, tuple, None], ...]) -> int:

        # Numeric constant or dummy
        if isinstance(node, mat.NumericNode) or isinstance(node, mat.DummyNode):
            return self.CONST_NODE

        # Declared entity
        elif isinstance(node, mat.DeclaredEntityNode):

            if node.get_type() == const.PARAM_TYPE:
                return self.CONST_NODE

            if node.symbol not in [mv.symbol for mv in self.gbd_problem.comp_meta_vars.values()]:
                return self.PURE_X_NODE

            entities = node.collect_declared_entities(self.gbd_problem.state, idx_set, dummy_syms)
            vars = {k: v for k, v in entities.items() if isinstance(v, mat.Variable)}

            has_x = False
            has_y = False
            for var_id, var in vars.items():
                if var_id in self.comp_vars:
                    has_y = True
                else:
                    has_x = True
                if has_x and has_y:
                    break

            if has_x and not has_y:
                return self.PURE_X_NODE
            elif has_y and not has_x:
                return self.PURE_Y_NODE
            else:
                return self.MIXED_NODE

        # Function
        elif isinstance(node, mat.ArithmeticTransformationNode):

            if node.is_reductive():
                inner_idx_sets = node.idx_set_node.evaluate(self.gbd_problem.state, idx_set, dummy_syms)
                inner_idx_set = OrderedSet().union(*inner_idx_sets)
                idx_set = mat.cartesian_product([idx_set, inner_idx_set])
                dummy_syms = node.idx_set_node.combined_dummy_syms

            statuses = []
            for operand in node.operands:
                statuses.append(self.__verify_complicating_node(operand, idx_set, dummy_syms))
            status = self.__combine_node_statuses(statuses)

            # Node is either non-complicating or pure complicating
            if status in [self.CONST_NODE, self.PURE_X_NODE, self.PURE_Y_NODE]:
                return status

            # Node contains complicating and non-complicating variables
            else:

                # Non-Reductive Function
                if not node.is_reductive():
                    # All non-reductive functions in AMPL are nonlinear
                    raise ValueError("GBD problem builder encountered a complicating function node"
                                     + " that violates Property P")

                # Reductive Function
                else:

                    # Nonlinear Reductive Function
                    if node.symbol != "sum":
                        raise ValueError("GBD problem builder encountered a complicating function node"
                                         + " that violates Property P")

                    # Summation
                    else:
                        return status

        # Unary Operation
        elif isinstance(node, mat.UnaryArithmeticOperationNode):
            return self.__verify_complicating_node(node.operand, idx_set, dummy_syms)

        # Binary Operation
        elif isinstance(node, mat.BinaryArithmeticOperationNode):

            lhs_status = self.__verify_complicating_node(node.lhs_operand, idx_set, dummy_syms)
            rhs_status = self.__verify_complicating_node(node.rhs_operand, idx_set, dummy_syms)
            statuses = [lhs_status, rhs_status]

            if self.CONST_NODE in statuses:
                if self.PURE_X_NODE in statuses:
                    return self.PURE_X_NODE
                elif self.PURE_Y_NODE in statuses:
                    return self.PURE_Y_NODE
                elif self.MIXED_NODE in statuses:
                    return self.MIXED_NODE
                else:
                    return self.CONST_NODE
            elif lhs_status == self.PURE_X_NODE and rhs_status == self.PURE_X_NODE:
                return self.PURE_X_NODE
            elif lhs_status == self.PURE_Y_NODE and rhs_status == self.PURE_Y_NODE:
                return self.PURE_Y_NODE
            else:
                if node.operator in ['+', '-', "less"]:
                    return self.MIXED_NODE
                else:
                    raise ValueError("GBD problem builder encountered a complicating binary arithmetic operation node"
                                     + " that violates Property P")

        # Multi Operation
        elif isinstance(node, mat.MultiArithmeticOperationNode):

            statuses = [self.__verify_complicating_node(o, idx_set, dummy_syms) for o in node.operands]

            if all([s == self.CONST_NODE for s in statuses]):
                return self.CONST_NODE
            if self.PURE_X_NODE in statuses and self.PURE_Y_NODE not in statuses and self.MIXED_NODE not in statuses:
                return self.PURE_X_NODE
            elif self.PURE_X_NODE not in statuses and self.PURE_Y_NODE in statuses and self.MIXED_NODE not in statuses:
                return self.PURE_Y_NODE
            else:
                if node.operator in ['+', '-', "less"]:
                    return self.MIXED_NODE
                else:
                    raise ValueError("GBD problem builder encountered a complicating multi arithmetic operation node"
                                     + " that violates Property P")

        # Conditional Operation
        elif isinstance(node, mat.ConditionalArithmeticExpressionNode):
            statuses = []
            for op_node in node.operands:
                statuses.append(self.__verify_complicating_node(op_node, idx_set, dummy_syms))
            return self.__combine_node_statuses(statuses)

        else:
            raise ValueError("GBD problem builder encountered an unexpected node"
                             + " while verifying the formulation of a complicating expression")

    def __combine_node_statuses(self, statuses: List[int]) -> int:
        has_x = False
        has_y = False
        for status in statuses:
            if status == self.PURE_X_NODE:
                has_x = True
            elif status == self.PURE_Y_NODE:
                has_y = True
            elif status == self.MIXED_NODE:
                has_x = True
                has_y = True
        if has_x and has_y:
            return self.MIXED_NODE
        elif has_x:
            return self.PURE_X_NODE
        elif has_y:
            return self.PURE_Y_NODE
        else:
            return self.CONST_NODE

    def __reformulate_node(self,
                           node: mat.ArithmeticExpressionNode,
                           idx_set: mat.IndexingSet,
                           dummy_syms: Tuple[Union[int, float, str, tuple, None], ...]
                           ) -> Tuple[Union[mat.ArithmeticExpressionNode, mat.DummyNode], int]:

        # Numeric constant
        if isinstance(node, mat.NumericNode) or isinstance(node, mat.DummyNode):
            return node, self.CONST_NODE

        # Declared entity
        elif isinstance(node, mat.DeclaredEntityNode):

            if node.get_type() == const.PARAM_TYPE:
                return node, self.CONST_NODE

            status = self.__verify_complicating_node(node, idx_set, dummy_syms)

            if status == self.PURE_X_NODE:
                return node, status

            else:

                # scalar variable node
                if node.idx_node is None:
                    return node, self.PURE_Y_NODE

                # indexed variable node
                else:
                    return self.__reformulate_complicating_entity_node(node, idx_set, dummy_syms), self.PURE_Y_NODE

        # Function
        elif isinstance(node, mat.ArithmeticTransformationNode):

            if node.is_reductive():
                inner_idx_sets = node.idx_set_node.evaluate(self.gbd_problem.state, idx_set, dummy_syms)
                inner_idx_set = OrderedSet().union(*inner_idx_sets)
                idx_set = mat.cartesian_product([idx_set, inner_idx_set])
                dummy_syms = node.idx_set_node.combined_dummy_syms

            statuses = []
            for i, operand in enumerate(node.operands):
                ref_operand, status = self.__reformulate_node(operand, idx_set, dummy_syms)
                statuses.append(status)
                node.operands[i] = ref_operand

            return node, self.__combine_node_statuses(statuses)

        # Unary Operation
        elif isinstance(node, mat.UnaryArithmeticOperationNode):
            return self.__reformulate_node(node.operand, idx_set, dummy_syms)

        # Binary Operation
        elif isinstance(node, mat.BinaryArithmeticOperationNode):

            lhs_operand, lhs_status = self.__reformulate_node(node.lhs_operand, idx_set, dummy_syms)
            rhs_operand, rhs_status = self.__reformulate_node(node.rhs_operand, idx_set, dummy_syms)

            node.lhs_operand = lhs_operand
            node.rhs_operand = rhs_operand

            # Linearly separable
            if node.operator in ['+', '-']:
                if lhs_status in self.COMP_STATUSES and rhs_status in self.COMP_STATUSES:
                    if lhs_status == self.PURE_Y_NODE and rhs_status == self.PURE_Y_NODE:
                        return node, self.PURE_Y_NODE
                    else:
                        return node, self.MIXED_NODE
                elif lhs_status in self.COMP_STATUSES:
                    return lhs_operand, lhs_status
                elif rhs_status in self.COMP_STATUSES:
                    if node.operator == '-':
                        rhs_operand = self.node_builder.append_negative_unity_coefficient(rhs_operand)
                        return rhs_operand, rhs_status
                    else:
                        return rhs_operand, rhs_status
                else:
                    if lhs_status == self.CONST_NODE and rhs_status == self.CONST_NODE:
                        return node, self.CONST_NODE
                    else:
                        return node, self.PURE_X_NODE

            # Not linearly separable
            else:
                if lhs_status in self.COMP_STATUSES and rhs_status in self.COMP_STATUSES:
                    if lhs_status == self.PURE_Y_NODE and rhs_status == self.PURE_Y_NODE:
                        return node, self.PURE_Y_NODE
                    else:
                        return node, self.MIXED_NODE
                elif lhs_status in self.COMP_STATUSES:
                    return node, lhs_status
                elif rhs_status in self.COMP_STATUSES:
                    return node, rhs_status
                else:
                    if lhs_status == self.CONST_NODE and rhs_status == self.CONST_NODE:
                        return node, self.CONST_NODE
                    else:
                        return node, self.PURE_X_NODE

        # Multi Operation
        elif isinstance(node, mat.MultiArithmeticOperationNode):

            ref_results = [self.__reformulate_node(o, idx_set, dummy_syms) for o in node.operands]
            statuses = [r[1] for r in ref_results]

            if all([s == self.CONST_NODE for s in statuses]):
                return node, self.CONST_NODE

            elif all([s not in self.COMP_STATUSES for s in statuses]):
                return node, self.PURE_X_NODE

            else:

                # Linearly separable
                if node.operator == '+':
                    comp_ref_results = [r for r in ref_results if r[1] in self.COMP_STATUSES]
                    comp_ref_nodes = [r[0] for r in comp_ref_results]
                    comp_statuses = [r[1] for r in comp_ref_results]
                    node.operands = comp_ref_nodes
                    if all([s == self.PURE_Y_NODE for s in comp_statuses]):
                        return node, self.PURE_Y_NODE
                    else:
                        return node, self.MIXED_NODE

                # Not linearly separable
                else:
                    node.operands = [r[0] for r in ref_results]
                    if all([s == self.PURE_Y_NODE for s in statuses]):
                        return node, self.PURE_Y_NODE
                    else:
                        return node, self.MIXED_NODE

        # Conditional Operation
        elif isinstance(node, mat.ConditionalArithmeticExpressionNode):

            statuses = []

            for i, op_node in enumerate(node.operands):

                ref_op_node, status = self.__reformulate_node(op_node, idx_set, dummy_syms)

                if status == self.PURE_X_NODE:
                    ref_op_node = mat.NumericNode(id=self.generate_free_node_id(),
                                                  value=0)
                    status = self.CONST_NODE

                statuses.append(status)
                node.operands[i] = ref_op_node

            return node, self.__combine_node_statuses(statuses)

        else:
            raise ValueError("GBD problem builder encountered an unexpected node"
                             + " while reformulating a complicating expression")

    def __reformulate_complicating_entity_node(self,
                                               entity_node: mat.DeclaredEntityNode,
                                               idx_set: mat.IndexingSet,
                                               dummy_syms: Tuple[Union[int, float, str, tuple, None], ...]):

        # Retrieve all variable entities
        entities = entity_node.collect_declared_entities(self.gbd_problem.state, idx_set, dummy_syms)
        vars = {k: v for k, v in entities.items() if isinstance(v, mat.Variable)}

        # Identify complicating meta-variables that own a complicating variable entity
        sub_comp_meta_vars = {}
        has_x = False
        has_y = False
        for var_id, var in vars.items():
            is_y = False
            for comp_meta_var_id, comp_meta_var in self.gbd_problem.comp_meta_vars.items():
                if comp_meta_var.is_owner(var, self.gbd_problem.state):
                    has_y = True
                    is_y = True
                    sub_comp_meta_vars[comp_meta_var_id] = comp_meta_var
                    break
            if not is_y:
                has_x = True

        # Check whether the entity node is purely non-complicating or purely complicating
        if not (has_x and has_y):
            return entity_node  # Return the entity node argument without reformulation

        # Build summation node
        sum_idx_set_node = self.__build_idx_set_node_for_mixed_comp_entity_node(entity_node,
                                                                                sub_comp_meta_vars)
        sum_node = mat.ArithmeticTransformationNode(id=self.generate_free_node_id(),
                                                    symbol="sum",
                                                    idx_set_node=sum_idx_set_node,
                                                    operands=entity_node)

        return sum_node

    def __build_idx_set_node_for_mixed_comp_entity_node(
            self,
            entity_node: mat.DeclaredEntityNode,
            sub_comp_meta_vars: Dict[str, mat.MetaVariable]):

        # === Setup ===

        # Identify fixed dimensions of the variable's indexing set
        parent_meta_var = self.gbd_problem.meta_vars[entity_node.symbol]
        is_dim_fixed = []
        full_to_reduced_dim_index_map = {}
        dim_index = -1
        red_dim_index = -1
        for ims in parent_meta_var.idx_meta_sets:
            for is_dim_fixed_i in ims.is_dim_fixed:
                dim_index += 1
                if not is_dim_fixed_i:
                    red_dim_index += 1
                    full_to_reduced_dim_index_map[dim_index] = red_dim_index
                is_dim_fixed.append(is_dim_fixed_i)

        # Retrieve index nodes of the entity node
        entity_idx_nodes = entity_node.idx_node.component_nodes

        # === Constraint Construction ===

        conjunction_nodes = []

        # Build a logical expression to verify membership in the indexing sets of the complicating variables
        for comp_meta_var_id, comp_meta_var in sub_comp_meta_vars.items():

            # Initialize empty list of logical conjunction operand nodes
            conjunction_operands = []

            # Retrieve dummy symbols of the indexing set of the complicating meta-variable
            def_idx_nodes = comp_meta_var.idx_set_node.get_dummy_component_nodes(self.gbd_problem.state)
            def_dummy_syms = []
            for i, def_idx_node in enumerate(def_idx_nodes):
                if isinstance(def_idx_node, mat.DummyNode) and not is_dim_fixed[i]:
                    def_dummy_syms.append(def_idx_node.symbol)

            # --- Build set membership operation nodes ---

            dim_index = -1
            for component_set_node in comp_meta_var.idx_set_node.set_nodes:

                # Copy challenged set node
                if isinstance(component_set_node, mat.IndexingSetNode):
                    challenged_set_node = deepcopy(component_set_node.set_node)
                else:
                    challenged_set_node = deepcopy(component_set_node)

                # Build challenge dummy node

                challenge_dummy_node = None
                dim = component_set_node.get_dim(self.gbd_problem.state)

                # Unique uncontrolled dummy
                if dim == 1:
                    dim_index += 1
                    challenge_dummy_node = deepcopy(entity_idx_nodes[full_to_reduced_dim_index_map[dim_index]])

                # Compound dummy
                else:

                    component_dummy_nodes = []
                    is_controlled = []
                    for _ in range(dim):
                        dim_index += 1
                        if is_dim_fixed[dim_index]:
                            idx_node = def_idx_nodes[dim_index]
                            idx_node = self.__replace_dummy_nodes(idx_node,
                                                                  def_dummy_syms,
                                                                  entity_idx_nodes)
                        else:
                            idx_node = entity_idx_nodes[full_to_reduced_dim_index_map[dim_index]]
                        component_dummy_nodes.append(deepcopy(idx_node))
                        is_controlled.append(entity_idx_nodes[dim_index].is_controlled())

                    if any(is_controlled):
                        challenge_dummy_node = mat.CompoundDummyNode(id=self.generate_free_node_id(),
                                                                     component_nodes=component_dummy_nodes)

                # Build set membership operation nodes
                if challenge_dummy_node is not None and challenged_set_node is not None:

                    # Replace uncontrolled dummies
                    challenged_set_node = self.__replace_dummy_nodes(challenged_set_node,
                                                                     def_dummy_syms,
                                                                     entity_idx_nodes)
                    if not isinstance(challenged_set_node, mat.SetExpressionNode):
                        raise ValueError("GBD problem builder expected a set expression node"
                                         + " while reformulating a mixed entity node")

                    # Build logical set-membership operation
                    membership_node = mat.SetMembershipOperationNode(id=self.generate_free_node_id(),
                                                                     operator="in",
                                                                     member_node=challenge_dummy_node,
                                                                     set_node=challenged_set_node)
                    conjunction_operands.append(membership_node)

            # --- Modify constraint of the indexing set ---

            if comp_meta_var.idx_set_node.constraint_node is not None:
                constraint_node = self.__replace_dummy_nodes(comp_meta_var.idx_set_node.constraint_node,
                                                             def_dummy_syms,
                                                             entity_idx_nodes)
                constraint_node.is_prioritized = True
                conjunction_operands.append(constraint_node)

            # --- Build logical conjunction node ---
            conjunction_node = mat.MultiLogicalOperationNode(id=self.generate_free_node_id(),
                                                             operator="&&",
                                                             operands=conjunction_operands)
            conjunction_node.is_prioritized = True
            conjunction_nodes.append(conjunction_node)

        # Build logical disjunction node
        if len(conjunction_nodes) > 0:
            disjunction_node = mat.MultiLogicalOperationNode(id=self.generate_free_node_id(),
                                                             operator="||",
                                                             operands=conjunction_nodes)
        else:
            disjunction_node = None

        # === Constrained Indexing Set Node Construction ===

        # Build indexing set node
        ord_set_node = mat.OrderedSetNode(id=self.generate_free_node_id(),
                                          start_node=mat.NumericNode(id=self.generate_free_node_id(), value=1),
                                          end_node=mat.NumericNode(id=self.generate_free_node_id(), value=1))
        sum_idx_set_node = mat.CompoundSetNode(id=self.generate_free_node_id(),
                                               set_nodes=[ord_set_node],
                                               constraint_node=disjunction_node)

        return sum_idx_set_node

    def __replace_dummy_nodes(self,
                              node: mat.ExpressionNode,
                              dummy_syms: List[str],
                              replacement_nodes: List[mat.ExpressionNode]) -> mat.ExpressionNode:

        # Node is a dummy
        if isinstance(node, mat.DummyNode):
            if node.symbol in dummy_syms:
                index = dummy_syms.index(node.symbol)
                return replacement_nodes[index]

        # Node is not a dummy
        else:
            modified_children = []
            for child in node.get_children():
                modified_children.append(self.__replace_dummy_nodes(child, dummy_syms, replacement_nodes))
            node.set_children(modified_children)
            return node

    def __build_complicating_subtraction_node(self,
                                              root_node: mat.ArithmeticExpressionNode,
                                              idx_set: mat.IndexingSet,
                                              dummy_syms: Tuple[Union[int, float, str, tuple, None], ...]):

        root_node.is_prioritized = True
        root_node_clone = deepcopy(root_node)

        queue = Queue()
        queue.put((root_node_clone, idx_set, dummy_syms))

        # modify complicating variable nodes
        while not queue.empty():

            node, idx_set, dummy_syms = queue.get()
            node: mat.ExpressionNode
            idx_set: mat.IndexingSet
            dummy_syms: mat.Element

            is_comp_var_node = False

            if isinstance(node, mat.DeclaredEntityNode):
                if node.get_type() == const.VAR_TYPE:

                    status = self.__verify_complicating_node(node, idx_set, dummy_syms)

                    if status in self.COMP_STATUSES:

                        is_comp_var_node = True

                        # build storage parameter index node
                        cuts_dummy_node = mat.DummyNode(id=self.generate_free_node_id(),
                                                        symbol=self.gbd_problem.cuts.dummy_symbols[0])
                        if node.idx_node is not None:
                            storage_param_idx_node = deepcopy(node.idx_node)
                            storage_param_idx_node.component_nodes.append(cuts_dummy_node)
                        else:
                            storage_param_idx_node = mat.CompoundDummyNode(id=self.generate_free_node_id(),
                                                                           component_nodes=[cuts_dummy_node])

                        # retrieve storage meta-parameter
                        storage_meta_param = self.gbd_problem.stored_comp_decisions[node.symbol]

                        # transform complicating variable node in storage parameter node
                        node.symbol = storage_meta_param.symbol
                        node.idx_node = storage_param_idx_node
                        node.set_type(const.PARAM_TYPE)

            elif isinstance(node, mat.ArithmeticTransformationNode):
                if node.is_reductive():
                    # add the indexing set of the current scope to that of the outer scope
                    inner_idx_sets = node.idx_set_node.evaluate(self.gbd_problem.state, idx_set, dummy_syms)
                    inner_idx_set = OrderedSet().union(*inner_idx_sets)
                    idx_set = mat.cartesian_product([idx_set, inner_idx_set])
                    dummy_syms = node.idx_set_node.combined_dummy_syms

            if not is_comp_var_node:
                children = node.get_children()
                for child in children:
                    queue.put((child, idx_set, dummy_syms))

        # build subtraction node
        subtraction_node = mat.SubtractionNode(id=self.generate_free_node_id(),
                                               lhs_operand=root_node,
                                               rhs_operand=root_node_clone,
                                               is_prioritized=True)

        return subtraction_node

    # Master Problem Cuts
    # ------------------------------------------------------------------------------------------------------------------

    def __build_cuts(self):

        # generate cutting plane meta-constraints for the master problem

        self.__build_optimality_cut()
        self.__build_feasibility_cut()

        # determine whether canonical integer cuts can be generated
        are_comp_vars_binary = True
        comp_var_syms = self.gbd_problem.get_comp_var_syms()
        for sym in comp_var_syms:
            parent_meta_var = self.gbd_problem.meta_vars[sym]
            if not parent_meta_var.is_binary:
                # if a single complicating variable is not binary, then set the flag to false
                are_comp_vars_binary = False
                break

        if are_comp_vars_binary:  # build the canonical integer cut if all complicating variables are binary
            self.__build_canonical_integer_cut()

    def __build_optimality_cut(self):

        # retrieve indexing meta sets
        idx_meta_sets = [self.gbd_problem.cuts]

        # build indexing set constraint node
        is_feasible_node = self.node_builder.build_default_entity_node(self.gbd_problem.is_feasible)
        idx_set_con_node = mat.RelationalOperationNode(operator='=',
                                                       lhs_operand=is_feasible_node,
                                                       rhs_operand=mat.NumericNode(1))

        # build indexing set node
        idx_set_node = self.node_builder.build_idx_set_node(idx_meta_sets)
        idx_set_node.constraint_node = idx_set_con_node

        # build expression
        expression = self.__build_optimality_cut_expression()

        # build meta-constraint
        meta_con = mat.MetaConstraint(symbol=self.gbd_problem.opt_cut_con_sym,
                                      idx_meta_sets=idx_meta_sets,
                                      idx_set_node=idx_set_node,
                                      expression=expression)

        # add meta-constraint to the problem
        self.gbd_problem.gbd_cuts[meta_con.symbol] = meta_con
        self.gbd_problem.add_meta_constraint(meta_con, is_in_model=False)

    def __build_feasibility_cut(self):

        # retrieve indexing meta sets
        idx_meta_sets = [self.gbd_problem.cuts]

        # build indexing set constraint node
        is_feasible_node = self.node_builder.build_default_entity_node(self.gbd_problem.is_feasible)
        idx_set_con_node = mat.RelationalOperationNode(operator='=',
                                                       lhs_operand=is_feasible_node,
                                                       rhs_operand=mat.NumericNode(0))

        # build indexing set node
        idx_set_node = self.node_builder.build_idx_set_node(idx_meta_sets)
        idx_set_node.constraint_node = idx_set_con_node

        # build expression
        expression = self.__build_feasibility_cut_expression()

        # build meta-constraint
        meta_con = mat.MetaConstraint(symbol=self.gbd_problem.fbl_cut_con_sym,
                                      idx_meta_sets=idx_meta_sets,
                                      idx_set_node=idx_set_node,
                                      expression=expression)

        # add meta-constraint to the problem
        self.gbd_problem.gbd_cuts[self.gbd_problem.fbl_cut_con_sym] = meta_con
        self.gbd_problem.add_meta_constraint(meta_con, is_in_model=False)

    def __build_canonical_integer_cut(self):

        # build indexing set node
        idx_meta_sets = [self.gbd_problem.cuts]
        idx_set_node = self.node_builder.build_idx_set_node(idx_meta_sets)

        # build expression
        expression = self.__build_canonical_integer_cut_expression()

        # build meta-constraint
        meta_con = mat.MetaConstraint(symbol=self.gbd_problem.can_int_cut_con_sym,
                                      idx_meta_sets=idx_meta_sets,
                                      idx_set_node=idx_set_node,
                                      expression=expression)

        # add meta-constraint to the problem
        self.gbd_problem.gbd_cuts[self.gbd_problem.can_int_cut_con_sym] = meta_con
        self.gbd_problem.add_meta_constraint(meta_con, is_in_model=False)

    def __build_optimality_cut_expression(self):

        sum_operands = []

        # Build stored objective node
        stored_obj = self.gbd_problem.stored_obj
        stored_obj_idx_node = self.node_builder.build_default_entity_index_node(stored_obj)
        stored_obj_node = mat.DeclaredEntityNode(id=self.generate_free_node_id(),
                                                 symbol=stored_obj.symbol,
                                                 entity_index_node=stored_obj_idx_node,
                                                 type=const.PARAM_TYPE)
        sum_operands.append(stored_obj_node)

        # Build F(y) node
        sum_operands.append(self.__build_aux_f_summation_node())

        # Build (lambda * G(y)) nodes
        g_ids = list(self.gbd_problem.aux_g_meta_vars.keys())
        for g_id in g_ids:
            sum_operands.append(self.__build_aux_g_summation_node(g_id))

        # Eta node
        eta = self.gbd_problem.eta
        eta_idx_node = None
        if eta.get_reduced_dimension() > 0:
            eta_idx_node = self.node_builder.build_default_entity_index_node(eta)
        eta_node = mat.DeclaredEntityNode(id=self.generate_free_node_id(),
                                          symbol=eta.symbol,
                                          entity_index_node=eta_idx_node,
                                          type=const.VAR_TYPE)

        # Lower bound
        lb_node = mat.MultiArithmeticOperationNode(id=self.generate_free_node_id(),
                                                   operator='+',
                                                   operands=sum_operands)

        # Inequality node
        ineq_op_node = mat.RelationalOperationNode(id=self.generate_free_node_id(), operator=">=")
        ineq_op_node.add_operand(eta_node)
        ineq_op_node.add_operand(lb_node)

        return mat.Expression(ineq_op_node)

    def __build_feasibility_cut_expression(self):

        sum_operands = []

        # Build stored objective node
        stored_obj = self.gbd_problem.stored_obj
        stored_obj_idx_node = self.node_builder.build_default_entity_index_node(stored_obj)
        stored_obj_node = mat.DeclaredEntityNode(id=self.generate_free_node_id(),
                                                 symbol=stored_obj.symbol,
                                                 entity_index_node=stored_obj_idx_node,
                                                 type=const.PARAM_TYPE)
        sum_operands.append(stored_obj_node)

        # Build (lambda * G(y)) nodes
        g_ids = list(self.gbd_problem.aux_g_meta_vars.keys())
        for g_id in g_ids:
            sum_operands.append(self.__build_aux_g_summation_node(g_id))

        # Zero node
        zero_node = mat.NumericNode(id=self.generate_free_node_id(), value=0)

        # Lower bound
        lb_node = mat.MultiArithmeticOperationNode(id=self.generate_free_node_id(),
                                                   operator='+',
                                                   operands=sum_operands)

        # Inequality node
        ineq_op_node = mat.RelationalOperationNode(id=self.generate_free_node_id(), operator=">=")
        ineq_op_node.add_operand(zero_node)
        ineq_op_node.add_operand(lb_node)

        return mat.Expression(ineq_op_node)

    def __build_aux_f_summation_node(self):
        aux_f_node = self.node_builder.build_default_entity_node(self.gbd_problem.aux_f_meta_var)

        # auxiliary variable is only indexed with respect to the cuts set
        if self.gbd_problem.aux_f_meta_var.get_reduced_dimension() <= 1:
            return aux_f_node

        # auxiliary variable is indexed with respect to other sets
        else:

            idx_set_node = deepcopy(self.gbd_problem.aux_f_meta_var.idx_set_node)
            idx_set_node.set_nodes.pop()  # Remove cuts set from indexing set node

            sum_f_node = mat.ArithmeticTransformationNode(id=self.generate_free_node_id(),
                                                          symbol="sum",
                                                          idx_set_node=idx_set_node,
                                                          operands=aux_f_node)
            return sum_f_node

    def __build_aux_g_summation_node(self, g_id: int):

        aux_g_meta_var = self.gbd_problem.aux_g_meta_vars[g_id]

        # Retrieve G(y) node
        aux_g_node = self.node_builder.build_default_entity_node(aux_g_meta_var)

        # Build index node of the duality multiplier node
        dual_mult_index_node = deepcopy(aux_g_node.idx_node)

        # Build duality multiplier node
        dual_mult_meta_param = self.gbd_problem.duality_multipliers[g_id]
        dual_mult_node = mat.DeclaredEntityNode(id=self.generate_free_node_id(),
                                                symbol=dual_mult_meta_param.symbol,
                                                entity_index_node=dual_mult_index_node,
                                                type=const.PARAM_TYPE)

        # build (lambda * G(y)) node
        prod_node = mat.MultiplicationNode(id=self.generate_free_node_id(),
                                           operands=[dual_mult_node, aux_g_node])

        # auxiliary variable is only indexed with respect to the cuts set
        if aux_g_meta_var.get_reduced_dimension() <= 1:
            return prod_node

        # auxiliary variable is indexed with respect to other sets
        else:

            # build sum (lambda * G(y)) node
            idx_set_node = deepcopy(aux_g_meta_var.idx_set_node)
            idx_set_node.set_nodes.pop()  # Remove CUTS set from indexing set node
            sum_g_node = mat.ArithmeticTransformationNode(id=self.generate_free_node_id(),
                                                          symbol="sum",
                                                          idx_set_node=idx_set_node,
                                                          operands=prod_node)

            return sum_g_node

    def __build_canonical_integer_cut_expression(self):

        def build_sum_node(binary_value: int, operand_node_generator: Callable[[mat.MetaEntity],
                                                                               mat.ArithmeticExpressionNode]):

            operands = []
            for _, comp_meta_var in self.gbd_problem.comp_meta_vars.items():

                # Retrieve component indexing set node
                storage_meta_param = self.gbd_problem.stored_comp_decisions[comp_meta_var.symbol]

                # Build storage parameter index node
                storage_param_idx_node = self.node_builder.build_default_entity_index_node(comp_meta_var)
                cuts_dummy_node = mat.DummyNode(id=self.generate_free_node_id(),
                                                symbol=self.gbd_problem.cuts.dummy_symbols[0])
                if storage_param_idx_node is None:
                    storage_param_idx_node = mat.CompoundDummyNode(id=self.generate_free_node_id(),
                                                                   component_nodes=[cuts_dummy_node])
                else:
                    storage_param_idx_node.component_nodes.append(cuts_dummy_node)

                # Build storage parameter node
                storage_param_node = mat.DeclaredEntityNode(id=self.generate_free_node_id(),
                                                            symbol=storage_meta_param.symbol,
                                                            entity_index_node=storage_param_idx_node,
                                                            type=const.PARAM_TYPE)

                # Build binary constant node
                bin_const_node = mat.NumericNode(id=self.generate_free_node_id(), value=binary_value)

                # Build equality node
                eq_node = mat.RelationalOperationNode(id=self.generate_free_node_id(),
                                                      operator='=',
                                                      lhs_operand=storage_param_node,
                                                      rhs_operand=bin_const_node)

                idx_set_con_node = comp_meta_var.idx_set_node.constraint_node
                if idx_set_con_node is not None:
                    idx_set_con_node = deepcopy(idx_set_con_node)
                    idx_set_con_node.is_prioritized = True
                    con_node = mat.BinaryLogicalOperationNode(id=self.generate_free_node_id(),
                                                              operator="&&",
                                                              lhs_operand=idx_set_con_node,
                                                              rhs_operand=eq_node)
                else:
                    con_node = eq_node

                idx_set_node = self.node_builder.build_entity_idx_set_node(comp_meta_var)
                idx_set_node.constraint_node = con_node

                # Build operand node
                operand_node = operand_node_generator(comp_meta_var)
                operand_node.is_prioritized = True

                # Build summation node
                var_sum_node = mat.ArithmeticTransformationNode(id=self.generate_free_node_id(),
                                                                symbol="sum",
                                                                idx_set_node=idx_set_node,
                                                                operands=operand_node)

                operands.append(var_sum_node)

            # Build addition node
            var_sum_node = mat.MultiArithmeticOperationNode(id=self.generate_free_node_id(),
                                                            operator="+",
                                                            operands=operands)
            var_sum_node.is_prioritized = True
            return var_sum_node

        # LHS

        def build_comp_var_node(meta_entity: mat.MetaEntity):
            return self.node_builder.build_default_entity_node(meta_entity)

        sum_nodes = []
        for bin_value in [1, 0]:
            sum_node = build_sum_node(bin_value, build_comp_var_node)
            sum_nodes.append(sum_node)

        lhs_node = mat.SubtractionNode(id=self.generate_free_node_id(),
                                       lhs_operand=sum_nodes[0],
                                       rhs_operand=sum_nodes[1])

        # RHS

        def build_one_node(_):
            return mat.NumericNode(id=self.generate_free_node_id(),
                                   value=1)

        sum_node = build_sum_node(1, build_one_node)
        one_node = mat.NumericNode(id=self.generate_free_node_id(),
                                   value=1)

        rhs_node = mat.SubtractionNode(id=self.generate_free_node_id(),
                                       lhs_operand=sum_node,
                                       rhs_operand=one_node)

        # Inequality node
        ineq_op_node = mat.RelationalOperationNode(id=self.generate_free_node_id(), operator="<=")
        ineq_op_node.add_operand(lhs_node)
        ineq_op_node.add_operand(rhs_node)

        return mat.Expression(ineq_op_node)

    # Master Problem
    # ------------------------------------------------------------------------------------------------------------------

    def __build_mp(self):

        meta_vars = [mv for _, mv in self.gbd_problem.comp_meta_vars.items()]
        meta_vars.append(self.gbd_problem.eta)
        meta_vars.append(self.gbd_problem.aux_f_meta_var)
        meta_vars.extend([mv for _, mv in self.gbd_problem.aux_g_meta_vars.items()])

        meta_obj = self.gbd_problem.master_obj

        meta_cons = [mc for _, mc in self.gbd_problem.pure_comp_cons.items()]
        meta_cons.extend([mc for _, mc in self.gbd_problem.gbd_cuts.items()])

        mp = BaseProblem(symbol=self.gbd_problem.mp_symbol,
                         description="Master problem")
        mp.model_meta_vars = meta_vars
        mp.model_meta_objs.append(meta_obj)
        mp.model_meta_cons = meta_cons

        self.gbd_problem.add_subproblem(mp)
        self.gbd_problem.mp = mp

    # Subproblem Containers
    # ------------------------------------------------------------------------------------------------------------------

    def __build_subproblem_containers(self):

        idx_sets = []
        for sym in self.gbd_problem.idx_meta_sets:
            idx_sets.append(self.gbd_problem.state.get_set(sym).elements)

        sp_idx_set = mat.cartesian_product(idx_sets)

        if len(sp_idx_set) == 0:  # scalar subproblems
            sp_idx_set.add(None)

        # generate subproblem containers
        for sp_index in sp_idx_set:
            for primal_sp, fbl_sp in zip(self.gbd_problem.primal_sps, self.gbd_problem.fbl_sps):
                sp_container = GBDSubproblemContainer(primal_sp=primal_sp,
                                                      fbl_sp=fbl_sp,
                                                      sp_index=sp_index)
                self.gbd_problem.sp_containers.append(sp_container)

        # retrieve indexing sets of all complicating meta-variables partitioned by subproblem
        comp_vars = list(self.gbd_problem.comp_meta_vars.values())
        idx_sets = self.__assemble_complete_entity_idx_sets_by_symbol(comp_vars)
        sp_sym_idx_sets = self.__partition_complete_entity_idx_sets_by_sp_sym(entity_type=const.VAR_TYPE,
                                                                              complete_idx_sets=idx_sets)
        sp_idx_sets = self.__partition_sp_entity_idx_sets_by_sp_index(sp_entity_idx_sets=sp_sym_idx_sets)

        # store indexing sets of all complicating meta-variables partitioned by subproblem
        for sp_idx_set_dict, sp_container in zip(sp_idx_sets, self.gbd_problem.sp_containers):
            sp_container.comp_var_idx_sets = sp_idx_set_dict

        # retrieve indexing sets of all mixed-complicating meta-constraints partitioned by subproblem
        mixed_comp_cons = list(self.gbd_problem.mixed_comp_cons.values())
        idx_sets = self.__assemble_complete_entity_idx_sets_by_symbol(mixed_comp_cons)
        sp_sym_idx_sets = self.__partition_complete_entity_idx_sets_by_sp_sym(entity_type=const.CON_TYPE,
                                                                              complete_idx_sets=idx_sets)
        sp_idx_sets = self.__partition_sp_entity_idx_sets_by_sp_index(sp_entity_idx_sets=sp_sym_idx_sets)

        # store indexing sets of all mixed-complicating meta-constraints partitioned by subproblem
        for sp_idx_set_dict, sp_container in zip(sp_idx_sets, self.gbd_problem.sp_containers):
            sp_container.mixed_comp_con_idx_set = sp_idx_set_dict

    def __assemble_complete_entity_idx_sets_by_symbol(self, meta_entities: List[mat.MetaEntity]):

        # retrieve the symbols of all entities
        entity_syms = set([me.symbol for me in meta_entities])

        # instantiate the collection of the indexing sets of each entity
        complete_idx_sets = {}  # key: entity symbol; value: indexing set

        # assemble the complete indexing set of each entity by symbol
        for entity_sym in entity_syms:

            # check if the entity is indexed
            if self.gbd_problem.get_meta_entity(entity_sym).get_reduced_dimension() == 0:
                complete_idx_sets[entity_sym] = None  # assign None for a scalar entity

            else:  # proceed with assembling the indexing set for an indexed entity

                entity_idx_subsets = []  # list of indexing subsets for a given entity symbol

                # find the meta-entities that match the current entity symbol
                for me in meta_entities:
                    if me.symbol == entity_sym:

                        # retrieve the indexing subset of the meta-entity
                        entity_idx_subset = me.idx_set_node.evaluate(state=self.gbd_problem.state)[0]

                        # add the indexing subset to the list
                        entity_idx_subsets.append(entity_idx_subset)

                # evaluate the union of all indexing subsets to obtain the complete indexing set
                complete_idx_sets[entity_sym] = OrderedSet.union(*entity_idx_subsets)

        return complete_idx_sets

    def __partition_complete_entity_idx_sets_by_sp_sym(self,
                                                       entity_type: str,
                                                       complete_idx_sets: Dict[str, Optional[OrderedSet]]):

        # instantiate the collection of the indexing sets by subproblem symbol and entity symbol
        sp_entity_idx_sets = {}  # key: primal subproblem symbol; value: dictionary of indexing sets

        # arrange the indexing sets by subproblem symbol and entity symbol
        for primal_sp in self.gbd_problem.primal_sps:

            # instantiate the collection of indexing sets by entity symbol for the current primal subproblem
            sp_entity_idx_sets_i = {}  # key: entity symbol; value: indexing set
            sp_entity_idx_sets[primal_sp.symbol] = sp_entity_idx_sets_i

            # retrieve the selected list of meta-entities
            if entity_type == const.VAR_TYPE:
                meta_entities = primal_sp.model_meta_vars
            elif entity_type == const.OBJ_TYPE:
                meta_entities = primal_sp.model_meta_objs
            elif entity_type == const.CON_TYPE:
                meta_entities = primal_sp.model_meta_cons
            else:
                raise ValueError("GBD problem builder encountered an unexpected entity type '{0}'".format(entity_type)
                                 + " while partitioning entity indexing sets by subproblem")

            # iterate over the relevant meta-entities of the current primal subproblem
            for me in meta_entities:
                if me.symbol in complete_idx_sets:

                    # check if the entity is indexed

                    if me.get_reduced_dimension() == 0:  # scalar entity
                        sp_entity_idx_sets_i[me.symbol] = None  # assign None for a scalar entity

                    else:  # indexed entity

                        # retrieve the indexing subset of the meta-entity for the current primal subproblem
                        entity_idx_set = me.idx_set_node.evaluate(state=self.gbd_problem.state)[0]

                        # retrieve the indexing set elements that designate relevant entities
                        sp_entity_idx_set = entity_idx_set.intersection(complete_idx_sets[me.symbol])

                        # update an existing indexing set
                        if me.symbol in sp_entity_idx_sets_i:
                            sp_entity_idx_sets_i[me.symbol] |= sp_entity_idx_set

                        # assign a new indexing set otherwise
                        else:
                            sp_entity_idx_sets_i[me.symbol] = sp_entity_idx_set

        return sp_entity_idx_sets

    def __partition_sp_entity_idx_sets_by_sp_index(self,
                                                   sp_entity_idx_sets: Dict[str, Dict[str, Optional[mat.IndexingSet]]]):

        partitioned_entity_idx_sets = []

        # arrange the entity indexing sets by subproblem symbol, subproblem index, and entity symbol
        for sp_container in self.gbd_problem.sp_containers:

            partitioned_entity_idx_sets_sp = {}
            partitioned_entity_idx_sets.append(partitioned_entity_idx_sets_sp)

            # retrieve the indexing sets for the current subproblem
            sp_entity_idx_sets_i = sp_entity_idx_sets[sp_container.primal_sp.symbol]

            # iterate over all relevant entity indexing sets that belong to the current subproblem
            for entity_sym, sp_entity_idx_set in sp_entity_idx_sets_i.items():

                if sp_entity_idx_set is None:  # scalar entity
                    partitioned_entity_idx_sets_sp[entity_sym] = None  # assign None

                else:  # indexed entity

                    # retrieve the parent meta-entity
                    parent_meta_entity = self.gbd_problem.get_meta_entity(entity_sym)

                    # instantiate a blank filter element
                    filter_element = [None] * parent_meta_entity.get_reduced_dimension()

                    # populate the filter element with sub-elements of the current subproblem index
                    filter_element = self.generate_entity_sp_index(sp_index=sp_container.sp_index,
                                                                   meta_entity=parent_meta_entity,
                                                                   entity_index=filter_element)

                    # filter the indexing set by the subproblem index
                    filtered_var_idx_set = mat.filter_set(sp_entity_idx_set, filter_element)

                    # assign the indexing set to the subproblem container
                    partitioned_entity_idx_sets_sp[entity_sym] = filtered_var_idx_set

        return partitioned_entity_idx_sets

    # Scripts
    # ------------------------------------------------------------------------------------------------------------------

    def __build_and_write_script(self):
        self.__clean_script()
        self.__build_model_scripts()
        self.__build_problem_declarations()

        self.gbd_problem.compound_script.write(dir_path=self.gbd_problem.working_dir_path,
                                               main_file_name=self.gbd_problem.symbol + ".run")

    def __clean_script(self):

        main_script = self.gbd_problem.compound_script.main_script

        # TODO: account for special commands and other types of file statements
        i = 0
        while i < len(main_script):  # terminate after final statement

            statement = main_script.statements[i]

            # Remove comments
            if isinstance(statement, stm.Comment):
                main_script.statements.pop(i)
                i -= 1

            # Remove display and print statements
            elif isinstance(statement, stm.DisplayStatement):
                main_script.statements.pop(i)
                i -= 1

            # Remove model file include statements
            elif isinstance(statement, stm.FileStatement):
                if statement.command == "model":
                    main_script.statements.pop(i)
                    i -= 1

            # Remove entity declarations
            elif isinstance(statement, stm.ModelEntityDeclaration):
                main_script.statements.pop(i)
                i -= 1

            # Remove solve statements
            elif isinstance(statement, stm.SolveStatement):
                main_script.statements.pop(i)
                i -= 1

            i += 1  # increment statement index

    def __build_model_scripts(self):

        ref_mod_script = self.__build_reformulated_model_script()
        mp_mod_script = self.__build_mp_model_script()
        fbl_mod_script = self.__build_fbl_sp_model_script()

        self.gbd_problem.compound_script.add_included_script(ref_mod_script, file_command="model")
        self.gbd_problem.compound_script.add_included_script(mp_mod_script,
                                                             file_command="model",
                                                             statement_index=1)
        self.gbd_problem.compound_script.add_included_script(fbl_mod_script,
                                                             file_command="model",
                                                             statement_index=2)

    def __build_reformulated_model_script(self):
        script_builder = ScriptBuilder()
        return script_builder.generate_problem_model_script(problem=self.gbd_problem,
                                                            model_file_extension=".modr")

    def __build_mp_model_script(self):

        meta_sets_params = [self.gbd_problem.cut_count,
                            self.gbd_problem.cuts,
                            self.gbd_problem.stored_obj,
                            self.gbd_problem.is_feasible]
        meta_sets_params.extend([mp for _, mp in self.gbd_problem.stored_comp_decisions.items()])
        meta_sets_params.extend([mp for _, mp in self.gbd_problem.duality_multipliers.items()])

        meta_vars = [self.gbd_problem.eta, self.gbd_problem.aux_f_meta_var]
        meta_vars += [aux_g for _, aux_g in self.gbd_problem.aux_g_meta_vars.items()]

        master_obj = self.gbd_problem.master_obj

        meta_cons = [gbd_cut for _, gbd_cut in self.gbd_problem.gbd_cuts.items()]

        script_builder = ScriptBuilder()
        return script_builder.generate_model_script(model_file_name=self.gbd_problem.symbol,
                                                    model_file_extension=".modm",
                                                    meta_sets_params=meta_sets_params,
                                                    meta_vars=meta_vars,
                                                    meta_objs=[master_obj],
                                                    meta_cons=meta_cons)

    def __build_fbl_sp_model_script(self):

        fbl_objs = [fp.model_meta_objs[0] for fp in self.gbd_problem.fbl_sps]

        script_builder = ScriptBuilder()
        return script_builder.generate_model_script(model_file_name=self.gbd_problem.symbol,
                                                    model_file_extension=".modrsl",
                                                    meta_vars=self.gbd_problem.slack_vars,
                                                    meta_objs=fbl_objs,
                                                    meta_cons=self.gbd_problem.sl_fbl_cons)

    def __build_problem_declarations(self):

        idx_meta_sets = [ms for ms in self.gbd_problem.idx_meta_sets.values()]

        main_script = self.gbd_problem.compound_script.main_script
        script_builder = ScriptBuilder()

        for primal_sp in self.gbd_problem.primal_sps:
            primal_sp_decl = script_builder.generate_subproblem_declaration(problem=self.gbd_problem,
                                                                            subproblem=primal_sp,
                                                                            idx_meta_sets=idx_meta_sets)
            main_script.statements.append(primal_sp_decl)

        for fbl_sp in self.gbd_problem.fbl_sps:
            fbl_sp_decl = script_builder.generate_subproblem_declaration(problem=self.gbd_problem,
                                                                         subproblem=fbl_sp,
                                                                         idx_meta_sets=idx_meta_sets)
            main_script.statements.append(fbl_sp_decl)

        mp_decl = script_builder.generate_subproblem_declaration(problem=self.gbd_problem,
                                                                 subproblem=self.gbd_problem.mp)
        main_script.statements.append(mp_decl)

    # Utility
    # ------------------------------------------------------------------------------------------------------------------

    def generate_free_node_id(self) -> int:
        return self.node_builder.generate_free_node_id()

    def __retrieve_meta_entity_from_definition(self, entity_def: str):

        ampl_parser = AMPLParser(self.gbd_problem)
        idx_set_node, entity_node = ampl_parser.parse_declared_entity_and_idx_set(entity_def)

        sym = entity_node.symbol
        if sym in self.gbd_problem.origin_to_std_con_map:
            std_syms = [mc.symbol for mc in self.gbd_problem.origin_to_std_con_map[sym]]
        else:
            std_syms = [sym]

        meta_entities = []

        for std_sym in std_syms:
            parent_meta_entity = self.gbd_problem.get_meta_entity(std_sym)
            if idx_set_node is None:
                meta_entities.append(parent_meta_entity)
            else:
                sub_meta_entity = self.entity_builder.build_sub_meta_entity(idx_subset_node=idx_set_node,
                                                                            meta_entity=parent_meta_entity,
                                                                            entity_idx_node=entity_node.idx_node)
                meta_entities.append(sub_meta_entity)

        return meta_entities

    def generate_entity_sp_index(self,
                                 sp_index: mat.Element,
                                 meta_entity: mat.MetaEntity,
                                 entity_index: List[Union[int, float, str, None]] = None):

        if entity_index is None:
            entity_index = list(meta_entity.get_dummy_symbols())  # retrieve default entity index

        sp_idx_pos = 0  # first index position of the current indexing meta-set
        for idx_meta_set in self.gbd_problem.idx_meta_sets.values():

            if meta_entity.is_indexed_with(idx_meta_set):

                idx_syms = sp_index[sp_idx_pos:sp_idx_pos + idx_meta_set.reduced_dimension]
                sp_idx_pos += idx_meta_set.reduced_dimension  # update position of the subproblem index

                ent_idx_pos = meta_entity.get_first_reduced_dim_index_of_idx_set(idx_meta_set)
                entity_index[ent_idx_pos:ent_idx_pos + idx_meta_set.reduced_dimension] = idx_syms  # update entity index

        return entity_index
