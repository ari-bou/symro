from typing import Dict, List, Optional, Union
import numpy as np

import symro.mat as mat
from symro.prob.problem import Problem
import symro.scripting.script as scr


class AMPLParser:

    # Constants
    # ------------------------------------------------------------------------------------------------------------------

    # --- Arithmetic Operation Symbols ---

    ARITH_UNA_OPR_SYMBOLS = ["+", "-"]

    ARITH_BIN_OPR_SYMBOLS_12 = ["+", "-", "less"]
    ARITH_BIN_OPR_SYMBOLS_14 = ["*", "/", "div", "mod"]
    ARITH_BIN_OPR_SYMBOLS_16 = ["^", "**"]
    ARITH_BIN_OPR_SYMBOLS = (
        ARITH_BIN_OPR_SYMBOLS_12 + ARITH_BIN_OPR_SYMBOLS_14 + ARITH_BIN_OPR_SYMBOLS_16
    )

    ARITH_FUNCTION_SYMBOLS = {
        "abs": mat.ABSOLUTE_VALUE_FUNCTION,
        "alias": mat.ALIAS_FUNCTION,
        "Beta": mat.BETA_RNG_FUNCTION,
        "Cauchy": mat.CAUCHY_RNG_FUNCTION,
        "ceil": mat.CEILING_FUNCTION,
        "floor": mat.FLOOR_FUNCTION,
        "cos": mat.COSINE_FUNCTION,
        "cosh": mat.HYPERBOLIC_COSINE_FUNCTION,
        "acos": mat.INVERSE_COSINE_FUNCTION,
        "acosh": mat.INVERSE_HYPERBOLIC_COSINE_FUNCTION,
        "exp": mat.EXPONENTIAL_FUNCTION,
        "Exponential": mat.EXPONENTIAL_RNG_FUNCTION,
        "Gamma": mat.GAMMA_RNG_FUNCTION,
        "Irand224": mat.Irand224_RNG_FUNCTION,
        "log": mat.NATURAL_LOGARITHM_FUNCTION,
        "log10": mat.BASE_10_LOGARITHM_FUNCTION,
        "ln": mat.NATURAL_LOGARITHM_FUNCTION,
        "max": mat.MAXIMUM_FUNCTION,
        "min": mat.MINIMUM_FUNCTION,
        "Normal": mat.NORMAL_RNG_FUNCTION,
        "Normal01": mat.STANDARD_NORMAL_RNG_FUNCTION,
        "Poisson": mat.POISSON_RNG_FUNCTION,
        "round": mat.ROUND_FUNCTION,
        "precision": mat.PRECISION_FUNCTION,
        "sin": mat.SINE_FUNCTION,
        "sinh": mat.HYPERBOLIC_SINE_FUNCTION,
        "asin": mat.INVERSE_SINE_FUNCTION,
        "asinh": mat.INVERSE_HYPERBOLIC_SINE_FUNCTION,
        "sqrt": mat.SQUARE_ROOT_FUNCTION,
        "sum": mat.SUMMATION_FUNCTION,
        "prod": mat.PRODUCT_FUNCTION,
        "tan": mat.TANGENT_FUNCTION,
        "tanh": mat.HYPERBOLIC_TANGENT_FUNCTION,
        "atan": mat.INVERSE_TANGENT_FUNCTION,
        "atanh": mat.INVERSE_HYPERBOLIC_TANGENT_FUNCTION,
        "atanh2": mat.INVERSE_HYPERBOLIC_TANGENT_FUNCTION_2,
        "time": mat.TIME_FUNCTION,
        "ctime": mat.CTIME_FUNCTION,
        "trunc": mat.TRUNCATION_FUNCTION,
        "Uniform": mat.UNIFORM_RNG_FUNCTION,
        "Uniform01": mat.STANDARD_UNIFORM_RNG_FUNCTION,
    }

    # --- Logical Operation Symbols ---

    REL_OPR_SYMBOLS = ["<", "<=", "<>", "=", "==", "!=", ">=", ">"]

    LOGIC_UNA_OPR_SYMBOLS = ["!", "not"]
    LOGIC_BIN_OPR_SYMBOLS = ["&&", "||", "and", "or"]

    # --- Set Operation Symbols ---

    INFINITE_SET_SYMBOLS = ["Reals", "Integers", "ASCII", "EBCDIC", "Display"]

    SET_OPR_SYMBOLS_8 = ["union", "diff", "symdiff"]
    SET_OPR_SYMBOLS_9 = ["inter"]
    SET_OPR_SYMBOLS = SET_OPR_SYMBOLS_8 + SET_OPR_SYMBOLS_9

    SET_FUNCTION_SYMBOLS = [
        "next",
        "nextw",
        "prev",
        "prevw",
        "first",
        "last",
        "member",
        "ord",
        "ord0",
        "card",
        "arity",
        "indexarity",
    ]

    # --- String Operation Symbols ---
    STR_FUNCTION_SYMBOLS = [
        "num",
        "num0",
        "ichar",
        "char",
        "length",
        "substr",
        "sprintf",
        "match",
        "sub",
        "gsub",
    ]

    # --- Suffixes ---
    # A.11
    VAR_SUFFIXES = [
        "init",
        "init0",
        "lb",
        "lb0",
        "lb1",
        "lb2",
        "ub",
        "ub0",
        "ub1",
        "ub2",
        "rc",
        "lrc",
        "urc",
        "relax",
        "slack",
        "lslack",
        "uslack",
        "status",
        "sstatus",
        "astatus",
        "val",
    ]
    OBJ_SUFFIXES = ["val"]
    CON_SUFFIXES = [
        "body",
        "dinit",
        "dinit0",
        "dual",
        "ldual",
        "udual",
        "lb",
        "lbs",
        "ub",
        "ubs",
        "slack",
        "lslack",
        "uslack",
        "status",
        "sstatus",
        "astatus",
    ]
    SUFFIX_DIRECTIONS = ["IN", "OUT", "INOUT", "LOCAL"]

    # --- Predefined Entities ---

    GENERIC_SET_SYMBOLS = [
        "_PARS",
        "_SETS",
        "_VARS",
        "_CONS",
        "_PROBS",
        "_ENVS",
        "_FUNCS",
    ]

    PREDEF_SET_SYMBOLS = INFINITE_SET_SYMBOLS + GENERIC_SET_SYMBOLS

    # A.19.4
    GENERIC_PARAM_SYMBOLS = [
        "_nvars",
        "_ncons",
        "_nobjs",
        "_varname",
        "_conname",
        "_objname",
        "_var",
        "_con",
        "_obj",
        "_snvars",
        "_sncons",
        "_snobjs",
        "_svarname",
        "_sconname",
        "_sobjname",
        "_svar",
        "_scon",
        "_sobj",
        "_nccons",
        "_cconname",
        "_scvar",
        "_snbvars",
        "_snccons",
        "_snivars",
        "_snlcc",
        "_snlnc",
        "_snnlcc",
        "_snnlcons",
        "_snnlnc",
        "_snnlobjs",
        "_snnlv",
        "_snzcons",
        "_snzobjs",
    ]

    PREDEF_PARAM_SYMBOLS = ["Infinity"] + GENERIC_PARAM_SYMBOLS

    # --- Command Symbols ---
    COMMAND_SYMBOLS = [
        "break",
        "call",
        "cd",
        "check",
        "close",
        "commands",
        "continue",
        "csvdisplay",
        "data",
        "delete",
        "display",
        "_display",
        "drop",
        "end",
        "environ",
        "exit",
        "expand",
        "fix",
        "function",
        "include",
        "let",
        "load",
        "model",
        "objective",
        "option",
        "print",
        "printf",
        "problem",
        "purge",
        "quit",
        "read",
        "redeclare",
        "reload",
        "remove",
        "reset",
        "restore",
        "shell",
        "show",
        "solexpand",
        "solution",
        "solve",
        "suffix",
        "table",
        "update",
        "unfix",
        "unload",
        "write",
        "xref",
    ]
    COMPOUND_COMMAND_SYMBOLS = ["if", "for", "repeat"]

    REDIRECTION_OPERATORS = ["<", ">", ">>"]

    # Construction
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, problem: Problem = None, working_dir_path: str = ""):

        self._lexer: AMPLLexer = AMPLLexer()

        self.problem: Optional[Problem] = problem

        self.expressions: Dict[str, mat.Expression] = {}

        self._active_script: Optional[scr.Script] = None
        self.compound_script: Optional[scr.CompoundScript] = None

        self.__free_node_id: int = -1

        self._can_evaluate: bool = True
        self._can_include_in_model: bool = True

        self.working_dir_path: str = working_dir_path

    def _setup(self):
        self._can_evaluate = True
        self._can_include_in_model = True
        self.expressions = {}

    # Tokenization
    # ------------------------------------------------------------------------------------------------------------------

    def tokenize(self, literal: str):
        return self._lexer.tokenize(literal)

    def _tokenize(self, literal: str) -> scr.Script:

        self._setup()

        tokens = self._lexer.tokenize(literal)

        main_script = scr.Script(
            name="main",
            script_type=scr.ScriptType.COMMANDS,
            raw_literal=literal,
            tokens=tokens,
        )

        self.compound_script = scr.CompoundScript(main_script=main_script)

        return main_script

    def _tokenize_included_file(
        self, literal: str, script_id: str, script_type: scr.ScriptType
    ) -> scr.Script:

        self._setup()

        tokens = self._lexer.tokenize(literal)

        script = scr.Script(
            name=script_id, script_type=script_type, raw_literal=literal, tokens=tokens
        )
        self.compound_script.included_scripts[script_id] = script

        return script

    # Expression Parsing
    # ------------------------------------------------------------------------------------------------------------------

    def parse_expression(self, literal: str):
        script = self._tokenize(literal)
        self._active_script = script
        return self._parse_expression()

    def _parse_expression(self):
        return self._parse_logical_expression()

    # Conditional Expression Parsing
    # ------------------------------------------------------------------------------------------------------------------

    def __parse_conditional_expression(
        self,
    ) -> Union[mat.ArithmeticConditionalNode, mat.SetConditionalNode]:

        operands = []
        conditions = []

        while self.get_token() in ("if", "else"):

            if self.get_token() == "else":
                self._next_token()  # skip 'else'

            # condition
            if self.get_token() == "if":

                self._next_token()  # skip 'if'

                conditions.append(self._parse_logical_expression())

                self._enforce_token_value("then")
                self._next_token()  # skip 'then'

            operands.append(self._parse_set_expression())

        if isinstance(operands[0], mat.SetExpressionNode):
            return mat.SetConditionalNode(operands=operands, conditions=conditions)

        else:
            return mat.ArithmeticConditionalNode(
                operands=operands, conditions=conditions
            )
            # return self.__convert_conditional_arithmetic_expression_to_addition_operation(operands, conditions)

    # Logical Expression Parsing
    # ------------------------------------------------------------------------------------------------------------------

    def parse_logical_expression(self, literal: str):
        script = self._tokenize(literal)
        self._active_script = script
        return self._parse_logical_expression()

    def _parse_logical_expression(
        self,
    ) -> Union[
        mat.LogicalExpressionNode,
        mat.SetExpressionNode,
        mat.ArithmeticExpressionNode,
        mat.StringExpressionNode,
        mat.BaseDummyNode,
    ]:

        root_operation = self.__parse_logical_operand()

        # parse next logical operation
        while self.get_token() in self.LOGIC_BIN_OPR_SYMBOLS:

            operator_sym = self.get_token()  # parse logical operator
            self._next_token()  # skip operator

            # retrieve operator
            if operator_sym in ("&&", "and"):
                operator = mat.CONJUNCTION_OPERATOR
            else:
                operator = mat.DISJUNCTION_OPERATOR

            # parse next logical operand
            rhs_operand = self.__parse_logical_operand()

            # build logical operation node
            root_operation = mat.LogicalOperationNode(
                operator=operator, operands=[root_operation, rhs_operand]
            )

        return root_operation

    def __parse_logical_operand(
        self,
    ) -> Union[
        mat.LogicalExpressionNode,
        mat.SetExpressionNode,
        mat.ArithmeticExpressionNode,
        mat.StringExpressionNode,
        mat.BaseDummyNode,
    ]:

        token = self.get_token()

        # unary negation
        if token in ["!", "not"]:
            self._next_token()
            operand = self.__parse_logical_operand()
            return mat.LogicalOperationNode(
                operator=mat.UNARY_INVERSION_OPERATOR, operands=[operand]
            )

        # logical reduction
        if token in ["exists", "forall"]:

            self._next_token()  # skip operator

            # parse indexing set
            idx_set_node = None
            if self.get_token() == "{":
                idx_set_node = self._parse_indexing_set_definition()

            operand = self.__parse_logical_operand()

            # retrieve operator
            if token == "exists":
                operator = mat.EXISTS_OPERATOR
            else:
                operator = mat.FOR_ALL_OPERATOR

            return mat.LogicalReductionNode(
                operator=operator, idx_set_node=idx_set_node, operand=operand
            )

        else:

            node = self._parse_relational_expression()

            is_member = False
            if (
                isinstance(node, mat.BaseDummyNode)
                or isinstance(node, mat.ArithmeticExpressionNode)
                or isinstance(node, mat.StringExpressionNode)
            ):
                is_member = True

            # Set Membership
            if is_member and self.get_token() in ["in", "not"]:
                return self.__parse_set_membership_operation(node)

            # Set Comparison
            elif isinstance(node, mat.SetExpressionNode) and self.get_token() in [
                "within",
                "not",
            ]:
                return self.__parse_set_comparison_operation(node)

            # Other
            else:
                return node

    def __parse_set_comparison_operation(
        self, lhs_operand: mat.SetExpressionNode
    ) -> mat.SetComparisonOperationNode:

        operator = ""
        if self.get_token() == "not":
            operator = "not "
            self._next_token()
        operator += "within"
        self._next_token()

        rhs_operand = self._parse_set_expression()
        return mat.SetComparisonOperationNode(
            operator=operator, lhs_operand=lhs_operand, rhs_operand=rhs_operand
        )

    def __parse_set_membership_operation(
        self,
        dummy_node: Union[
            mat.BaseDummyNode, mat.ArithmeticExpressionNode, mat.StringExpressionNode
        ],
    ) -> mat.SetMembershipOperationNode:

        operator = ""
        if self.get_token() == "not":
            operator = "not "
            self._next_token()
        operator += "in"
        self._next_token()

        rhs_operand = self._parse_set_expression()
        return mat.SetMembershipOperationNode(
            operator=operator, member_node=dummy_node, set_node=rhs_operand
        )

    def parse_relational_expression(self, literal: str):
        script = self._tokenize(literal)
        self._active_script = script
        return self._parse_relational_expression()

    def _parse_relational_expression(
        self,
    ) -> Union[
        mat.RelationalOperationNode,
        mat.SetExpressionNode,
        mat.ArithmeticExpressionNode,
        mat.StringExpressionNode,
        mat.BaseDummyNode,
    ]:

        root_operation = self._parse_set_expression()

        while self.get_token() in self.REL_OPR_SYMBOLS:

            operator_sym = self.get_token()
            self._next_token()  # skip operator

            # retrieve operator

            if operator_sym in ("=", "=="):
                operator = mat.EQUALITY_OPERATOR

            elif operator_sym in ("!=", "<>"):
                operator = mat.STRICT_INEQUALITY_OPERATOR

            elif operator_sym == "<":
                operator = mat.LESS_INEQUALITY_OPERATOR

            elif operator_sym == "<=":
                operator = mat.LESS_EQUAL_INEQUALITY_OPERATOR

            elif operator_sym == ">":
                operator = mat.GREATER_INEQUALITY_OPERATOR

            else:
                operator = mat.GREATER_EQUAL_INEQUALITY_OPERATOR

            rhs_operand = self._parse_set_expression()  # parse next operand

            # build relational operation node
            root_operation = mat.RelationalOperationNode(
                operator=operator, lhs_operand=root_operation, rhs_operand=rhs_operand
            )

        return root_operation

    # Set Expression Parsing
    # ------------------------------------------------------------------------------------------------------------------

    def parse_set_expression(self, literal: str):
        script = self._tokenize(literal)
        self._active_script = script
        return self._parse_set_expression()

    def _parse_set_expression(
        self, precedence: int = 8
    ) -> Union[
        mat.SetExpressionNode,
        mat.ArithmeticExpressionNode,
        mat.StringExpressionNode,
        mat.BaseDummyNode,
    ]:

        # parse the first operand
        if precedence == 8:
            root_operation = self._parse_set_expression(precedence=9)
        else:
            root_operation = self.__parse_set()

        if precedence == 8:  # precedence level 8
            operators = self.SET_OPR_SYMBOLS_8
        else:  # precedence level 9
            operators = self.SET_OPR_SYMBOLS_9

        # parse next set operation
        while self.get_token() in operators:

            operator_sym = self.get_token()
            self._next_token()  # skip operator

            # retrieve operator
            if operator_sym == "union":
                operator = mat.UNION_OPERATOR
            elif operator_sym == "inter":
                operator = mat.INTERSECTION_OPERATOR
            elif operator_sym == "diff":
                operator = mat.DIFFERENCE_OPERATOR
            else:
                operator = mat.SYMMETRIC_DIFFERENCE_OPERATOR

            # parse next RHS operand
            if precedence == 8:  # precedence level 8
                rhs_operand = self._parse_set_expression(precedence=9)
            else:  # precedence level 9
                rhs_operand = self.__parse_set()

            # build a new binary set operation node
            root_operation = mat.SetOperationNode(
                operator=operator, lhs_operand=root_operation, rhs_operand=rhs_operand
            )

        return root_operation

    def __parse_set(
        self,
    ) -> Union[
        mat.SetExpressionNode,
        mat.ArithmeticExpressionNode,
        mat.StringExpressionNode,
        mat.BaseDummyNode,
    ]:

        token = self.get_token()

        # Opening curly brace
        if self.get_token() == "{":
            return self._parse_indexing_set_definition()

        # Set Reduction Operation
        elif token in ["union", "inter", "setof"]:

            self._next_token()

            # Indexing set
            idx_set_node = None
            if self.get_token() == "{":
                idx_set_node = self._parse_indexing_set_definition()

            if token == "setof":
                # TODO: implement parsing logic for setof operator
                raise NotImplementedError(
                    "Parsing logic for 'setof' has not yet been implemented"
                )
            else:
                operand = self._parse_set_expression()

            # retrieve operator
            if token == "union":
                operator = mat.UNION_OPERATOR
            elif token == "inter":
                operator = mat.INTERSECTION_OPERATOR
            else:
                operator = mat.SETOF_OPERATOR

            set_reduc_op = mat.SetReductionNode(
                operator=operator, idx_set_node=idx_set_node, operand=operand
            )
            return set_reduc_op

        # Declared Set
        elif token in self.problem.meta_sets:
            return self._parse_declared_entity()

        # Predefined Set
        elif token in self.PREDEF_SET_SYMBOLS:
            if token in self.INFINITE_SET_SYMBOLS:
                # TODO: implement parsing logic for infinite sets
                raise NotImplementedError(
                    "Parsing logic for infinite sets has not been implemented"
                )
            else:
                # TODO: implement parsing logic for generic sets
                raise NotImplementedError(
                    "Parsing logic for generic sets has not been implemented"
                )

        else:
            node = self._parse_arithmetic_expression()

            # Ordered Set
            if self.get_token() == "..":
                return self.__parse_ordered_set(node)

            else:
                return node

    def parse_indexing_set_definition(self, literal: str) -> mat.CompoundSetNode:
        script = self._tokenize(literal)
        self._active_script = script
        return self._parse_indexing_set_definition()

    def _parse_indexing_set_definition(
        self,
    ) -> Union[mat.CompoundSetNode, mat.EnumeratedSetNode]:

        # Start at opening brace '{'
        self._next_token()  # skip opening brace '{'

        if self.get_token() == "}":  # empty set
            set_node = mat.EnumeratedSetNode()

        else:
            set_node = self.__parse_set_definition_body()
            if not isinstance(set_node, mat.CompoundSetNode) and not isinstance(
                set_node, mat.EnumeratedSetNode
            ):
                raise ValueError(
                    "AMPL parser expected either compound set or an enumerate set"
                    + " while parsing indexing set expression"
                )

        self._enforce_token_value("}")
        self._next_token()  # skip closing brace '}'

        return set_node

    def __parse_set_definition_body(self) -> mat.BaseSetNode:

        nodes = []  # list of set or element nodes
        con_node = None  # set constraint
        is_explicit: bool = False  # denotes whether the set is explicitly enumerated

        # Build the component nodes
        while True:

            node = self._parse_set_expression()

            if isinstance(node, mat.BaseDummyNode):
                if self.get_token() == "in":
                    self._next_token()
                    node = self.__parse_indexing_set(node)
                else:
                    is_explicit = True
            elif isinstance(node, mat.ArithmeticExpressionNode) or isinstance(
                node, mat.StringExpressionNode
            ):
                is_explicit = True

            nodes.append(node)

            if self.get_token() == ",":
                self._next_token()
            elif self.get_token() == ":":
                self._next_token()
                con_node = self._parse_logical_expression()
                break
            elif self.get_token() == "}":
                break

        # Build the set node
        if not is_explicit:
            return mat.CompoundSetNode(set_nodes=nodes, constraint_node=con_node)
        else:
            return mat.EnumeratedSetNode(element_nodes=nodes)

    def __parse_indexing_set(
        self, dummy_node: mat.BaseDummyNode
    ) -> mat.IndexingSetNode:

        # add unbound symbol(s) to problem

        if isinstance(dummy_node, mat.DummyNode):
            self.problem.unbound_symbols.add(dummy_node.symbol)

        elif isinstance(dummy_node, mat.CompoundDummyNode):
            for cmpt_node in dummy_node.component_nodes:
                if isinstance(cmpt_node, mat.DummyNode):
                    self.problem.unbound_symbols.add(cmpt_node.symbol)

        set_node = self._parse_set_expression()  # parse set expression

        # build indexing set node
        return mat.IndexingSetNode(dummy_node=dummy_node, set_node=set_node)

    def __parse_ordered_set(
        self, start_node: Union[mat.ArithmeticExpressionNode, mat.DummyNode]
    ) -> mat.OrderedSetNode:

        self._enforce_token_value("..")
        self._next_token()  # skip '..'

        end_node = self._parse_arithmetic_expression()

        ordered_set_node = mat.OrderedSetNode(start_node=start_node, end_node=end_node)
        return ordered_set_node

    # Arithmetic Expression Parsing
    # ------------------------------------------------------------------------------------------------------------------

    def parse_arithmetic_expression(self, literal: str):
        script = self._tokenize(literal)
        self._active_script = script
        return self._parse_arithmetic_expression()

    def _parse_arithmetic_expression(
        self, precedence: int = 12
    ) -> Union[mat.ArithmeticExpressionNode, mat.StringExpressionNode]:
        # parse first operand

        if precedence == 12:
            root_operation = self._parse_arithmetic_expression(precedence=14)
        elif precedence == 14:
            root_operation = self.__parse_unary_arithmetic_operation()
        else:  # precedence level 16
            root_operation = self.__parse_arithmetic_operand()

        arith_operation: Optional[mat.ArithmeticOperationNode] = None

        if precedence == 12:  # precedence level 12
            operators = self.ARITH_BIN_OPR_SYMBOLS_12
        elif precedence == 14:  # precedence level 14
            operators = self.ARITH_BIN_OPR_SYMBOLS_14
        else:  # precedence level 16
            operators = self.ARITH_BIN_OPR_SYMBOLS_16

        while self.get_token() in operators:  # parse next arithmetic operation

            operator_sym = self.get_token()
            self._next_token()  # skip operator

            if precedence == 12:  # precedence level 12
                rhs_operand = self._parse_arithmetic_expression(precedence=14)
            elif precedence == 14:  # precedence level 14
                rhs_operand = self.__parse_unary_arithmetic_operation()
            else:  # precedence level 16
                rhs_operand = self.__parse_arithmetic_operand()

            # build a binary arithmetic operation node for the following operations:
            # - non-negative subtraction
            # - integer division
            # - modulus
            # - exponentiation
            if operator_sym in ["less", "div", "mod", "^", "**"]:

                # less
                if operator_sym == "less":
                    root_operation = self.__convert_less_operation_to_non_negative_max(
                        lhs_operand=root_operation, rhs_operand=rhs_operand
                    )

                # div or mod
                elif operator_sym == "div" or operator_sym == "mod":
                    root_operation = mat.ArithmeticTransformationNode(
                        fcn=self.ARITH_FUNCTION_SYMBOLS[operator_sym],
                        operands=[root_operation, rhs_operand],
                    )

                # exponentiation
                else:
                    root_operation = mat.ExponentiationNode(
                        lhs_operand=root_operation, rhs_operand=rhs_operand
                    )

                arith_operation = None

            # build a multi arithmetic operation node for the following operations:
            # - addition
            # - subtraction
            # - multiplication
            # - division
            else:

                # convert subtraction operation to addition operation
                if operator_sym == "-":
                    operator_sym = "+"  # change the operator
                    rhs_operand = self.append_negative_unity_coefficient(rhs_operand)

                # convert division operation to multiplication operation
                elif operator_sym == "/":
                    operator_sym = "*"  # change the operator
                    rhs_operand = self.__convert_divisor_to_fraction(
                        divisor=rhs_operand
                    )

                # build a new arithmetic operation node if it does not exist yet
                if arith_operation is None:
                    if operator_sym == "+":
                        arith_operation = mat.AdditionNode(
                            operands=[root_operation, rhs_operand]
                        )
                    else:
                        arith_operation = mat.MultiplicationNode(
                            operands=[root_operation, rhs_operand]
                        )
                    root_operation = arith_operation

                # otherwise, add the RHS operand to the existing arithmetic operation node
                else:
                    arith_operation.operands.append(rhs_operand)

        return root_operation

    def __convert_less_operation_to_non_negative_max(
        self,
        lhs_operand: mat.ArithmeticExpressionNode,
        rhs_operand: mat.ArithmeticExpressionNode,
    ):
        zero_node = mat.NumericNode(value=0)
        rhs_operand = self.append_negative_unity_coefficient(rhs_operand)
        sub_node = mat.AdditionNode(operands=[lhs_operand, rhs_operand])
        return mat.ArithmeticTransformationNode(
            fcn=mat.MAXIMUM_FUNCTION, idx_set_node=None, operands=[sub_node, zero_node]
        )

    @staticmethod
    def __convert_divisor_to_fraction(divisor: mat.ArithmeticExpressionNode):

        # build a numeric node with value 1
        one_node = mat.NumericNode(value=1)

        # build a division operation node
        fraction = mat.DivisionNode(
            lhs_operand=one_node, rhs_operand=divisor, is_prioritized=True
        )

        return fraction

    def __parse_unary_arithmetic_operation(
        self,
    ) -> Union[
        mat.ArithmeticExpressionNode,
        mat.SetExpressionNode,
        mat.StringExpressionNode,
        mat.BaseDummyNode,
    ]:

        token = self.get_token()

        # Unary operator
        if token in self.ARITH_UNA_OPR_SYMBOLS:

            operator = self.get_token()
            self._next_token()  # skip operator

            operand = self._parse_arithmetic_expression(precedence=16)  # parse operand

            # special case: unary positive
            if operator == "+":
                node = operand

            # special case: unary negation
            elif operator == "-":
                node = self.append_negative_unity_coefficient(operand)

            else:
                raise ValueError(
                    "AMPL parser encountered an unexpected unary arithmetic operator '{0}'".format(
                        operator
                    )
                )

            return node

        else:
            return self._parse_arithmetic_expression(precedence=16)

    @staticmethod
    def append_negative_unity_coefficient(node: mat.ArithmeticExpressionNode):

        # special case 1: node is numeric
        if isinstance(node, mat.NumericNode):
            node.value *= -1
            return node

        # special case 2: node is a multi-operand multiplication node
        elif (
            isinstance(node, mat.ArithmeticOperationNode)
            and node.operator == mat.MULTIPLICATION_OPERATOR
        ):

            has_numeric_factor_node = False

            # look for existing numeric factor nodes in the multiplication node
            for factor in node.operands:
                if isinstance(factor, mat.NumericNode):
                    factor.value *= -1  # negate the coefficient
                    has_numeric_factor_node = True
                    break

            if has_numeric_factor_node:
                return node

            # add -1 coefficient to the node
            else:
                coeff_node = mat.NumericNode(value=-1)
                node.operands.insert(0, coeff_node)
                return node

        # generic case
        else:
            coeff_node = mat.NumericNode(value=-1)
            node.is_prioritized = True
            mult_node = mat.MultiplicationNode(operands=[coeff_node, node])
            return mult_node

    def __parse_arithmetic_operand(
        self,
    ) -> Union[
        mat.ArithmeticExpressionNode,
        mat.SetExpressionNode,
        mat.StringExpressionNode,
        mat.BaseDummyNode,
    ]:

        token = self.get_token()

        # Conditional Expression
        if token == "if":
            return self.__parse_conditional_expression()

        # Variable or Parameter
        elif (
            token in self.problem.meta_vars
            or token in self.problem.meta_params
            or token in self.problem.meta_objs
            or token in self.problem.meta_cons
            or token in self.problem.meta_tables
        ):
            return self._parse_declared_entity()

        # Predefined Parameter
        elif token in self.PREDEF_PARAM_SYMBOLS:
            return self.__parse_predefined_parameter()

        # Function
        elif token in self.ARITH_FUNCTION_SYMBOLS:
            return self.__parse_function()

        # Constant Symbol
        elif token.isnumeric():
            return self.__parse_numeric_constant(token)

        else:
            return self._parse_string_expression()

    def __parse_function(self):

        function_sym = self.get_token()

        # Indexing set
        self._next_token()
        idx_set_node = None
        if self.get_token() == "{":
            idx_set_node = (
                self._parse_indexing_set_definition()
            )  # skip opening and closing curly braces

        # Reductive Function
        if idx_set_node is not None:
            operand = self._parse_arithmetic_expression(precedence=14)
            operand.is_prioritized = True
            operands = [operand]

        # Non-Reductive Function
        else:
            # the arguments of a non-reductive function must be contained in parentheses
            self._enforce_token_value("(")
            self._next_token()  # skip opening parenthesis '('

            operands = []
            while True:
                operands.append(self._parse_arithmetic_expression())
                if self.get_token() != ",":
                    break
                else:
                    self._next_token()  # skip comma

            self._next_token()  # skip closing parenthesis ')'

        function_operation = mat.ArithmeticTransformationNode(
            fcn=self.ARITH_FUNCTION_SYMBOLS[function_sym],
            idx_set_node=idx_set_node,
            operands=operands,
        )

        return function_operation

    # String Expression Parsing
    # ------------------------------------------------------------------------------------------------------------------

    def _parse_string_expression(
        self,
    ) -> Union[
        mat.LogicalExpressionNode,
        mat.SetExpressionNode,
        mat.ArithmeticExpressionNode,
        mat.StringExpressionNode,
        mat.BaseDummyNode,
    ]:

        root_operation = self.__parse_string_term()
        str_operation: Optional[mat.StringOperationNode] = None

        while self.get_token() == "&":

            self.get_token()
            self._next_token()  # skip operator

            rhs_operand = self.__parse_string_term()

            if str_operation is None:
                str_operation = mat.StringOperationNode(
                    operator=mat.CONCATENATION_OPERATOR,
                    operands=[root_operation, rhs_operand],
                )
                root_operation = str_operation

            else:
                str_operation.operands.append(rhs_operand)

        return root_operation

    def __parse_string_term(
        self,
    ) -> Union[
        mat.LogicalExpressionNode,
        mat.SetExpressionNode,
        mat.ArithmeticExpressionNode,
        mat.StringExpressionNode,
        mat.BaseDummyNode,
    ]:

        token = self.get_token()

        # Opening parenthesis
        if token == "(":

            self._next_token()  # skip opening parenthesis
            operand = self._parse_expression()

            token = self.get_token()
            if token == ")":
                self._next_token()  # skip closing parenthesis
                operand.is_prioritized = True
                return operand
            elif token == ",":
                operand = self.__parse_compound_dummy(operand)
                self._next_token()  # skip closing parenthesis
                return operand

        # String Literal
        elif token[0] in ["'", '"']:
            return self._parse_string_literal()

        # Dummy
        else:
            self._next_token()
            return mat.DummyNode(symbol=token)

    def _parse_string_literal(self):

        token = self.get_token()
        self._next_token()  # skip literal

        delimiter = token[0]
        literal = token[1 : len(token) - 1]

        return mat.StringNode(literal=literal, delimiter=delimiter)

    # Entity Parsing
    # ------------------------------------------------------------------------------------------------------------------

    def parse_entity(
        self, literal: str
    ) -> Union[
        mat.DeclaredEntityNode, mat.NumericNode, mat.StringExpressionNode, mat.DummyNode
    ]:
        script = self._tokenize(literal)
        self._active_script = script

        token = self.get_token()
        if (
            token in self.problem.meta_vars
            or token in self.problem.meta_params
            or token in self.problem.meta_objs
            or token in self.problem.meta_cons
            or token in self.problem.meta_tables
        ):
            return self._parse_declared_entity()
        elif token in self.PREDEF_PARAM_SYMBOLS:
            return self.__parse_predefined_parameter()
        elif token.isnumeric():
            return self.__parse_numeric_constant(token)
        else:
            return self._parse_string_expression()

    def parse_declared_entity_and_idx_set(self, literal: str):

        script = self._tokenize(literal)
        self._active_script = script

        idx_set_node = None
        if self.get_token() == "{":
            idx_set_node = self._parse_indexing_set_definition()

        entity_node = self._parse_declared_entity()

        return idx_set_node, entity_node

    def parse_declared_entity(self, literal: str) -> mat.DeclaredEntityNode:
        script = self._tokenize(literal)
        self._active_script = script
        return self._parse_declared_entity()

    def _parse_declared_entity(
        self,
    ) -> Union[mat.DeclaredEntityNode, mat.DeclaredSetNode]:

        symbol = self.get_token()
        self._next_token()  # skip symbol

        entity_type = mat.PARAM_TYPE
        if symbol in self.problem.meta_sets:
            entity_type = mat.SET_TYPE
        if symbol in self.problem.meta_params:
            entity_type = mat.PARAM_TYPE
        elif symbol in self.problem.meta_vars:
            entity_type = mat.VAR_TYPE
        elif symbol in self.problem.meta_objs:
            entity_type = mat.OBJ_TYPE
        elif symbol in self.problem.meta_cons:
            entity_type = mat.CON_TYPE
        elif symbol in self.problem.meta_tables:
            entity_type = mat.TABLE_TYPE
        elif symbol in self.problem.subproblems:
            entity_type = mat.PROB_TYPE

        # index
        index_node = None
        if self.get_token() == "[":
            index_node = self.__parse_entity_index()

        # suffix
        suffix = None
        if self.get_token() == ".":
            self._next_token()  # skip '.'
            suffix = self.get_token()
            self._next_token()  # skip suffix

        if entity_type == mat.SET_TYPE:
            return mat.DeclaredSetNode(
                symbol=symbol, idx_node=index_node, suffix=suffix
            )
        else:
            return mat.DeclaredEntityNode(
                symbol=symbol, idx_node=index_node, suffix=suffix, type=entity_type
            )

    def parse_entity_index(self, literal: str) -> mat.CompoundDummyNode:
        script = self._tokenize(literal)
        self._active_script = script
        return self.__parse_entity_index()

    def __parse_entity_index(self) -> mat.CompoundDummyNode:

        self._next_token()  # skip opening bracket

        nodes = []
        while True:
            nodes.append(self._parse_arithmetic_expression())
            if self.get_token() == ",":
                self._next_token()
            elif self.get_token() == "]":
                break  # end at closing parenthesis
            else:
                raise ValueError(
                    "AMPL parser encountered unexpected token '{0}'".format(
                        self.get_token()
                    )
                    + " while parsing a compound set"
                )

        self._next_token()  # skip closing bracket

        return mat.CompoundDummyNode(component_nodes=nodes)

    def __parse_predefined_parameter(self):

        token = self.get_token()

        # A.7.2
        if token == "Infinity":
            self._next_token()  # skip symbol
            return mat.NumericNode(value=np.inf)
        else:
            # treat it as a declared entity
            return self._parse_declared_entity()  # symbol is skipped in this method

    def __parse_numeric_constant(self, token: str):

        integral = token
        fractional = None

        if not self._is_last_token():
            self._next_token()
            if self.get_token() != ".":
                self._prev_token()
            else:
                self._next_token()
                token = self.get_token()
                if token.isnumeric():
                    fractional = token
                else:
                    self._prev_token()

        if fractional is None:
            coeff_sym = integral
        else:
            coeff_sym = "{0}.{1}".format(integral, fractional)

        is_sci_not = False
        sci_not_sym = ""
        power_sign = ""
        power = ""
        if not self._is_last_token():
            self._next_token()
            if self.get_token() in ["D", "d", "E", "e"]:
                sci_not_sym = self.get_token()
                is_sci_not = True
                self._next_token()
                if self.get_token() in ["+", "-"]:
                    power_sign = self.get_token()
                    self._next_token()
                power = self.get_token()
            else:
                self._prev_token()

        self._next_token()

        symbol = "{0}{1}{2}{3}".format(coeff_sym, sci_not_sym, power_sign, power)
        return mat.NumericNode(
            value=symbol,
            sci_not=is_sci_not,
            coeff_sym=coeff_sym,
            power_sign=power_sign,
            power_sym=power,
        )

    def __parse_compound_dummy(
        self,
        first_dummy_node: Union[
            mat.DummyNode, mat.ArithmeticExpressionNode, mat.StringExpressionNode
        ],
    ) -> mat.CompoundDummyNode:
        component_nodes = [first_dummy_node]
        self._next_token()  # skip first comma in dummy list
        while True:
            component_nodes.append(self._parse_arithmetic_expression())
            if self.get_token() == ",":
                self._next_token()
            elif self.get_token() == ")":
                break  # end at closing parenthesis
            else:
                raise ValueError(
                    "AMPL parser encountered unexpected token '{0}'".format(
                        self.get_token()
                    )
                    + " while parsing a compound set"
                )
        return mat.CompoundDummyNode(component_nodes=component_nodes)

    # Utility
    # ------------------------------------------------------------------------------------------------------------------

    def _is_last_token(self) -> bool:
        return self._active_script.token_index == len(self._active_script.tokens) - 1

    def _enforce_token_value(self, expected_token: str):
        if self.get_token() != expected_token:
            msg = "AMPL parser encountered an unexpected token '{0}' ".format(
                self.get_token()
            ) + "while expecting the token '{0}'".format(expected_token)

            raise ValueError(msg)

    def get_token(self) -> str:
        return self._active_script.tokens[self._active_script.token_index]

    def _prev_token(self, can_skip_whitespace: bool = True) -> bool:
        if self._active_script.token_index != 0:
            while True:
                self._active_script.token_index -= 1
                if not can_skip_whitespace or self.get_token() != " ":
                    break
            return True
        else:
            return False

    def _next_token(self, can_skip_whitespace: bool = True) -> bool:
        """
        Update the token of the active script to the next token.

        :param can_skip_whitespace: if true, skip all whitespace tokens until the first non-whitespace token is found.
        :return: true if a succeeding token exists, false if no succeeding token exists
        """
        if not self._active_script.token_index == len(self._active_script.tokens) - 1:
            while True:
                self._active_script.token_index += 1
                if not can_skip_whitespace or self.get_token() != " ":
                    break
            return True
        else:
            return False

    def _skip_until_token(self, token) -> bool:
        while True:
            if self.get_token() == token:
                return True
            elif not self._next_token():
                return False

    def _extract_string(self, delimiters: Union[str, List[str]] = None):
        if delimiters is None:
            delimiters = [" "]
        elif isinstance(delimiters, str):
            delimiters = [delimiters]
        token = ""
        while True:
            token += self.get_token()
            if not self._next_token(can_skip_whitespace=False):
                break  # reached last token
            if self.get_token() in delimiters:
                break  # reached whitespace or end of statement
        return token


class AMPLLexer:
    def __init__(self):

        self.__index: int = 0
        self.__literal: str = ""
        self.__token: str = ""

        self.__comment_tokens: Optional[List[str]] = None
        self.script_tokens: Optional[List[str]] = None

        self.__statement_level: int = 0
        self.__compound_statement_levels: Optional[List[int]] = None

        self.__is_commented_single = False
        self.__is_commented_multi = False
        self.__is_mid_statement: bool = False

    def tokenize(self, literal: str) -> List[str]:

        self.__index = 0
        self.__literal = literal
        self.__token = ""

        self.__comment_tokens = []
        self.script_tokens = []

        self.__statement_level = 0
        self.__compound_statement_levels = []

        is_numeric = False
        is_string_single = False
        is_string_double = False
        self.__is_commented_single = False
        self.__is_commented_multi = False
        self.__is_mid_statement = False

        while literal != "":

            c = literal[self.__index]

            if c == "\\":
                if not self.__next_char():
                    raise ValueError(
                        "AMPL lexer expected an escaped character but encountered an EOF"
                    )
                c += literal[self.__index]

            if is_string_single:
                self.__token += c
                if c == "'":
                    is_string_single = False
                    self.__add_token()

            elif is_string_double:
                self.__token += c
                if c == '"':
                    is_string_double = False
                    self.__add_token()

            elif self.__is_commented_single:

                # closing delimiter: new line character
                if c in ["\n", "\r"] or self.__is_last_char():
                    self.__is_commented_single = False
                    self.__add_comment_token()
                    self.__add_comment_token("\n")
                    if not self.__is_mid_statement:
                        self.__add_comment_tokens()

                # not a closing delimiter
                else:
                    self.__token += c

            elif self.__is_commented_multi:

                # closing delimiter '*/'
                if c == "/" and literal[self.__index - 1] == "*":

                    self.__is_commented_multi = False

                    self.__token = self.__token[
                        : len(self.__token) - 1
                    ]  # remove '*' from the comment token

                    self.__add_comment_token()
                    self.__add_comment_token("*/")

                    if not self.__is_mid_statement:
                        self.__add_comment_tokens()

                # not a closing delimiter
                else:
                    self.__token += c

            else:

                if c in ["\n", "\r", "\t", " "]:
                    self.__add_token()
                    self.__add_token(" ")

                elif c == "#":
                    self.__is_commented_single = True
                    self.__add_token()
                    self.__add_comment_token("#")

                elif c == '"':
                    is_string_double = True
                    self.__token += '"'

                elif c == "'":
                    is_string_single = True
                    self.__token += "'"

                elif c in [",", "~", "+", "^", "(", ")", "[", "]", "{", "}"]:
                    self.__add_token()
                    self.__token += c
                    self.__add_token()

                elif c == ":":
                    self.__match_token([":", ":="])

                elif c == "-":
                    self.__match_token(["-", "->"])

                elif c == "*":
                    self.__match_token(["*", "**"])

                elif c == "/":
                    self.__add_token()
                    self.__token += c
                    if not self.__next_char():
                        raise ValueError("AMPL lexer encountered an unexpected EOF")
                    c2_candidate = self.__literal[self.__index]
                    if c2_candidate == "*":
                        self.__is_commented_multi = True
                        self.__add_comment_token("/*")
                    else:
                        self.__prev_char()
                        self.__add_token()

                elif c == "=":
                    self.__match_token(["=", "=="])

                elif c == ">":
                    self.__match_token([">", ">=", ">>"])

                elif c == "<":
                    self.__match_token(["<", "<-", "<=", "<>", "<->"])

                elif c == "!":
                    self.__match_token(["!", "!="])

                elif c == "&":
                    self.__match_token(["&", "&&"])

                elif c == "|":
                    self.__match_token("||")

                elif c == ".":
                    self.__match_token([".", ".."])

                elif c == ";":
                    self.__add_token()
                    self.__token += c
                    self.__add_token()
                    self.__add_comment_tokens()

                else:
                    if self.__token == "":
                        if c.isnumeric():
                            is_numeric = True
                        else:
                            is_numeric = False
                        self.__token += c
                    else:
                        if is_numeric:
                            if c in ["D", "d", "E", "e"] and not self.__is_last_char():
                                self.__next_char()
                                c_next = self.__literal[self.__index]
                                if c_next.isnumeric() or c_next in ["+", "-"]:
                                    self.__add_token()
                                    self.__add_token(c)
                                    if c_next in ["+", "-"]:
                                        self.__add_token(c_next)
                                    else:
                                        self.__token = c_next
                                else:
                                    is_numeric = False
                                    self.__prev_char()
                                    self.__token += c
                            else:
                                if not c.isnumeric():
                                    is_numeric = False
                                self.__token += c
                        else:
                            self.__token += c

            if not self.__next_char():
                self.__add_token()
                break

        # Remove trailing whitespace tokens
        while len(self.script_tokens) > 0:
            if self.script_tokens[len(self.script_tokens) - 1] == " ":
                self.script_tokens.pop()
            else:
                break

        return self.script_tokens

    def __match_token(self, candidates: Union[str, List[str]]):

        self.__add_token()

        if isinstance(candidates, str):
            candidates = [candidates]
        candidates.sort(key=lambda s: len(s), reverse=True)

        max_length = max([len(s) for s in candidates])
        end_index = self.__index + max_length

        # Retrieve target
        if end_index > len(self.__literal):  # end index is out of range
            target = self.__literal[self.__index :]
        else:
            target = self.__literal[self.__index : end_index]

        # Identify token among candidates
        token = ""
        for candidate in candidates:
            if len(candidate) <= len(target):
                if target[: len(candidate)] == candidate:
                    token = candidate
                    break

        # Skip characters of token
        for i in range(len(token) - 1):
            self.__next_char()

        self.__add_token(token)

    def __is_last_char(self) -> bool:
        return self.__index >= len(self.__literal) - 1

    def __is_current_level_compound(self) -> bool:
        if len(self.__compound_statement_levels) > 0:
            i = len(self.__compound_statement_levels) - 1
            if self.__statement_level == self.__compound_statement_levels[i]:
                return True
        return False

    def __next_char(self) -> bool:
        if not self.__is_last_char():
            self.__index += 1
            return True
        else:
            return False

    def __prev_char(self) -> bool:
        if self.__index > 0:
            self.__index -= 1
            return True
        else:
            return False

    def __add_token(self, token: str = None):

        if token is not None:
            self.__token = token

        self.__process_token()

        if self.__token != "":
            if self.__token == " ":
                if len(self.script_tokens) > 0:
                    if self.script_tokens[len(self.script_tokens) - 1] != " ":
                        self.script_tokens.append(self.__token)
            else:
                self.script_tokens.append(self.__token)
            self.__token = ""

    def __process_token(self):

        if self.__token == ";":
            self.__is_mid_statement = False

        elif self.__token in ["if", "else", "for", "repeat"]:
            self.__is_mid_statement = True
            self.__statement_level += 1
            self.__compound_statement_levels.append(self.__statement_level)

        elif self.__token in ["{", "}"]:
            if self.__is_current_level_compound():
                self.__is_mid_statement = False
            else:
                self.__is_mid_statement = True

        elif self.__token in ["", " "]:
            pass  # no change

        else:
            self.__is_mid_statement = True

    def __add_comment_token(self, token: str = None):
        if token is not None:
            self.__token = token
        if self.__token != "":
            self.__comment_tokens.append(self.__token)
            self.__token = ""

    def __add_comment_tokens(self):
        if len(self.__comment_tokens) > 0:
            self.script_tokens.extend(self.__comment_tokens)
            self.__comment_tokens.clear()
