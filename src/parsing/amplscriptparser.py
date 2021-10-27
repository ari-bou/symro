from typing import List, Tuple, Union

import symro.src.constants as const
import symro.src.mat as mat
from symro.src.prob.problem import Problem
import symro.src.scripting.script as scr
import symro.src.scripting.amplstatement as ampl_stm
from symro.src.parsing.amplparser import AMPLParser
from symro.src.parsing.specialcommandparser import SpecialCommandParser
import symro.src.util.util as util


class AMPLScriptParser(AMPLParser):

    # Construction
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self,
                 problem: Problem = None,
                 working_dir_path: str = ""):
        super().__init__(problem=problem,
                         working_dir_path=working_dir_path)

    # Script Parsing
    # ------------------------------------------------------------------------------------------------------------------

    def parse_script(self, literal: str) -> ampl_stm.CompoundScript:
        script = self._tokenize(literal)
        self._active_script = script
        self.__parse_script()
        return self.compound_script

    def __parse_script(self):

        if len(self._active_script.tokens) == 0:
            return

        while True:

            can_continue, statements = self.__parse_sentence()

            # Add statement
            if statements is not None:
                if isinstance(statements, list):
                    self._active_script.statements.extend(statements)
                else:
                    self._active_script.statements.append(statements)

            if not can_continue:
                break

    def __parse_included_script(self, file_name: str):

        literal = util.read_file(self.working_dir_path, file_name)
        included_script = self._tokenize(literal, script_id=file_name)

        prev_script = self._active_script
        self._active_script = included_script
        self.__parse_script()
        self._active_script = prev_script

    def __parse_sentence(self) -> Tuple[bool, Union[ampl_stm.BaseStatement, List[ampl_stm.BaseStatement]]]:

        token = self.get_token()
        sentence = None

        # Flags
        is_statement = True  # true if the sentence is a single statement (i.e. terminates with a ';')
        is_compound = False  # true if the sentence is a compound statement
        can_continue = True  # true if a following sentence needs to be parsed, false otherwise

        # Empty Statement
        if token == ';':
            pass

        # --- Comments ---

        # Comment
        elif token in ('#', "/*"):
            is_statement = False
            sentence = self.__parse_comment()

        # --- General Commands ---

        elif token == "include":
            is_statement = False
            sentence = self.__parse_file_command()

        elif token == "commands":
            sentence = self.__parse_file_command()

        elif token in ("model", "data"):
            sentence = self.__parse_mode_command()

        elif token == "option":
            sentence = self.__parse_option_statement()

        # --- Printing/Display Commands ---

        elif token in ("display", "_display", "csvdisplay", "print", "printf"):
            sentence = self.__parse_display_print_statement()

        # --- Modelling Commands ---

        elif token == "solve":
            sentence = self.__parse_solve_statement()

        elif token in ("read", "write"):
            sentence = self.__parse_io_statement()

        elif token in ("drop", "restore", "objective"):
            sentence = self.__parse_entity_inclusion_command()

        elif token in ("fix", "unfix"):
            sentence = self.__parse_fix_command()

        elif token in ("reset", "update", "let"):
            sentence = self.__parse_data_manipulation_command()

        elif token == "problem":
            sentence = self.__parse_problem_statement()

        elif token == "environ":
            raise NotImplementedError("Parsing logic for environ command not implemented yet")

        elif token in ("solution", "delete", "purge", "redeclare"):
            raise NotImplementedError("Parsing logic for command {0} not yet implemented".format(token))

        # --- Model Examination Commands ---

        elif token in ("show", "xref", "expand", "check"):
            raise NotImplementedError("Parsing logic for command {0} not yet implemented".format(token))

        # --- Computational Environment Commands ---

        elif token in ("shell", "cd", "quit", "exit", "end"):
            raise NotImplementedError("Parsing logic for command {0} not yet implemented".format(token))

        # --- Control Logic ---

        # If Statement
        elif token == "if":
            is_compound = True
            sentence = self.__parse_if_statement()

        # For Loop
        elif token == "for":
            is_compound = True
            sentence = self.__parse_for_loop()

        # While/Until Loop
        elif token == "repeat":
            # TODO: implement parsing logic for while/until loop
            raise NotImplementedError("Parsing logic for while/until loop not yet implemented")

        # --- Entity Declarations ---

        elif token == "table":
            sentence = self.__parse_table_declaration()

        elif token == "set":
            sentence = self.__parse_set_declaration()

        elif token == "param":
            sentence = self.__parse_param_declaration()

        elif token == "var":
            sentence = self.__parse_var_declaration()

        elif token in ["maximize", "minimize"]:
            sentence = self.__parse_obj_declaration(token)

        elif token == "subject":
            self._next_token()  # skip subject
            self._enforce_token_value("to")
            self._next_token()  # skip 'to'
            sentence = self.__parse_con_declaration()

        # Catch-all
        elif token in self.COMMAND_SYMBOLS:
            self._skip_until_token(';')

        else:
            sentence = self.__parse_con_declaration()

        # Single sentence
        if not is_compound:

            # Sentence that is not a statement
            if not is_statement:
                # current token is either:
                #   (i) the last token of the current sentence, and the last token overall
                #   (ii) the first token of the following sentence
                if self._is_last_token():
                    can_continue = False  # if the current token is last, then terminate parsing

            # Statement
            else:
                # current token is ';'
                self._enforce_token_value(';')
                if not self._next_token():
                    can_continue = False  # if there is no following token, then terminate parsing

        # Compound statement
        else:
            # current token is either:
            #   (i) a closing curly brace '}' (in which case it must be the last token overall)
            #   (ii) the first token of the following sentence
            if self._is_last_token() and self.get_token() == '}':
                can_continue = False  # if there is no following token, then terminate parsing

        return can_continue, sentence

    def __parse_compound_statement(self):

        statements = []

        self._next_token()  # skip opening curly brace

        while True:

            _, statement = self.__parse_sentence()
            if isinstance(statement, list):
                statements.extend(statement)
            else:
                statements.append(statement)

            if self.get_token() == '}':
                break

        self._enforce_token_value('}')
        self._next_token()  # skip closing curly brace
        return statements

    # Comment Parsing
    # ------------------------------------------------------------------------------------------------------------------

    # A.1
    def __parse_comment(self):

        opening_delimiter = self.get_token()
        self._next_token(can_skip_whitespace=False)  # skip opening delimiter

        is_multi = True
        if opening_delimiter == '#':
            is_multi = False

        token = self.get_token()

        if is_multi:
            if token == "*/":
                comment = ""
            else:
                comment = token
                self._next_token()  # skip comment

        else:
            if token == '\n':
                comment = ""
            else:
                comment = token
                self._next_token()  # skip comment

        self._next_token()  # skip closing delimiter

        statements = [ampl_stm.Comment(comment, is_multi=is_multi)]
        statements.extend(self.__parse_special_commands(comment))

        return statements

    def __parse_special_commands(self, literal: str):

        script_command_parser = SpecialCommandParser()
        script_commands = script_command_parser.parse_script_commands(literal)

        statements = []
        for script_command in script_commands:

            if script_command.symbol == const.SPECIAL_COMMAND_OMIT_DECLARATIONS:
                self._can_include_in_model = False
            elif script_command.symbol == const.SPECIAL_COMMAND_INCLUDE_DECLARATIONS:
                self._can_include_in_model = True
            elif script_command.symbol == const.SPECIAL_COMMAND_EVAL:
                self._can_evaluate = True
            elif script_command.symbol == const.SPECIAL_COMMAND_NOEVAL:
                self._can_evaluate = False

            self.problem.add_script_command(script_command)
            statements.append(scr.SpecialCommandStatement(script_command))

        return statements

    # Modelling Entity Declaration Parsing
    # ------------------------------------------------------------------------------------------------------------------

    # A.6
    def __parse_set_declaration(self):

        self._next_token()  # skip 'set'

        # symbol
        symbol = self.get_token()
        self._next_token()  # skip symbol

        alias = None
        idx_set_node = None
        super_set_node = None
        defined_value_node = None
        default_value_node = None

        while self.get_token() != ';':

            token = self.get_token()

            # indexing
            if token == '{':
                idx_set_node = self._parse_indexing_set_definition()  # skip opening and closing curly braces

            # super set
            elif token == "within":
                self._next_token()
                super_set_node = self._parse_set_expression()

            # defined value
            elif token in ('=', ":="):
                self._next_token()
                defined_value_node = self._parse_set_expression()

            # default value
            elif token == "default":
                self._next_token()
                default_value_node = self._parse_set_expression()

            # alias
            else:
                alias = self.get_token()
                self._next_token()  # skip alias

        # build meta-set
        meta_set = mat.MetaSet(symbol=symbol,
                               alias=alias,
                               idx_set_node=idx_set_node,
                               super_set_node=super_set_node,
                               defined_value_node=defined_value_node,
                               default_value_node=default_value_node)

        is_auxiliary = not self._can_include_in_model or not self._can_evaluate
        self.problem.add_meta_set(meta_set, is_auxiliary=is_auxiliary)

        # build statement
        return ampl_stm.ModelEntityDeclaration(meta_set)

    # A.7
    def __parse_param_declaration(self):

        self._next_token()  # skip 'param'

        # Symbol
        symbol = self.get_token()
        self._next_token()  # skip symbol

        alias = None
        idx_set_node = None

        is_binary = False
        is_integer = False
        is_symbolic = False
        relational_constraints = {}
        defined_value_node = None
        default_node = None
        super_set_node = None

        while self.get_token() != ';':

            token = self.get_token()

            # Indexing
            if self.get_token() == '{':
                idx_set_node = self._parse_indexing_set_definition()  # skip opening and closing curly braces

            # Attributes

            elif token == "binary":
                is_binary = True
                self._next_token()

            elif token == "integer":
                is_integer = True
                self._next_token()

            elif token == "symbolic":
                is_symbolic = True
                self._next_token()

            elif token in ['<', "<=", "!=", "<>", '>', ">=", ":="]:
                rel_op = self.get_token()
                self._next_token()
                ari_node = self._parse_arithmetic_expression()
                relational_constraints[rel_op] = ari_node

            elif token in ('=', "=="):
                self._next_token()
                defined_value_node = self._parse_arithmetic_expression()

            elif token == "default":
                self._next_token()
                default_node = self._parse_arithmetic_expression()

            elif token == "in":
                self._next_token()
                super_set_node = self._parse_set_expression()

            # Alias
            else:
                alias = self.get_token()
                self._next_token()  # skip alias

            if self.get_token() == ',':
                self._next_token()

        # Meta-Parameter

        meta_param = mat.MetaParameter(symbol=symbol,
                                       alias=alias,
                                       idx_set_node=idx_set_node,
                                       is_binary=is_binary,
                                       is_integer=is_integer,
                                       is_symbolic=is_symbolic,
                                       relational_constraints=relational_constraints,
                                       defined_value=defined_value_node,
                                       default_value=default_node,
                                       super_set_node=super_set_node)

        is_auxiliary = not self._can_include_in_model or not self._can_evaluate
        self.problem.add_meta_parameter(meta_param, is_auxiliary=is_auxiliary)

        # Statement
        return ampl_stm.ModelEntityDeclaration(meta_param)

    # A.8
    def __parse_var_declaration(self):

        self._next_token()  # skip 'var'

        # Symbol
        symbol = self.get_token()
        self._next_token()  # skip symbol

        alias = None
        idx_set_node = None

        attributes = []
        is_binary = False
        is_integer = False
        is_symbolic = False
        default_value_node = None
        defined_value_node = None
        upper_bound_node = None
        lower_bound_node = None

        while self.get_token() != ';':

            token = self.get_token()

            # Indexing
            if token == '{':
                idx_set_node = self._parse_indexing_set_definition()  # skip opening and closing curly braces

            # Attributes

            elif token == "binary":
                attributes.append("binary")
                is_binary = True
                self._next_token()

            elif token == "integer":
                attributes.append("integer")
                is_integer = True
                self._next_token()

            elif token == "symbolic":
                attributes.append("symbolic")
                is_symbolic = True
                self._next_token()

            elif token in ["<=", '=', ":=", ">="]:

                operator = self.get_token()
                self._next_token()
                ari_node = self._parse_arithmetic_expression()

                attributes.append("{0} {1}".format(operator, ari_node))

                if operator == "<=":
                    upper_bound_node = ari_node
                elif operator == ">=":
                    lower_bound_node = ari_node
                elif operator == '=':
                    defined_value_node = ari_node
                elif operator == ":=":
                    default_value_node = ari_node

            elif token == "default":
                self._next_token()
                default_value_node = self._parse_arithmetic_expression()
                attributes.append("default {0}".format(default_value_node))

            # Alias
            else:
                alias = self.get_token()
                self._next_token()  # skip alias

            if self.get_token() == ',':
                self._next_token()

        # Meta-Variable

        meta_var = mat.MetaVariable(symbol=symbol,
                                    alias=alias,
                                    idx_set_node=idx_set_node,
                                    is_binary=is_binary,
                                    is_integer=is_integer,
                                    is_symbolic=is_symbolic,
                                    default_value=default_value_node,
                                    defined_value=defined_value_node,
                                    lower_bound=lower_bound_node,
                                    upper_bound=upper_bound_node)

        is_auxiliary = not self._can_include_in_model or not self._can_evaluate
        self.problem.add_meta_variable(meta_var, is_auxiliary=is_auxiliary)

        # Statement
        return ampl_stm.ModelEntityDeclaration(meta_var)

    # A.10
    def __parse_obj_declaration(self, direction: str):

        self._next_token()  # skip direction

        # Symbol
        symbol = self.get_token()
        self._next_token()  # skip symbol

        alias = None
        idx_set_node = None
        expression = None

        while self.get_token() != ';':

            # Indexing Set Definition
            if self.get_token() == '{':
                idx_set_node = self._parse_indexing_set_definition()  # skip opening and closing curly braces

            # Expression
            elif self.get_token() == ':':
                self._next_token()  # skip ':'
                expression_node = self._parse_arithmetic_expression()
                expression = mat.Expression(expression_node, idx_set_node, id=symbol)
                self.expressions[symbol] = expression

            # Suffixes
            elif self.get_token() == "suffix":
                # TODO: suffix initializations
                raise NotImplementedError("Parsing logic for suffix initializations not implemented yet")

            # Alias
            else:
                alias = self.get_token()
                self._next_token()  # skip alias

        # Meta-Objective

        meta_obj = mat.MetaObjective(symbol=symbol,
                                     alias=alias,
                                     idx_set_node=idx_set_node,
                                     direction=direction,
                                     expression=expression)

        is_auxiliary = not self._can_include_in_model or not self._can_evaluate
        self.problem.add_meta_objective(meta_obj, is_auxiliary=is_auxiliary)

        # Statement
        return ampl_stm.ModelEntityDeclaration(meta_obj)

    # A.9
    def __parse_con_declaration(self):

        symbol = self.get_token()
        self._next_token()  # skip symbol

        alias = None
        idx_set_node = None
        expression = None

        while self.get_token() != ';':

            # Indexing
            if self.get_token() == '{':
                idx_set_node = self._parse_indexing_set_definition()  # skip opening and closing curly braces

            # Constraint Definition Delimiter
            elif self.get_token() == ':':
                self._next_token()  # skip colon ':'
                expression_node = self._parse_relational_expression()
                expression = mat.Expression(expression_node, idx_set_node, id=symbol)
                self.expressions[symbol] = expression

            # Suffixes
            elif self.get_token() == "suffix":
                # TODO: suffix initializations
                raise NotImplementedError("Parsing logic for suffix initializations not implemented yet")

            # Alias
            elif self.get_token() not in ['{', ':', ":=", "default"]:
                alias = self.get_token()
                self._next_token()  # skip alias

        # Meta-Constraint

        meta_con = mat.MetaConstraint(symbol=symbol,
                                      alias=alias,
                                      idx_set_node=idx_set_node,
                                      expression=expression)

        is_auxiliary = not self._can_include_in_model or not self._can_evaluate
        self.problem.add_meta_constraint(meta_con, is_auxiliary=is_auxiliary)

        meta_con.elicit_constraint_type()

        # Statement
        return ampl_stm.ModelEntityDeclaration(meta_con)

    # Table Declaration and Access
    # ------------------------------------------------------------------------------------------------------------------

    # A.13
    def __parse_table_declaration(self):

        self._next_token()  # skip 'table'

        # Table Symbol
        table_sym = self.get_token()
        self.problem.meta_tables[table_sym] = None
        self._next_token()  # skip table symbol

        # Indexing Set Node
        idx_set_node = None
        if self.get_token() == '{':
            idx_set_node = self._parse_indexing_set_definition()  # skip opening and closing curly braces

        # Direction
        direction = None
        if self.get_token() in ["IN", "OUT", "INOUT"]:
            direction = self.get_token()
            self._next_token()  # skip direction

        # String list
        opt_nodes = []
        while True:
            if self.get_token() == ':':
                self._next_token()  # skip ':'
                break
            else:
                str_node = self._parse_string_expression()
                opt_nodes.append(str_node)

        # Key Spec
        key_spec = self.__parse_key_spec()

        data_specs = []
        if self.get_token() == ',':
            self._next_token()  # skip comma ','

            # Data Specs
            while True:

                data_specs.append(self.__parse_data_spec())

                if self.get_token() == ',':
                    self._next_token()  # skip comma
                elif self.get_token() == ';':
                    break

        return ampl_stm.TableDeclaration(
            table_sym=table_sym,
            idx_set_node=idx_set_node,
            direction=direction,
            opt_nodes=opt_nodes,
            key_spec=key_spec,
            data_specs=data_specs
        )

    # A.13
    def __parse_key_spec(self):

        # Set-IO
        set_expr_node = None
        arrow_token = None
        if self.get_token() != '[':
            set_expr_node = self._parse_set_expression()
            arrow_token = self.get_token()
            self._next_token()  # skip arrow

        # Key-Col-Specs
        key_col_specs = []
        self._next_token()  # skip opening bracket '['
        while True:

            data_col_node = self._parse_string_expression()
            if self.get_token() == '~':
                self._next_token()  # skip '~'
                idx_node = data_col_node
                data_col_node = self._parse_string_expression()
            else:
                idx_node = None

            key_col_specs.append((idx_node, data_col_node))

            if self.get_token() == ']':
                self._next_token()  # skip closing bracket ']'
                break
            else:
                self._next_token()  # skip comma ','

        return ampl_stm.TableDeclaration.build_table_key_spec(
            set_expr_node=set_expr_node,
            arrow_token=arrow_token,
            key_col_specs=key_col_specs
        )

    # A.13
    def __parse_data_spec(self) -> Union[ampl_stm.IndexedTableDataSpec, ampl_stm.TableDataSpec]:

        node = self._parse_set_expression()

        # Indexed Data Spec
        if isinstance(node, mat.SetExpressionNode):

            idx_set_node = node

            # Compound Indexed Data Spec (Form 1)
            if self.get_token() == '(':
                return self.__parse_indexed_data_spec_form_1(idx_set_node)

            # Compound Indexed Data Spec (Form 2)
            elif self.get_token() == '<':

                self._next_token()  # skip opening delimiter '<'

                data_spec_components = []
                while True:
                    data_spec_components.append(self.__parse_data_spec())
                    if self.get_token() == ',':
                        self._next_token()  # skip comma
                    else:
                        break

                self._next_token()  # skip closing delimiter '>'

                return ampl_stm.IndexedTableDataSpec(
                    idx_set_node=idx_set_node,
                    data_spec_components=data_spec_components,
                    form=2
                )

            # Single Indexed Data Spec
            else:
                data_col_node = self._parse_arithmetic_expression()
                data_spec_component = self.__parse_data_spec_component(data_col_node)
                return ampl_stm.IndexedTableDataSpec(
                    idx_set_node=idx_set_node,
                    data_spec_components=[data_spec_component],
                    form=0
                )

        # Single Data Spec
        else:
            return self.__parse_data_spec_component(node)

    # A.13
    def __parse_indexed_data_spec_form_1(self, idx_set_node: mat.SetExpressionNode):

        self._next_token()  # skip opening parenthesis '('

        data_specs = []
        while True:
            data_col_node = self._parse_arithmetic_expression()
            data_specs.append(self.__parse_data_spec_component(data_col_node))
            if self.get_token() == ',':
                self._next_token()  # skip comma
            else:
                break

        self._next_token()  # skip closing parenthesis ')'

        if len(data_specs) == 1:
            data_spec = data_specs[0]
            if data_spec.data_expr_node is not None:
                data_spec.data_expr_node.is_prioritized = True
            else:
                data_spec.data_col_node.is_prioritized = True
            form = 0

        else:
            form = 1

        return ampl_stm.IndexedTableDataSpec(
            idx_set_node=idx_set_node,
            data_spec_components=data_specs,
            form=form
        )

    # A.13
    def __parse_data_spec_component(self, data_col_node: Union[mat.ArithmeticExpressionNode,
                                                               mat.StringExpressionNode,
                                                               mat.BaseDummyNode]):
        data_expr_node = None
        if self.get_token() == '~':
            self._next_token()  # skip '~'
            data_expr_node = data_col_node
            data_col_node = self._parse_arithmetic_expression()

        if self.get_token() in ["IN", "OUT", "INOUT"]:
            direction = self.get_token()
            self._next_token()  # skip direction
        else:
            direction = None

        return ampl_stm.TableDataSpec(
            data_expr_node=data_expr_node,
            data_col_node=data_col_node,
            direction=direction
        )

    # A.13
    def __parse_table_access_statement(self, command: str):
        self._next_token()  # skip 'table'
        table_node = self._parse_declared_entity()
        return ampl_stm.TableAccessStatement(
            command=command,
            table_node=table_node
        )

    # Environment Commands Parsing
    # ------------------------------------------------------------------------------------------------------------------

    # A.14: model and data
    def __parse_mode_command(self):
        command = self.get_token()
        self._next_token()
        if self.get_token() == ';':
            return ampl_stm.ModeStatement(command)
        else:
            self._prev_token()
            return self.__parse_file_command()

    # A.14: include, commands, model, and data
    def __parse_file_command(self):

        command = self.get_token()
        self._next_token()  # skip 'include', 'commands', 'model', or 'data'

        file_name_node = self.__parse_argument()

        # TODO: parsing logic for included data files
        if command == "commands":
            print("AMPL Parser ignored script '{0}'".format(file_name_node)
                  + " referenced in a 'commands' statement")
        elif command == "data":
            print("AMPL Parser ignored script '{0}'".format(file_name_node)
                  + " referenced in a 'data' statement")
        else:
            file_name = file_name_node.evaluate(self.problem.state)[0]
            self.__parse_included_script(file_name)

        return ampl_stm.FileStatement(
            command=command,
            file_name_node=file_name_node
        )

    # A.14.1: option
    def __parse_option_statement(self):

        self._next_token()  # skip 'option'

        arg_nodes = []
        redirection = None

        while True:

            if self.get_token() in self.REDIRECTION_OPERATORS:
                redirection = self.__parse_redirection()
                break
            else:
                if self.get_token() == ' ':
                    self._next_token()
                arg_nodes.append(self.__parse_argument())

            if self.get_token() == ';':
                break

        return ampl_stm.OptionStatement(
            arg_nodes=arg_nodes,
            redirection=redirection
        )

    # A.15
    def __parse_redirection(self):
        operator = self.get_token()
        if operator in self.REDIRECTION_OPERATORS:
            self._next_token()  # skip redirection operator
            file_name_node = self.__parse_argument()
            return ampl_stm.Redirection(operator=operator, file_name_node=file_name_node)
        else:
            return None

    def __parse_argument(self):

        # Delimited string literal
        if self.get_token()[0] in ['"', "'"]:
            return self._parse_string_literal()

        # Parenthesized string expression
        elif self.get_token() == '(':
            self._next_token()  # skip opening parenthesis '('
            node = self._parse_string_expression()
            node.is_prioritized = True
            self._next_token()  # skip closing parenthesis ')'
            return node

        # Non-delimited string literal
        else:
            literal = self._extract_string([' ', ';'])
            if self.get_token() == ' ':
                self._next_token()  # skip whitespace
            return mat.DummyNode(symbol=literal)

    # A.16: display, print, and printf
    def __parse_display_print_statement(self):

        command = self.get_token()
        self._next_token()  # skip command

        idx_set_node = None
        arg_idx_set_node = None

        # indexing set node of the print statement
        if self.get_token() == '{':

            idx_set_node = self._parse_indexing_set_definition()

            # if the indexing set node is followed by a ':', then it controls the entire statement
            if self.get_token() == ':':
                self._next_token()  # skip colon ':'

            # otherwise, the indexing set node only controls the first argument
            else:
                arg_idx_set_node = idx_set_node
                idx_set_node = None

        # argument list
        arg_list = []
        is_parsing_arg_list = True
        while is_parsing_arg_list:

            if arg_idx_set_node is None:

                # check if the current node is an indexing set node
                if self.get_token() == '{':
                    arg_idx_set_node = self._parse_indexing_set_definition()

            # argument begins with an indexing set node
            if arg_idx_set_node is not None:

                token = self.get_token()

                # indexing set node controls a parenthesized list of argument nodes
                if self.get_token() == '(':

                    indexed_arg_nodes = []

                    self._next_token()  # skip opening parenthesis '('

                    is_parsing_indexed_arg_list = True
                    while is_parsing_indexed_arg_list:
                        indexed_arg_nodes.append(self._parse_set_expression())
                        if self.get_token() == ')':
                            is_parsing_indexed_arg_list = False
                        else:
                            self._next_token()  # skip comma

                    self._next_token()  # skip closing parenthesis ')'

                    arg_list.append(ampl_stm.DisplayStatement.build_indexed_arg_list(
                        idx_set_node=arg_idx_set_node,
                        arg_nodes=indexed_arg_nodes))

                # indexing set node is also the argument node
                elif token == ',' or token in self.REDIRECTION_OPERATORS or token == ';':
                    arg_list.append(arg_idx_set_node)

                # indexing set node controls a single argument node
                else:
                    arg_node = self._parse_set_expression()
                    arg_list.append(ampl_stm.DisplayStatement.build_indexed_arg_list(
                        idx_set_node=arg_idx_set_node,
                        arg_nodes=[arg_node]))

            # argument is a single non-indexed expression node
            else:
                arg_list.append(self._parse_set_expression())

            if self.get_token() != ',':
                is_parsing_arg_list = False
            else:
                self._next_token()  # skip comma

            arg_idx_set_node = None  # reset the current argument indexing set node

        # Redirection
        redirection = self.__parse_redirection()

        return ampl_stm.DisplayStatement(
            command=command,
            idx_set_node=idx_set_node,
            arg_list=arg_list,
            redirection=redirection
        )

    # A.17 and A.18.3: read and write
    # TODO: parsing logic for 'read' and 'write' commands
    def __parse_io_statement(self):
        command = self.get_token()
        self._next_token()  # skip command

        if self.get_token() == "table":
            return self.__parse_table_access_statement(command)

        else:
            raise NotImplementedError("Parsing logic for IO command not yet implemented")

    # Modelling Commands Parsing
    # ------------------------------------------------------------------------------------------------------------------

    # A.18.1: solve
    def __parse_solve_statement(self):
        self._next_token()  # skip 'solve'
        redirection = self.__parse_redirection()
        return ampl_stm.SolveStatement(redirection=redirection)

    # A.18.2: solution
    # TODO: parsing logic for 'solution' command

    # A.18.5: delete and purge
    # TODO: parsing logic for 'delete' and 'purge' commands

    # A.18.5: redeclare
    # TODO: parsing logic for 'redeclare' command

    # A.18.6: drop, restore, and objective
    def __parse_entity_inclusion_command(self):

        command = self.get_token()
        self._next_token()  # skip command

        idx_set_node = None
        if self.get_token() == '{':
            idx_set_node = self._parse_indexing_set_definition()

        entity_node = self._parse_declared_entity()

        redirection = self.__parse_redirection()

        return ampl_stm.EntityInclusionStatement(
            command=command,
            idx_set_node=idx_set_node,
            entity_node=entity_node,
            redirection=redirection
        )

    # A.18.7 and A.18.9: fix and unfix
    def __parse_fix_command(self):

        command = self.get_token()
        self._next_token()  # skip command

        idx_set_node = None
        if self.get_token() == '{':
            idx_set_node = self._parse_indexing_set_definition()

        entity_node = self._parse_declared_entity()

        expr_node = None
        if self.get_token() == ":=":
            self._next_token()  # skip assignment operator ':='
            expr_node = self._parse_expression()

        redirection = self.__parse_redirection()

        return ampl_stm.FixStatement(
            command=command,
            idx_set_node=idx_set_node,
            entity_node=entity_node,
            expr_node=expr_node,
            redirection=redirection
        )

    # A.18.8: problem
    def __parse_problem_statement(self):

        self._next_token()  # skip 'problem'

        prob_node = None
        idx_set_node = None
        env_sym = None
        entities = []
        redirection = None

        # print problem symbol
        if self.get_token() == ';':
            pass

        # print problem symbol to redirection
        elif self.get_token() in self.REDIRECTION_OPERATORS:
            redirection = self.__parse_redirection()

        # set problem or declare problem
        else:

            prob_node = self._parse_declared_entity()

            if self.get_token() == '{':
                idx_set_node = self._parse_indexing_set_definition()

            if self.get_token() not in ["suffix", ':', ';'] + self.REDIRECTION_OPERATORS:
                env_sym = self.get_token()
                self._next_token()  # skip environment symbol

            if self.get_token() == "suffix":
                # TODO: suffix initializations
                raise NotImplementedError("Parsing logic for suffix initializations not implemented yet")

            if self.get_token() == ':':
                self._next_token()  # skip colon ':'

                while True:
                    if self.get_token() == ';':
                        break
                    entity_idx_set_node = None
                    if self.get_token() == '{':
                        entity_idx_set_node = self._parse_indexing_set_definition()
                    entity_node = self._parse_declared_entity()
                    entities.append((entity_idx_set_node, entity_node))
                    if self.get_token() == ',':
                        self._next_token()  # skip comma

            # check if the statement is a problem declaration
            if prob_node.symbol not in self.problem.subproblems:
                prob_node.type = mat.PROB_TYPE  # set entity type

        return ampl_stm.ProblemStatement(
            prob_node=prob_node,
            idx_set_node=idx_set_node,
            env_sym=env_sym,
            item_nodes=entities,
            redirection=redirection
        )

    # A.18.8: environ
    # TODO: parsing logic for 'environ' command

    # A.18.9: reset and update
    # TODO: parsing logic for 'reset', 'update', and 'let' commands
    def __parse_data_manipulation_command(self):

        command = self.get_token()
        self._next_token()  # skip command

        subcommand = None
        idx_set_node = None
        expr_node = None

        if command == "let":

            idx_set_node = None
            if self.get_token() == '{':
                idx_set_node = self._parse_indexing_set_definition()

            entity_nodes = [self._parse_declared_entity()]

            expr_node = None
            if self.get_token() == ":=":
                self._next_token()  # skip assignment operator ':='
                expr_node = self._parse_expression()

        else:  # update or reset

            if self.get_token() in ("options", "data", "problem"):
                subcommand = self.get_token()
                self._next_token()  # skip subcommand

            entity_nodes = []
            while self.get_token() != ';':  # terminate at end of statement
                entity_nodes.append(self._parse_declared_entity())  # parse declared entity symbol
                if self.get_token() == ',':
                    self._next_token()  # skip comma

        return ampl_stm.DataManipulationStatement(
            command=command,
            subcommand=subcommand,
            idx_set_node=idx_set_node,
            entity_nodes=entity_nodes,
            expr_node=expr_node
        )

    # A.19: show, xref, expand, check
    # TODO: parsing logic for 'show', 'xref', 'expand', and 'check' commands

    # Control Logic Parsing
    # ------------------------------------------------------------------------------------------------------------------

    # A.20.1
    def __parse_if_statement(self):
        clauses = self.__parse_if_statement_clause()
        return ampl_stm.IfStatement(clauses)

    # A.20.1
    def __parse_if_statement_clause(self, is_first_clause: bool = True):

        clauses = []

        self._next_token()  # skip 'if'
        cdn_expr_node = self._parse_logical_expression()
        self._enforce_token_value("then")
        self._next_token()  # skip 'then'

        if self.get_token() == '{':
            statements = self.__parse_compound_statement()
        else:
            _, statements = self.__parse_sentence()

        trailing_comments = []
        while True:
            if self.get_token() in ['#', "/*"]:
                comment = self.__parse_comment()
                if isinstance(comment, list):
                    trailing_comments.extend(comment)
                else:
                    trailing_comments.append(comment)
            else:
                break

        clauses.append(ampl_stm.IfStatement.build_clause(
            operator="if" if is_first_clause else "else if",
            cdn_expr_node=cdn_expr_node,
            statements=statements,
            trailing_comments=trailing_comments
        ))

        if self.get_token() == "else":

            self._next_token()
            token = self.get_token()

            # else if
            if token == "if":
                clauses.extend(self.__parse_if_statement_clause(is_first_clause=False))

            # else
            else:
                if token == "{":  # compound statement
                    statements = self.__parse_compound_statement()
                else:  # single statement
                    _, statements = self.__parse_sentence()
                clauses.append(ampl_stm.IfStatement.build_clause(
                    operator="else",
                    cdn_expr_node=None,
                    statements=statements
                ))

        return clauses

    # A.20.1
    def __parse_for_loop(self):

        self._next_token()  # skip 'for'

        if self.get_token() != '{':
            self._next_token()  # skip loop name

        loop_sym = None
        if self.get_token() != '{':
            loop_sym = self.get_token()
            self._next_token()  # skip loop symbol

        idx_set_node = self._parse_set_expression()

        if self.get_token() == '{':
            statements = self.__parse_compound_statement()
        else:
            _, statements = self.__parse_sentence()

        return ampl_stm.ForLoopStatement(
            loop_sym=loop_sym,
            idx_set_node=idx_set_node,
            statements=statements
        )
