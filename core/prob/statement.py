from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Tuple, Union

import symro.core.util.util as util
import symro.core.mat as mat
from symro.core.prob.specialcommand import SpecialCommand


# Base Statement
# ----------------------------------------------------------------------------------------------------------------------
class BaseStatement(ABC):

    def __init__(self):
        pass

    def __str__(self):
        return self.get_literal()

    @abstractmethod
    def get_literal(self, indent_level: int = 0) -> str:
        pass

    def get_validated_literal(self,
                              indent_level: int = 0,
                              validator: Callable[["BaseStatement"], bool] = None) -> str:
        return self.get_literal(indent_level)


class Statement(BaseStatement):

    def __init__(self, literal: str):
        super(Statement, self).__init__()
        self.literal: str = literal

    def get_literal(self, indent_level: int = 0) -> str:
        return "{0}".format(indent_level * '\t') + self.literal


# Comment
# ----------------------------------------------------------------------------------------------------------------------
class Comment(BaseStatement):

    def __init__(self, comment: str, is_multi: bool):
        super(Comment, self).__init__()
        self.comment: str = comment
        self.is_multi: bool = is_multi

    def get_literal(self, indent_level: int = 0) -> str:
        if not self.is_multi:
            return "{0}#{1}".format(indent_level * '\t', self.comment)
        else:
            return "{0}/*{1}*/".format(indent_level * '\t', self.comment)


# File I/O
# ----------------------------------------------------------------------------------------------------------------------

class Redirection:

    def __init__(self,
                 operator: str,
                 file_name_node: Union[mat.StringExpressionNode, mat.DummyNode]):
        self.operator: str = operator
        self.file_name_node: Union[mat.StringExpressionNode, mat.DummyNode] = file_name_node

    def __str__(self):
        return self.get_literal()

    def get_literal(self) -> str:
        return "{0} {1}".format(self.operator, self.file_name_node)


class FileStatement(BaseStatement):

    def __init__(self,
                 command: str,
                 file_name_node: Union[mat.StringExpressionNode, mat.DummyNode]):
        super(FileStatement, self).__init__()
        self.command: str = command
        self.file_name_node: Union[mat.StringExpressionNode, mat.DummyNode] = file_name_node

    def get_literal(self, indent_level: int = 0) -> str:
        return "{0}{1} {2};".format(indent_level * '\t', self.command, self.file_name_node)


# Option Statement
# ----------------------------------------------------------------------------------------------------------------------
class OptionStatement(BaseStatement):

    def __init__(self,
                 arg_nodes: List[mat.ExpressionNode],
                 redirection: Redirection):
        super(OptionStatement, self).__init__()
        self.arg_nodes: List[mat.ExpressionNode] = arg_nodes
        self.redirection: Redirection = redirection

    def get_literal(self, indent_level: int = 0) -> str:
        literal = "{0}option".format(indent_level * '\t')
        if len(self.arg_nodes) > 0:
            literal += " {0}".format(' '.join([arg.get_literal() for arg in self.arg_nodes]))
        if self.redirection is not None:
            literal += " {0}".format(self.redirection)
        literal += ';'
        return literal


# Model Entity Declaration
# ----------------------------------------------------------------------------------------------------------------------
class ModelEntityDeclaration(BaseStatement):

    def __init__(self, meta_entity: mat.MetaEntity):
        super(ModelEntityDeclaration, self).__init__()
        self.meta_entity: mat.MetaEntity = meta_entity

    def get_literal(self, indent_level: int = 0) -> str:
        return "{0}".format(indent_level * '\t') + self.meta_entity.get_declaration()


# Problem Statement
# ----------------------------------------------------------------------------------------------------------------------
class ProblemStatement(BaseStatement):

    def __init__(self,
                 prob_node: mat.DeclaredEntityNode,
                 idx_set_node: mat.CompoundSetNode = None,
                 env_sym: str = None,
                 item_nodes: List[Tuple[mat.CompoundSetNode, mat.DeclaredEntityNode]] = None,
                 redirection: Redirection = None):

        super(ProblemStatement, self).__init__()

        self.prob_node: mat.DeclaredEntityNode = prob_node
        self.idx_set_node: mat.CompoundSetNode = idx_set_node
        self.env_sym: str = env_sym
        self.item_nodes: List[Tuple[mat.CompoundSetNode, mat.DeclaredEntityNode]] = item_nodes
        self.redirection: Redirection = redirection

        if self.item_nodes is None:
            self.item_nodes = []

    def get_literal(self, indent_level: int = 0) -> str:

        literal = "{0}problem".format(indent_level * '\t')

        if self.prob_node is not None:
            literal += " {0}".format(self.prob_node)
        if self.idx_set_node is not None:
            literal += " {0}".format(self.idx_set_node)
        if self.env_sym is not None:
            literal += " {0}".format(self.env_sym)

        if len(self.item_nodes) > 0:
            literal += ":\n"
            i = 0
            for ent_idx_set_node, entity_node in self.item_nodes:
                entity_str = "{0}".format('\t' * (indent_level + 1))
                if ent_idx_set_node is None:
                    entity_str += entity_node.get_literal()
                else:
                    entity_str += "{0} {1}".format(ent_idx_set_node, entity_node)
                if i < len(self.item_nodes) - 1:
                    entity_str += ",\n"
                else:
                    entity_str += '\n'
                literal += entity_str
                i += 1

        if self.redirection is not None:
            literal += " {0}".format(self.redirection)

        literal += ';'

        return literal


# Table Statements
# ----------------------------------------------------------------------------------------------------------------------

class TableKeySpec:

    def __init__(self,
                 set_expr_node: mat.SetExpressionNode,
                 arrow_token: str,
                 key_col_specs: List[Tuple[mat.ExpressionNode, mat.ExpressionNode]]):
        self.set_expr_node: mat.SetExpressionNode = set_expr_node
        self.arrow_token: str = arrow_token
        self.key_col_specs: List[Tuple[mat.ExpressionNode, mat.ExpressionNode]] = key_col_specs

    def __str__(self):
        return self.get_literal()

    def get_literal(self) -> str:

        literal = ""

        if self.set_expr_node is not None:
            literal += "{0} {1} ".format(self.set_expr_node, self.arrow_token)

        key_col_specs_str = []
        for idx_node, data_col_node in self.key_col_specs:
            if idx_node is None:
                key_col_specs_str.append(data_col_node.get_literal())
            else:
                key_col_specs_str.append("{0} ~ {1}".format(idx_node, data_col_node))

        literal += "[{0}]".format(", ".join(key_col_specs_str))

        return literal


class BaseTableDataSpec(ABC):

    def __init__(self):
        pass

    def __str__(self):
        return self.get_literal()

    @abstractmethod
    def get_literal(self) -> str:
        pass


class TableDataSpec(BaseTableDataSpec):

    def __init__(self,
                 data_expr_node: mat.ArithmeticExpressionNode,
                 data_col_node: mat.ArithmeticExpressionNode,
                 direction: str):
        super(TableDataSpec, self).__init__()
        self.data_expr_node: mat.ArithmeticExpressionNode = data_expr_node
        self.data_col_node: mat.ArithmeticExpressionNode = data_col_node
        self.direction: str = direction

    def get_literal(self) -> str:
        if self.data_expr_node is None:
            literal = self.data_col_node.get_literal()
        else:
            literal = "{0} ~ {1}".format(self.data_expr_node, self.data_col_node)
        if self.direction is not None:
            literal += " {0}".format(self.direction)
        return literal


class IndexedTableDataSpec(BaseTableDataSpec):

    def __init__(self,
                 idx_set_node: mat.SetExpressionNode,
                 data_spec_components: List[BaseTableDataSpec],
                 form: int = 0):
        super(IndexedTableDataSpec, self).__init__()
        self.idx_set_node: mat.SetExpressionNode = idx_set_node
        self.data_spec_components: List[BaseTableDataSpec] = data_spec_components
        self.form: int = form

    def get_literal(self) -> str:
        if self.form == 0:
            return "{0} {1}".format(self.idx_set_node, self.data_spec_components[0])
        else:
            data_spec_str = ", ".join([ds.get_literal() for ds in self.data_spec_components])
            if self.form == 1:
                return "{0} ({1})".format(self.idx_set_node, data_spec_str)
            else:
                return "{0} <{1}>".format(self.idx_set_node, data_spec_str)


# table
class TableDeclaration(BaseStatement):

    def __init__(self,
                 table_sym: str,
                 idx_set_node: mat.SetExpressionNode,
                 direction: str,
                 opt_nodes: List[mat.StringExpressionNode],
                 key_spec: TableKeySpec,
                 data_specs: List[BaseTableDataSpec]):
        super(TableDeclaration, self).__init__()
        self.table_sym: str = table_sym
        self.idx_set_node: mat.SetExpressionNode = idx_set_node
        self.direction: str = direction
        self.opt_nodes: List[mat.StringExpressionNode] = opt_nodes
        self.key_spec: TableKeySpec = key_spec
        self.data_specs: List[BaseTableDataSpec] = data_specs

    def get_literal(self, indent_level: int = 0) -> str:

        literal = "{0}table {1}".format(indent_level * '\t', self.table_sym)

        if self.idx_set_node is not None:
            literal += " {0}".format(self.idx_set_node)

        if self.direction is not None:
            literal += " {0}".format(self.direction)

        if len(self.opt_nodes) > 0:
            literal += " {0}".format(' '.join([n.get_literal() for n in self.opt_nodes]))

        literal += ':'

        specs = [self.key_spec]
        specs.extend(self.data_specs)
        literal += " {0};".format(", ".join([str(s) for s in specs]))

        return literal


class TableAccessStatement(BaseStatement):

    def __init__(self,
                 command: str,
                 table_node: mat.DeclaredEntityNode):
        super(TableAccessStatement, self).__init__()
        self.command: str = command
        self.table_node: mat.DeclaredEntityNode = table_node

    def get_literal(self, indent_level: int = 0) -> str:
        return "{0}{1} table {2};".format(indent_level * '\t', self.command, self.table_node)


# Data Statements
# ----------------------------------------------------------------------------------------------------------------------

def clean_data_statement_element(sub_element: Union[int, float, str]):
    if isinstance(sub_element, str):
        if sub_element.isnumeric():
            sub_element = "'{0}'".format(sub_element)
    return sub_element


class SetDataStatement(BaseStatement):

    def __init__(self,
                 symbol: str,
                 elements: mat.IndexingSet):
        super(SetDataStatement, self).__init__()
        self.symbol: str = symbol
        self.elements: mat.IndexingSet = elements

    @staticmethod
    def generate_element_literal(element: mat.Element):

        sub_elements = [clean_data_statement_element(se) for se in element]

        if len(sub_elements) == 0:
            return ""
        elif len(sub_elements) == 1:
            return str(sub_elements[0])
        else:
            return "({0})".format(','.join(sub_elements))

    def get_literal(self, indent_level: int = 0) -> str:
        literal = "{0}set {1} :=\n".format(indent_level * '\t', self.symbol)

        for element in self.elements:
            literal += "{0}{1}\n".format((indent_level + 1) * '\t', self.generate_element_literal(element))

        literal += ';'

        return literal


class ParameterDataStatement(BaseStatement):

    def __init__(self,
                 symbol: str,
                 type: str = "param",
                 default_value: Union[int, float, str] = None,
                 values: Dict[mat.Element, Union[int, float, str]] = None):
        super(ParameterDataStatement, self).__init__()
        self.default_value: Union[int, float, str] = default_value
        self.type: str = type  # param or var
        self.symbol: str = symbol
        self.values: Dict[mat.Element, Union[int, float, str]] = values if values is not None else {}

    @staticmethod
    def generate_element_value_pair_literal(element: mat.Element, value: Union[int, float, str]):

        value = clean_data_statement_element(value)

        # scalar entity
        if element is None:
            return value

        # indexed entity
        else:
            sub_elements = [clean_data_statement_element(se) for se in element]
            return "{0} {1}".format(' '.join(sub_elements), value)

    def get_literal(self, indent_level: int = 0) -> str:

        literal = "{0}{1} {2}".format(indent_level * '\t', self.type, self.symbol)

        if self.default_value is not None:
            literal += " default {0}".format(clean_data_statement_element(self.default_value))

        literal += " :=\n"

        for element, value in self.values.items():
            literal += "{0}{1}\n".format((indent_level + 1) * '\t',
                                         self.generate_element_value_pair_literal(element, value))

        literal += ';'

        return literal


# Display Statement
# ----------------------------------------------------------------------------------------------------------------------

class IndexedArgList:

    def __init__(self,
                 idx_set_node: mat.SetExpressionNode,
                 arg_nodes: List[mat.ExpressionNode]):
        self.idx_set_node: mat.SetExpressionNode = idx_set_node
        self.arg_nodes: List[mat.ExpressionNode] = arg_nodes

    def __str__(self):
        return self.get_literal()

    def get_literal(self) -> str:
        literal = self.idx_set_node.get_literal()
        if len(self.arg_nodes) == 1:
            literal += " {0}".format(self.arg_nodes[0])
        else:
            literal += " ({0})".format(", ".join([n.get_literal() for n in self.arg_nodes]))
        return literal


# display, print, printf
class DisplayStatement(BaseStatement):

    def __init__(self,
                 command: str,
                 idx_set_node: mat.SetExpressionNode,
                 arg_list: List[Union[mat.ExpressionNode, IndexedArgList]],
                 redirection: Redirection):
        super(DisplayStatement, self).__init__()
        self.command: str = command
        self.idx_set_node: mat.SetExpressionNode = idx_set_node
        self.arg_list: List[Union[mat.ExpressionNode, IndexedArgList]] = arg_list
        self.redirection: Redirection = redirection

    def get_literal(self, indent_level: int = 0) -> str:
        literal = "{0}".format(indent_level * '\t') + self.command
        if self.idx_set_node is not None:
            literal += " {0}:".format(self.idx_set_node)
        if len(self.arg_list) > 0:
            literal += " {0}".format(", ".join([str(a) for a in self.arg_list]))
        if self.redirection is not None:
            literal += " {0}".format(self.redirection)
        literal += ";"
        return literal


# Modelling Statements
# ----------------------------------------------------------------------------------------------------------------------

# drop, restore, objective
class NonAssignmentStatement(BaseStatement):

    def __init__(self,
                 command: str,
                 idx_set_node: mat.SetExpressionNode,
                 entity_node: mat.DeclaredEntityNode,
                 redirection: Redirection):
        super(NonAssignmentStatement, self).__init__()
        self.command: str = command
        self.idx_set_node: mat.SetExpressionNode = idx_set_node
        self.entity_node: mat.DeclaredEntityNode = entity_node
        self.redirection: Redirection = redirection

    def get_literal(self, indent_level: int = 0) -> str:
        literal = "{0}".format(indent_level * '\t') + self.command
        if self.idx_set_node is not None:
            literal += " {0}".format(self.idx_set_node)
        literal += " {0}".format(self.entity_node)
        if self.redirection is not None:
            literal += " {0}".format(self.redirection)
        literal += ';'
        return literal


# fix, unfix, let
class AssignmentStatement(BaseStatement):

    def __init__(self,
                 command: str,
                 idx_set_node: mat.SetExpressionNode,
                 entity_node: mat.DeclaredEntityNode,
                 expr_node: mat.ExpressionNode,
                 redirection: Redirection):
        super(AssignmentStatement, self).__init__()
        self.command: str = command
        self.idx_set_node: mat.SetExpressionNode = idx_set_node
        self.entity_node: mat.DeclaredEntityNode = entity_node
        self.expr_node: mat.ExpressionNode = expr_node
        self.redirection: Redirection = redirection

    def get_literal(self, indent_level: int = 0) -> str:
        literal = "{0}".format(indent_level * '\t') + self.command
        if self.idx_set_node is not None:
            literal += " {0}".format(self.idx_set_node)
        literal += " {0}".format(self.entity_node)
        if self.expr_node is not None:
            literal += " := {0}".format(self.expr_node)
        if self.redirection is not None:
            literal += " {0}".format(self.redirection)
        literal += ';'
        return literal


# Solve Statement
# ----------------------------------------------------------------------------------------------------------------------
class SolveStatement(BaseStatement):

    def __init__(self, redirection: Redirection):
        super(SolveStatement, self).__init__()
        self.redirection: Redirection = redirection

    def get_literal(self, indent_level: int = 0) -> str:
        if self.redirection is None:
            return "{0}solve;".format(indent_level * '\t')
        else:
            return "{0}solve {1};".format(indent_level * '\t', self.redirection)


# Control Flow Statements
# ----------------------------------------------------------------------------------------------------------------------

class IfStatementClause:

    def __init__(self,
                 operator: str,
                 cdn_expr_node: Optional[mat.LogicalExpressionNode],
                 statements: Union[BaseStatement, List[BaseStatement]],
                 trailing_comments: List[BaseStatement] = None):
        self.operator: str = operator
        self.cdn_expr_node: mat.LogicalExpressionNode = cdn_expr_node
        self.statements: List[BaseStatement] = statements if isinstance(statements, list) else [statements]
        self.trailing_comments: List[BaseStatement] = trailing_comments if trailing_comments is not None else []

    def __str__(self):
        return self.get_literal()

    def get_literal(self, indent_level: int = 0) -> str:

        literal = "{0}".format('\t' * indent_level) + self.operator

        if self.operator != "else":
            literal += " {0} then".format(self.cdn_expr_node)

        literal += " {\n"
        for statement in self.statements:
            literal += statement.get_literal(indent_level + 1) + "\n"
        literal += "{0}".format('\t' * indent_level) + '}'

        if len(self.trailing_comments) > 0:
            literal += '\n'
            for statement in self.trailing_comments:
                literal += statement.get_literal(indent_level) + "\n"

        return literal

    def get_validated_literal(self,
                              indent_level: int = 0,
                              validator: Callable[[BaseStatement], bool] = None) -> str:

        literal = "{0}".format('\t' * indent_level) + self.operator

        if self.operator != "else":
            literal += " {0} then".format(self.cdn_expr_node)

        valid_statements = [s for s in self.statements if validator(s)]

        literal += " {\n"
        for statement in valid_statements:
            literal += statement.get_validated_literal(indent_level + 1, validator) + "\n"
        literal += "{0}".format('\t' * indent_level) + '}'

        if len(self.trailing_comments) > 0:
            literal += '\n'
            for statement in self.trailing_comments:
                literal += statement.get_validated_literal(indent_level) + "\n"

        return literal


class IfStatement(BaseStatement):

    def __init__(self, clauses: List[IfStatementClause]):
        super(IfStatement, self).__init__()
        self.clauses: List[IfStatementClause] = clauses

    def get_literal(self, indent_level: int = 0) -> str:
        return '\n'.join([c.get_literal(indent_level) for c in self.clauses])

    def get_validated_literal(self,
                              indent_level: int = 0,
                              validator: Callable[[BaseStatement], bool] = None) -> str:
        return '\n'.join([c.get_validated_literal(indent_level, validator) for c in self.clauses])


class ForLoopStatement(BaseStatement):

    def __init__(self,
                 loop_sym: str,
                 idx_set_node: mat.SetExpressionNode,
                 statements: List[BaseStatement]):
        super(ForLoopStatement, self).__init__()
        self.loop_sym: str = loop_sym
        self.idx_set_node: mat.SetExpressionNode = idx_set_node
        self.statements: List[BaseStatement] = statements if isinstance(statements, list) else [statements]

    def get_literal(self, indent_level: int = 0) -> str:

        literal = "{0}for".format('\t' * indent_level)
        if self.loop_sym is not None:
            literal += ' ' + self.loop_sym
        literal += ' ' + self.idx_set_node.get_literal()

        literal += " {\n"
        for statement in self.statements:
            literal += statement.get_literal(indent_level + 1) + "\n"
        literal += "{0}".format('\t' * indent_level) + '}'

        return literal

    def get_validated_literal(self,
                              indent_level: int = 0,
                              validator: Callable[[BaseStatement], bool] = None) -> str:

        literal = "{0}for".format('\t' * indent_level)
        if self.loop_sym is not None:
            literal += ' ' + self.loop_sym
        literal += ' ' + self.idx_set_node.get_literal()

        valid_statements = [s for s in self.statements if validator(s)]

        literal += " {\n"
        for statement in valid_statements:
            literal += statement.get_validated_literal(indent_level + 1, validator) + "\n"
        literal += "{0}".format('\t' * indent_level) + '}'

        return literal


# Special Command Statement
# ----------------------------------------------------------------------------------------------------------------------

class SpecialCommandStatement(BaseStatement):

    def __init__(self, script_command: SpecialCommand):
        super(SpecialCommandStatement, self).__init__()
        self.special_command: SpecialCommand = script_command

    def get_literal(self, indent_level: int = 0) -> str:
        return ""


# Statement Collection
# ----------------------------------------------------------------------------------------------------------------------
class StatementCollection(BaseStatement):

    def __init__(self, statements: List[BaseStatement]):
        super(StatementCollection, self).__init__()
        self.statements: List[BaseStatement] = statements

    def get_literal(self, indent_level: int = 0) -> str:
        literals = []
        for statement in self.statements:
            literals.append("{0}".format(statement.get_literal(indent_level)))
        return '\n'.join(literals)

    def get_validated_literal(self,
                              indent_level: int = 0,
                              validator: Callable[[BaseStatement], bool] = None) -> str:
        literals = []
        for statement in self.statements:
            literals.append("{0}".format(statement.get_validated_literal(indent_level, validator)))
        return '\n'.join(literals)


# Scripts
# ----------------------------------------------------------------------------------------------------------------------

class Script:

    def __init__(self,
                 id: str = "main",
                 literal: str = None,
                 tokens: List[str] = None,
                 statements: List[BaseStatement] = None):
        self.id: str = id
        self.literal: str = literal
        self.tokens: List[str] = tokens
        self.token_index: int = 0
        self.statements: List[BaseStatement] = statements if statements is not None else []

    def __str__(self):
        return self.get_literal()

    def __len__(self):
        return len(self.statements)

    def copy(self, source: "Script"):
        self.id = source.id
        self.literal = source.literal
        self.tokens = list(source.tokens)
        self.token_index = source.token_index
        self.statements = list(source.statements)

    def write(self, dir_path: str, file_name: str = None):
        if file_name is None:
            file_name = self.id
        util.write_file(dir_path, file_name, self.get_literal())

    def get_literal(self, indent_level: int = 0) -> str:
        literals = []
        for statement in self.statements:
            if not isinstance(statement, SpecialCommandStatement):
                literals.append("{0}".format(statement.get_literal(indent_level)))
        return '\n'.join(literals)

    def get_validated_literal(self,
                              indent_level: int = 0,
                              validator: Callable[[BaseStatement], bool] = None) -> str:
        literals = []
        for statement in self.statements:
            if validator(statement) and not isinstance(statement, SpecialCommandStatement):
                literals.append("{0}".format(statement.get_validated_literal(indent_level, validator)))
        return '\n'.join(literals)


class CompoundScript:

    def __init__(self,
                 main_script: Script = None,
                 included_scripts: Union[List[Script], Dict[str, Script]] = None):

        if included_scripts is None:
            included_scripts = {}
        if isinstance(included_scripts, list):
            included_scripts = {script.id: script for script in included_scripts}

        self.main_script: Optional[Script] = main_script
        self.included_scripts: Dict[str, Script] = included_scripts

    def copy(self, source: "CompoundScript"):

        self.main_script = Script()
        self.main_script.copy(source.main_script)

        self.included_scripts = {}
        for id, script in source.included_scripts.items():
            self.included_scripts[id] = Script()
            self.included_scripts[id].copy(script)

    def add_included_script(self,
                            script: Script,
                            include_in_main: bool = True,
                            file_command: str = "include",
                            statement_index: int = 0):
        self.included_scripts[script.id] = script
        if include_in_main:
            file_name_node = mat.StringNode(literal=script.id, delimiter='"')
            file_statement = FileStatement(command=file_command,
                                           file_name_node=file_name_node)
            self.main_script.statements.insert(statement_index, file_statement)

    def write(self, dir_path: str, main_file_name: str = None):
        self.main_script.write(dir_path, main_file_name)
        for id, included_script in self.included_scripts.items():
            included_script.write(dir_path)
