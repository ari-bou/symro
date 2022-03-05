from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Tuple, Union

import symro.mat as mat
from symro.scripting.script import BaseStatement, Script, CompoundScript


# Comment
# ----------------------------------------------------------------------------------------------------------------------
class Comment(BaseStatement):
    def __init__(self, comment: str, is_multi: bool):
        super(Comment, self).__init__()
        self.comment: str = comment
        self.is_multi: bool = is_multi

    def get_literal(self, indent_level: int = 0) -> str:
        if not self.is_multi:
            return "{0}#{1}".format(indent_level * "\t", self.comment)
        else:
            return "{0}/*{1}*/".format(indent_level * "\t", self.comment)


# Mode
# ----------------------------------------------------------------------------------------------------------------------
class ModeStatement(BaseStatement):
    def __init__(self, literal: str):
        super(ModeStatement, self).__init__()
        self.literal: str = literal

    def get_literal(self, indent_level: int = 0) -> str:
        return "{0}".format(indent_level * "\t") + self.literal


# File I/O
# ----------------------------------------------------------------------------------------------------------------------


class Redirection:
    def __init__(
        self,
        operator: str,
        file_name_node: Union[mat.StringExpressionNode, mat.DummyNode],
    ):
        self.operator: str = operator
        self.file_name_node: Union[
            mat.StringExpressionNode, mat.DummyNode
        ] = file_name_node

    def __str__(self):
        return self.get_literal()

    def get_literal(self) -> str:
        return "{0} {1}".format(self.operator, self.file_name_node)


class FileStatement(BaseStatement):
    def __init__(
        self,
        command: str,
        file_name_node: Union[mat.StringExpressionNode, mat.DummyNode],
    ):
        super(FileStatement, self).__init__()
        self.command: str = command
        self.file_name_node: Union[
            mat.StringExpressionNode, mat.DummyNode
        ] = file_name_node

    def get_literal(self, indent_level: int = 0) -> str:
        return "{0}{1} {2};".format(
            indent_level * "\t", self.command, self.file_name_node
        )


# Option Statement
# ----------------------------------------------------------------------------------------------------------------------
class OptionStatement(BaseStatement):
    def __init__(self, arg_nodes: List[mat.ExpressionNode], redirection: Redirection):
        super(OptionStatement, self).__init__()
        self.arg_nodes: List[mat.ExpressionNode] = arg_nodes
        self.redirection: Redirection = redirection

    def get_literal(self, indent_level: int = 0) -> str:
        literal = "{0}option".format(indent_level * "\t")
        if len(self.arg_nodes) > 0:
            literal += " {0}".format(
                " ".join([arg.get_literal() for arg in self.arg_nodes])
            )
        if self.redirection is not None:
            literal += " {0}".format(self.redirection)
        literal += ";"
        return literal


# Model Entity Declaration
# ----------------------------------------------------------------------------------------------------------------------
class ModelEntityDeclaration(BaseStatement):
    def __init__(self, meta_entity: mat.MetaEntity):
        super(ModelEntityDeclaration, self).__init__()
        self.meta_entity: mat.MetaEntity = meta_entity

    def get_literal(self, indent_level: int = 0) -> str:
        return (
            "{0}".format(indent_level * "\t") + self.meta_entity.generate_declaration()
        )


# Problem Statement
# ----------------------------------------------------------------------------------------------------------------------
class ProblemStatement(BaseStatement):
    def __init__(
        self,
        prob_node: mat.DeclaredEntityNode,
        idx_set_node: mat.CompoundSetNode = None,
        env_sym: str = None,
        item_nodes: List[Tuple[mat.CompoundSetNode, mat.DeclaredEntityNode]] = None,
        redirection: Redirection = None,
    ):

        super(ProblemStatement, self).__init__()

        self.prob_node: mat.DeclaredEntityNode = prob_node
        self.idx_set_node: mat.CompoundSetNode = idx_set_node
        self.env_sym: str = env_sym
        self.item_nodes: List[
            Tuple[mat.CompoundSetNode, mat.DeclaredEntityNode]
        ] = item_nodes
        self.redirection: Redirection = redirection

        if self.item_nodes is None:
            self.item_nodes = []

    def get_literal(self, indent_level: int = 0) -> str:

        literal = "{0}problem".format(indent_level * "\t")

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
                entity_str = "{0}".format("\t" * (indent_level + 1))
                if ent_idx_set_node is None:
                    entity_str += entity_node.get_literal()
                else:
                    entity_str += "{0} {1}".format(ent_idx_set_node, entity_node)
                if i < len(self.item_nodes) - 1:
                    entity_str += ",\n"
                else:
                    entity_str += "\n"
                literal += entity_str
                i += 1

        if self.redirection is not None:
            literal += " {0}".format(self.redirection)

        literal += ";"

        return literal


# Table Statements
# ----------------------------------------------------------------------------------------------------------------------


class BaseTableDataSpec(ABC):
    def __init__(self):
        pass

    def __str__(self):
        return self.get_literal()

    @abstractmethod
    def get_literal(self) -> str:
        pass


class TableDataSpec(BaseTableDataSpec):
    def __init__(
        self,
        data_expr_node: mat.ArithmeticExpressionNode,
        data_col_node: mat.ArithmeticExpressionNode,
        direction: str,
    ):
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
    def __init__(
        self,
        idx_set_node: mat.SetExpressionNode,
        data_spec_components: List[BaseTableDataSpec],
        form: int = 0,
    ):
        super(IndexedTableDataSpec, self).__init__()
        self.idx_set_node: mat.SetExpressionNode = idx_set_node
        self.data_spec_components: List[BaseTableDataSpec] = data_spec_components
        self.form: int = form

    def get_literal(self) -> str:
        if self.form == 0:
            return "{0} {1}".format(self.idx_set_node, self.data_spec_components[0])
        else:
            data_spec_str = ", ".join(
                [ds.get_literal() for ds in self.data_spec_components]
            )
            if self.form == 1:
                return "{0} ({1})".format(self.idx_set_node, data_spec_str)
            else:
                return "{0} <{1}>".format(self.idx_set_node, data_spec_str)


# table
class TableDeclaration(BaseStatement):
    class TableKeySpec:
        def __init__(
            self,
            set_expr_node: mat.SetExpressionNode,
            arrow_token: str,
            key_col_specs: List[Tuple[mat.ExpressionNode, mat.ExpressionNode]],
        ):
            self.set_expr_node: mat.SetExpressionNode = set_expr_node
            self.arrow_token: str = arrow_token
            self.key_col_specs: List[
                Tuple[mat.ExpressionNode, mat.ExpressionNode]
            ] = key_col_specs

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
                    key_col_specs_str.append(
                        "{0} ~ {1}".format(idx_node, data_col_node)
                    )

            literal += "[{0}]".format(", ".join(key_col_specs_str))

            return literal

    def __init__(
        self,
        table_sym: str,
        idx_set_node: mat.SetExpressionNode,
        direction: str,
        opt_nodes: List[mat.StringExpressionNode],
        key_spec: TableKeySpec,
        data_specs: List[BaseTableDataSpec],
    ):
        super(TableDeclaration, self).__init__()
        self.table_sym: str = table_sym
        self.idx_set_node: mat.SetExpressionNode = idx_set_node
        self.direction: str = direction
        self.opt_nodes: List[mat.StringExpressionNode] = opt_nodes
        self.key_spec: TableDeclaration.TableKeySpec = key_spec
        self.data_specs: List[BaseTableDataSpec] = data_specs

    @staticmethod
    def build_table_key_spec(
        set_expr_node: mat.SetExpressionNode,
        arrow_token: str,
        key_col_specs: List[Tuple[mat.ExpressionNode, mat.ExpressionNode]],
    ):
        return TableDeclaration.TableKeySpec(
            set_expr_node=set_expr_node,
            arrow_token=arrow_token,
            key_col_specs=key_col_specs,
        )

    def get_literal(self, indent_level: int = 0) -> str:

        literal = "{0}table {1}".format(indent_level * "\t", self.table_sym)

        if self.idx_set_node is not None:
            literal += " {0}".format(self.idx_set_node)

        if self.direction is not None:
            literal += " {0}".format(self.direction)

        if len(self.opt_nodes) > 0:
            literal += " {0}".format(
                " ".join([n.get_literal() for n in self.opt_nodes])
            )

        literal += ":"

        specs = [self.key_spec]
        specs.extend(self.data_specs)
        literal += " {0};".format(", ".join([str(s) for s in specs]))

        return literal


class TableAccessStatement(BaseStatement):
    def __init__(self, command: str, table_node: mat.DeclaredEntityNode):
        super(TableAccessStatement, self).__init__()
        self.command: str = command
        self.table_node: mat.DeclaredEntityNode = table_node

    def get_literal(self, indent_level: int = 0) -> str:
        return "{0}{1} table {2};".format(
            indent_level * "\t", self.command, self.table_node
        )


# Data Statements
# ----------------------------------------------------------------------------------------------------------------------

SPECIAL_CHARS = "!@#$%^&*()-+?=,<>/ "


def clean_data_statement_element(sub_element: Union[int, float, str]):
    if isinstance(sub_element, str):

        if sub_element.isnumeric():
            sub_element = "'{0}'".format(sub_element)

        elif any(c in SPECIAL_CHARS for c in sub_element):
            sub_element = "'{0}'".format(sub_element)

    else:
        sub_element = str(sub_element)

    return sub_element


class SetDataStatement(BaseStatement):
    def __init__(self, symbol: str, elements: mat.IndexingSet):
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
            return "({0})".format(",".join(sub_elements))

    def get_literal(self, indent_level: int = 0) -> str:
        literal = "{0}set {1} :=\n".format(indent_level * "\t", self.symbol)

        for element in self.elements:
            literal += "{0}{1}\n".format(
                (indent_level + 1) * "\t", self.generate_element_literal(element)
            )

        literal += ";"

        return literal


class ParameterDataStatement(BaseStatement):
    def __init__(
        self,
        symbol: str,
        type: str = "param",
        default_value: Union[int, float, str] = None,
        values: Dict[mat.Element, Union[int, float, str]] = None,
    ):
        super(ParameterDataStatement, self).__init__()
        self.default_value: Union[int, float, str] = default_value
        self.type: str = type  # param or var
        self.symbol: str = symbol
        self.values: Dict[mat.Element, Union[int, float, str]] = (
            values if values is not None else {}
        )

    @staticmethod
    def generate_element_value_pair_literal(
        element: mat.Element, value: Union[int, float, str]
    ):

        value = clean_data_statement_element(value)

        # scalar entity
        if element is None or len(element) == 0:
            return value

        # indexed entity
        else:
            sub_elements = [clean_data_statement_element(se) for se in element]
            return "{0} {1}".format(" ".join(sub_elements), value)

    def get_literal(self, indent_level: int = 0) -> str:

        literal = "{0}{1} {2}".format(indent_level * "\t", self.type, self.symbol)

        if self.default_value is not None:
            literal += " default {0}".format(
                clean_data_statement_element(self.default_value)
            )

        literal += " :=\n"

        for element, value in self.values.items():
            literal += "{0}{1}\n".format(
                (indent_level + 1) * "\t",
                self.generate_element_value_pair_literal(element, value),
            )

        literal += ";"

        return literal


# Display Statement
# ----------------------------------------------------------------------------------------------------------------------

# display, print, printf
class DisplayStatement(BaseStatement):
    class IndexedArgList:
        def __init__(
            self,
            idx_set_node: mat.SetExpressionNode,
            arg_nodes: List[mat.ExpressionNode],
        ):
            self.idx_set_node: mat.SetExpressionNode = idx_set_node
            self.arg_nodes: List[mat.ExpressionNode] = arg_nodes

        def __str__(self):
            return self.get_literal()

        def get_literal(self) -> str:
            literal = self.idx_set_node.get_literal()
            if len(self.arg_nodes) == 1:
                literal += " {0}".format(self.arg_nodes[0])
            else:
                literal += " ({0})".format(
                    ", ".join([n.get_literal() for n in self.arg_nodes])
                )
            return literal

    def __init__(
        self,
        command: str,
        idx_set_node: mat.SetExpressionNode,
        arg_list: List[Union[mat.ExpressionNode, IndexedArgList]],
        redirection: Redirection,
    ):
        super(DisplayStatement, self).__init__()
        self.command: str = command
        self.idx_set_node: mat.SetExpressionNode = idx_set_node
        self.arg_list: List[
            Union[mat.ExpressionNode, DisplayStatement.IndexedArgList]
        ] = arg_list
        self.redirection: Redirection = redirection

    @staticmethod
    def build_indexed_arg_list(
        idx_set_node: mat.SetExpressionNode, arg_nodes: List[mat.ExpressionNode]
    ):
        return DisplayStatement.IndexedArgList(
            idx_set_node=idx_set_node, arg_nodes=arg_nodes
        )

    def get_literal(self, indent_level: int = 0) -> str:
        literal = "{0}".format(indent_level * "\t") + self.command
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
class EntityInclusionStatement(BaseStatement):
    def __init__(
        self,
        command: str,
        idx_set_node: mat.SetExpressionNode,
        entity_node: mat.DeclaredEntityNode,
        redirection: Redirection,
    ):
        super(EntityInclusionStatement, self).__init__()
        self.command: str = command
        self.idx_set_node: mat.SetExpressionNode = idx_set_node
        self.entity_node: mat.DeclaredEntityNode = entity_node
        self.redirection: Redirection = redirection

    def get_literal(self, indent_level: int = 0) -> str:
        literal = "{0}".format(indent_level * "\t") + self.command
        if self.idx_set_node is not None:
            literal += " {0}".format(self.idx_set_node)
        literal += " {0}".format(self.entity_node)
        if self.redirection is not None:
            literal += " {0}".format(self.redirection)
        literal += ";"
        return literal


# fix, unfix
class FixStatement(BaseStatement):
    def __init__(
        self,
        command: str,
        idx_set_node: mat.SetExpressionNode,
        entity_node: mat.DeclaredEntityNode,
        expr_node: mat.ExpressionNode,
        redirection: Redirection,
    ):
        super(FixStatement, self).__init__()
        self.command: str = command
        self.idx_set_node: mat.SetExpressionNode = idx_set_node
        self.entity_node: mat.DeclaredEntityNode = entity_node
        self.expr_node: mat.ExpressionNode = expr_node
        self.redirection: Redirection = redirection

    def get_literal(self, indent_level: int = 0) -> str:
        literal = "{0}".format(indent_level * "\t") + self.command
        if self.idx_set_node is not None:
            literal += " {0}".format(self.idx_set_node)
        literal += " {0}".format(self.entity_node)
        if self.expr_node is not None:
            literal += " := {0}".format(self.expr_node)
        if self.redirection is not None:
            literal += " {0}".format(self.redirection)
        literal += ";"
        return literal


# update, reset, let
class DataManipulationStatement(BaseStatement):
    def __init__(
        self,
        command: str,
        subcommand: str = None,
        idx_set_node: mat.SetExpressionNode = None,
        entity_nodes: List[mat.DeclaredEntityNode] = None,
        expr_node: mat.ExpressionNode = None,
    ):

        super(DataManipulationStatement, self).__init__()

        self.command: str = command
        self.subcommand: str = subcommand
        self.idx_set_node: mat.SetExpressionNode = idx_set_node
        self.entity_nodes: List[mat.DeclaredEntityNode] = entity_nodes
        self.expr_node: mat.ExpressionNode = expr_node

    def get_literal(self, indent_level: int = 0) -> str:

        literal = "{0}".format(indent_level * "\t") + self.command

        if self.subcommand is not None:
            literal += " " + self.subcommand

        if self.idx_set_node is not None:
            literal += " {0}".format(self.idx_set_node)

        if self.entity_nodes is not None:
            literal += " {0}".format(", ".join([str(n) for n in self.entity_nodes]))

        if self.expr_node is not None:
            literal += " := {0}".format(self.expr_node)

        literal += ";"

        return literal


# Solve Statement
# ----------------------------------------------------------------------------------------------------------------------
class SolveStatement(BaseStatement):
    def __init__(self, redirection: Redirection):
        super(SolveStatement, self).__init__()
        self.redirection: Redirection = redirection

    def get_literal(self, indent_level: int = 0) -> str:
        if self.redirection is None:
            return "{0}solve;".format(indent_level * "\t")
        else:
            return "{0}solve {1};".format(indent_level * "\t", self.redirection)


# Control Flow Statements
# ----------------------------------------------------------------------------------------------------------------------


class IfStatement(BaseStatement):
    class IfStatementClause:
        def __init__(
            self,
            operator: str,
            cdn_expr_node: Optional[mat.LogicalExpressionNode],
            statements: Union[BaseStatement, List[BaseStatement]],
            trailing_comments: List[BaseStatement] = None,
        ):
            self.operator: str = operator
            self.cdn_expr_node: mat.LogicalExpressionNode = cdn_expr_node
            self.statements: List[BaseStatement] = (
                statements if isinstance(statements, list) else [statements]
            )
            self.trailing_comments: List[BaseStatement] = (
                trailing_comments if trailing_comments is not None else []
            )

        def __str__(self):
            return self.get_literal()

        def get_literal(self, indent_level: int = 0) -> str:

            literal = "{0}".format("\t" * indent_level) + self.operator

            if self.operator != "else":
                literal += " {0} then".format(self.cdn_expr_node)

            literal += " {\n"
            for statement in self.statements:
                literal += statement.get_literal(indent_level + 1) + "\n"
            literal += "{0}".format("\t" * indent_level) + "}"

            if len(self.trailing_comments) > 0:
                literal += "\n"
                for statement in self.trailing_comments:
                    literal += statement.get_literal(indent_level) + "\n"

            return literal

        def get_validated_literal(
            self,
            indent_level: int = 0,
            validator: Callable[[BaseStatement], bool] = None,
        ) -> str:

            literal = "{0}".format("\t" * indent_level) + self.operator

            if self.operator != "else":
                literal += " {0} then".format(self.cdn_expr_node)

            valid_statements = [s for s in self.statements if validator(s)]

            literal += " {\n"
            for statement in valid_statements:
                literal += (
                    statement.get_validated_literal(indent_level + 1, validator) + "\n"
                )
            literal += "{0}".format("\t" * indent_level) + "}"

            if len(self.trailing_comments) > 0:
                literal += "\n"
                for statement in self.trailing_comments:
                    literal += statement.get_validated_literal(indent_level) + "\n"

            return literal

    def __init__(self, clauses: List[IfStatementClause]):
        super(IfStatement, self).__init__()
        self.clauses: List[IfStatement.IfStatementClause] = clauses

    @staticmethod
    def build_clause(
        operator: str,
        cdn_expr_node: Optional[mat.LogicalExpressionNode],
        statements: Union[BaseStatement, List[BaseStatement]],
        trailing_comments: List[BaseStatement] = None,
    ):
        return IfStatement.IfStatementClause(
            operator=operator,
            cdn_expr_node=cdn_expr_node,
            statements=statements,
            trailing_comments=trailing_comments,
        )

    def get_literal(self, indent_level: int = 0) -> str:
        return "\n".join([c.get_literal(indent_level) for c in self.clauses])

    def get_validated_literal(
        self, indent_level: int = 0, validator: Callable[[BaseStatement], bool] = None
    ) -> str:
        return "\n".join(
            [c.get_validated_literal(indent_level, validator) for c in self.clauses]
        )


class ForLoopStatement(BaseStatement):
    def __init__(
        self,
        loop_sym: str,
        idx_set_node: mat.SetExpressionNode,
        statements: List[BaseStatement],
    ):
        super(ForLoopStatement, self).__init__()
        self.loop_sym: str = loop_sym
        self.idx_set_node: mat.SetExpressionNode = idx_set_node
        self.statements: List[BaseStatement] = (
            statements if isinstance(statements, list) else [statements]
        )

    def get_literal(self, indent_level: int = 0) -> str:

        literal = "{0}for".format("\t" * indent_level)
        if self.loop_sym is not None:
            literal += " " + self.loop_sym
        literal += " " + self.idx_set_node.get_literal()

        literal += " {\n"
        for statement in self.statements:
            literal += statement.get_literal(indent_level + 1) + "\n"
        literal += "{0}".format("\t" * indent_level) + "}"

        return literal

    def get_validated_literal(
        self, indent_level: int = 0, validator: Callable[[BaseStatement], bool] = None
    ) -> str:

        literal = "{0}for".format("\t" * indent_level)
        if self.loop_sym is not None:
            literal += " " + self.loop_sym
        literal += " " + self.idx_set_node.get_literal()

        valid_statements = [s for s in self.statements if validator(s)]

        literal += " {\n"
        for statement in valid_statements:
            literal += (
                statement.get_validated_literal(indent_level + 1, validator) + "\n"
            )
        literal += "{0}".format("\t" * indent_level) + "}"

        return literal


def add_included_script_to_compound_script(
    compound_script: CompoundScript,
    included_script: Script,
    include_in_main: bool = True,
    file_command: str = "include",
    statement_index: int = 0,
):
    compound_script.included_scripts[included_script.name] = included_script
    if include_in_main:
        file_name_node = mat.StringNode(literal=included_script.name, delimiter='"')
        file_statement = FileStatement(
            command=file_command, file_name_node=file_name_node
        )
        compound_script.main_script.statements.insert(statement_index, file_statement)
