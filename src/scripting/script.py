from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Union

import symro.src.util.util as util
from symro.src.scripting.specialcommand import SpecialCommand


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
                 name: str = "main",
                 raw_literal: str = None,
                 tokens: List[str] = None,
                 statements: List[BaseStatement] = None):
        self.name: str = name
        self.raw_literal: str = raw_literal
        self.tokens: List[str] = tokens
        self.token_index: int = 0
        self.statements: List[BaseStatement] = statements if statements is not None else []

    def __str__(self):
        return self.get_literal()

    def __len__(self):
        return len(self.statements)

    def copy(self, source: "Script"):
        self.name = source.name
        self.raw_literal = source.raw_literal
        self.tokens = list(source.tokens)
        self.token_index = source.token_index
        self.statements = list(source.statements)

    def write(self, dir_path: str, file_name: str = None):
        if file_name is None:
            file_name = self.name
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
            included_scripts = {script.name: script for script in included_scripts}

        self.main_script: Optional[Script] = main_script
        self.included_scripts: Dict[str, Script] = included_scripts

    def copy(self, source: "CompoundScript"):

        self.main_script = Script()
        self.main_script.copy(source.main_script)

        self.included_scripts = {}
        for id, script in source.included_scripts.items():
            self.included_scripts[id] = Script()
            self.included_scripts[id].copy(script)

    def write(self, dir_path: str, main_file_name: str = None):
        self.main_script.write(dir_path, main_file_name)
        for id, included_script in self.included_scripts.items():
            included_script.write(dir_path)
