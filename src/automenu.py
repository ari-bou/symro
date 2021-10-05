from abc import ABC, abstractmethod
from typing import Dict, List, Optional


class MetaArgument:

    def __init__(self,
                 name: str,
                 dtype: str = None,
                 is_optional: bool = False,
                 default_value: str = None):
        self.name: str = name
        self.dtype: str = dtype
        self.is_optional: bool = is_optional
        self.default_value: str = self.transform(default_value)

    def transform(self, value):
        if self.dtype is None:
            return value
        elif self.dtype == "bool":
            value = str(value)
            value = value.lower().strip()
            if value in ['t', "true", '1', 1]:
                return True
            elif value in ['f', "false", '0', 0]:
                return False
            else:
                return None
        elif self.dtype == "int":
            try:
                return int(value)
            except TypeError:
                return None
        elif self.dtype == "float":
            try:
                return float(value)
            except TypeError:
                return None
        elif self.dtype == "str":
            return str(value)
        return value


class Argument:

    def __init__(self, arg_id: int, name: str, value):
        self.arg_id: int = arg_id
        self.name: str = name
        self.value = value


class Command:

    def __init__(self):

        self.__new_id: int = -1

        self.__args: Dict[int, Argument] = {}
        self.__ordered_args: List[Argument] = []
        self.__named_args: Dict[str, Argument] = {}
        self.__options: Dict[str, str] = {}

        self.executed: bool = False

    def __len__(self):
        return len(self.__args)

    def __generate_arg_id(self) -> int:
        self.__new_id += 1
        return self.__new_id

    def add_arg(self, value, name: str = None):
        arg = Argument(self.__generate_arg_id(), name, value)
        self.__args[arg.arg_id] = arg
        if name is None:
            self.__ordered_args.append(arg)
        else:
            self.__named_args[name] = arg

    def remove_arg(self, index: int = None, name: str = None):

        arg = None

        if name is not None:
            try:
                arg = self.__named_args[name]
            except KeyError:
                raise ValueError("Invalid argument identifier: '{0}'.".format(name))

        if index is not None:
            try:
                arg = self.__ordered_args[index]
            except IndexError:
                raise ValueError("Argument index out of range: '{0}'.".format(index))

        if arg is None:
            raise ValueError("No valid argument identifier was supplied.")

        self.__args.pop(arg.arg_id)
        if arg.name is None:
            self.__ordered_args.pop(index)
        else:
            self.__named_args.pop(arg.name)

    def get_arg(self, index: int = None, name: str = None) -> str:

        if name is not None:
            try:
                return self.__named_args[name].value
            except KeyError:
                if index is None:
                    raise ValueError("Invalid argument identifier: '{0}'.".format(name))

        if index is not None:
            if index < len(self.__ordered_args):
                return self.__ordered_args[index].value
            elif index < len(self.__ordered_args) + len(self.__named_args):
                index = index - len(self.__ordered_args)
                name = list(self.__named_args.keys())[index]
                return self.__named_args[name].value
            else:
                raise ValueError("Argument index out of range: '{0}'.".format(index))

        if len(self.__args) > 0:
            for arg_id, arg in self.__args.items():
                return arg.value

        else:
            raise ValueError("Command has no arguments.")

    def get_ordered_args(self) -> list:
        return [a.value for a in self.__ordered_args]

    def get_named_args(self) -> dict:
        return {k: v.value for k, v in self.__named_args.items()}

    def add_option(self, key: str, val: str = ""):
        if key[0] == '-':
            key = key[1:]
        self.__options[key] = val

    def contains_option(self, key: str) -> bool:
        return key in self.__options.keys()


class Server(ABC):

    def __init__(self, action: str):
        self.action = action

    @abstractmethod
    def run(self, cmd: Command):
        pass


class Dispatcher(Server):

    def __init__(self,
                 action: str,
                 routes: List[str]):

        super(Dispatcher, self).__init__(action)
        self.routes: List[str] = routes

    def run(self, cmd: Command):
        action = cmd.get_arg(index=1)
        if action in self.routes:
            cmd.remove_arg(index=0)
        else:
            raise ValueError("Invalid routing symbol encountered: '{0}'.".format(action))


class Executor(Server):

    def __init__(self,
                 action: str,
                 method,
                 meta_args: List[MetaArgument] = None,
                 help_message: str = None,
                 can_accept_unspecified_args: bool = False):

        super(Executor, self).__init__(action)

        self.method = method
        self.meta_args: Dict[str, MetaArgument] = {}
        self.arg_order: List[MetaArgument] = []
        self.help_message: str = help_message if help_message is not None else ""
        self.can_accept_unspecified_args: bool = can_accept_unspecified_args

        if meta_args is not None:
            for meta_arg in meta_args:
                self.meta_args[meta_arg.name] = meta_arg
                self.arg_order.append(meta_arg)

    def add_meta_arg(self,
                     name: str,
                     dtype: str,
                     is_optional: bool,
                     default_value: str = None):
        meta_arg = MetaArgument(name=name,
                                dtype=dtype,
                                is_optional=is_optional,
                                default_value=default_value)
        self.meta_args[name] = meta_arg
        self.arg_order.append(meta_arg)

    def run(self, cmd: Command):

        proc_arg_names: List[str] = []

        args = cmd.get_ordered_args()
        kwargs = cmd.get_named_args()

        action = args.pop(0)
        if action != self.action:
            raise ValueError("Command action cannot be a named argument.")

        if len(args) + len(kwargs) - 1 > len(self.arg_order) and not self.can_accept_unspecified_args:
            raise ValueError("Too many arguments were supplied.")

        for i, arg in enumerate(args):
            if i < len(self.arg_order):
                meta_arg = self.arg_order[i]
                proc_arg_names.append(meta_arg.name)
                args[i] = meta_arg.transform(arg)

        for name, arg in kwargs.items():
            if name in self.meta_args:
                proc_arg_names.append(name)
                kwargs[name] = self.meta_args[name].transform(arg)
            elif not self.can_accept_unspecified_args:
                raise ValueError("Encountered invalid argument identifier '{0}'.".format(name))

        for name, meta_arg in self.meta_args.items():
            if not meta_arg.is_optional and meta_arg.name not in proc_arg_names:
                kwargs[name] = meta_arg.default_value

        self.method(*args, **kwargs)
        cmd.executed = True


class CommandParser:

    def __init__(self):
        self.token_index: int = 0
        self.token: str = ""
        self.tokens: List[str] = []

    def tokenize_and_parse(self, literal: str) -> Command:
        self.__tokenize_command(literal)
        return self.__parse_command()

    def __tokenize_command(self, cmd_lit: str):

        cmd_lit = cmd_lit.strip()

        self.tokens = []
        self.token = ""
        is_literal_single = False
        is_literal_double = False

        for i, c in enumerate(cmd_lit):

            if is_literal_single:
                if c == "'":
                    is_literal_single = False
                    self.__add_token()
                    self.__add_token("'")
                else:
                    self.token += c
            elif is_literal_double:
                if c == '"':
                    is_literal_double = False
                    self.__add_token()
                    self.__add_token('"')
                else:
                    self.token += c

            elif c == ' ':
                self.__add_token()
            elif c in ['=', ':', ',', '[', ']', '{', '}']:
                self.__add_token()
                self.__add_token(c)
            elif c == "'":
                self.__add_token()
                self.__add_token("'")
                is_literal_single = True
            elif c == '"':
                self.__add_token()
                self.__add_token('"')
                is_literal_double = True
            else:
                self.token += c

            if i == len(cmd_lit) - 1:
                self.__add_token()

    def __parse_command(self) -> Command:

        cmd = Command()

        self.token_index = 0

        while self.token_index < len(self.tokens):

            token = self.get_token()

            if token[0] == '-':
                cmd.add_option(token)
                self.__next_token()

            else:

                arg_val = self.__parse_argument_value()  # skips current token

                if self.token_index < len(self.tokens):
                    if self.get_token() == "=":
                        arg_name = arg_val
                        self.__next_token()  # skip '='
                        arg_val = self.__parse_argument_value()
                        cmd.add_arg(arg_val, name=arg_name)
                    else:
                        cmd.add_arg(arg_val)
                else:
                    cmd.add_arg(arg_val)

        return cmd

    def __parse_argument_value(self):

        # List of arguments
        if self.get_token() == '[':
            arg_val = self.__parse_list()  # skips opening and closing brackets

        # Dictionary of arguments
        elif self.get_token() == '{':
            arg_val = self.__parse_dict()  # skips opening and closing braces

        # String literal argument
        elif self.get_token() in ['"', "'"]:
            self.__next_token()  # skip opening string delimiter
            arg_val = self.get_token()
            self.__next_token()
            self.__next_token()  # skip closing string delimiter

        # Normal argument
        else:
            arg_val = self.get_token()
            self.__next_token()

        return arg_val

    def __parse_list(self):

        self.__next_token()  # skip opening bracket

        arg_vals = []
        while self.get_token() != ']':
            arg_val = self.__parse_argument_value()
            arg_vals.append(arg_val)
            if self.get_token() == ',':
                self.__next_token()

        self.__next_token()  # skip closing bracket
        return arg_vals

    def __parse_dict(self):

        self.__next_token()  # skip opening brace

        arg_dict = {}
        while self.get_token() != '}':
            arg_key = self.__parse_argument_value()
            self.__next_token()  # skip ':'
            arg_val = self.__parse_argument_value()
            arg_dict[arg_key] = arg_val
            if self.get_token() == ',':
                self.__next_token()

        self.__next_token()  # skip closing brace
        return arg_dict

    def is_last_token(self) -> bool:
        return self.token_index >= len(self.tokens) - 1

    def __add_token(self, token: str = None):
        if token is not None:
            self.token = token
        if self.token != "":
            self.tokens.append(self.token)
            self.token = ""

    def get_token(self) -> str:
        return self.tokens[self.token_index]

    def __prev_token(self) -> bool:
        if self.token_index != 0:
            self.token_index -= 1
            return True
        else:
            return False

    def __next_token(self) -> bool:
        if not self.is_last_token():
            self.token_index += 1
            return True
        else:
            self.token_index += 1
            return False


class Menu:

    def __init__(self,
                 servers: Dict[str, Server],
                 quit_symbols: List[str] = None,
                 help_symbols: List[str] = None,
                 pbcmd_symbols: List[str] = None,
                 prebuilt_commands: List[str] = None,
                 header: str = None,
                 preamble: str = None,
                 instructions: str = None,
                 tag: str = None,
                 can_display_startup_pbcmd: bool = True,
                 can_display_startup_help: bool = True,
                 can_catch_exceptions: bool = True,
                 can_loop: bool = False):

        self.command: Optional[Command] = None
        self.servers: Dict[str, Server] = servers

        self.quit_symbols: List[str] = quit_symbols
        self.help_symbols: List[str] = help_symbols
        self.pbcmd_symbols: List[str] = pbcmd_symbols

        self.prebuilt_commands: List[str] = prebuilt_commands

        self.header: str = header
        self.preamble: str = preamble
        self.instructions: str = instructions
        self.help_message: str = ""
        self.tag: str = tag if header is not None else ""

        self.can_display_startup_pbcmd: bool = can_display_startup_pbcmd
        self.can_display_startup_help: bool = can_display_startup_help
        self.can_catch_exceptions: bool = can_catch_exceptions
        self.can_loop: bool = can_loop

        if self.quit_symbols is None:
            self.quit_symbols = []
        if len(self.quit_symbols) == 0:
            self.quit_symbols.append("quit")

        if self.help_symbols is None:
            self.help_symbols = []
        if len(self.help_symbols) == 0:
            self.help_symbols.append("help")

        if self.pbcmd_symbols is None:
            self.pbcmd_symbols = []

        if self.prebuilt_commands is None:
            self.prebuilt_commands = []

    def run(self):

        self.__generate_help_message()
        prompt = self.__generate_prompt()

        print(prompt)

        is_running = True
        while is_running:
            try:
                command_literal = input("\n" + self.tag + '>')
                is_running = self.__parse_and_execute_command(command_literal)
                if not self.can_loop:
                    is_running = False
            except Exception as e:
                if self.can_catch_exceptions:
                    print("Error: {0}".format(e))
                else:
                    raise e

    def __generate_help_message(self):

        help_message = ""
        help_messages = {}

        for action, server in self.servers.items():
            if isinstance(server, Executor):
                help_messages[action] = server.help_message
            elif isinstance(server, Dispatcher):
                help_messages[action] = '|'.join(server.routes)

        help_msg_list = [(action, msg) for action, msg in help_messages.items()]
        help_msg_list.sort(key=lambda t: t[0])

        if self.instructions is not None:
            help_message = self.instructions + "\n"

        help_message += "Actions:\n"
        for action, msg in help_msg_list:
            help_message += "\t{0}: {1}\n".format(action, msg)

        self.help_message = help_message

    def __generate_prompt(self) -> str:

        prompt = "\n"

        if self.header is not None:
            prompt += "{0}\n{1}\n".format(self.header, '-' * 50)

        if self.preamble is not None:
            prompt += "{0}\n".format(self.preamble)

        if self.can_display_startup_pbcmd and len(self.prebuilt_commands) > 0:
            prompt += "\nCommands: \n"
            for i, pbcmd_literal in enumerate(self.prebuilt_commands):
                prompt += "\t {0}. {1}\n".format(i, pbcmd_literal)

        if self.can_display_startup_help and self.instructions is not None:
            prompt += "{0}".format(self.help_message)

        return prompt

    def __parse_and_execute_command(self, command_literal: str) -> bool:
        command_parser = CommandParser()
        self.command = command_parser.tokenize_and_parse(command_literal)
        is_running = self.__execute_command(self.command)
        return is_running

    def __execute_command(self, command: Command) -> bool:

        is_routing = True
        is_running = True

        while is_routing:

            action = command.get_arg(index=0)

            if action in self.quit_symbols:
                is_running = False
                is_routing = False

            elif action in self.help_symbols:
                print("\n{0}".format(self.help_message))
                is_routing = False

            elif action in self.pbcmd_symbols:
                is_running = self.__execute_prebuilt_command(command)
                is_routing = False

            else:
                self.servers[action].run(command)
                if command.executed:
                    is_routing = False

        return is_running

    def __execute_prebuilt_command(self, command: Command) -> bool:

        index = int(command.get_arg(1))

        try:
            pbcmd_literal = self.prebuilt_commands[index]
        except IndexError:
            raise ValueError("Index of prebuilt command is out of range.")

        return self.__parse_and_execute_command(pbcmd_literal)
