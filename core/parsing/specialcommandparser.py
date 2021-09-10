import json
from typing import List

from symro.core.prob.specialcommand import SpecialCommand


class SpecialCommandParser:

    SPECIAL_COMMAND_DELIMITER_SYMBOL = '@'

    def __init__(self):
        self.literal: str = ""
        self.index: int = 0

    # Script Command Parsing
    # ------------------------------------------------------------------------------------------------------------------

    def parse_script_commands(self, literal: str) -> List[SpecialCommand]:

        self.index = 0
        self.literal = literal.strip()

        return self.__parse_special_commands()

    def __parse_special_commands(self) -> List[SpecialCommand]:

        script_commands: List[SpecialCommand] = []

        while self.index < len(self.literal):

            if not self.__skip_until_char(self.SPECIAL_COMMAND_DELIMITER_SYMBOL):
                break
            self.__next()

            script_flag_symbol = self.__extract_token(delimiters=[' ', '('])

            self.__skip_char(' ')
            args_str = []
            if self.char() == '(':

                self.__next()
                self.__skip_char(' ')

                while self.char() != ')':

                    if self.char() == '[':
                        value = self.__skip_nested_delimiters('[', ']')
                        args_str.append((None, value))

                    elif self.char() == '{':
                        value = self.__skip_nested_delimiters('{', '}')
                        args_str.append((None, value))

                    else:
                        token = self.__extract_token(delimiters=[' ', '=', ',', ')'])
                        self.__skip_char(' ')

                        if self.char() == '=':

                            self.__next()
                            self.__skip_char(' ')

                            if self.char() == '[':
                                value = self.__skip_nested_delimiters('[', ']')
                                args_str.append((token, value))
                            elif self.char() == '{':
                                value = self.__skip_nested_delimiters('{', '}')
                                args_str.append((token, value))
                            else:
                                value = self.__extract_token(delimiters=[' ', ',', ')'])
                                args_str.append((token, value))

                        else:
                            args_str.append((None, token))

                    self.__skip_char(' ')
                    if self.char() == ',':
                        self.__next()
                        self.__skip_char(' ')

            script_command = SpecialCommand(symbol=script_flag_symbol)
            for name, value in args_str:
                try:
                    if value[0] in ['[', '{']:
                        value = json.loads(value)
                    else:
                        value = json.loads("[{0}]".format(value))[0]
                except:
                    pass
                script_command.add_arg(value=value, name=name)

            script_commands.append(script_command)

        return script_commands

    # Utility
    # ------------------------------------------------------------------------------------------------------------------

    def __next(self, count: int = 1) -> bool:
        self.index += count
        return self.index < len(self.literal)

    def char(self):
        try:
            return self.literal[self.index]
        except IndexError:
            return None

    def __is_char_escaped(self) -> bool:
        if self.index > 0:
            return self.literal[self.index - 1] == '\\'
        else:
            return False

    def __skip_char(self, c: str):
        while self.char() == c:
            self.__next()

    def __skip_until_char(self, c, can_skip_escaped: bool = False) -> bool:

        def check_char(_c):
            return _c == c

        def check_list(_c):
            return _c in c

        if isinstance(c, list):
            checker = check_list
        else:
            checker = check_char

        while self.index < len(self.literal):
            if checker(self.char()):
                if not can_skip_escaped:
                    return True
                elif not self.__is_char_escaped():
                    return True
            self.__next()
        return False

    def __skip_nested_delimiters(self, left: str, right: str) -> str:
        start_index = self.index
        if self.char() != left:  # starting character must be the opening delimiter
            raise ValueError("Missing opening delimiter.")
        nested_count = 1
        while True:
            self.__next()
            self.__skip_until_char([left, right], can_skip_escaped=True)
            if self.char() == left:  # found left delimiter
                nested_count += 1
            else:
                nested_count -= 1  # found right delimiter
            if nested_count == 0:  # found last right delimiter
                break
        self.__next()  # skip last right delimiter
        return self.literal[start_index:self.index]  # return <left><string><right>

    def __extract_token(self, delimiters: List[str] = None) -> str:

        if delimiters is None:
            delimiters = []

        # start at first character of the token
        start_index = self.index

        at_delimiter = False
        while self.index < len(self.literal):
            for d in delimiters:
                if len(d) == 1:
                    if self.char() == d:
                        at_delimiter = True
                        break
                elif len(d) > 1:
                    if self.__read_ahead(len(d)) == d:
                        at_delimiter = True
            if at_delimiter:
                break  # end at first character immediately after the token (i.e. at delimiter)
            self.__next()

        if start_index == self.index:
            return ""
        else:
            return self.literal[start_index: self.index]

    def __read_ahead(self, count: int) -> str:
        if len(self.literal) - self.index < count:
            return self.literal[self.index:]
        else:
            return self.literal[self.index:self.index + count]
