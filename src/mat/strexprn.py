from functools import partial
import numpy as np
from typing import Callable, List

from symro.src.mat.exprn import StringExpressionNode
from symro.src.mat.opern import RelationalOperationNode, StringOperationNode
from symro.src.mat.lexprn import BooleanNode
from symro.src.mat.util import IndexingSet, Element
from symro.src.mat.state import State


class StringNode(StringExpressionNode):

    def __init__(self, literal: str, delimiter: str = None):

        super().__init__()

        self.literal: str = literal  # string literal without delimiter
        self.delimiter: str = delimiter

        if self.delimiter is None:
            if '"' not in self.literal:
                self.delimiter = '"'
            elif "'" not in self.literal:
                self.delimiter = "'"
            else:
                raise ValueError("Encountered a string literal ({0})".format(self.literal)
                                 + " containing both string delimiters ' and \" ")

    def __and__(self, other: StringExpressionNode):

        if isinstance(other, StringNode):
            return StringNode(self.literal + other.literal)

        return StringOperationNode.concatenate(self, other)

    def __eq__(self, other: StringExpressionNode):
        if isinstance(other, StringNode):
            return BooleanNode(value=self.literal == other.literal)
        else:
            return RelationalOperationNode.equal(self, other)

    def __ne__(self, other: StringExpressionNode):
        if isinstance(other, StringNode):
            return BooleanNode(value=self.literal != other.literal)
        else:
            return RelationalOperationNode.not_equal(self, other)

    def __lt__(self, other: StringExpressionNode):
        if isinstance(other, StringNode):
            return BooleanNode(value=self.literal < other.literal)
        else:
            return RelationalOperationNode.less_than(self, other)

    def __le__(self, other: StringExpressionNode):
        if isinstance(other, StringNode):
            return BooleanNode(value=self.literal <= other.literal)
        else:
            return RelationalOperationNode.less_than_or_equal(self, other)

    def __gt__(self, other: StringExpressionNode):
        if isinstance(other, StringNode):
            return BooleanNode(value=self.literal > other.literal)
        else:
            return RelationalOperationNode.greater_than(self, other)

    def __ge__(self, other: StringExpressionNode):
        if isinstance(other, StringNode):
            return BooleanNode(value=self.literal >= other.literal)
        else:
            return RelationalOperationNode.greater_than_or_equal(self, other)

    def evaluate(self,
                 state: State,
                 idx_set: IndexingSet = None,
                 dummy_element: Element = None) -> np.ndarray:
        mp = 1 if idx_set is None else len(idx_set)
        return np.full(shape=mp, fill_value=self.literal)

    def to_lambda(self,
                  state: State,
                  idx_set_member: Element = None,
                  dummy_element: Element = None) -> Callable:
        return partial(lambda l: l, self.literal)

    def get_children(self) -> List:
        return []

    def set_children(self, operands: list):
        pass

    def get_literal(self) -> str:
        return self.delimiter + self.literal + self.delimiter
