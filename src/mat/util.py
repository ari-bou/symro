import itertools
from ordered_set import OrderedSet
from typing import Iterable, List, Tuple, Union


# Types
# ----------------------------------------------------------------------------------------------------------------------
Element = Tuple[Union[int, float, str, None], ...]
IndexingSet = OrderedSet[Element]

# Operators
# ----------------------------------------------------------------------------------------------------------------------

# Unary Arithmetic
UNARY_POSITIVE_OPERATOR = 1
UNARY_NEGATION_OPERATOR = 2

# Binary Arithmetic
ADDITION_OPERATOR = 11
SUBTRACTION_OPERATOR = 12
MULTIPLICATION_OPERATOR = 13
DIVISION_OPERATOR = 14
EXPONENTIATION_OPERATOR = 15

# Unary Logical
UNARY_INVERSION_OPERATOR = 101

# Binary Logical
CONJUNCTION_OPERATOR = 111
DISJUNCTION_OPERATOR = 112

# Relational
EQUALITY_OPERATOR = 121
STRICT_INEQUALITY_OPERATOR = 122
LESS_INEQUALITY_OPERATOR = 123
LESS_EQUAL_INEQUALITY_OPERATOR = 124
GREATER_INEQUALITY_OPERATOR = 125
GREATER_EQUAL_INEQUALITY_OPERATOR = 126

# Set

UNION_OPERATOR = 201
INTERSECTION_OPERATOR = 202
DIFFERENCE_OPERATOR = 203
SYMMETRIC_DIFFERENCE_OPERATOR = 204

SETOF_OPERATOR = 211

# String
CONCATENATION_OPERATOR = 301

# Symbols
AMPL_OPERATOR_SYMBOLS = {

    UNARY_NEGATION_OPERATOR: '-',

    ADDITION_OPERATOR: '+',
    SUBTRACTION_OPERATOR: '-',
    MULTIPLICATION_OPERATOR: '*',
    DIVISION_OPERATOR: '/',
    EXPONENTIATION_OPERATOR: '^',

    UNARY_INVERSION_OPERATOR: '!',

    CONJUNCTION_OPERATOR: '&&',
    DISJUNCTION_OPERATOR: '||',

    EQUALITY_OPERATOR: '==',
    STRICT_INEQUALITY_OPERATOR: '!=',
    LESS_INEQUALITY_OPERATOR: '<',
    LESS_EQUAL_INEQUALITY_OPERATOR: '<=',
    GREATER_INEQUALITY_OPERATOR: '>',
    GREATER_EQUAL_INEQUALITY_OPERATOR: '>=',

    UNION_OPERATOR: 'union',
    INTERSECTION_OPERATOR: 'inter',
    DIFFERENCE_OPERATOR: 'diff',
    SYMMETRIC_DIFFERENCE_OPERATOR: 'symdiff',

    SETOF_OPERATOR: 'setof',

    CONCATENATION_OPERATOR: '&'

}


# Element Operations
# ----------------------------------------------------------------------------------------------------------------------

def get_element_literal(element: Union[int, float, str]):
    """
    Transform an element into a string literal. Delimiters are added to string elements, whereas numeric elements are
    converted to strings. Element cannot be a dummy or a parameter.
    :param element: a non-dummy element
    :return: element string literal
    """
    if isinstance(element, str):
        return "'{0}'".format(element)
    elif isinstance(element, float):
        if element.is_integer():
            element = int(element)
    return str(element)


# Set Operations
# ----------------------------------------------------------------------------------------------------------------------

def cartesian_product(sets: List[IndexingSet]) -> IndexingSet:
    """
    Evaluate the cartesian product of two or more sets. Each element of the combined set is a unique combination of 1
    element from each of the constituent sets.
    :param sets: list of constituent ordered sets
    :return: ordered set of elements comprising the combined set
    """

    if len(sets) == 0:
        return OrderedSet()

    else:
        sets = [s for s in sets if s is not None]
        combined_elements = itertools.product(*sets)
        flattened_elements = [flatten_element(e) for e in combined_elements]
        return OrderedSet(flattened_elements)


def flatten_element(element: Tuple[Union[int, float, str, Element], ...]) -> Element:

    flattened_element = []

    for sub_element in element:

        if isinstance(sub_element, Iterable):
            flattened_element.extend(sub_element)

        else:
            flattened_element.append(sub_element)

    return tuple(flattened_element)


def aggregate_set(set_in: IndexingSet,
                  dim_positions: List[int],
                  agg_values: List[Union[int, float, str, None]]):
    agg_set = OrderedSet()
    for element in set_in:
        tfm_element = list(element)
        for dim_pos, agg_val in zip(dim_positions, agg_values):
            tfm_element[dim_pos] = (agg_val,)
        agg_set.add(tuple(tfm_element))
    return agg_set


def filter_set(set_in: IndexingSet,
               filter_element: Union[List[Union[int, float, str, None]],
                                     Tuple[Union[int, float, str, None], ...]]) -> IndexingSet:
    filtered_set = OrderedSet()
    for element in set_in:
        is_in_filter = [e == f or f is None for e, f in zip(element, filter_element)]
        if all(is_in_filter):
            filtered_set.add(element)
    return filtered_set


def remove_set_dimensions(set_in: IndexingSet, dim_positions: List[int]):
    dim_positions.sort(reverse=True)
    set_out = OrderedSet()
    for element in set_in:
        element_list = list(element)
        for dim_pos in dim_positions:
            element_list.pop(dim_pos)
        set_out.add(tuple(element_list))
    return set_out
