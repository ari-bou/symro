import itertools
from ordered_set import OrderedSet
from typing import Iterable, List, Tuple, Union


# Types
# ----------------------------------------------------------------------------------------------------------------------
Element = Tuple[Union[int, float, str, None], ...]
IndexingSet = OrderedSet[Element]

# Entities
SET_TYPE = "set"
PARAM_TYPE = "param"
VAR_TYPE = "var"
OBJ_TYPE = "obj"
CON_TYPE = "con"
TABLE_TYPE = "table"
PROB_TYPE = "prob"
ENV_TYPE = "env"

# Functions
CONSTANT = 0
LINEAR = 1
BILINEAR = 2
TRILINEAR = 3
FRACTIONAL = 4
FRACTIONAL_BILINEAR = 5
FRACTIONAL_TRILINEAR = 6
UNIVARIATE_NONLINEAR = 7
GENERAL_NONCONVEX = 8


# Operators and Functions
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

# Basic Arithmetic Functions
SUMMATION_FUNCTION = 21
PRODUCT_FUNCTION = 22
MINIMUM_FUNCTION = 23
MAXIMUM_FUNCTION = 24
EXPONENTIAL_FUNCTION = 25
NATURAL_LOGARITHM_FUNCTION = 26
BASE_10_LOGARITHM_FUNCTION = 27
SQUARE_ROOT_FUNCTION = 28
ABSOLUTE_VALUE_FUNCTION = 29
MODULUS_FUNCTION = 30
INTEGER_DIVISION_FUNCTION = 31

# Trigonometric Functions
SINE_FUNCTION = 41
COSINE_FUNCTION = 42
TANGENT_FUNCTION = 43
HYPERBOLIC_SINE_FUNCTION = 44
HYPERBOLIC_COSINE_FUNCTION = 45
HYPERBOLIC_TANGENT_FUNCTION = 46
INVERSE_SINE_FUNCTION = 47
INVERSE_COSINE_FUNCTION = 48
INVERSE_TANGENT_FUNCTION = 49
INVERSE_HYPERBOLIC_SINE_FUNCTION = 50
INVERSE_HYPERBOLIC_COSINE_FUNCTION = 51
INVERSE_HYPERBOLIC_TANGENT_FUNCTION = 52
INVERSE_HYPERBOLIC_TANGENT_FUNCTION_2 = 53

# Rounding Functions
ROUND_FUNCTION = 61
PRECISION_FUNCTION = 62
TRUNCATION_FUNCTION = 63
CEILING_FUNCTION = 64
FLOOR_FUNCTION = 65

# Random Number Functions
BETA_RNG_FUNCTION = 71
CAUCHY_RNG_FUNCTION = 72
EXPONENTIAL_RNG_FUNCTION = 73
GAMMA_RNG_FUNCTION = 74
Irand224_RNG_FUNCTION = 75
NORMAL_RNG_FUNCTION = 76
STANDARD_NORMAL_RNG_FUNCTION = 77
POISSON_RNG_FUNCTION = 78
UNIFORM_RNG_FUNCTION = 79
STANDARD_UNIFORM_RNG_FUNCTION = 80

# Miscellaneous Functions
TIME_FUNCTION = 91
CTIME_FUNCTION = 92
ALIAS_FUNCTION = 93

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

# Reductive Logical
EXISTS_OPERATOR = 131
FOR_ALL_OPERATOR = 132

# Set

UNION_OPERATOR = 201
INTERSECTION_OPERATOR = 202
DIFFERENCE_OPERATOR = 203
SYMMETRIC_DIFFERENCE_OPERATOR = 204

SETOF_OPERATOR = 211

# String
CONCATENATION_OPERATOR = 301

# Operator Symbols
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

    EXISTS_OPERATOR: "exists",
    FOR_ALL_OPERATOR: "forall",

    UNION_OPERATOR: 'union',
    INTERSECTION_OPERATOR: 'inter',
    DIFFERENCE_OPERATOR: 'diff',
    SYMMETRIC_DIFFERENCE_OPERATOR: 'symdiff',

    SETOF_OPERATOR: 'setof',

    CONCATENATION_OPERATOR: '&'

}

# Function Symbols

AMPL_FUNCTION_SYMBOLS = {

    SUMMATION_FUNCTION: "sum",
    PRODUCT_FUNCTION: "prod",
    MINIMUM_FUNCTION: "min",
    MAXIMUM_FUNCTION: "max",
    EXPONENTIAL_FUNCTION: "exp",
    NATURAL_LOGARITHM_FUNCTION: "log",
    BASE_10_LOGARITHM_FUNCTION: "log10",
    SQUARE_ROOT_FUNCTION: "sqrt",
    ABSOLUTE_VALUE_FUNCTION: "abs",
    MODULUS_FUNCTION: "mod",
    INTEGER_DIVISION_FUNCTION: "div",

    SINE_FUNCTION: "sin",
    COSINE_FUNCTION: "cos",
    TANGENT_FUNCTION: "tan",
    HYPERBOLIC_SINE_FUNCTION: "sinh",
    HYPERBOLIC_COSINE_FUNCTION: "cosh",
    HYPERBOLIC_TANGENT_FUNCTION: "tanh",
    INVERSE_SINE_FUNCTION: "asin",
    INVERSE_COSINE_FUNCTION: "acos",
    INVERSE_TANGENT_FUNCTION: "atan",
    INVERSE_HYPERBOLIC_SINE_FUNCTION: "asinh",
    INVERSE_HYPERBOLIC_COSINE_FUNCTION: "acosh",
    INVERSE_HYPERBOLIC_TANGENT_FUNCTION: "atanh",
    INVERSE_HYPERBOLIC_TANGENT_FUNCTION_2: "atanh2",

    ROUND_FUNCTION: "round",
    PRECISION_FUNCTION: "precision",
    TRUNCATION_FUNCTION: "trunc",
    CEILING_FUNCTION: "ceil",
    FLOOR_FUNCTION: "floor",

    BETA_RNG_FUNCTION: "Beta",
    CAUCHY_RNG_FUNCTION: "Cauchy",
    EXPONENTIAL_RNG_FUNCTION: "Exponential",
    GAMMA_RNG_FUNCTION: "Gamma",
    Irand224_RNG_FUNCTION: "Irand224",
    NORMAL_RNG_FUNCTION: "Normal",
    STANDARD_NORMAL_RNG_FUNCTION: "Normal01",
    POISSON_RNG_FUNCTION: "Poisson",
    UNIFORM_RNG_FUNCTION: "Uniform",
    STANDARD_UNIFORM_RNG_FUNCTION: "Uniform01",

    TIME_FUNCTION: "time",
    CTIME_FUNCTION: "ctime",
    ALIAS_FUNCTION: "alias",

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
            tfm_element[dim_pos] = agg_val
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
