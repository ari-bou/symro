import itertools
from typing import Iterable, List, Union

from symro.src.mat.orderedset import OrderedSet
from symro.src.mat.types import Element


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


def flatten_element(element: tuple) -> Element:

    flattened_element = []

    for sub_element in element:

        if isinstance(sub_element, Iterable):
            flattened_element.extend(sub_element)

        else:
            flattened_element.append(sub_element)

    return tuple(flattened_element)


# Set Operations
# ----------------------------------------------------------------------------------------------------------------------

def cartesian_product(sets: List[OrderedSet]) -> OrderedSet:
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


def aggregate_set(set_in: OrderedSet,
                  dim_positions: List[int],
                  agg_values: list):
    agg_set = OrderedSet()
    for element in set_in:
        tfm_element = list(element)
        for dim_pos, agg_val in zip(dim_positions, agg_values):
            tfm_element[dim_pos] = agg_val
        agg_set.add(tuple(tfm_element))
    return agg_set


def filter_set(set_in: OrderedSet,
               filter_element: Union[list, tuple]) -> OrderedSet:
    filtered_set = OrderedSet()
    for element in set_in:
        is_in_filter = [e == f or f is None for e, f in zip(element, filter_element)]
        if all(is_in_filter):
            filtered_set.add(element)
    return filtered_set


def remove_set_dimensions(set_in: OrderedSet, dim_positions: List[int]):
    dim_positions.sort(reverse=True)
    set_out = OrderedSet()
    for element in set_in:
        element_list = list(element)
        for dim_pos in dim_positions:
            element_list.pop(dim_pos)
        set_out.add(tuple(element_list))
    return set_out
