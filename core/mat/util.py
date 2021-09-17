from ordered_set import OrderedSet
from typing import List, Optional, Tuple, Union

Element = Tuple[Union[int, float, str, None], ...]
IndexingSet = OrderedSet[Tuple[Union[int, float, str, None], ...]]


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
    Combine individual sets together. Each element of the combined set is a unique combination of 1 element from each
    of the constituent sets.
    :param sets: list of constituent ordered sets
    :return: ordered set of elements comprising the combined set
    """

    combined_sets: IndexingSet = OrderedSet()
    set_count = len(sets)

    def combine(index: List[Union[int, float, str]] = None, set_index: int = 0):
        if index is None:
            index = []
        set_i = sets[set_index]
        if len(set_i) > 0:
            for element in set_i:
                index_copy: List[Optional[Union[int, float, str]]] = list(index)
                index_copy.extend(list(element))
                if set_index == set_count - 1:
                    combined_sets.add(tuple(index_copy))
                else:
                    combine(index_copy, set_index + 1)
        else:
            index_copy: List[Optional[Union[int, float, str]]] = list(index)
            index_copy.append(None)
            if set_index == set_count - 1:
                combined_sets.add(tuple(index_copy))
            else:
                combine(index_copy, set_index + 1)

    if set_count > 0:
        combine()

    return combined_sets


def flatten_set(set_in: OrderedSet[Tuple[Optional[Tuple[Union[int, float, str], ...]], ...]]) -> IndexingSet:
    flattened_set = OrderedSet()
    for element in set_in:
        flattened_element = []
        for sub_element in element:
            if sub_element is not None:
                flattened_element.extend(list(sub_element))
            else:
                flattened_element.append(None)
        flattened_set.add(tuple(flattened_element))
    return flattened_set


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
