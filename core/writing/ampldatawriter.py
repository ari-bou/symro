import numpy as np
from typing import Dict, List, Tuple, Union

from symro.core.prob.problem import Problem
import symro.core.util.util as util


def write_initialization_data_file(problem: Problem,
                                   data_init_dir_path: str,
                                   data_init_file_name: str):

    declarations = ""

    for var_sym, var_collection in problem.state.var_collections.items():

        meta_var = problem.meta_vars[var_sym]
        var_map = var_collection.entity_map

        param_sym = "{0}_INIT".format(var_sym)
        param_dim = meta_var.get_reduced_dimension()
        values = {k: v.value for k, v in var_map.items()}

        declaration = generate_param_data_statement(param_sym, param_dim, values)
        declarations += declaration + "\n\n"

    util.write_file(data_init_dir_path, data_init_file_name, declarations)


def generate_param_data_statement(param_sym: str,
                                  dim: int,
                                  values: Dict[Tuple[Union[int, float, str], ...], Union[float, str]]) -> str:
    declaration = "param {0} := \n".format(param_sym)
    if dim == 0:
        data = __generate_scalar_param_data_block(values)
    elif dim == 1:
        data = __generate_1d_param_data_block(values)
    elif dim == 2:
        data = __generate_2d_param_data_block(values, 0, 1)
    else:
        data = __generate_nd_param_data_block(values, dim - 2, dim - 1)
    declaration += data + ";"
    return declaration


def __generate_scalar_param_data_block(values: Dict[Tuple[Union[int, float, str], ...], Union[float, str]]) -> str:
    value = values[list(values.keys())[0]]
    data = str(value)
    return data


def __generate_1d_param_data_block(values: Dict[Tuple[Union[int, float, str], ...], Union[float, str]]) -> str:
    data = ""
    for index, value in values.items():
        data += "{0}\t{1}\n".format(index[0], __process_value(value))
    return data


def __generate_2d_param_data_block(values: Dict[Tuple[Union[int, float, str], ...], Union[float, str]],
                                   row_dim_pos: int,
                                   col_dim_pos: int) -> str:
    data = ""
    for multi_index, value in values.items():
        index_1 = multi_index[row_dim_pos]
        index_2 = multi_index[col_dim_pos]
        data += "\t{0} {1} {2}".format(index_1, index_2, __process_value(value))
    return data


def __generate_nd_param_data_block(values: Dict[Tuple[Union[int, float, str], ...], Union[float, str]],
                                   row_dim_pos: int,
                                   col_dim_pos: int) -> str:

    value_maps: Dict[tuple, Dict[tuple, float]] = {}
    floating_index_positions = [row_dim_pos, col_dim_pos]
    data = ""

    for multi_index, value in values.items():
        fixed_multi_index = __replace_floating_indices_with_placeholders(multi_index,
                                                                         floating_index_positions)
        if fixed_multi_index in value_maps:
            value_maps[fixed_multi_index][multi_index] = value
        else:
            value_maps[fixed_multi_index] = {multi_index: value}

    for raw_fixed_multi_index, values in value_maps.items():
        data_2d = __generate_2d_param_data_block(values=values,
                                                 row_dim_pos=row_dim_pos,
                                                 col_dim_pos=col_dim_pos)
        fixed_multi_index = []
        for index in raw_fixed_multi_index:
            index_str = str(index)
            if isinstance(index, int) or isinstance(index, float):
                index_str = str(index)
            elif index != '*':
                index_str = "'{0}'".format(index)
            fixed_multi_index.append(index_str)

        data += "[{0}]\t{1}\n".format(",".join(fixed_multi_index), data_2d)

    return data


def __replace_floating_indices_with_placeholders(multi_index: tuple,
                                                 floating_index_positions: List[int]) -> tuple:
    multi_index = list(multi_index)
    for i in range(len(multi_index)):
        if i in floating_index_positions:
            multi_index[i] = '*'
    return tuple(multi_index)


def __process_value(value: Union[int, float, str]):
    if isinstance(value, str):
        if value in ["inf", "-inf"]:
            return 0
    else:
        if value == np.inf or value == -np.inf:
            return 0
    return value
