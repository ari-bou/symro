from typing import Callable, Dict, List, Optional, Tuple, Union

import symro.mat as mat


# TODO: remove this module

# Fundamental Blocks
# ------------------------------------------------------------------------------------------------------------------


def generate_entity_instance(
    meta_entity: mat.MetaEntity,
    can_write_index: bool,
    name_modifier: Callable = None,
    symbol_surrogates: List[Tuple[mat.MetaSet, str]] = None,
    suffix: str = None,
) -> str:

    if symbol_surrogates is None:
        symbol_surrogates = []

    name = (
        meta_entity.symbol
        if name_modifier is None
        else name_modifier(meta_entity.symbol)
    )
    instance = name

    if can_write_index and len(meta_entity.idx_meta_sets) > 0:

        index_symbols = list(meta_entity.idx_set_reduced_dummy_element)

        for meta_set, symbol in symbol_surrogates:
            if meta_entity.is_indexed_with(meta_set):
                index_symbols[
                    meta_entity.get_first_reduced_dim_index_of_idx_set(meta_set)
                ] = symbol

        instance += generate_entity_index(index_symbols)

    if suffix is not None:
        if suffix[0] != ".":
            suffix = "." + suffix
        instance += suffix

    return instance


def generate_entity_index(
    index_symbols: Optional[List[Union[str, mat.MetaEntity]]]
) -> str:

    if index_symbols is None:
        return ""
    if len(index_symbols) == 0:
        return ""

    def get_symbol(sym) -> str:
        if isinstance(sym, str):
            return sym
        elif isinstance(sym, mat.MetaSet):
            return sym.reduced_dummy_element[0]
        elif isinstance(sym, mat.MetaParameter):
            return sym.symbol
        else:
            return str(sym)

    index_symbols = [get_symbol(symbol) for symbol in index_symbols]

    return "[" + ",".join(index_symbols) + "]"


def generate_entity_indexing_set_definition(
    meta_entity: mat.MetaEntity, remove_sets: List[Union[str, mat.MetaSet]] = None
) -> str:
    def get_set_name(s) -> str:
        if isinstance(s, str):
            return s
        elif isinstance(s, mat.MetaSet):
            return s.symbol

    indexing_meta_sets = [ms for ms in meta_entity.idx_meta_sets]

    # Remove controlled sets
    if remove_sets is not None:
        remove_sets = [get_set_name(s) for s in remove_sets]
        indexing_meta_sets = [
            ms for ms in indexing_meta_sets if ms.symbol not in remove_sets
        ]

    return generate_indexing_set_definition(
        indexing_meta_sets, meta_entity.idx_set_con_literal
    )


def generate_indexing_set_definition(
    idx_meta_sets: Optional[Union[List[mat.MetaSet], Dict[str, mat.MetaSet]]],
    idx_set_con: str = None,
):

    if idx_meta_sets is None:
        return ""
    if isinstance(idx_meta_sets, dict):
        idx_meta_sets = [ms for _, ms in idx_meta_sets.items()]

    indexing_set_declarations = [
        meta_set.generate_idx_set_literal() for meta_set in idx_meta_sets
    ]

    # Generate definition
    definition = ""
    if len(indexing_set_declarations) > 0:
        definition = "{" + ", ".join(indexing_set_declarations)
        if idx_set_con is not None and idx_set_con != "":
            definition += ": " + idx_set_con
        definition += "}"

    return definition


# Assignment
# ------------------------------------------------------------------------------------------------------------------


def generate_apply_statement(
    command: str,
    entity: mat.MetaEntity,
    index_symbols: Optional[List[Union[str, mat.MetaEntity]]] = None,
    indexing_sets: List[mat.MetaSet] = None,
    indexing_set_constraint: str = None,
) -> str:

    indexing_set_definition = generate_indexing_set_definition(
        indexing_sets, indexing_set_constraint
    )
    if indexing_set_definition != "":
        indexing_set_definition += " "

    index = generate_entity_index(index_symbols)

    return "{0} {1}{2}{3};".format(
        command, indexing_set_definition, entity.symbol, index
    )


def generate_assignment_statement(
    command: str,
    entity: Union[str, mat.MetaEntity],
    source_entity: Union[str, mat.MetaEntity],
    index_symbols: Optional[List[Union[str, mat.MetaEntity]]] = None,
    source_index_symbols: Optional[List[Union[str, mat.MetaEntity]]] = None,
    source_suffix: str = None,
    indexing_sets: Union[List[mat.MetaSet], Dict[str, mat.MetaSet], None] = None,
    indexing_set_constraint: str = None,
) -> str:

    indexing_set_definition = generate_indexing_set_definition(
        indexing_sets, indexing_set_constraint
    )
    if indexing_set_definition != "":
        indexing_set_definition += " "

    index = generate_entity_index(index_symbols)
    source_index = generate_entity_index(source_index_symbols)

    if isinstance(entity, mat.MetaEntity):
        entity = entity.symbol
    if isinstance(source_entity, mat.MetaEntity):
        source_entity = source_entity.symbol

    if source_suffix is None:
        source_suffix = ""
    if len(source_suffix) > 0:
        if source_suffix[0] != ".":
            source_suffix = "." + source_suffix

    return "{0} {1}{2}{3} := {4}{5}{6};".format(
        command,
        indexing_set_definition,
        entity,
        index,
        source_entity,
        source_index,
        source_suffix,
    )


# Problem Initialization
# ------------------------------------------------------------------------------------------------------------------


def __generate_initial_value_assignment_statement(
    meta_variable: mat.MetaEntity,
    controlled_sets: List[mat.MetaSet],
    indexing_param_symbols: Dict[str, str],
    can_write_indexing_sets: bool,
    can_write_index: bool,
    indent_count: int = 1,
) -> str:

    if can_write_indexing_sets:
        indexing_set_decl = generate_entity_indexing_set_definition(
            meta_variable, remove_sets=controlled_sets
        )
    else:
        indexing_set_decl = ""

    var_instance = generate_entity_instance(meta_variable, can_write_index)

    symbol_surrogates = []
    for meta_set in controlled_sets:
        if meta_set.symbol in indexing_param_symbols:
            symbol_surrogates.append(
                (meta_set, indexing_param_symbols[meta_set.symbol])
            )
    param_instance = generate_entity_instance(
        meta_variable,
        can_write_index=can_write_index,
        name_modifier=lambda sl: sl + "_INIT",
        symbol_surrogates=symbol_surrogates,
    )

    statement = "{0}let {1} {2} := {3};".format(
        "\t" * indent_count, indexing_set_decl, var_instance, param_instance
    )
    return statement


# Output
# ------------------------------------------------------------------------------------------------------------------


def generate_global_file_output_statement(entity_type: str, file_name: str) -> str:

    if entity_type == "var":
        length_symbol = "_nvars"
        name_symbol = "_varname"
        entity_symbol = "_var"
        suffixes = ["", "lb", "ub", "slack"]
    elif entity_type == "con":
        length_symbol = "_ncons"
        name_symbol = "_conname"
        entity_symbol = "_con"
        suffixes = ["body", "lb", "ub", "slack"]
    else:
        return ""

    idx_set_def = "{" + "i in 1..{0}".format(length_symbol) + "}"

    for i, sfx in enumerate(suffixes):
        if sfx != "":
            suffixes[i] = "." + sfx

    values = ["{0}[i]".format(name_symbol)]
    values.extend(["{0}[i]{1}".format(entity_symbol, sfx) for sfx in suffixes])
    value_list = ", ".join(values)

    return "display {0} ({1}) > {2};".format(idx_set_def, value_list, file_name)


def generate_file_output_statement(meta_entity: mat.MetaEntity, file_name: str) -> str:

    if isinstance(meta_entity, mat.MetaVariable):
        suffixes = ["", "lb", "ub", "slack"]
    elif isinstance(meta_entity, mat.MetaConstraint):
        suffixes = ["body", "lb", "ub", "slack"]
    elif isinstance(meta_entity, mat.MetaParameter):
        suffixes = [""]
    else:
        return ""

    idx_set_def = generate_entity_indexing_set_definition(meta_entity)
    entity_instance = generate_entity_instance(meta_entity, can_write_index=True)

    for i, sfx in enumerate(suffixes):
        if sfx != "":
            suffixes[i] = "." + sfx
    values = [entity_instance + sfx for sfx in suffixes]
    value_list = "({0})".format(", ".join(values))

    return "display {0} {1} > {2};".format(idx_set_def, value_list, file_name)
