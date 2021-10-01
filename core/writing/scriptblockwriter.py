from typing import Callable, Dict, List, Optional, Tuple, Union

import symro.core.mat as mat


# Fundamental Blocks
# ------------------------------------------------------------------------------------------------------------------

def generate_entity_instance(meta_entity: mat.MetaEntity,
                             can_write_index: bool,
                             name_modifier: Callable = None,
                             symbol_surrogates: List[Tuple[mat.MetaSet, str]] = None,
                             suffix: str = None) -> str:

    if symbol_surrogates is None:
        symbol_surrogates = []

    name = meta_entity.get_symbol() if name_modifier is None else name_modifier(meta_entity.get_symbol())
    instance = name

    if can_write_index and len(meta_entity.get_idx_meta_sets()) > 0:

        index_symbols = list(meta_entity.get_idx_set_dummy_element())

        for meta_set, symbol in symbol_surrogates:
            if meta_entity.is_indexed_with(meta_set):
                index_symbols[meta_entity.get_first_reduced_dim_index_of_idx_set(meta_set)] = symbol

        instance += generate_entity_index(index_symbols)

    if suffix is not None:
        if suffix[0] != '.':
            suffix = '.' + suffix
        instance += suffix

    return instance


def generate_entity_index(index_symbols: Optional[List[Union[str, mat.MetaEntity]]]) -> str:

    if index_symbols is None:
        return ""
    if len(index_symbols) == 0:
        return ""

    def get_symbol(sym) -> str:
        if isinstance(sym, str):
            return sym
        elif isinstance(sym, mat.MetaSet):
            return sym.get_reduced_dummy_element()[0]
        elif isinstance(sym, mat.MetaParameter):
            return sym.get_symbol()
        else:
            return str(sym)

    index_symbols = [get_symbol(symbol) for symbol in index_symbols]

    return '[' + ','.join(index_symbols) + ']'


def generate_entity_indexing_set_definition(meta_entity: mat.MetaEntity,
                                            remove_sets: List[Union[str, mat.MetaSet]] = None) -> str:
    def get_set_name(s) -> str:
        if isinstance(s, str):
            return s
        elif isinstance(s, mat.MetaSet):
            return s.get_symbol()

    indexing_meta_sets = [ms for ms in meta_entity.get_idx_meta_sets()]

    # Remove controlled sets
    if remove_sets is not None:
        remove_sets = [get_set_name(s) for s in remove_sets]
        indexing_meta_sets = [ms for ms in indexing_meta_sets if ms.get_symbol() not in remove_sets]

    return generate_indexing_set_definition(indexing_meta_sets,
                                            meta_entity.get_idx_set_con_literal())


def generate_indexing_set_definition(idx_meta_sets: Optional[Union[List[mat.MetaSet], Dict[str, mat.MetaSet]]],
                                     idx_set_con: str = None):

    if idx_meta_sets is None:
        return ""
    if isinstance(idx_meta_sets, dict):
        idx_meta_sets = [ms for _, ms in idx_meta_sets.items()]

    indexing_set_declarations = [meta_set.generate_idx_set_literal() for meta_set in idx_meta_sets]

    # Generate definition
    definition = ""
    if len(indexing_set_declarations) > 0:
        definition = '{' + ", ".join(indexing_set_declarations)
        if idx_set_con is not None and idx_set_con != "":
            definition += ": " + idx_set_con
        definition += '}'

    return definition


# Assignment
# ------------------------------------------------------------------------------------------------------------------

def generate_apply_statement(command: str,
                             entity: mat.MetaEntity,
                             index_symbols: Optional[List[Union[str, mat.MetaEntity]]] = None,
                             indexing_sets: List[mat.MetaSet] = None,
                             indexing_set_constraint: str = None) -> str:

    indexing_set_definition = generate_indexing_set_definition(indexing_sets,
                                                               indexing_set_constraint)
    if indexing_set_definition != "":
        indexing_set_definition += " "

    index = generate_entity_index(index_symbols)

    return "{0} {1}{2}{3};".format(command,
                                   indexing_set_definition,
                                   entity.get_symbol(),
                                   index)


def generate_assignment_statement(command: str,
                                  entity: Union[str, mat.MetaEntity],
                                  source_entity: Union[str, mat.MetaEntity],
                                  index_symbols: Optional[List[Union[str, mat.MetaEntity]]] = None,
                                  source_index_symbols: Optional[List[Union[str, mat.MetaEntity]]] = None,
                                  source_suffix: str = None,
                                  indexing_sets: Union[List[mat.MetaSet], Dict[str, mat.MetaSet], None] = None,
                                  indexing_set_constraint: str = None) -> str:

    indexing_set_definition = generate_indexing_set_definition(indexing_sets,
                                                               indexing_set_constraint)
    if indexing_set_definition != "":
        indexing_set_definition += " "

    index = generate_entity_index(index_symbols)
    source_index = generate_entity_index(source_index_symbols)

    if isinstance(entity, mat.MetaEntity):
        entity = entity.get_symbol()
    if isinstance(source_entity, mat.MetaEntity):
        source_entity = source_entity.get_symbol()

    if source_suffix is None:
        source_suffix = ""
    if len(source_suffix) > 0:
        if source_suffix[0] != '.':
            source_suffix = '.' + source_suffix

    return "{0} {1}{2}{3} := {4}{5}{6};".format(command,
                                                indexing_set_definition,
                                                entity,
                                                index,
                                                source_entity,
                                                source_index,
                                                source_suffix)


# Problem Initialization
# ------------------------------------------------------------------------------------------------------------------

def generate_initialization_subroutine(meta_vars: Dict[str, mat.MetaVariable],
                                       omitted_vars: Union[Dict[str, Union[str, mat.MetaVariable]],
                                                           List[Union[str, mat.MetaVariable]]] = None,
                                       is_looped: bool = True,
                                       controlled_sets: List[mat.MetaSet] = None,
                                       dummy_syms: Union[List, Dict] = None,
                                       default_values: Union[List, Dict] = None,
                                       can_declare_indexing_param: Union[List[bool], Dict[str, bool]] = None,
                                       indent_count: int = 0) -> List[str]:

    def get_var_sym(v) -> str:
        if isinstance(v, mat.MetaVariable):
            return v.get_symbol()
        else:
            return v

    if omitted_vars is None:
        omitted_vars = []
    if isinstance(omitted_vars, dict):
        omitted_vars = [v for k, v in omitted_vars.items()]
    omitted_vars = [get_var_sym(v) for v in omitted_vars]

    if controlled_sets is None:
        controlled_sets = []

    if dummy_syms is None:
        dummy_syms = []
    if isinstance(dummy_syms, list):
        idx_par_sym_dict = {}
        for i, meta_set in enumerate(controlled_sets):
            try:
                sym = dummy_syms[i]
                idx_par_sym_dict[meta_set.get_symbol()] = sym
            except IndexError:
                pass
        dummy_syms = idx_par_sym_dict

    if default_values is None:
        default_values = []
    if isinstance(default_values, list):
        dft_val_dict = {}
        for i, meta_set in enumerate(controlled_sets):
            try:
                dft_val = default_values[i]
                dft_val_dict[meta_set.get_symbol()] = dft_val
            except IndexError:
                pass
        default_values = dft_val_dict

    if can_declare_indexing_param is None:
        can_declare_indexing_param = []
    if isinstance(can_declare_indexing_param, list):
        can_decl_idx_par_dict = {}
        for i, meta_set in enumerate(controlled_sets):
            try:
                can_decl = can_declare_indexing_param[i]
                can_decl_idx_par_dict[meta_set.get_symbol()] = can_decl
            except IndexError:
                pass
        can_declare_indexing_param = can_decl_idx_par_dict

    assignment_indent_count = indent_count
    lines = []

    for meta_set in controlled_sets:
        if can_declare_indexing_param.get(meta_set.get_symbol(), False):
            sym = dummy_syms.get(meta_set.get_symbol(), meta_set.get_reduced_dummy_element()[0])
            def_val = default_values.get(meta_set.get_symbol(), None)
            indexing_param = mat.MetaParameter(symbol=sym,
                                               default_value=def_val)
            lines.append("{0}{1}".format("\t" * indent_count,
                                         indexing_param.generate_declaration()))

    if is_looped and len(controlled_sets) > 0:
        idx_set_def = generate_indexing_set_definition(controlled_sets)
        line = ("\t" * indent_count) + "for " + idx_set_def + " {"
        lines.append(line)
        assignment_indent_count += 1

    for name, meta_var in meta_vars.items():
        if name not in omitted_vars:
            line = __generate_initial_value_assignment_statement(meta_var,
                                                                 controlled_sets=controlled_sets,
                                                                 indexing_param_symbols=dummy_syms,
                                                                 can_write_indexing_sets=True,
                                                                 can_write_index=True,
                                                                 indent_count=assignment_indent_count)
            lines.append(line)

    if is_looped and len(controlled_sets) > 0:
        lines.append(("\t" * indent_count) + '}')

    return lines


def __generate_initial_value_assignment_statement(meta_variable: mat.MetaEntity,
                                                  controlled_sets: List[mat.MetaSet],
                                                  indexing_param_symbols: Dict[str, str],
                                                  can_write_indexing_sets: bool,
                                                  can_write_index: bool,
                                                  indent_count: int = 1) -> str:

    if can_write_indexing_sets:
        indexing_set_decl = generate_entity_indexing_set_definition(meta_variable,
                                                                    remove_sets=controlled_sets)
    else:
        indexing_set_decl = ""

    var_instance = generate_entity_instance(meta_variable, can_write_index)

    symbol_surrogates = []
    for meta_set in controlled_sets:
        if meta_set.get_symbol() in indexing_param_symbols:
            symbol_surrogates.append((meta_set, indexing_param_symbols[meta_set.get_symbol()]))
    param_instance = generate_entity_instance(meta_variable,
                                              can_write_index=can_write_index,
                                              name_modifier=lambda sl: sl + "_INIT",
                                              symbol_surrogates=symbol_surrogates)

    statement = "{0}let {1} {2} := {3};".format("\t" * indent_count,
                                                indexing_set_decl,
                                                var_instance,
                                                param_instance)
    return statement


# Output
# ------------------------------------------------------------------------------------------------------------------

def generate_global_file_output_statement(entity_type: str,
                                          file_name: str) -> str:

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

    idx_set_def = '{' + "i in 1..{0}".format(length_symbol) + '}'

    for i, sfx in enumerate(suffixes):
        if sfx != "":
            suffixes[i] = '.' + sfx

    values = ["{0}[i]".format(name_symbol)]
    values.extend(["{0}[i]{1}".format(entity_symbol, sfx) for sfx in suffixes])
    value_list = ', '.join(values)

    return "display {0} ({1}) > {2};".format(idx_set_def, value_list, file_name)


def generate_file_output_statement(meta_entity: mat.MetaEntity,
                                   file_name: str) -> str:

    if isinstance(meta_entity, mat.MetaVariable):
        suffixes = ["", "lb", "ub", "slack"]
    elif isinstance(meta_entity, mat.MetaConstraint):
        suffixes = ["body", "lb", "ub", "slack"]
    elif isinstance(meta_entity, mat.MetaParameter):
        suffixes = [""]
    else:
        return ""

    idx_set_def = generate_entity_indexing_set_definition(meta_entity)
    entity_instance = generate_entity_instance(meta_entity,
                                               can_write_index=True)

    for i, sfx in enumerate(suffixes):
        if sfx != "":
            suffixes[i] = '.' + sfx
    values = [entity_instance + sfx for sfx in suffixes]
    value_list = "({0})".format(', '.join(values))

    return "display {0} {1} > {2};".format(idx_set_def,
                                           value_list,
                                           file_name)
