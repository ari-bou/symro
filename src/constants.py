# Paths
# ----------------------------------------------------------------------------------------------------------------------
WORKING_REL_DIR_PATH = r"working"
REF_SOLUTION_REL_DIR_PATH = r"working/ref"
INIT_SOLUTION_REL_DIR_PATH = r"working/init_solution"
REPORT_TEMPLATE_REL_DIR_PATH = r"report"

# File Names
# ----------------------------------------------------------------------------------------------------------------------
DEFAULT_SOLUTION_FILE_NAME = r"soln"
DEFAULT_SUB_SOLUTION_FILE_NAME = r"sub_soln"
DEFAULT_INITIAL_SOLUTION_FILE_NAME = r"init"
DEFAULT_RUN_FILE_NAME = "problem.run"
DEFAULT_MODEL_FILE_NAME = "problem.mod"
DEFAULT_MODEL_INIT_FILE_NAME = "problem.modi"
DEFAULT_DATA_INIT_FILE_NAME = "problem_init.dati"
DEFAULT_REPORT_FILE_NAME = "report"
DEFAULT_CUSTOM_COMMANDS_FILE_NAME = "custom_commands.json"
DEFAULT_META_REPORTS_FILE_NAME = "meta_reports.json"
DEFAULT_CONFIG_FILE_NAME = "config.json"

# Command Line Interface Symbols
# ----------------------------------------------------------------------------------------------------------------------
NO_REFERENCE_PROBLEM_OPTION = "nr"
NO_EXECUTE_OPTION = "ne"
NO_SAVE_OPTION = "ns"

# Custom Command Parameter Keys
# ----------------------------------------------------------------------------------------------------------------------
COMMAND_TYPE_KEY = "command_type"
POSITIONAL_ARGUMENTS_KEY = "positional_arguments"
NAMED_ARGUMENTS_KEY = "named_arguments"

# Meta-Report Parameter Keys
# ----------------------------------------------------------------------------------------------------------------------
REPORT_REL_DIR_PATH_KEY = "report_rel_dir_path"
REPORT_FILE_NAME_KEY = "report_file_name"
CASE_LIST_KEY = "cases"
PROBLEM_NAME_KEY = "name"
PROBLEM_DESCRIPTION_KEY = "description"
DEFINITION_DIR_PATH_KEY = "definition_dir_path"
TEMPLATE_FILE_NAME_KEY = "template_file_name"
RUN_FILE_NAME_KEY = "run_file_name"
SOLUTION_FILE_NAME_KEY = "solution_file_name"
CAN_GENERATE_SCRIPT_FLAG_KEY = "can_generate_script"
CAN_EXECUTE_SCRIPT_FLAG_KEY = "can_execute_script"

# Script Flags
# ----------------------------------------------------------------------------------------------------------------------
SPECIAL_COMMAND_MODEL = "MODEL"
SPECIAL_COMMAND_ADDITIONAL_MODEL = "ADDITIONAL_MODELS"
SPECIAL_COMMAND_INIT_DATA = "INIT_DATA"
SPECIAL_COMMAND_SETUP = "SETUP"
SPECIAL_COMMAND_NOEVAL = "NOEVAL"
SPECIAL_COMMAND_EVAL = "EVAL"
SPECIAL_COMMAND_OMIT_DECLARATIONS = "OMIT_DECL"
SPECIAL_COMMAND_INCLUDE_DECLARATIONS = "INCL_DECL"
SPECIAL_COMMAND_FSPROBLEM = "FSPROBLEM"
SPECIAL_COMMAND_SUBPROBLEM = "SUBPROBLEM"
SPECIAL_COMMAND_MASTER_PROBLEM = "MASTER_PROBLEM"
SPECIAL_COMMAND_PRIMAL_SUBPROBLEM = "PRIMAL_SUBPROBLEM"
SPECIAL_COMMAND_FEASIBILITY_SUBPROBLEM = "FEASIBILITY_SUBPROBLEM"
SPECIAL_COMMAND_INITIALIZATION = "INITIALIZATION"
SPECIAL_COMMAND_GBD_ALGORITHM = "GBD_ALGORITHM"
SPECIAL_COMMAND_OUTPUT = "OUTPUT"

SPECIAL_COMMAND_SYMBOLS = [SPECIAL_COMMAND_MODEL,
                           SPECIAL_COMMAND_ADDITIONAL_MODEL,
                           SPECIAL_COMMAND_INIT_DATA,
                           SPECIAL_COMMAND_SETUP,
                           SPECIAL_COMMAND_NOEVAL,
                           SPECIAL_COMMAND_EVAL,
                           SPECIAL_COMMAND_OMIT_DECLARATIONS,
                           SPECIAL_COMMAND_INCLUDE_DECLARATIONS,
                           SPECIAL_COMMAND_FSPROBLEM,
                           SPECIAL_COMMAND_SUBPROBLEM,
                           SPECIAL_COMMAND_MASTER_PROBLEM,
                           SPECIAL_COMMAND_PRIMAL_SUBPROBLEM,
                           SPECIAL_COMMAND_FEASIBILITY_SUBPROBLEM,
                           SPECIAL_COMMAND_INITIALIZATION,
                           SPECIAL_COMMAND_GBD_ALGORITHM,
                           SPECIAL_COMMAND_OUTPUT]

# Raw Data File Attributes
# ----------------------------------------------------------------------------------------------------------------------
DF_NAME_COL = "_varname[j]"
DF_VAL_COL = "_var[j]"
DF_LB_COL = "_var[j].lb"
DF_UB_COL = "_var[j].ub"

# Entity Types
# ----------------------------------------------------------------------------------------------------------------------
SET_TYPE = "set"
PARAM_TYPE = "param"
VAR_TYPE = "var"
OBJ_TYPE = "obj"
CON_TYPE = "con"
TABLE_TYPE = "table"
PROB_TYPE = "prob"
ENV_TYPE = "env"

# Term Types
# ----------------------------------------------------------------------------------------------------------------------
CONSTANT = 0
LINEAR = 1
BILINEAR = 2
TRILINEAR = 3
FRACTIONAL = 4
FRACTIONAL_BILINEAR = 5
FRACTIONAL_TRILINEAR = 6
UNIVARIATE_CONCAVE = 7
GENERAL_NONCONVEX = 8

# Data Sheet
# ----------------------------------------------------------------------------------------------------------------------

DATASHEET_ID_COL_INDEX = 1
DATASHEET_ALIAS_COL_INDEX = 2
DATASHEET_AMPL_SYM_COL_INDEX = 3
DATASHEET_VAL_COL_INDEX = 6
DATASHEET_MIN_VAL_COL_INDEX = 7
DATASHEET_MAX_VAL_COL_INDEX = 8
DATASHEET_CON_SYM_COL_INDEX = 9
DATASHEET_CON_IDX_SET_DEF_COL_INDEX = 10

DATASHEET_ID_COL_HEADER = "Id"
DATASHEET_MODEL_SYM_COL_HEADER = "Model Symbol"
DATASHEET_COL_HEADERS = [DATASHEET_ID_COL_HEADER,
                         DATASHEET_MODEL_SYM_COL_HEADER,
                         "AMPL Symbol",
                         "Description",
                         "Unit",
                         "Value",
                         "Min",
                         "Max",
                         "Constraint Symbol",
                         "Indexing Set Definition"]

DATASHEET_FUNCTION_SECTION_HEADER = "FUNCTIONS"
DATASHEET_OUTPUT_SECTION_HEADER = "OUTPUTS"
DATASHEET_INPUT_SECTION_HEADER = "INPUTS"
DATASHEET_RAW_INPUT_SECTION_HEADER = "RAW INPUTS"
DATASHEET_TFM_INPUT_SECTION_HEADER = "TFM INPUTS"
DATASHEET_AUXILIARY_SECTION_HEADER = "AUXILIARY"
DATASHEET_CONSTANT_SECTION_HEADER = "CONSTANTS"
DATASHEET_SECTION_HEADERS = [DATASHEET_FUNCTION_SECTION_HEADER,
                             DATASHEET_OUTPUT_SECTION_HEADER,
                             DATASHEET_INPUT_SECTION_HEADER,
                             DATASHEET_RAW_INPUT_SECTION_HEADER,
                             DATASHEET_TFM_INPUT_SECTION_HEADER,
                             DATASHEET_AUXILIARY_SECTION_HEADER,
                             DATASHEET_CONSTANT_SECTION_HEADER]
