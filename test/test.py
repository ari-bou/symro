import symro
from symro.test.test_util import *
from symro.test.test_parsing import run_ampl_parser_test_group
from symro.test.test_node_builder import run_node_builder_test_group
from symro.test.test_formulator import run_formulator_test_group
from symro.test.test_entity_builder import run_entity_builder_test_group
from symro.test.test_convexifier import run_convexifier_test_group
from symro.test.test_gbd import run_gbd_test_group


# General Tests
# ----------------------------------------------------------------------------------------------------------------------

def run_general_test_group():
    tests = [("Build problem from file", build_problem_from_file_test),
             ("Build problem from literal", build_problem_from_literal_test)]
    return run_tests(tests)


def build_problem_from_file_test() -> bool:
    try:
        symro.read_ampl("diet.run",
                        working_dir_path=SCRIPT_DIR_PATH)
        return True
    except Exception as e:
        warnings.warn(str(e))
        return False


def build_problem_from_literal_test():
    try:
        symro.read_ampl(script_literal="set I;")
        return True
    except Exception as e:
        warnings.warn(str(e))
        return False


# Run
# ----------------------------------------------------------------------------------------------------------------------

def run_all_tests():
    true_count, total_count = run_test_groups([run_ampl_parser_test_group,
                                               run_general_test_group,
                                               run_node_builder_test_group,
                                               run_formulator_test_group,
                                               run_entity_builder_test_group,
                                               run_convexifier_test_group,
                                               run_gbd_test_group])

    print("\nSummary")
    print('-' * 50)
    print("{0}/{1} tests were conducted successfully.".format(true_count, total_count))


# Execution
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    run_convexifier_test_group()
    #run_all_tests()
