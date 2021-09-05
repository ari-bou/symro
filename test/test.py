import symro
from symro.test.test_util import *
from symro.test.test_node_builder import run_node_builder_test_group
from symro.test.test_entity_builder import run_entity_builder_test_group
from symro.test.test_gbd import run_gbd_test_group


# General Tests
# ----------------------------------------------------------------------------------------------------------------------

def run_general_test_group():
    tests = [("Build problem from file", build_problem_from_file_test),
             ("Build problem from literal", build_problem_from_literal_test)]
    return run_tests(tests)


def build_problem_from_file_test() -> bool:
    # noinspection PyBroadException
    try:
        symro.build_problem("diet.run",
                            working_dir_path=SCRIPT_DIR_PATH)
        return True
    except Exception as e:
        print(e)
        return False


def build_problem_from_literal_test():
    # noinspection PyBroadException
    try:
        symro.build_problem(script_literal="set I;")
        return True
    except Exception as e:
        print(e)
        return False


# Run
# ----------------------------------------------------------------------------------------------------------------------

def run_all_tests():
    true_count, total_count = run_test_groups([run_general_test_group,
                                               run_node_builder_test_group,
                                               run_entity_builder_test_group,
                                               run_gbd_test_group])
    print("\n{0}/{1} tests were conducted successfully.".format(true_count, total_count))


# Execution
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    run_all_tests()
