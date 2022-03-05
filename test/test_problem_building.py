import warnings

import symro
from .test_util import *


# Tests
# ----------------------------------------------------------------------------------------------------------------------


def test_build_problem_from_file():
    try:
        symro.read_ampl("diet.run", working_dir_path=SCRIPT_DIR_PATH)
        assert True
    except Exception as e:
        warnings.warn(str(e))
        assert False


def test_build_problem_from_literal():
    try:
        symro.read_ampl(script_literal="set I;")
        assert True
    except Exception as e:
        warnings.warn(str(e))
        assert False
