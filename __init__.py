import os
import pathlib

from symro.src import *


# The directory containing this file
ROOT_DIR = pathlib.Path(__file__).parent

version_file = open(os.path.join(ROOT_DIR, 'VERSION'))
version = version_file.read().strip()
__version__ = version
