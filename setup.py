import os
import pathlib
import setuptools

# The directory containing this file
ROOT_DIR = pathlib.Path(__file__).parent

# The text of the README file
README = (ROOT_DIR / "README.md").read_text()

version_file = open(os.path.join(ROOT_DIR, "symro", "symro", "VERSION"))
version = version_file.read().strip()
__version__ = version

with open(os.path.join(ROOT_DIR, "requirements.txt")) as f:
    requirements = [line.strip() for line in f]

with open(os.path.join(ROOT_DIR, "test", "test-requirements.txt")) as f:
    test_requirements = [line.strip() for line in f]

# This call to setup() does all the work
setuptools.setup(
    name="symro",
    version=version,
    description="SYMbolic Reformulation and Optimization (SYMRO) package",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/ari-bou/symro",
    author="Ariel A. Boucheikhchoukh",
    author_email="ariel.boucheikh@gmail.com",
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Intended Audience :: Science/Research",
        "Development Status :: 2 - Pre-Alpha",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.8",
    include_package_data=True,
    install_requires=requirements,
    extras_require={"test": test_requirements},
)
