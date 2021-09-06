import pathlib
import setuptools

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# This call to setup() does all the work
setuptools.setup(
    name="symro",
    version="0.0.1",
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
    package_dir={"": "core"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.8",
    include_package_data=True,
    install_requires=["amplpy>=0.7.1", "numpy>=1.21.2", "ordered-set>=4.0.2"]
)
