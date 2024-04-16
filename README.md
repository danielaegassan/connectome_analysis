# connectome_analysis

This package provides a library of general functions to analyze connectomes from a topological persepective.  For documentation and examples see our documentation

[![PyPI](https://img.shields.io/pypi/v/connectome_analysis?style=flat-square)](https://pypi.python.org/pypi/connectome_analysis/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/connectome_analysis?style=flat-square)](https://pypi.python.org/pypi/connectome_analysis/)
[![PyPI - License](https://img.shields.io/pypi/l/connectome_analysis?style=flat-square)](https://pypi.python.org/pypi/connectome_analysis/)


---

**Documentation**: [https://danielaegassan.github.io/connectome_analysis](https://danielaegassan.github.io/connectome_analysis)

**Source Code**: [https://github.com/danielaegassan/connectome_analysis](https://github.com/danielaegassan/connectome_analysis)

**PyPI**: [https://pypi.org/project/connectome_analysis/](https://pypi.org/project/connectome_analysis/)

---

## User installation

```sh
pip install git+https://github.com/danielaegassan/connectome_analysis.git
```

## Development installation

* Clone this repository
* Requirements:
  * [Poetry](https://python-poetry.org/)
  * [gcc](https://gcc.gnu.org/) 9+
  * [CMake](https://cmake.org/)
  * Python 3.8+

* Create a virtual environment and install the dependencies

```sh
poetry install
```

CMake may have difficulties to find the right compilers to compile the C++ code. 
If that is the case, you have to specify the path to the compilers yourself:

```sh
CC=/path/to/gcc CXX=/path/to/g++ poetry install
```

This is especially important on MacOS.

* Activate the virtual environment

```sh
poetry shell
```

### Testing

```sh
poetry run pytest tests
```

### Documentation

The documentation is automatically generated from the content of the [docs directory](./docs) and from the docstrings
 of the public signatures of the source code. The documentation is updated and published as a [Github project page
 ](https://pages.github.com/) automatically as part each release.

 TODO: 
 [ ] Are we putting this on pypy? 
 [ ] What about the badges on top?
