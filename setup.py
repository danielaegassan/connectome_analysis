# -*- coding: utf-8 -*-
from setuptools import setup

package_dir = \
{'': 'src'}

packages = \
['connalysis',
 'connalysis.modelling',
 'connalysis.network',
 'connalysis.randomization']

package_data = \
{'': ['*']}

install_requires = \
['matplotlib',
 'numpy',
 'pandas',
 'progressbar==2.5',
 'pyflagser',
 'pyflagsercount>=0.2.41',
 'scipy',
 'tables',
 'tqdm']

setup_kwargs = {
    'name': 'connectome-analysis',
    'version': '0.0.1',
    'description': 'Functions for network analysis of graphs coming brain models and activity on them',
    'long_description': '# connectome_analysis\n\n[![PyPI](https://img.shields.io/pypi/v/connectome_analysis?style=flat-square)](https://pypi.python.org/pypi/connectome_analysis/)\n[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/connectome_analysis?style=flat-square)](https://pypi.python.org/pypi/connectome_analysis/)\n[![PyPI - License](https://img.shields.io/pypi/l/connectome_analysis?style=flat-square)](https://pypi.python.org/pypi/connectome_analysis/)\n\n\n---\n\n**Documentation**: [https://danielaegassan.github.io/connectome_analysis](https://danielaegassan.github.io/connectome_analysis)\n\n**Source Code**: [https://github.com/danielaegassan/connectome_analysis](https://github.com/danielaegassan/connectome_analysis)\n\n**PyPI**: [https://pypi.org/project/connectome_analysis/](https://pypi.org/project/connectome_analysis/)\n\n---\n\nLibrary of general functions to analyze connectoms.\n\n## Installation\n\n```sh\npip install connectome_analysis\n```\n\n## Development\n\n* Clone this repository\n* Requirements:\n  * [Poetry](https://python-poetry.org/)\n  * [gcc](https://gcc.gnu.org/) 9\n  * [CMake](https://cmake.org/)\n  * Python 3.8+\n\n* Create a virtual environment and install the dependencies\n\n```sh\npoetry install\n```\n\nCMake may have difficulties to find the right compilers to compile the C++ code. \nIf that is the case, you have to specify the path to the compilers yourself:\n\n```sh\nCC=/path/to/gcc CXX=/path/to/g++ poetry install\n```\n\nThis is especially important on MacOS.\n\n* Activate the virtual environment\n\n```sh\npoetry shell\n```\n\n### Testing\n\n```sh\npytest tests\n```\n\n### Documentation\n\n#### TODO: Make this happen\nThe documentation is automatically generated from the content of the [docs directory](./docs) and from the docstrings\n of the public signatures of the source code. The documentation is updated and published as a [Github project page\n ](https://pages.github.com/) automatically as part each release.\n',
    'author': 'Daniela Egas Santander',
    'author_email': 'daniela.egassantander@epfl.ch',
    'maintainer': 'None',
    'maintainer_email': 'None',
    'url': 'https://danielaegassan.github.io/connectome_analysis',
    'package_dir': package_dir,
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'python_requires': '>=3.8.3,<4',
}
from build import *
build(setup_kwargs)

setup(**setup_kwargs)
