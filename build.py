# SPDX-FileCopyrightText: 2024 Blue Brain Project / EPFL
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import os
import re
import sys
import platform
import subprocess

from typing import Any, Dict

from setuptools import Extension
from setuptools.command.build_ext import build_ext
from distutils.version import LooseVersion


# This is exactly what is contained in src/generate_model_cpp/setup.py
# In the future, we should probably avoid code duplication and import those classes
# If you do that, be careful to handle `dir_pybind11`
class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                               ", ".join(e.name for e in self.extensions))

        if platform.system() == "Windows":
            cmake_version = LooseVersion(re.search(r'version\s*([\d.]+)', out.decode()).group(1))
            if cmake_version < '3.1.0':
                raise RuntimeError("CMake >= 3.1.0 is required on Windows")

        self.install_dependencies()

        for ext in self.extensions:
            self.build_extension(ext)

    def install_dependencies(self):
        dir_start = os.getcwd()
        dir_pybind11 = os.path.join(dir_start, 'src/generate_model/pybind11')
        if os.path.exists(dir_pybind11):
            return 0
        os.mkdir(dir_pybind11)
        subprocess.check_call(['git', 'clone',
                               'https://github.com/pybind/pybind11.git',
                               dir_pybind11])
        #subprocess.check_call(['git', 'checkout', '0bd8896'])


    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        # required for auto-detection of auxiliary "native" libs
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                      '-DPYTHON_EXECUTABLE=' + sys.executable]

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        if platform.system() == "Windows":
            cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(cfg.upper(), extdir)]
            if sys.maxsize > 2**32:
                cmake_args += ['-A', 'x64']
            build_args += ['--', '/m']
        else:
            cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
            build_args += ['--', '-j2']

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(env.get('CXXFLAGS', ''),
                                                              self.distribution.get_version())
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)


def build(setup_kwargs: Dict[str, Any]) -> None:
    setup_kwargs.update(
        {
            "ext_modules": [CMakeExtension(f"generate_model", sourcedir="src/generate_model")],
            "cmdclass": dict(build_ext=CMakeBuild),
            "zip_safe": False,
        }
    )