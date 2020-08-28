import os
import platform
import subprocess
from pathlib import Path
from setuptools import Extension
from setuptools.command.build_ext import build_ext

import torch

__all__ = [
    'get_ext_modules',
    'CMakeBuild',
]

_THIS_DIR = Path(__file__).parent.resolve()
_ROOT_DIR = _THIS_DIR.parent.parent.resolve()
_TORCHAUDIO_DIR = _ROOT_DIR / 'torchaudio'


def get_ext_modules():
    if platform.system() == 'Windows':
        return None
    return [_CMakeExtension('torchaudio._torchaudio')]


def _get_cxx11_abi():
    try:
        return int(torch._C._GLIBCXX_USE_CXX11_ABI)
    except ImportError:
        return 0


class _CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: "
                f'{", ".join(e.name for e in self.extensions)}')

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        config_cmd = [
            'cmake',
            '-GNinja',
            f"-DCMAKE_PREFIX_PATH={torch.utils.cmake_prefix_path}",
            f"_GLIBCXX_USE_CXX11_ABI={_get_cxx11_abi()}",
            "-DBUILD_PYTHON_EXTENSION:BOOL=ON",
            "-DBUILD_LIBTORCHAUDIO:BOOL=OFF",
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={_TORCHAUDIO_DIR}",
            ext.sourcedir
        ]

        build_cmd = [
            'cmake',
            '--build', '.',
            '--config', 'RELEASE',
            '--target', '_torchaudio'
        ]

        env = os.environ.copy()
        os.makedirs(self.build_temp, exist_ok=True)
        subprocess.check_call(config_cmd, cwd=self.build_temp, env=env)
        subprocess.check_call(build_cmd, cwd=self.build_temp)
