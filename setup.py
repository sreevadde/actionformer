"""ActionFormer package setup with optional C++ extensions."""

import os
import sys
from setuptools import setup, find_packages

# Try to build C++ extension, but make it optional
ext_modules = []
cmdclass = {}

try:
    from torch.utils.cpp_extension import BuildExtension, CppExtension

    ext_modules = [
        CppExtension(
            name='nms_1d_cpu',
            sources=['actionformer/utils/csrc/nms_cpu.cpp'],
            extra_compile_args=['-fopenmp'] if sys.platform != 'darwin' else []
        )
    ]
    cmdclass = {'build_ext': BuildExtension}
except ImportError:
    print("Warning: torch not found, skipping C++ extension build")

setup(
    name='actionformer',
    version='1.5.0',
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    install_requires=[
        'torch>=2.0.0',
        'torchvision>=0.15.0',
        'numpy>=1.24.0',
        'pyyaml>=6.0',
        'tensorboard>=2.10.0',
        'tqdm>=4.65.0',
        'pandas>=1.5.0',
        'scipy>=1.10.0',
        'joblib>=1.2.0',
    ],
)
