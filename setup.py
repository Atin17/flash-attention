"""
Setup script for FlashAttention CUDA extension.
"""

from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# CUDA extension
ext_modules = [
    CUDAExtension(
        name='flash_attention_cuda',
        sources=[
            'csrc/flash_attention.cu',
            'csrc/bindings.cpp',
        ],
        extra_compile_args={
            'cxx': ['-O3', '-std=c++14'],
            'nvcc': [
                '-O3',
                '--use_fast_math',
                '-std=c++14',
                '--expt-relaxed-constexpr',
                '-U__CUDA_NO_HALF_OPERATORS__',
                '-U__CUDA_NO_HALF_CONVERSIONS__',
                '-U__CUDA_NO_HALF2_OPERATORS__',
            ]
        }
    )
]

setup(
    name='flash-attention',
    version='0.1.0',
    description='FlashAttention: Fast and Memory-Efficient Attention',
    author='Your Name',
    author_email='your.email@example.com',
    url='https://github.com/yourusername/flash-attention',
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={
        'build_ext': BuildExtension
    },
    install_requires=[
        'torch>=2.0.0',
        'numpy>=1.21.0',
    ],
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
