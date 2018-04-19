
from distutils.core import setup, Extension
from Cython.Build import cythonize

setup(ext_modules=cythonize(
    Extension(
        # the extension name
        "log_uniform",
        # the Cython source and additional C++ source files
        sources=["log_uniform.pyx", "Log_Uniform_Sampler.cpp"],
        # generate and compile C++ code
        language="c++",
        extra_compile_args=["-std=c++11"]
    )
))
