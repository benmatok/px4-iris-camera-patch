from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        "vision.ice_ba",
        sources=["vision/ice_ba.pyx", "vision/cpp/ice_ba.cpp"],
        include_dirs=[numpy.get_include(), "vision/cpp"],
        language="c++",
        extra_compile_args=["-std=c++11"]
    )
]

setup(
    name="drone_env_pure_python",
    packages=find_packages(),
    ext_modules=cythonize(extensions),
    install_requires=[
        "numpy",
        "scipy",
        "Cython"
    ]
)
