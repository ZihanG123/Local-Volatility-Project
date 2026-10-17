from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext


setup(
    name="compfi-optimization-native",
    ext_modules=[
        Pybind11Extension(
            "_optimization_cpp",
            ["optimization_cpp.cpp"],
            cxx_std=17,
            extra_compile_args=["-O3"],
        )
    ],
    cmdclass={"build_ext": build_ext},
)