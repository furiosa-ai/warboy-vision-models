from distutils.command.build_ext import build_ext as build_ext_orig

from setuptools import Extension, find_packages
from skbuild import setup


class CTypesExtension(Extension):
    pass


class build_ext(build_ext_orig):
    def build_extension(self, ext):
        self._ctypes = isinstance(ext, CTypesExtension)
        return super().build_extension(ext)

    def get_export_symbols(self, ext):
        if self._ctypes:
            return ext.export_symbols
        return super().get_export_symbols(ext)

    def get_ext_filename(self, ext_name):
        if self._ctypes:
            return ext_name + ".so"
        return super().get_ext_filename(ext_name)


setup(
    name="warboy-vision-models",
    version="v0.1.0",
    packages=find_packages(""),
    package_dir={"": ""},
    cmake_source_dir="utils/postprocess_func/tracking/cbytetrack",
    cmake_install_dir="utils/postprocess_func/tracking/cbytetrack/build",
    cmake_args=[
        # 여기서 추가 CMake 인자를 지정할 수 있습니다.
    ],
    include_package_data=True,
    ext_modules=[
        CTypesExtension(
            "cpose_decode",
            ["utils/postprocess_func/cpose_decode/pose_decode.cpp"],
            extra_compile_args=["-ffast-math", "-O3"],
        ),
        CTypesExtension(
            "cbox_decode",
            ["utils/postprocess_func/cbox_decode/box_decode.cpp"],
            extra_compile_args=["-ffast-math", "-O3"],
        ),
    ],
    cmdclass={"build_ext": build_ext},
)
