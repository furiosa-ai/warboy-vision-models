from distutils.command.build_ext import build_ext as build_ext_orig
from distutils.dir_util import mkpath

from setuptools import Extension, find_packages
from skbuild import setup

postproc_root = "utils/postprocess_func/"

mkpath(f"{postproc_root}/tracking/cbytetrack/build")


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
    cmake_source_dir=f"{postproc_root}/tracking/cbytetrack",
    cmake_install_dir=f"{postproc_root}/tracking/cbytetrack/build",
    cmake_args=[],
    include_package_data=True,
    ext_modules=[
        CTypesExtension(
            f"{postproc_root}.cpose_decode.cpose_decode",
            [f"{postproc_root}/cpose_decode/pose_decode.cpp"],
            extra_compile_args=["-ffast-math", "-O3"],
        ),
        CTypesExtension(
            f"{postproc_root}.cbox_decode.cbox_decode",
            [f"{postproc_root}/cbox_decode/box_decode.cpp"],
            extra_compile_args=["-ffast-math", "-O3"],
        ),
    ],
    cmdclass={"build_ext": build_ext},
)
