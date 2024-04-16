cd utils/postprocess_func/cbox_decode
rm -rf build cpose_decode.so
python build.py build_ext --inplace
cd -

cd utils/postprocess_func/cpose_decode
rm -rf build cbox_decode.so
python build.py build_ext --inplace
cd -

cd utils/postprocess_func/cseg_decode
rm -rf build cseg_decode.so
python build.py build_ext --inplace
cd -

cd utils/postprocess_func/tracking/cbytetrack
rm -rf build
mkdir build
cd -
cd utils/postprocess_func/tracking/cbytetrack/build
cmake ..
make
cd -
