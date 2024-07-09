cd warboy/utils/decoder/cbox_decode
rm -rf build cpose_decode.so
python build.py build_ext --inplace
cd -

cd warboy/utils/decoder/cpose_decode
rm -rf build cbox_decode.so
python build.py build_ext --inplace
cd -

cd warboy/utils/decoder/tracking/cbytetrack
rm -rf build
mkdir build
cd -
cd warboy/utils/decoder/tracking/cbytetrack/build
cmake ..
make
cd -
