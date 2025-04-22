cd warboy/yolo/cbox_decode
rm -rf build cbox_decode.so
python build.py build_ext --inplace
cd -

cd warboy/yolo/cpose_decode
rm -rf build cpose_decode.so
python build.py build_ext --inplace
cd -

cd warboy/yolo/tracking/cbytetrack
rm -rf build
mkdir build
cd -
cd warboy/yolo/tracking/cbytetrack/build
cmake ..
make
cd -
