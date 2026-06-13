cd cpp_inference
rm -rf build
mkdir -p build
cd build

cmake .. \
  -DCUDA_ROOT=/usr/local/cuda-12.2 \
  -DTensorRT_ROOT=/usr
  
make -j$(( $(nproc) / 2 ))