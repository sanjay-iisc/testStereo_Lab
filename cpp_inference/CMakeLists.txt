cmake_minimum_required(VERSION 3.18)

project(multitask_trt_inference VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


# package
find_package(OpenCV 4 REQUIRED)

# cuda 
set(CUDA_ROOT "" CACHE PATH "Optional CUDA root directory")

find_path(CUDA_INCLUDE_DIR
    cuda_runtime_api.h
    HINTS ${CUDA_ROOT} $ENV{CUDA_ROOT}
    PATH_SUFFIXES include
    PATHS
        /usr/local/cuda
        /usr/local/cuda-12
        /usr/local/cuda-12.2)

find_library(CUDA_CUDART_LIBRARY
    cudart
    HINTS ${CUDA_ROOT} $ENV{CUDA_ROOT}
    PATH_SUFFIXES lib64 lib
    PATHS
        /usr/local/cuda
        /usr/local/cuda-12
        /usr/local/cuda-12.2)

if(NOT CUDA_INCLUDE_DIR)
    message(FATAL_ERROR "CUDA include directory not found")
endif()

if(NOT CUDA_CUDART_LIBRARY)
    message(FATAL_ERROR "CUDA cudart library not found")
endif()

# TensorRT
set(TensorRT_ROOT "" CACHE PATH "Optional TensorRT root directory")

find_path(TENSORRT_INCLUDE_DIR
    NvInfer.h
    HINTS ${TensorRT_ROOT} $ENV{TensorRT_ROOT}
    PATH_SUFFIXES include include/x86_64-linux-gnu
    PATHS
        /usr
        /usr/local/TensorRT)

find_library(TENSORRT_LIBRARY
    nvinfer
    HINTS ${TensorRT_ROOT} $ENV{TensorRT_ROOT}
    PATH_SUFFIXES lib lib64 lib/x86_64-linux-gnu
    PATHS
        /usr
        /usr/local/TensorRT)

find_library(TENSORRT_ONNXPARSER_LIBRARY
    nvonnxparser
    HINTS ${TensorRT_ROOT} $ENV{TensorRT_ROOT}
    PATH_SUFFIXES lib lib64 lib/x86_64-linux-gnu
    PATHS
        /usr
        /usr/local/TensorRT)

if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_LIBRARY OR NOT TENSORRT_ONNXPARSER_LIBRARY)
    message(FATAL_ERROR
        "TensorRT not found.\n"
        "Install TensorRT 10.x or configure with:\n"
        "cmake .. -DTensorRT_ROOT=/path/to/TensorRT")
endif()

file(READ "${TENSORRT_INCLUDE_DIR}/NvInferVersion.h" TRT_VERSION_HEADER)

string(REGEX MATCH
    "#define[ \t]+TRT_MAJOR_ENTERPRISE[ \t]+([0-9]+)"
    _major_match
    "${TRT_VERSION_HEADER}"
)
set(TENSORRT_MAJOR "${CMAKE_MATCH_1}")

string(REGEX MATCH
    "#define[ \t]+TRT_MINOR_ENTERPRISE[ \t]+([0-9]+)"
    _minor_match
    "${TRT_VERSION_HEADER}"
)
set(TENSORRT_MINOR "${CMAKE_MATCH_1}")

string(REGEX MATCH
    "#define[ \t]+TRT_PATCH_ENTERPRISE[ \t]+([0-9]+)"
    _patch_match
    "${TRT_VERSION_HEADER}"
)
set(TENSORRT_PATCH "${CMAKE_MATCH_1}")

if(TENSORRT_MAJOR STREQUAL "")
    message(WARNING "Could not parse TensorRT version from NvInferVersion.h")
else()
    message(STATUS "TensorRT Version : ${TENSORRT_MAJOR}.${TENSORRT_MINOR}.${TENSORRT_PATCH}")
    message(STATUS "TensorRT Include : ${TENSORRT_INCLUDE_DIR}")
    message(STATUS "TensorRT Library : ${TENSORRT_LIBRARY}")
    message(STATUS "TensorRT ONNX Parser : ${TENSORRT_ONNXPARSER_LIBRARY}")
endif()



# executable
add_executable(multitask_infer
    src/main.cpp
    src/TensorRTModel.cpp
    src/Preprocessor.cpp
    src/Postprocessor.cpp
)

## binary file
set_target_properties(multitask_infer PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)


target_include_directories(multitask_infer PRIVATE 
    include
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIR}
    ${TENSORRT_INCLUDE_DIR})

target_link_libraries(multitask_infer PRIVATE
    ${OpenCV_LIBS}
    ${CUDA_CUDART_LIBRARY}
    ${TENSORRT_LIBRARY}
    ${TENSORRT_ONNXPARSER_LIBRARY})


target_compile_options(multitask_infer PRIVATE $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>)



message(STATUS "OpenCV Version : ${OpenCV_VERSION}")
message(STATUS "OpenCV Include : ${OpenCV_INCLUDE_DIRS}")
# message(STATUS "OpenCV Libraries : ${OpenCV_LIBS}")

message(STATUS "CUDA Root    : ${CUDA_ROOT}")
message(STATUS "CUDA Include : ${CUDA_INCLUDE_DIR}")
message(STATUS "CUDA CUDART  : ${CUDA_CUDART_LIBRARY}")





# ------------------------

# find_package(OpenCV REQUIRED)

# include_directories(
#     /usr/local/cuda/include
#     /usr/include/x86_64-linux-gnu
#     ${OpenCV_INCLUDE_DIRS}
# )

# link_directories(
#     /usr/local/cuda/lib64
#     /usr/lib/x86_64-linux-gnu
# )

# add_executable(multitask_infer main.cpp)

# target_link_libraries(multitask_infer
#     ${OpenCV_LIBS}
#     nvinfer
#     nvonnxparser
#     cudart
# )
