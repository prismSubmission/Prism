cmake_minimum_required(VERSION 3.21)
project(Prism)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optional dependency roots. Set these from the command line if CMake cannot
# find the libraries automatically on your system.
set(ONNXRUNTIME_ROOT "" CACHE PATH "Root directory of ONNX Runtime GPU")
set(TOKENIZERS_CPP_ROOT "" CACHE PATH "Root directory of tokenizers-cpp")
set(SENTENCEPIECE_ROOT "" CACHE PATH "Root directory of sentencepiece")
set(CONDA_ENV_ROOT "$ENV{CONDA_PREFIX}" CACHE PATH "Root directory of the Conda environment that provides GSL, if any")

find_package(OpenMP REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Boost REQUIRED COMPONENTS filesystem system)
find_package(CURL REQUIRED)
find_library(TIFF_LIB tiff REQUIRED)

find_path(GSL_INCLUDE_DIR gsl/gsl_rng.h
    HINTS ${CONDA_ENV_ROOT}
    PATH_SUFFIXES include
)
find_library(GSL_LIB gsl
    HINTS ${CONDA_ENV_ROOT}
    PATH_SUFFIXES lib
    REQUIRED
)
find_library(GSLCBLAS_LIB gslcblas
    HINTS ${CONDA_ENV_ROOT}
    PATH_SUFFIXES lib
    REQUIRED
)

find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
    HINTS ${ONNXRUNTIME_ROOT}
    PATH_SUFFIXES include include/onnxruntime/core/session
    REQUIRED
)
find_library(ONNXRUNTIME_LIB onnxruntime
    HINTS ${ONNXRUNTIME_ROOT}
    PATH_SUFFIXES lib lib64
    REQUIRED
)

find_path(TOKENIZERS_CPP_INCLUDE_DIR tokenizers_cpp.h
    HINTS ${TOKENIZERS_CPP_ROOT}
    PATH_SUFFIXES include
    REQUIRED
)
find_library(TOKENIZERS_CPP_LIB tokenizers_cpp
    HINTS ${TOKENIZERS_CPP_ROOT}
    PATH_SUFFIXES build build/release lib lib64
    REQUIRED
)
find_library(TOKENIZERS_C_LIB tokenizers_c
    HINTS ${TOKENIZERS_CPP_ROOT}
    PATH_SUFFIXES build build/release lib lib64
    REQUIRED
)

find_path(SENTENCEPIECE_INCLUDE_DIR sentencepiece_processor.h
    HINTS ${SENTENCEPIECE_ROOT}
    PATH_SUFFIXES include local/usr/local/include
    REQUIRED
)
find_library(SENTENCEPIECE_LIB sentencepiece
    HINTS ${SENTENCEPIECE_ROOT}
    PATH_SUFFIXES lib lib64 local/usr/local/lib
    REQUIRED
)

include_directories(
    ${Boost_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${CURL_INCLUDE_DIRS}
    ${GSL_INCLUDE_DIR}
    ${ONNXRUNTIME_INCLUDE_DIR}
    ${TOKENIZERS_CPP_INCLUDE_DIR}
    ${SENTENCEPIECE_INCLUDE_DIR}
    ${PROJECT_SOURCE_DIR}/lib 
)

add_executable(Prism
    driver/subgraph-cardinality-estimation.cc
    lib/SubgraphMatching/getOracle.cpp
)

target_link_libraries(Prism
    PUBLIC
    OpenMP::OpenMP_CXX
    ${Boost_LIBRARIES}
    ${OpenCV_LIBS}           
    ${GSL_LIB}
    ${GSLCBLAS_LIB}
    ${CURL_LIBRARIES}        
    ${TIFF_LIB}
    ${ONNXRUNTIME_LIB}
    ${TOKENIZERS_CPP_LIB}
    ${TOKENIZERS_C_LIB}
    ${SENTENCEPIECE_LIB}
    pthread
    dl
)

message(STATUS "------------------------------------------------")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Compiler Flags: ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${CMAKE_BUILD_TYPE}}")
message(STATUS "ONNX Runtime include: ${ONNXRUNTIME_INCLUDE_DIR}")
message(STATUS "ONNX Runtime lib: ${ONNXRUNTIME_LIB}")
message(STATUS "tokenizers-cpp include: ${TOKENIZERS_CPP_INCLUDE_DIR}")
message(STATUS "tokenizers-cpp lib: ${TOKENIZERS_CPP_LIB}")
message(STATUS "tokenizers-c lib: ${TOKENIZERS_C_LIB}")
message(STATUS "sentencepiece include: ${SENTENCEPIECE_INCLUDE_DIR}")
message(STATUS "sentencepiece lib: ${SENTENCEPIECE_LIB}")
message(STATUS "GSL include: ${GSL_INCLUDE_DIR}")
message(STATUS "GSL lib: ${GSL_LIB}")
message(STATUS "------------------------------------------------")
