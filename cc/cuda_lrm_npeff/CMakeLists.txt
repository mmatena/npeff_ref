cmake_minimum_required(VERSION 3.25)

set(CMAKE_CXX_COMPILER "/usr/bin/g++")
set(CMAKE_C_COMPILER "/usr/bin/gcc")
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")

project(
    Cuda_M_NPEFF
    VERSION 0.1
    DESCRIPTION "Decomposition of low-rank per-example Fisher matrices."
    LANGUAGES CUDA CXX)

# Note this needs to be done in the main CMakeLists
# since it calls enable_testing, which must be in the
# main CMakeLists.
include(CTest)
enable_testing()

# FetchContent added in CMake 3.11, downloads during the configure step
# FetchContent_MakeAvailable was added in CMake 3.14; simpler usage
include(FetchContent)



# Hdf5 info.
set(HDF5_INCLUDE_DIR "/usr/lib/x86_64-linux-gnu/hdf5/serial/include")
set(HDF5_LIB_DIR "/usr/lib/x86_64-linux-gnu/hdf5/serial/lib")

# CUDA info.
set(CUDA_INCLUDE_DIR "/usr/local/cuda/include")
set(CUDA_LIB_DIR "/usr/local/cuda/lib64")

include_directories(${HDF5_INCLUDE_DIR})
include_directories(${CUDA_INCLUDE_DIR})



# The compiled library code is here
add_subdirectory(src)

# The executable code is here
add_subdirectory(mains)


# Testing library
FetchContent_Declare(
  catch
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG v3.3.1)
FetchContent_MakeAvailable(catch)
# Adds Catch2::Catch2


# gflags -> for command line flags.
FetchContent_Declare(
  gflags
  GIT_REPOSITORY https://github.com/gflags/gflags.git
  GIT_TAG v2.2.2)
FetchContent_MakeAvailable(gflags)
# Adds gflags::gflags

