cmake_minimum_required(VERSION 3.10)
project(OMSimTest)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

# Path to LibTorch
set(Torch_DIR "${CMAKE_SOURCE_DIR}/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)

# Executable
add_executable(OMSimTest main.cpp)

# Link LibTorch
target_link_libraries(OMSimTest "${TORCH_LIBRARIES}")

# Set RPATH to LibTorch for runtime
set_property(TARGET OMSimTest PROPERTY CXX_STANDARD 14)
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${Torch_DIR})
