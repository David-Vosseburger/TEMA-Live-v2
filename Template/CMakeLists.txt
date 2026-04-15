cmake_minimum_required(VERSION 3.5)
project(tema_cpp_bridge)

# Minimal CMake to build shared libs for the Template C++ files.
# Usage (optional):
# mkdir -p Template/build && cd Template/build && cmake .. && make

set(CMAKE_CXX_STANDARD 14)
set(SOURCE_DIR ${CMAKE_SOURCE_DIR})

# grid_signals
if(EXISTS ${SOURCE_DIR}/grid_signals.cpp)
  add_library(grid_signals SHARED ${SOURCE_DIR}/grid_signals.cpp)
  target_include_directories(grid_signals PRIVATE ${SOURCE_DIR})
endif()

# hmm_regime
if(EXISTS ${SOURCE_DIR}/hmm_regime.cpp)
  add_library(hmm_regime SHARED ${SOURCE_DIR}/hmm_regime.cpp)
  target_include_directories(hmm_regime PRIVATE ${SOURCE_DIR})
endif()

message(STATUS "Configured minimal C++ bridge build (if sources exist)")
