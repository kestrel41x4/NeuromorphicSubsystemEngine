cmake_minimum_required(VERSION 3.26)
project(NeuromorphicSubsystemEngine LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Shared protocol headers interface
add_subdirectory(protocol)

# Host execution runtime
add_subdirectory(runtime)