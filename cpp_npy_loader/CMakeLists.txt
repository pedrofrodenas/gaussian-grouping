cmake_minimum_required(VERSION 3.16)

# Project name and version
project(NumpyArrayLoader VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(PkgConfig REQUIRED)

# Find zlib (required by cnpy)
find_package(ZLIB REQUIRED)

# Create cnpy library from source
add_library(cnpy STATIC cnpy.cpp cnpy.h)

# Set include directories for cnpy
target_include_directories(cnpy PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Link zlib to cnpy
target_link_libraries(cnpy PUBLIC ZLIB::ZLIB)

# Create the main executable
add_executable(numpy_loader main.cpp)

# Link cnpy library to the executable
target_link_libraries(numpy_loader PRIVATE cnpy)

# Set compiler-specific options
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(numpy_loader PRIVATE -Wall -Wextra -O2)
endif()

# For MSVC
if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_compile_options(numpy_loader PRIVATE /W4)
endif()

# Install target (optional)
install(TARGETS numpy_loader
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)

# Print some information
message(STATUS "Building NumpyArrayLoader with C++${CMAKE_CXX_STANDARD}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")