cmake_minimum_required(VERSION 3.12)

project(ReplicatedTokenizer VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(PkgConfig REQUIRED)
find_package(ZLIB REQUIRED)

# Find CURL
find_package(CURL REQUIRED)
if(NOT CURL_FOUND)
    message(FATAL_ERROR "CURL library not found")
endif()

# Create executable
add_executable(tokenizer tokenizer.cpp)

# Set target properties
target_compile_features(tokenizer PRIVATE cxx_std_17)

# Link libraries
target_link_libraries(tokenizer 
    PRIVATE 
    CURL::libcurl
    ZLIB::ZLIB
)

# Compiler-specific options
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(tokenizer PRIVATE -Wall -Wextra -O2)
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_compile_options(tokenizer PRIVATE /W4 /O2)
endif()

# Installation
install(TARGETS tokenizer DESTINATION bin)

# Print build information
message(STATUS "Building ${PROJECT_NAME} v${PROJECT_VERSION}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "CURL Version: ${CURL_VERSION_STRING}")
message(STATUS "ZLIB Version: ${ZLIB_VERSION_STRING}")
