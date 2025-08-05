cmake_minimum_required(VERSION 3.16)
project(OpenCLIP_Inference)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set build type if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Add compiler flags
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# Function to download and extract ONNX Runtime
function(download_onnxruntime)
    set(ONNXRUNTIME_VERSION "1.16.3")
    set(ONNXRUNTIME_PLATFORM "linux-x64")
    set(ONNXRUNTIME_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-${ONNXRUNTIME_PLATFORM}-${ONNXRUNTIME_VERSION}.tgz")
    set(ONNXRUNTIME_DIR "${CMAKE_BINARY_DIR}/onnxruntime")
    set(ONNXRUNTIME_ARCHIVE "${CMAKE_BINARY_DIR}/onnxruntime.tgz")

    # Check if ONNX Runtime is already downloaded
    if(NOT EXISTS "${ONNXRUNTIME_DIR}")
        message(STATUS "Downloading ONNX Runtime ${ONNXRUNTIME_VERSION}...")
        
        # Download the archive
        file(DOWNLOAD 
            ${ONNXRUNTIME_URL} 
            ${ONNXRUNTIME_ARCHIVE}
            SHOW_PROGRESS
            STATUS download_status
        )
        
        # Check download status
        list(GET download_status 0 status_code)
        if(NOT status_code EQUAL 0)
            message(FATAL_ERROR "Failed to download ONNX Runtime")
        endif()
        
        # Extract the archive
        message(STATUS "Extracting ONNX Runtime...")
        file(ARCHIVE_EXTRACT 
            INPUT ${ONNXRUNTIME_ARCHIVE}
            DESTINATION ${CMAKE_BINARY_DIR}
        )
        
        # Rename the extracted directory to a consistent name
        file(GLOB EXTRACTED_DIRS "${CMAKE_BINARY_DIR}/onnxruntime-*")
        list(GET EXTRACTED_DIRS 0 EXTRACTED_DIR)
        file(RENAME ${EXTRACTED_DIR} ${ONNXRUNTIME_DIR})
        
        # Clean up the archive
        file(REMOVE ${ONNXRUNTIME_ARCHIVE})
        
        message(STATUS "ONNX Runtime downloaded and extracted successfully")
    else()
        message(STATUS "ONNX Runtime already exists at ${ONNXRUNTIME_DIR}")
    endif()
    
    # Set parent scope variables
    set(ONNXRUNTIME_ROOT_PATH ${ONNXRUNTIME_DIR} PARENT_SCOPE)
endfunction()

# Download ONNX Runtime
download_onnxruntime()

# Set ONNX Runtime paths
set(ONNXRUNTIME_INCLUDE_DIRS "${ONNXRUNTIME_ROOT_PATH}/include")
set(ONNXRUNTIME_LIB_DIR "${ONNXRUNTIME_ROOT_PATH}/lib")

# Find ONNX Runtime library
find_library(ONNXRUNTIME_LIB
    NAMES onnxruntime
    PATHS ${ONNXRUNTIME_LIB_DIR}
    NO_DEFAULT_PATH
)

if(NOT ONNXRUNTIME_LIB)
    message(FATAL_ERROR "ONNX Runtime library not found in ${ONNXRUNTIME_LIB_DIR}")
endif()

message(STATUS "ONNX Runtime include directory: ${ONNXRUNTIME_INCLUDE_DIRS}")
message(STATUS "ONNX Runtime library: ${ONNXRUNTIME_LIB}")

find_package(ZLIB REQUIRED)

# Create cnpy library from source
add_library(cnpy STATIC cnpy.cpp cnpy.h)

target_include_directories(cnpy PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(cnpy PUBLIC ZLIB::ZLIB)

# Create the executable
set(SOURCES
    clip_inference.cpp  # Assuming your main file is named main.cpp
)

set(SOURCES_B
        text_matching.cpp  # Assuming your main file is named main.cpp
)

add_executable(${PROJECT_NAME} ${SOURCES})

add_executable(FULL_MATCHER ${SOURCES_B})

# Include directories
target_include_directories(${PROJECT_NAME} PRIVATE 
    ${ONNXRUNTIME_INCLUDE_DIRS}
)

target_include_directories(FULL_MATCHER PRIVATE
        ${ONNXRUNTIME_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(${PROJECT_NAME} PRIVATE 
    ${ONNXRUNTIME_LIB}
)

target_link_libraries(FULL_MATCHER PRIVATE
        ${ONNXRUNTIME_LIB} cnpy
)

# Copy ONNX Runtime shared library to output directory (for runtime)
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${ONNXRUNTIME_LIB_DIR}/libonnxruntime.so*"
    "$<TARGET_FILE_DIR:${PROJECT_NAME}>"
    COMMENT "Copying ONNX Runtime shared library to output directory"
)

# Alternative: Set RPATH so the executable can find the shared library
set_target_properties(${PROJECT_NAME} PROPERTIES
    INSTALL_RPATH "${ONNXRUNTIME_LIB_DIR}"
    BUILD_WITH_INSTALL_RPATH TRUE
)

# Print build information
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")

# Optional: Create install target
install(TARGETS ${PROJECT_NAME}
    RUNTIME DESTINATION bin
)

install(FILES "${ONNXRUNTIME_LIB_DIR}/libonnxruntime.so"
    DESTINATION lib
)

# Optional: Add a target to clean downloaded dependencies
add_custom_target(clean-deps
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${ONNXRUNTIME_ROOT_PATH}
    COMMENT "Cleaning downloaded dependencies"
)
