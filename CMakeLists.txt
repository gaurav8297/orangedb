cmake_minimum_required(VERSION 3.25)
project(OrangeDB LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_C_VISIBILITY_PRESET hidden)
set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)
set(CMAKE_FIND_PACKAGE_RESOLVE_SYMLINKS TRUE)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

# On Linux, symbols in executables are not accessible by loaded shared libraries (e.g. via dlopen(3)). However, we need to export public symbols in executables so that extensions can access public symbols. This enables that behaviour.
set(CMAKE_ENABLE_EXPORTS TRUE)

# Place binaries and libraries according to GNU standards
include(GNUInstallDirs)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR})

find_package(OpenMP REQUIRED)

# Global compiler flags for debug and release
set(CMAKE_CXX_FLAGS_DEBUG "-fopenmp -g -fsanitize=address -fno-omit-frame-pointer")
set(CMAKE_CXX_FLAGS_RELEASE "-fopenmp -O3")
set(CMAKE_C_FLAGS_DEBUG "-fopenmp -g -fsanitize=address -fno-omit-frame-pointer")
set(CMAKE_C_FLAGS_RELEASE "-fopenmp -O3")

# Add include path
include_directories(${PROJECT_SOURCE_DIR}/src/include)
include_directories(${PROJECT_SOURCE_DIR}/third_party/spdlog)
include_directories(${PROJECT_SOURCE_DIR}/third_party/simsimd/include)
# Add source path
add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/simsimd)
add_subdirectory(${PROJECT_SOURCE_DIR}/src)

# Add executable path
add_executable(orangedb_main ${PROJECT_SOURCE_DIR}/src/main.cpp)
target_link_libraries(orangedb_main PUBLIC orangedb OpenMP::OpenMP_CXX simsimd)
target_include_directories(orangedb_main PUBLIC
        ${PROJECT_SOURCE_DIR}/src/include
        ${PROJECT_SOURCE_DIR}/third_party/spdlog
        ${PROJECT_SOURCE_DIR}/third_party/simsimd/include)
