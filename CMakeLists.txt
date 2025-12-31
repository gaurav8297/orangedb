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

include(FetchContent)

# Also requires one of: libbfd (gnu binutils), libdwarf, libdw (elfutils)
FetchContent_Declare(backward
        GIT_REPOSITORY https://github.com/bombela/backward-cpp
        GIT_TAG master  # or a version tag, such as v1.6
        SYSTEM          # optional, the Backward include directory will be treated as system directory
)
FetchContent_MakeAvailable(backward)

# Fetch knncolle for nearest neighbor search (required by umappp)
FetchContent_Declare(knncolle
        GIT_REPOSITORY https://github.com/LTLA/knncolle
        GIT_TAG master
        SYSTEM
)
FetchContent_MakeAvailable(knncolle)

# Fetch umappp for UMAP dimensionality reduction
FetchContent_Declare(umappp
        GIT_REPOSITORY https://github.com/LTLA/umappp
        GIT_TAG master
        SYSTEM
)
FetchContent_MakeAvailable(umappp)

# Configure OpenMP for macOS
if(APPLE)
    set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp")
    set(OpenMP_C_LIB_NAMES "omp")
    set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp")
    set(OpenMP_CXX_LIB_NAMES "omp")
    set(OpenMP_omp_LIBRARY "/opt/homebrew/opt/libomp/lib/libomp.dylib")
    include_directories("/opt/homebrew/opt/libomp/include")
    
    # Add Homebrew prefix to CMAKE_PREFIX_PATH for Arrow/Parquet
    list(APPEND CMAKE_PREFIX_PATH "/opt/homebrew/opt/apache-arrow")
endif()

find_package(OpenMP REQUIRED)
find_package(Arrow CONFIG REQUIRED)
find_package(Parquet CONFIG REQUIRED)

get_target_property(ARROW_INCLUDE Arrow::arrow_shared INTERFACE_INCLUDE_DIRECTORIES)
message(STATUS "Arrow includes: ${ARROW_INCLUDE}")
get_target_property(PARQUET_INCLUDE Parquet::parquet_shared INTERFACE_INCLUDE_DIRECTORIES)
message(STATUS "Parquet includes: ${PARQUET_INCLUDE}")

# Global compiler flags for debug and release
if(APPLE)
    # macOS requires -Xpreprocessor -fopenmp instead of -fopenmp
    set(OPENMP_FLAG "-Xpreprocessor -fopenmp")
    set(RDYNAMIC_FLAG "")  # macOS doesn't support -rdynamic
else()
    set(OPENMP_FLAG "-fopenmp")
    set(RDYNAMIC_FLAG "-rdynamic")  # Linux supports -rdynamic
endif()

#set(CMAKE_CXX_FLAGS_DEBUG "-fopenmp -g -fsanitize=address -fno-omit-frame-pointer -mavx2 -mfma -mavx512f -mavx512cd -mavx512vl -mavx512dq -mavx512bw")
# set(CMAKE_CXX_FLAGS_DEBUG "-fopenmp -g -fsanitize=address -rdynamic -fno-omit-frame-pointer")
set(CMAKE_CXX_FLAGS_DEBUG "${OPENMP_FLAG} -g -fsanitize=address ${RDYNAMIC_FLAG} -fno-omit-frame-pointer")

# for now remove -march=native
# set(CMAKE_CXX_FLAGS_RELEASE "-fopenmp -O3 -g -rdynamic -fno-omit-frame-pointer")
set(CMAKE_CXX_FLAGS_RELEASE "${OPENMP_FLAG} -O3 -g ${RDYNAMIC_FLAG} -fno-omit-frame-pointer")

#set(CMAKE_C_FLAGS_DEBUG "-fopenmp -g -fsanitize=address -fno-omit-frame-pointer -mavx2 -mfma -mavx512f -mavx512cd -mavx512vl -mavx512dq -mavx512bw")
# set(CMAKE_C_FLAGS_DEBUG "-fopenmp -g -fsanitize=address -rdynamic -fno-omit-frame-pointer")
set(CMAKE_C_FLAGS_DEBUG "${OPENMP_FLAG} -g -fsanitize=address ${RDYNAMIC_FLAG} -fno-omit-frame-pointer")

# for now remove -march=native
# set(CMAKE_C_FLAGS_RELEASE "-fopenmp -O3 -g -rdynamic -fno-omit-frame-pointer")
set(CMAKE_C_FLAGS_RELEASE "${OPENMP_FLAG} -O3 -g ${RDYNAMIC_FLAG} -fno-omit-frame-pointer")
#set(CMAKE_OSX_ARCHITECTURES "x86_64")

# Add include path
include_directories(${PROJECT_SOURCE_DIR}/third_party/spdlog)
include_directories(${PROJECT_SOURCE_DIR}/third_party/simsimd/include)
include_directories(${PROJECT_SOURCE_DIR}/third_party/faiss)
include_directories(${PROJECT_SOURCE_DIR}/third_party/iRangeGraph/include)
include_directories(${PROJECT_SOURCE_DIR}/third_party/json/include)
include_directories(${ARROW_INCLUDE})
include_directories(${PARQUET_INCLUDE})
include_directories(${PROJECT_SOURCE_DIR}/src/include)

# Get the source directories for FetchContent dependencies
FetchContent_GetProperties(umappp)
FetchContent_GetProperties(knncolle)
include_directories(${umappp_SOURCE_DIR}/include)
include_directories(${knncolle_SOURCE_DIR}/include)

# Add include directories for umappp's transitive dependencies
# These are automatically fetched by umappp's CMakeLists.txt
FetchContent_GetProperties(sanisizer)
if(NOT sanisizer_POPULATED)
    FetchContent_GetProperties(sanisizer SOURCE_DIR sanisizer_SOURCE_DIR)
endif()
include_directories(${sanisizer_SOURCE_DIR}/include)

FetchContent_GetProperties(irlba)
if(NOT irlba_POPULATED)
    FetchContent_GetProperties(irlba SOURCE_DIR irlba_SOURCE_DIR)
endif()
include_directories(${irlba_SOURCE_DIR}/include)

FetchContent_GetProperties(aarand)
if(NOT aarand_POPULATED)
    FetchContent_GetProperties(aarand SOURCE_DIR aarand_SOURCE_DIR)
endif()
include_directories(${aarand_SOURCE_DIR}/include)

FetchContent_GetProperties(subpar)
if(NOT subpar_POPULATED)
    FetchContent_GetProperties(subpar SOURCE_DIR subpar_SOURCE_DIR)
endif()
include_directories(${subpar_SOURCE_DIR}/include)

FetchContent_GetProperties(eigen)
if(NOT eigen_POPULATED)
    FetchContent_GetProperties(eigen SOURCE_DIR eigen_SOURCE_DIR)
endif()
include_directories(${eigen_SOURCE_DIR})
# Add source path
add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/simsimd)
add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/faiss)
add_subdirectory(${PROJECT_SOURCE_DIR}/src)

# check for linux

#
#include_directories(${PROJECT_SOURCE_DIR}/third_party/liburing/src/include)
#set(LIBURING_DIR ${CMAKE_SOURCE_DIR}/third_party/liburing/build)
#include_directories(${LIBURING_DIR}/include)
## Link the liburing library
#link_directories(${LIBURING_DIR}/lib)

# Add executable path
add_executable(orangedb_main ${PROJECT_SOURCE_DIR}/src/main.cpp)
target_link_libraries(orangedb_main PUBLIC orangedb OpenMP::OpenMP_CXX simsimd faiss Arrow::arrow_shared Parquet::parquet_shared Backward::Interface)
target_include_directories(orangedb_main PUBLIC
        ${PROJECT_SOURCE_DIR}/src/include
        ${PROJECT_SOURCE_DIR}/third_party/spdlog
        ${PROJECT_SOURCE_DIR}/third_party/simsimd/include
        ${PROJECT_SOURCE_DIR}/third_party/faiss
        ${PROJECT_SOURCE_DIR}/third_party/iRangeGraph/include
        ${PROJECT_SOURCE_DIR}/third_party/json/include
        ${ARROW_INCLUDE}
        ${PARQUET_INCLUDE}
        ${umappp_SOURCE_DIR}/include
        ${knncolle_SOURCE_DIR}/include
        ${sanisizer_SOURCE_DIR}/include
        ${irlba_SOURCE_DIR}/include
        ${aarand_SOURCE_DIR}/include
        ${subpar_SOURCE_DIR}/include
        ${eigen_SOURCE_DIR}
#        ${PROJECT_SOURCE_DIR}/third_party/liburing/src/include
        ${LIBURING_DIR}/include)
