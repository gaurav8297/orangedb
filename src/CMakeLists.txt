include_directories(${CMAKE_CURRENT_BINARY_DIR})
add_library(orangedb
        OBJECT
        helper_ds.cpp
        hnsw.cpp
        clustering.cpp
        scalar_quantizer.cpp
        partitioned_index.cpp
        file_handle.cpp)

set(ALL_OBJECT_FILES
        ${ALL_OBJECT_FILES} $<TARGET_OBJECTS:orangedb>
        PARENT_SCOPE)

#target_link_libraries(vector_index PUBLIC faiss)
set(ORANGE_DB_INCLUDES $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}> ${CMAKE_CURRENT_BINARY_DIR})
target_include_directories(orangedb PUBLIC ORANGE_DB_INCLUDES)
install(TARGETS orangedb)
