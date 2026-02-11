include_directories(${CMAKE_CURRENT_BINARY_DIR})

set(ORANGEDB_SOURCES
        helper_ds.cpp
        hnsw.cpp
        clustering.cpp
        partitioned_index.cpp
        file_handle.cpp
        reclustering_index.cpp
        incremental_index.cpp
)

if(ENABLE_CUVS)
    file(GLOB ORANGEDB_CU_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cu)
    list(APPEND ORANGEDB_SOURCES ${ORANGEDB_CU_SOURCES})
endif()

add_library(orangedb
        OBJECT
        ${ORANGEDB_SOURCES}
)

set(ALL_OBJECT_FILES
        ${ALL_OBJECT_FILES} $<TARGET_OBJECTS:orangedb>
        PARENT_SCOPE)

#target_link_libraries(vector_index PUBLIC faiss)
set(ORANGE_DB_INCLUDES $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}> ${CMAKE_CURRENT_BINARY_DIR})
target_include_directories(orangedb PUBLIC ORANGE_DB_INCLUDES)
if(ENABLE_CUVS)
target_link_libraries(orangedb PUBLIC cuvs::cuvs)
endif()
install(TARGETS orangedb)
