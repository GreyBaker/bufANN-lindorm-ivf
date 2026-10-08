# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

include(${PROJECT_SOURCE_DIR}/cmake/IVFContracts.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/IVFCentroids.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/IVFLists.cmake)

if(MSVC)
	add_subdirectory(dll)
else()
	#file(GLOB CPP_SOURCES *.cpp)
	set(CPP_SOURCES ann_exception.cpp aux_utils.cpp index.cpp
        linux_aligned_file_reader.cpp math_utils.cpp memory_mapper.cpp
        partition_and_pq.cpp pq_flash_index.cpp logger.cpp distance.cpp
		 utils.cpp bufann/graph_delta.cpp bufann/index_merger.cpp
        bufann/merge_insert.cpp bufann/inplace_backend.cpp bufann/inplace_graph_ops.cpp
        bufann/bufann_api.cpp)
	add_library(${PROJECT_NAME} ${CPP_SOURCES})
	add_library(${PROJECT_NAME}_s STATIC ${CPP_SOURCES})
    add_library(ivf_heap STATIC bufann/ivf_heap.cpp)
    target_link_libraries(ivf_heap PUBLIC ${PROJECT_NAME} ivf_contracts aio)
    # Finite-coordinate validation must survive the parent's -Ofast flags.
    target_compile_options(ivf_heap PRIVATE -fno-fast-math)
    if(TARGET ivf_lists)
        add_library(ivf_search STATIC bufann/ivf_search.cpp)
        target_link_libraries(ivf_search PUBLIC ivf_heap ivf_lists)
        target_compile_options(ivf_search PRIVATE -fno-fast-math)
        add_library(ivf_index STATIC bufann/ivf_index.cpp)
        target_link_libraries(ivf_index PUBLIC ivf_search)
        target_compile_options(ivf_index PRIVATE -fno-fast-math)
    endif()
endif()
install()
