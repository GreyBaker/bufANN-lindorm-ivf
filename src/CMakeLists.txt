# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

include(${PROJECT_SOURCE_DIR}/cmake/IVF.cmake)

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
    # Use the unchanged backend archive without inheriting its Intel OpenMP /
    # threaded MKL link interface into the FAISS engine.
    target_link_libraries(ivf_heap PUBLIC "$<TARGET_FILE:${PROJECT_NAME}>" ivf_contracts aio)
    add_dependencies(ivf_heap ${PROJECT_NAME})
    if(COMMAND ivf_native_link_policy)
        ivf_native_link_policy(ivf_heap)
    endif()
    if(TARGET ivf_lists)
        add_library(ivf_search STATIC bufann/ivf_search.cpp)
        target_link_libraries(ivf_search PUBLIC ivf_heap ivf_lists)
        add_library(ivf_index STATIC bufann/ivf_index.cpp)
        target_link_libraries(ivf_index PUBLIC ivf_search)
        ivf_native_link_policy(ivf_search)
        ivf_native_link_policy(ivf_index)
    endif()
endif()
install()
