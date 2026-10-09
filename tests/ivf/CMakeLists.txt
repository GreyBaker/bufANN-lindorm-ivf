cmake_minimum_required(VERSION 3.16)

# This subdirectory also configures on its own without MKL/libaio/folly.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(IVFContracts LANGUAGES CXX)
    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS OFF)
    enable_testing()
endif()

include(${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/IVFContracts.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/IVFCentroids.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/IVFLists.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/IVFCostModel.cmake)
find_package(Threads REQUIRED)

foreach(tool ivf_build ivf_driver)
    add_executable(${tool} "${IVF_ROOT}/tests/${tool}.cpp")
    target_link_libraries(${tool} PRIVATE ivf_contracts)
    set_target_properties(${tool} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests")
    if(NOT MSVC)
        target_compile_options(${tool} PRIVATE -fno-fast-math)
    endif()
endforeach()

# Native execution is available only with the actual Linux heap backend and
# opt-in centroid routing, never in dependency-isolated contract tools.
if(TARGET ivf_index AND NOT MSVC)
    add_executable(ivf_cost_model "${IVF_ROOT}/tests/ivf_cost_model.cpp")
    target_link_libraries(ivf_cost_model PRIVATE ivf_cost_model_core BufANN)
    target_compile_options(ivf_cost_model PRIVATE -fno-fast-math)
    set_target_properties(ivf_cost_model PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests")
    add_library(ivf_cli_execution STATIC "${IVF_ROOT}/tests/ivf_cli_execution.cpp")
    target_link_libraries(ivf_cli_execution PUBLIC ivf_index Threads::Threads)
    target_compile_options(ivf_cli_execution PRIVATE -fno-fast-math)
    foreach(tool ivf_build ivf_driver)
        target_compile_definitions(${tool} PRIVATE IVF_NATIVE_EXECUTION=1)
        target_link_libraries(${tool} PRIVATE ivf_cli_execution)
    endforeach()
endif()

if(TARGET ivf_router AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
    add_executable(ivf_router_diagnostics "${IVF_ROOT}/tests/ivf_router_diagnostics.cpp")
    target_link_libraries(ivf_router_diagnostics PRIVATE ivf_router Threads::Threads)
    target_compile_options(ivf_router_diagnostics PRIVATE -fno-fast-math)
    set_target_properties(ivf_router_diagnostics PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests")
    add_executable(ivf_training_diagnostics ivf_training_diagnostics.cpp)
    target_link_libraries(ivf_training_diagnostics PRIVATE ivf_router Threads::Threads)
    target_compile_options(ivf_training_diagnostics PRIVATE -fno-fast-math)
    set_target_properties(ivf_training_diagnostics PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests")
endif()

option(IVF_BUILD_CONTRACT_TESTS "Build IVF input and PQ contract tests" ON)
if(IVF_BUILD_CONTRACT_TESTS)
    enable_testing()
    add_executable(ivf_mixed_scheduler_tests ivf_mixed_scheduler_tests.cpp)
    target_link_libraries(ivf_mixed_scheduler_tests PRIVATE Threads::Threads)
    add_test(NAME ivf_mixed_scheduler COMMAND ivf_mixed_scheduler_tests)
    set_tests_properties(ivf_mixed_scheduler PROPERTIES TIMEOUT 30)
    add_executable(ivf_centroid_tests ivf_centroid_tests.cpp)
    target_link_libraries(ivf_centroid_tests PRIVATE ivf_centroids)
    if(NOT MSVC)
        target_compile_options(ivf_centroid_tests PRIVATE -fno-fast-math)
    endif()
    add_test(NAME ivf_centroid_training COMMAND ivf_centroid_tests
        "${CMAKE_CURRENT_BINARY_DIR}/centroid_fixtures")
    if(TARGET ivf_router)
        add_executable(ivf_cost_model_tests ivf_cost_model_tests.cpp)
        target_link_libraries(ivf_cost_model_tests PRIVATE ivf_cost_model_core)
        if(NOT MSVC)
            target_compile_options(ivf_cost_model_tests PRIVATE -fno-fast-math)
        endif()
        add_test(NAME ivf_cost_model_core COMMAND ivf_cost_model_tests)
        add_executable(ivf_ann_training_tests ivf_ann_training_tests.cpp)
        target_link_libraries(ivf_ann_training_tests PRIVATE ivf_router Threads::Threads)
        get_target_property(IVF_PATCHED_INCLUDE ivf_router INCLUDE_DIRECTORIES)
        target_include_directories(ivf_ann_training_tests SYSTEM PRIVATE ${IVF_PATCHED_INCLUDE})
        if(NOT MSVC)
            target_compile_options(ivf_ann_training_tests PRIVATE -fno-fast-math)
        endif()
        add_test(NAME ivf_ann_training COMMAND ivf_ann_training_tests "${CMAKE_CURRENT_BINARY_DIR}/ann_training_fixtures")
        set_tests_properties(ivf_ann_training PROPERTIES TIMEOUT 180)
        add_executable(ivf_patched_router_tests ivf_patched_router_tests.cpp)
        target_link_libraries(ivf_patched_router_tests PRIVATE ivf_router Threads::Threads)
        if(NOT MSVC)
            target_compile_options(ivf_patched_router_tests PRIVATE -fno-fast-math)
        endif()
        add_test(NAME ivf_patched_centroid_routing COMMAND ivf_patched_router_tests
            "${CMAKE_CURRENT_BINARY_DIR}/patched_router_fixtures")
        set_tests_properties(ivf_patched_centroid_routing PROPERTIES TIMEOUT 180)
        add_executable(ivf_mutable_list_tests ivf_mutable_list_tests.cpp)
        target_link_libraries(ivf_mutable_list_tests PRIVATE ivf_lists Threads::Threads)
        if(NOT MSVC)
            target_compile_options(ivf_mutable_list_tests PRIVATE -fno-fast-math)
        endif()
        add_test(NAME ivf_mutable_posting_lists COMMAND ivf_mutable_list_tests
            "${CMAKE_CURRENT_BINARY_DIR}/mutable_list_fixtures")
        add_executable(ivf_list_tests ivf_list_tests.cpp)
        target_link_libraries(ivf_list_tests PRIVATE ivf_lists)
        if(NOT MSVC)
            target_compile_options(ivf_list_tests PRIVATE -fno-fast-math)
        endif()
        add_test(NAME ivf_posting_lists COMMAND ivf_list_tests
            "${CMAKE_CURRENT_BINARY_DIR}/list_fixtures")
        add_executable(ivf_router_tests ivf_router_tests.cpp)
        target_link_libraries(ivf_router_tests PRIVATE ivf_router)
        if(NOT MSVC)
            target_compile_options(ivf_router_tests PRIVATE -fno-fast-math)
        endif()
        add_test(NAME ivf_centroid_routing COMMAND ivf_router_tests
            "${CMAKE_CURRENT_BINARY_DIR}/router_fixtures")
        # Explicit resource smoke; not part of routine CTest or canonical metrics.
        add_executable(ivf_centroid_scale_smoke ivf_centroid_scale_smoke.cpp)
        target_link_libraries(ivf_centroid_scale_smoke PRIVATE ivf_router)
        if(NOT MSVC)
            target_compile_options(ivf_centroid_scale_smoke PRIVATE -fno-fast-math)
        endif()
    endif()
    if(TARGET BufANN AND NOT MSVC)
        if(TARGET ivf_search)
            add_executable(ivf_mutable_index_tests ivf_mutable_index_tests.cpp)
            target_link_libraries(ivf_mutable_index_tests PRIVATE ivf_index Threads::Threads)
            target_compile_options(ivf_mutable_index_tests PRIVATE -fno-fast-math)
            add_test(NAME ivf_mutable_index_session COMMAND ivf_mutable_index_tests
                "${CMAKE_CURRENT_BINARY_DIR}/mutable_index_fixtures")
            add_executable(ivf_search_tests ivf_search_tests.cpp)
            target_link_libraries(ivf_search_tests PRIVATE ivf_search Threads::Threads)
            target_link_libraries(ivf_search_tests PRIVATE ivf_index)
            target_compile_options(ivf_search_tests PRIVATE -fno-fast-math)
            add_test(NAME ivf_query_composition COMMAND ivf_search_tests
                "${CMAKE_CURRENT_BINARY_DIR}/search_fixtures")
        endif()
        add_executable(ivf_heap_tests ivf_heap_tests.cpp)
        target_link_libraries(ivf_heap_tests PRIVATE BufANN aio -ltcmalloc)
        target_compile_options(ivf_heap_tests PRIVATE -fno-fast-math)
        add_test(NAME ivf_heap_storage COMMAND ivf_heap_tests
            "${CMAKE_CURRENT_BINARY_DIR}/heap_fixtures")
        add_executable(ivf_heap_adapter_tests ivf_heap_adapter_tests.cpp)
        target_link_libraries(ivf_heap_adapter_tests PRIVATE ivf_heap Threads::Threads)
        target_compile_options(ivf_heap_adapter_tests PRIVATE -fno-fast-math)
        add_test(NAME ivf_heap_adapter COMMAND ivf_heap_adapter_tests
            "${CMAKE_CURRENT_BINARY_DIR}/adapter_fixtures")
    endif()
    add_executable(ivf_contract_tests ivf_contract_tests.cpp
        "${IVF_ROOT}/src/logger.cpp" "${IVF_ROOT}/src/ann_exception.cpp")
    target_link_libraries(ivf_contract_tests PRIVATE ivf_contracts Threads::Threads)
    if(NOT MSVC)
        target_compile_options(ivf_contract_tests PRIVATE -fno-fast-math)
    endif()
    add_test(NAME ivf_contracts COMMAND ivf_contract_tests "${CMAKE_CURRENT_BINARY_DIR}/fixtures")
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    if(TARGET ivf_router_diagnostics)
        add_test(NAME ivf_patched_router_diagnostics COMMAND ${Python3_EXECUTABLE}
            "${CMAKE_CURRENT_SOURCE_DIR}/test_router_diagnostics.py"
            --fixtures $<TARGET_FILE:ivf_patched_router_tests> --diagnostic $<TARGET_FILE:ivf_router_diagnostics>
            --scratch "${CMAKE_CURRENT_BINARY_DIR}/router_diagnostic_fixtures")
        set_tests_properties(ivf_patched_router_diagnostics PROPERTIES TIMEOUT 180)
    endif()
    if(TARGET ivf_training_diagnostics)
        add_test(NAME ivf_training_diagnostics COMMAND ${Python3_EXECUTABLE}
            "${CMAKE_CURRENT_SOURCE_DIR}/test_training_diagnostics.py"
            --trainer $<TARGET_FILE:ivf_training_diagnostics>
            --scratch "${CMAKE_CURRENT_BINARY_DIR}/training_diagnostic_fixtures")
    endif()
    add_executable(ivf_pq_helper_parity ivf_pq_helper_parity.cpp
        "${IVF_ROOT}/src/logger.cpp" "${IVF_ROOT}/src/ann_exception.cpp")
    target_link_libraries(ivf_pq_helper_parity PRIVATE ivf_contracts)
    if(NOT MSVC)
        target_compile_options(ivf_pq_helper_parity PRIVATE -fno-fast-math)
    endif()
    if(TARGET ivf_cli_execution)
        add_test(NAME ivf_cost_model_cli COMMAND ${Python3_EXECUTABLE}
            "${CMAKE_CURRENT_SOURCE_DIR}/test_cost_model.py"
            --tool $<TARGET_FILE:ivf_cost_model> --build-dir "${CMAKE_BINARY_DIR}"
            --scratch "${CMAKE_CURRENT_BINARY_DIR}/cost_model_fixtures")
        set_tests_properties(ivf_cost_model_cli PROPERTIES TIMEOUT 180)
        add_test(NAME ivf_ann_training_cli COMMAND ${Python3_EXECUTABLE}
            "${CMAKE_CURRENT_SOURCE_DIR}/test_s09_cli.py"
            --build $<TARGET_FILE:ivf_build> --driver $<TARGET_FILE:ivf_driver>
            --scratch "${CMAKE_CURRENT_BINARY_DIR}/s09_cli_fixtures")
        set_tests_properties(ivf_ann_training_cli PROPERTIES TIMEOUT 180)
        add_test(NAME ivf_patched_cli_smoke COMMAND ${Python3_EXECUTABLE}
            "${CMAKE_CURRENT_SOURCE_DIR}/test_s08_cli.py"
            --build $<TARGET_FILE:ivf_build> --driver $<TARGET_FILE:ivf_driver>
            --scratch "${CMAKE_CURRENT_BINARY_DIR}/s08_cli_fixtures")
        set_tests_properties(ivf_patched_cli_smoke PROPERTIES TIMEOUT 180)
        set(IVF_PQ_PIVOT_HELPER "" CACHE FILEPATH "Optional existing standalone PQ pivot helper for trained integration tests")
        set(IVF_PQ_CODES_HELPER "" CACHE FILEPATH "Optional existing standalone PQ code helper for trained integration tests")
        if(IVF_PQ_PIVOT_HELPER OR IVF_PQ_CODES_HELPER)
            if(NOT EXISTS "${IVF_PQ_PIVOT_HELPER}" OR NOT EXISTS "${IVF_PQ_CODES_HELPER}")
                message(FATAL_ERROR "Set both IVF_PQ_PIVOT_HELPER and IVF_PQ_CODES_HELPER to existing binaries")
            endif()
            add_test(NAME ivf_trained_helper_integration COMMAND ${Python3_EXECUTABLE}
                "${CMAKE_CURRENT_SOURCE_DIR}/test_trained_helpers.py"
                --pivot-helper "${IVF_PQ_PIVOT_HELPER}" --codes-helper "${IVF_PQ_CODES_HELPER}"
                --parity $<TARGET_FILE:ivf_pq_helper_parity> --build-dir "${CMAKE_BINARY_DIR}"
                --scratch "${CMAKE_CURRENT_BINARY_DIR}/trained_fixtures")
        endif()
        add_test(NAME ivf_query_harness COMMAND ${Python3_EXECUTABLE}
            "${CMAKE_CURRENT_SOURCE_DIR}/test_harness.py"
            --build-dir "${CMAKE_BINARY_DIR}" --scratch "${CMAKE_CURRENT_BINARY_DIR}/harness_fixtures")
        add_test(NAME ivf_mixed_execution COMMAND ${Python3_EXECUTABLE}
            "${CMAKE_CURRENT_SOURCE_DIR}/test_mixed.py"
            --build-dir "${CMAKE_BINARY_DIR}" --scratch "${CMAKE_CURRENT_BINARY_DIR}/mixed_fixtures")
        set_tests_properties(ivf_mixed_execution PROPERTIES TIMEOUT 300)
        add_test(NAME ivf_cli_execution COMMAND ${Python3_EXECUTABLE}
            "${CMAKE_CURRENT_SOURCE_DIR}/test_cli_execution.py"
            --build $<TARGET_FILE:ivf_build> --driver $<TARGET_FILE:ivf_driver>
            --scratch "${CMAKE_CURRENT_BINARY_DIR}/execution_fixtures")
        foreach(mode patched-full refined pca pca-refined)
            string(REPLACE "-" "_" label "${mode}")
            add_test(NAME ivf_${label}_query_composition COMMAND ivf_search_tests
                "${CMAKE_CURRENT_BINARY_DIR}/${label}_search_fixtures" "${mode}")
            add_test(NAME ivf_${label}_mutable_session COMMAND ivf_mutable_index_tests
                "${CMAKE_CURRENT_BINARY_DIR}/${label}_mutable_fixtures" "${mode}")
            set_tests_properties(ivf_${label}_query_composition ivf_${label}_mutable_session PROPERTIES TIMEOUT 300)
        endforeach()
    endif()
    add_test(NAME ivf_cli_contracts COMMAND ${Python3_EXECUTABLE}
        "${CMAKE_CURRENT_SOURCE_DIR}/test_cli.py"
        --build $<TARGET_FILE:ivf_build> --driver $<TARGET_FILE:ivf_driver>
        --contracts $<TARGET_FILE:ivf_contract_tests>
        --scratch "${CMAKE_CURRENT_BINARY_DIR}/cli_fixtures")
endif()
