cmake_minimum_required(VERSION 3.16)

# This subdirectory also configures on its own without MKL/libaio/folly.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(IVFContracts LANGUAGES CXX)
    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS OFF)
    enable_testing()
endif()

include(${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/IVF.cmake)
option(IVF_BUILD_CONTRACT_TESTS "Build IVF contract, training, routing, storage, and integration tests" ON)
include(${CMAKE_CURRENT_SOURCE_DIR}/tools.cmake)

if(IVF_BUILD_CONTRACT_TESTS)
    enable_testing()
    add_executable(ivf_mixed_scheduler_tests ivf_mixed_scheduler_tests.cpp)
    target_link_libraries(ivf_mixed_scheduler_tests PRIVATE ivf_numeric_policy Threads::Threads)
    add_test(NAME ivf_mixed_scheduler COMMAND ivf_mixed_scheduler_tests)
    set_tests_properties(ivf_mixed_scheduler PROPERTIES TIMEOUT 30)
    add_executable(ivf_centroid_tests ivf_centroid_tests.cpp)
    target_link_libraries(ivf_centroid_tests PRIVATE ivf_centroids)
    add_test(NAME ivf_centroid_training COMMAND ivf_centroid_tests
        "${CMAKE_CURRENT_BINARY_DIR}/centroid_fixtures")
    # Keep the existing combined core/scan fixture with routing. The isolated
    # fixture links only the std-only core, proving OFF has no list dependency.
    if(TARGET ivf_lists)
        add_executable(ivf_cost_model_tests ivf_cost_model_tests.cpp)
        target_link_libraries(ivf_cost_model_tests PRIVATE ivf_lists)
    else()
        add_executable(ivf_cost_model_tests ivf_cost_model_core_tests.cpp)
    endif()
    target_link_libraries(ivf_cost_model_tests PRIVATE ivf_cost_model_core)
    add_test(NAME ivf_cost_model_core COMMAND ivf_cost_model_tests)
    if(TARGET ivf_router)
        add_executable(ivf_ann_training_tests ivf_ann_training_tests.cpp)
        target_link_libraries(ivf_ann_training_tests PRIVATE ivf_router ivf_hnswlib_headers Threads::Threads)
        add_test(NAME ivf_ann_training COMMAND ivf_ann_training_tests "${CMAKE_CURRENT_BINARY_DIR}/ann_training_fixtures")
        set_tests_properties(ivf_ann_training PROPERTIES TIMEOUT 180)
        add_executable(ivf_patched_router_tests ivf_patched_router_tests.cpp)
        target_link_libraries(ivf_patched_router_tests PRIVATE ivf_router Threads::Threads)
        add_test(NAME ivf_patched_centroid_routing COMMAND ivf_patched_router_tests
            "${CMAKE_CURRENT_BINARY_DIR}/patched_router_fixtures")
        set_tests_properties(ivf_patched_centroid_routing PROPERTIES TIMEOUT 180)
        add_executable(ivf_mutable_list_tests ivf_mutable_list_tests.cpp)
        target_link_libraries(ivf_mutable_list_tests PRIVATE ivf_lists Threads::Threads)
        add_test(NAME ivf_mutable_posting_lists COMMAND ivf_mutable_list_tests
            "${CMAKE_CURRENT_BINARY_DIR}/mutable_list_fixtures")
        add_executable(ivf_list_tests ivf_list_tests.cpp)
        target_link_libraries(ivf_list_tests PRIVATE ivf_lists)
        add_test(NAME ivf_posting_lists COMMAND ivf_list_tests
            "${CMAKE_CURRENT_BINARY_DIR}/list_fixtures")
        add_executable(ivf_router_tests ivf_router_tests.cpp)
        target_link_libraries(ivf_router_tests PRIVATE ivf_router)
        add_test(NAME ivf_centroid_routing COMMAND ivf_router_tests
            "${CMAKE_CURRENT_BINARY_DIR}/router_fixtures")
    endif()
    if(TARGET BufANN AND NOT MSVC)
        if(TARGET ivf_search)
            add_executable(ivf_mutable_index_tests ivf_mutable_index_tests.cpp)
            target_link_libraries(ivf_mutable_index_tests PRIVATE ivf_index Threads::Threads)
            add_test(NAME ivf_mutable_index_session COMMAND ivf_mutable_index_tests
                "${CMAKE_CURRENT_BINARY_DIR}/mutable_index_fixtures")
            add_executable(ivf_search_tests ivf_search_tests.cpp)
            target_link_libraries(ivf_search_tests PRIVATE ivf_search ivf_index Threads::Threads)
            add_test(NAME ivf_query_composition COMMAND ivf_search_tests
                "${CMAKE_CURRENT_BINARY_DIR}/search_fixtures")
        endif()
        add_executable(ivf_heap_tests ivf_heap_tests.cpp)
        target_link_libraries(ivf_heap_tests PRIVATE ivf_numeric_policy BufANN aio -ltcmalloc)
        add_test(NAME ivf_heap_storage COMMAND ivf_heap_tests
            "${CMAKE_CURRENT_BINARY_DIR}/heap_fixtures")
        add_executable(ivf_heap_adapter_tests ivf_heap_adapter_tests.cpp)
        target_link_libraries(ivf_heap_adapter_tests PRIVATE ivf_heap Threads::Threads)
        add_test(NAME ivf_heap_adapter COMMAND ivf_heap_adapter_tests
            "${CMAKE_CURRENT_BINARY_DIR}/adapter_fixtures")
    endif()
    add_executable(ivf_contract_tests ivf_contract_tests.cpp
        "${IVF_ROOT}/src/logger.cpp" "${IVF_ROOT}/src/ann_exception.cpp")
    target_link_libraries(ivf_contract_tests PRIVATE ivf_contracts Threads::Threads)
    add_test(NAME ivf_contracts COMMAND ivf_contract_tests "${CMAKE_CURRENT_BINARY_DIR}/fixtures")
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    add_test(NAME ivf_shared_orchestration COMMAND ${Python3_EXECUTABLE}
        "${CMAKE_CURRENT_SOURCE_DIR}/test_orchestration.py"
        --scratch "${CMAKE_CURRENT_BINARY_DIR}/orchestration_fixtures")
    set_tests_properties(ivf_shared_orchestration PROPERTIES TIMEOUT 30)
    set(IVF_DEPENDENCY_TEST_SOURCE_ARGS)
    if(TARGET ivf_hnswlib_headers)
        get_target_property(IVF_DEPENDENCY_TEST_SOURCE ivf_hnswlib_headers IVF_UPSTREAM_SOURCE_DIR)
        list(APPEND IVF_DEPENDENCY_TEST_SOURCE_ARGS --source "${IVF_DEPENDENCY_TEST_SOURCE}")
    endif()
    # OFF checks the local variant without fetching; ON verifies stock upstream.
    add_test(NAME ivf_hnswlib_dependency COMMAND ${Python3_EXECUTABLE}
        "${CMAKE_CURRENT_SOURCE_DIR}/test_hnswlib_dependency.py"
        --dependency "${IVF_ROOT}/dependencies/hnswlib"
        ${IVF_DEPENDENCY_TEST_SOURCE_ARGS})
    set_tests_properties(ivf_hnswlib_dependency PROPERTIES TIMEOUT 30)
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
