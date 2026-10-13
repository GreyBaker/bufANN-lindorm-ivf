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
option(IVF_BUILD_CONTRACT_TESTS
       "Build IVF contract, training, routing, storage, and integration tests" ON)
include(${CMAKE_CURRENT_SOURCE_DIR}/tools.cmake)

if(IVF_BUILD_CONTRACT_TESTS)
    enable_testing()
    add_executable(ivf_artifact_tests ivf_artifact_tests.cpp)
    target_link_libraries(ivf_artifact_tests PRIVATE ivf_contracts)
    add_test(NAME ivf_artifact_utilities COMMAND ivf_artifact_tests
                                                 "${CMAKE_CURRENT_BINARY_DIR}/artifact_fixtures")
    add_executable(ivf_mixed_scheduler_tests ivf_mixed_scheduler_tests.cpp)
    target_link_libraries(ivf_mixed_scheduler_tests PRIVATE ivf_numeric_policy Threads::Threads)
    add_test(NAME ivf_mixed_scheduler COMMAND ivf_mixed_scheduler_tests)
    set_tests_properties(ivf_mixed_scheduler PROPERTIES TIMEOUT 30)
    add_executable(ivf_list_routing_tests ivf_list_routing_tests.cpp)
    target_link_libraries(ivf_list_routing_tests PRIVATE ivf_numeric_policy Threads::Threads)
    add_test(NAME ivf_list_routing_schedule COMMAND ivf_list_routing_tests)
    set_tests_properties(ivf_list_routing_schedule PROPERTIES TIMEOUT 30)
    add_executable(ivf_insert_scheduler_tests ivf_insert_scheduler_tests.cpp)
    target_link_libraries(ivf_insert_scheduler_tests PRIVATE ivf_numeric_policy Threads::Threads)
    add_test(NAME ivf_insert_scheduler COMMAND ivf_insert_scheduler_tests)
    set_tests_properties(ivf_insert_scheduler PROPERTIES TIMEOUT 30)
    add_executable(ivf_insert_input_tests ivf_insert_input_tests.cpp)
    target_link_libraries(ivf_insert_input_tests PRIVATE ivf_numeric_policy Threads::Threads)
    add_test(NAME ivf_insert_input COMMAND ivf_insert_input_tests)
    set_tests_properties(ivf_insert_input PROPERTIES TIMEOUT 30)
    add_executable(ivf_driver_progress_tests ivf_driver_progress_tests.cpp)
    target_link_libraries(ivf_driver_progress_tests PRIVATE Threads::Threads)
    add_test(NAME ivf_driver_progress COMMAND ivf_driver_progress_tests)
    set_tests_properties(ivf_driver_progress PROPERTIES TIMEOUT 30)
    add_executable(ivf_fetch_timing_tests ivf_fetch_timing_tests.cpp)
    target_include_directories(ivf_fetch_timing_tests PRIVATE "${IVF_ROOT}/include")
    target_link_libraries(ivf_fetch_timing_tests PRIVATE ivf_numeric_policy Threads::Threads)
    add_test(NAME ivf_fetch_timing COMMAND ivf_fetch_timing_tests)
    set_tests_properties(ivf_fetch_timing PROPERTIES TIMEOUT 30)
    add_executable(ivf_pin_admission_tests ivf_pin_admission_tests.cpp)
    target_include_directories(ivf_pin_admission_tests PRIVATE "${IVF_ROOT}/include")
    target_link_libraries(ivf_pin_admission_tests PRIVATE ivf_numeric_policy Threads::Threads)
    add_test(NAME ivf_pin_admission COMMAND ivf_pin_admission_tests)
    set_tests_properties(ivf_pin_admission PROPERTIES TIMEOUT 30)
    add_executable(ivf_centroid_tests ivf_centroid_tests.cpp)
    target_link_libraries(ivf_centroid_tests PRIVATE ivf_centroids)
    add_test(NAME ivf_centroid_training COMMAND ivf_centroid_tests
                                                "${CMAKE_CURRENT_BINARY_DIR}/centroid_fixtures")
    # Keep the existing combined core/scan fixture with routing. The isolated fixture links only the
    # std-only core, proving OFF has no list dependency.
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
        target_link_libraries(ivf_ann_training_tests PRIVATE ivf_router ivf_hnswlib_headers
                                                             Threads::Threads)
        add_test(NAME ivf_ann_training COMMAND ivf_ann_training_tests
                                               "${CMAKE_CURRENT_BINARY_DIR}/ann_training_fixtures")
        set_tests_properties(ivf_ann_training PROPERTIES TIMEOUT 180)
        add_executable(ivf_patched_router_tests ivf_patched_router_tests.cpp)
        target_link_libraries(ivf_patched_router_tests PRIVATE ivf_router Threads::Threads)
        add_test(NAME ivf_patched_centroid_routing
                 COMMAND ivf_patched_router_tests
                         "${CMAKE_CURRENT_BINARY_DIR}/patched_router_fixtures")
        set_tests_properties(ivf_patched_centroid_routing PROPERTIES TIMEOUT 180)
        add_executable(ivf_mutable_list_tests ivf_mutable_list_tests.cpp)
        target_link_libraries(ivf_mutable_list_tests PRIVATE ivf_lists Threads::Threads)
        add_test(NAME ivf_mutable_posting_lists
                 COMMAND ivf_mutable_list_tests "${CMAKE_CURRENT_BINARY_DIR}/mutable_list_fixtures")
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
            add_test(NAME ivf_mutable_index_session
                     COMMAND ivf_mutable_index_tests
                             "${CMAKE_CURRENT_BINARY_DIR}/mutable_index_fixtures")
            add_executable(ivf_search_tests ivf_search_tests.cpp)
            target_link_libraries(ivf_search_tests PRIVATE ivf_search ivf_index Threads::Threads)
            add_test(NAME ivf_query_composition
                     COMMAND ivf_search_tests "${CMAKE_CURRENT_BINARY_DIR}/search_fixtures")
        endif()
        add_executable(ivf_heap_tests ivf_heap_tests.cpp)
        target_link_libraries(ivf_heap_tests PRIVATE ivf_numeric_policy BufANN aio -ltcmalloc)
        add_test(NAME ivf_heap_storage COMMAND ivf_heap_tests
                                               "${CMAKE_CURRENT_BINARY_DIR}/heap_fixtures")
        add_executable(ivf_heap_adapter_tests ivf_heap_adapter_tests.cpp)
        target_link_libraries(ivf_heap_adapter_tests PRIVATE ivf_heap Threads::Threads)
        add_test(NAME ivf_heap_adapter COMMAND ivf_heap_adapter_tests
                                               "${CMAKE_CURRENT_BINARY_DIR}/adapter_fixtures")
        add_executable(ivf_direct_io_tests ivf_direct_io_tests.cpp)
        target_link_libraries(ivf_direct_io_tests PRIVATE ivf_heap Threads::Threads)
        add_test(NAME ivf_direct_io COMMAND ivf_direct_io_tests
                                            "${CMAKE_CURRENT_BINARY_DIR}/direct_io_fixtures")
        set_tests_properties(ivf_direct_io PROPERTIES TIMEOUT 60)
        add_executable(ivf_read_io_tests ivf_read_io_tests.cpp)
        target_link_libraries(ivf_read_io_tests PRIVATE ivf_heap Threads::Threads)
        add_test(NAME ivf_read_io COMMAND ivf_read_io_tests
                                         "${CMAKE_CURRENT_BINARY_DIR}/read_io_fixtures")
        set_tests_properties(ivf_read_io PROPERTIES TIMEOUT 30)
        add_executable(ivf_exact_aio_tests ivf_exact_aio_tests.cpp)
        target_include_directories(ivf_exact_aio_tests PRIVATE "${IVF_ROOT}/include")
        target_link_libraries(ivf_exact_aio_tests PRIVATE ivf_numeric_policy aio)
        add_test(NAME ivf_exact_aio COMMAND ivf_exact_aio_tests)
        set_tests_properties(ivf_exact_aio PROPERTIES TIMEOUT 30)
        add_executable(ivf_batch_read_io_tests ivf_batch_read_io_tests.cpp)
        target_link_libraries(ivf_batch_read_io_tests PRIVATE ivf_heap Threads::Threads)
        add_test(NAME ivf_batch_read_io COMMAND ivf_batch_read_io_tests
                                              "${CMAKE_CURRENT_BINARY_DIR}/batch_read_io_fixtures")
        set_tests_properties(ivf_batch_read_io PROPERTIES TIMEOUT 30)
        add_executable(ivf_mutable_locality_tests ivf_mutable_locality_tests.cpp)
        target_link_libraries(ivf_mutable_locality_tests PRIVATE ivf_heap Threads::Threads)
        add_test(NAME ivf_mutable_locality COMMAND ivf_mutable_locality_tests
                                                 "${CMAKE_CURRENT_BINARY_DIR}/mutable_locality_fixtures")
        set_tests_properties(ivf_mutable_locality PROPERTIES TIMEOUT 120)
        add_executable(ivf_concurrent_fetch_tests ivf_concurrent_fetch_tests.cpp)
        target_link_libraries(ivf_concurrent_fetch_tests PRIVATE ivf_heap Threads::Threads)
        add_test(NAME ivf_concurrent_fetch COMMAND ivf_concurrent_fetch_tests
                                                "${CMAKE_CURRENT_BINARY_DIR}/concurrent_fetch_fixtures")
        set_tests_properties(ivf_concurrent_fetch PROPERTIES TIMEOUT 120)
    endif()
    add_executable(ivf_contract_tests ivf_contract_tests.cpp "${IVF_ROOT}/src/logger.cpp"
                                      "${IVF_ROOT}/src/ann_exception.cpp")
    target_link_libraries(ivf_contract_tests PRIVATE ivf_contracts Threads::Threads)
    add_test(NAME ivf_contracts COMMAND ivf_contract_tests "${CMAKE_CURRENT_BINARY_DIR}/fixtures")
    find_package(
        Python3
        COMPONENTS Interpreter
        REQUIRED)
    add_test(
        NAME ivf_cmake_scope
        COMMAND
            ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/test_cmake_scope.py" --cmake
            "${CMAKE_COMMAND}" --module "${IVF_ROOT}/cmake/IVF.cmake" --scratch
            "${CMAKE_CURRENT_BINARY_DIR}/cmake_scope_fixtures")
    set_tests_properties(ivf_cmake_scope PROPERTIES TIMEOUT 60)
    if(UNIX)
        add_test(NAME ivf_prepare_guards
                 COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/test_prepare_guards.py"
                         --repo "${IVF_ROOT}")
    endif()
    add_test(NAME ivf_shared_orchestration
             COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/test_orchestration.py"
                     --scratch "${CMAKE_CURRENT_BINARY_DIR}/orchestration_fixtures")
    set_tests_properties(ivf_shared_orchestration PROPERTIES TIMEOUT 30)
    add_test(NAME ivf_sift_download
             COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/test_sift_download.py")
    set_tests_properties(ivf_sift_download PROPERTIES TIMEOUT 30)
    add_test(NAME ivf_gt_coverage
             COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/test_gt_coverage.py")
    set_tests_properties(ivf_gt_coverage PROPERTIES TIMEOUT 60)
    add_test(NAME ivf_gt_depth
             COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/test_gt_depth.py")
    set_tests_properties(ivf_gt_depth PROPERTIES TIMEOUT 120 SKIP_RETURN_CODE 77)
    add_test(NAME ivf_gt_management
             COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/test_gt_management.py")
    set_tests_properties(ivf_gt_management PROPERTIES TIMEOUT 120)
    set(IVF_DEPENDENCY_TEST_SOURCE_ARGS)
    if(TARGET ivf_hnswlib_headers)
        get_target_property(IVF_DEPENDENCY_TEST_SOURCE ivf_hnswlib_headers IVF_UPSTREAM_SOURCE_DIR)
        list(APPEND IVF_DEPENDENCY_TEST_SOURCE_ARGS --source "${IVF_DEPENDENCY_TEST_SOURCE}")
    endif()
    add_test(
        NAME ivf_cmake_routing_no_python
        COMMAND
            ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/test_cmake_routing.py" --cmake
            "${CMAKE_COMMAND}" --repo "${IVF_ROOT}" --scratch
            "${CMAKE_CURRENT_BINARY_DIR}/cmake_routing_fixtures" ${IVF_DEPENDENCY_TEST_SOURCE_ARGS})
    set_tests_properties(ivf_cmake_routing_no_python PROPERTIES TIMEOUT 120)
    # OFF checks the local variant without fetching; ON verifies stock upstream.
    add_test(
        NAME ivf_hnswlib_dependency
        COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/test_hnswlib_dependency.py"
                --dependency "${IVF_ROOT}/dependencies/hnswlib" ${IVF_DEPENDENCY_TEST_SOURCE_ARGS})
    set_tests_properties(ivf_hnswlib_dependency PROPERTIES TIMEOUT 30)
    if(TARGET ivf_router_diagnostics)
        add_test(
            NAME ivf_patched_router_diagnostics
            COMMAND
                ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/test_router_diagnostics.py"
                --fixtures $<TARGET_FILE:ivf_patched_router_tests> --diagnostic
                $<TARGET_FILE:ivf_router_diagnostics> --scratch
                "${CMAKE_CURRENT_BINARY_DIR}/router_diagnostic_fixtures")
        set_tests_properties(ivf_patched_router_diagnostics PROPERTIES TIMEOUT 180)
    endif()
    if(TARGET ivf_training_diagnostics)
        add_test(
            NAME ivf_training_diagnostics
            COMMAND
                ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/test_training_diagnostics.py"
                --trainer $<TARGET_FILE:ivf_training_diagnostics> --scratch
                "${CMAKE_CURRENT_BINARY_DIR}/training_diagnostic_fixtures")
    endif()
    add_executable(ivf_pq_helper_parity ivf_pq_helper_parity.cpp "${IVF_ROOT}/src/logger.cpp"
                                        "${IVF_ROOT}/src/ann_exception.cpp")
    target_link_libraries(ivf_pq_helper_parity PRIVATE ivf_contracts)
    if(TARGET ivf_cli_execution)
        add_test(NAME ivf_parallel_insert
                 COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/test_parallel_insert.py"
                         --build $<TARGET_FILE:ivf_build> --driver $<TARGET_FILE:ivf_driver>
                         --scratch "${CMAKE_CURRENT_BINARY_DIR}/parallel_insert_fixtures")
        set_tests_properties(ivf_parallel_insert PROPERTIES TIMEOUT 240)
        add_test(NAME ivf_insert_input_policy
                 COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/test_insert_policy.py"
                         --build $<TARGET_FILE:ivf_build> --driver $<TARGET_FILE:ivf_driver>
                         --scratch "${CMAKE_CURRENT_BINARY_DIR}/insert_policy_fixtures")
        set_tests_properties(ivf_insert_input_policy PROPERTIES TIMEOUT 240)
        add_test(NAME ivf_parallel_list_build
                 COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/test_list_parallel_build.py"
                         --build $<TARGET_FILE:ivf_build>
                         --scratch "${CMAKE_CURRENT_BINARY_DIR}/parallel_list_fixtures")
        set_tests_properties(ivf_parallel_list_build PROPERTIES TIMEOUT 180)
        add_test(
            NAME ivf_cost_model_cli
            COMMAND
                ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/test_cost_model.py" --tool
                $<TARGET_FILE:ivf_cost_model> --build-dir "${CMAKE_BINARY_DIR}" --scratch
                "${CMAKE_CURRENT_BINARY_DIR}/cost_model_fixtures")
        set_tests_properties(ivf_cost_model_cli PROPERTIES TIMEOUT 180)
        add_test(
            NAME ivf_ann_training_cli
            COMMAND
                ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/test_s09_cli.py" --build
                $<TARGET_FILE:ivf_build> --driver $<TARGET_FILE:ivf_driver> --scratch
                "${CMAKE_CURRENT_BINARY_DIR}/s09_cli_fixtures")
        set_tests_properties(ivf_ann_training_cli PROPERTIES TIMEOUT 180)
        add_test(
            NAME ivf_patched_cli_smoke
            COMMAND
                ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/test_s08_cli.py" --build
                $<TARGET_FILE:ivf_build> --driver $<TARGET_FILE:ivf_driver> --scratch
                "${CMAKE_CURRENT_BINARY_DIR}/s08_cli_fixtures")
        set_tests_properties(ivf_patched_cli_smoke PROPERTIES TIMEOUT 180)
        set(IVF_PQ_PIVOT_HELPER
            ""
            CACHE FILEPATH
                  "Optional existing standalone PQ pivot helper for trained integration tests")
        set(IVF_PQ_CODES_HELPER
            ""
            CACHE FILEPATH
                  "Optional existing standalone PQ code helper for trained integration tests")
        if(IVF_PQ_PIVOT_HELPER OR IVF_PQ_CODES_HELPER)
            if(NOT EXISTS "${IVF_PQ_PIVOT_HELPER}" OR NOT EXISTS "${IVF_PQ_CODES_HELPER}")
                message(
                    FATAL_ERROR
                        "Set both IVF_PQ_PIVOT_HELPER and IVF_PQ_CODES_HELPER to existing binaries")
            endif()
            add_test(
                NAME ivf_trained_helper_integration
                COMMAND
                    ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/test_trained_helpers.py"
                    --pivot-helper "${IVF_PQ_PIVOT_HELPER}" --codes-helper "${IVF_PQ_CODES_HELPER}"
                    --parity $<TARGET_FILE:ivf_pq_helper_parity> --build-dir "${CMAKE_BINARY_DIR}"
                    --scratch "${CMAKE_CURRENT_BINARY_DIR}/trained_fixtures")
        endif()
        add_test(
            NAME ivf_query_harness
            COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/test_harness.py" --build-dir
                    "${CMAKE_BINARY_DIR}" --scratch "${CMAKE_CURRENT_BINARY_DIR}/harness_fixtures")
        add_test(
            NAME ivf_mixed_execution
            COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/test_mixed.py" --build-dir
                    "${CMAKE_BINARY_DIR}" --scratch "${CMAKE_CURRENT_BINARY_DIR}/mixed_fixtures")
        set_tests_properties(ivf_mixed_execution PROPERTIES TIMEOUT 300)
        add_test(
            NAME ivf_cli_execution
            COMMAND
                ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/test_cli_execution.py" --build
                $<TARGET_FILE:ivf_build> --driver $<TARGET_FILE:ivf_driver> --scratch
                "${CMAKE_CURRENT_BINARY_DIR}/execution_fixtures")
        foreach(mode patched-full refined pca pca-refined)
            string(REPLACE "-" "_" label "${mode}")
            add_test(NAME ivf_${label}_query_composition
                     COMMAND ivf_search_tests
                             "${CMAKE_CURRENT_BINARY_DIR}/${label}_search_fixtures" "${mode}")
            add_test(NAME ivf_${label}_mutable_session
                     COMMAND ivf_mutable_index_tests
                             "${CMAKE_CURRENT_BINARY_DIR}/${label}_mutable_fixtures" "${mode}")
            set_tests_properties(ivf_${label}_query_composition ivf_${label}_mutable_session
                                 PROPERTIES TIMEOUT 300)
        endforeach()
    endif()
    add_test(
        NAME ivf_cli_contracts
        COMMAND
            ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/test_cli.py" --build
            $<TARGET_FILE:ivf_build> --driver $<TARGET_FILE:ivf_driver> --contracts
            $<TARGET_FILE:ivf_contract_tests> --scratch "${CMAKE_CURRENT_BINARY_DIR}/cli_fixtures")
endif()
