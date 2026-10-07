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
find_package(Threads REQUIRED)

foreach(tool ivf_build ivf_driver)
    add_executable(${tool} "${IVF_ROOT}/tests/${tool}.cpp")
    target_link_libraries(${tool} PRIVATE ivf_contracts)
    set_target_properties(${tool} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests")
    if(NOT MSVC)
        target_compile_options(${tool} PRIVATE -fno-fast-math)
    endif()
endforeach()

option(IVF_BUILD_CONTRACT_TESTS "Build IVF input and PQ contract tests" ON)
if(IVF_BUILD_CONTRACT_TESTS)
    enable_testing()
    if(TARGET BufANN AND NOT MSVC)
        add_executable(ivf_heap_tests ivf_heap_tests.cpp)
        target_link_libraries(ivf_heap_tests PRIVATE BufANN aio -ltcmalloc)
        target_compile_options(ivf_heap_tests PRIVATE -fno-fast-math)
        add_test(NAME ivf_heap_storage COMMAND ivf_heap_tests
            "${CMAKE_CURRENT_BINARY_DIR}/heap_fixtures")
    endif()
    add_executable(ivf_contract_tests ivf_contract_tests.cpp
        "${IVF_ROOT}/src/logger.cpp" "${IVF_ROOT}/src/ann_exception.cpp")
    target_link_libraries(ivf_contract_tests PRIVATE ivf_contracts Threads::Threads)
    if(NOT MSVC)
        target_compile_options(ivf_contract_tests PRIVATE -fno-fast-math)
    endif()
    add_test(NAME ivf_contracts COMMAND ivf_contract_tests "${CMAKE_CURRENT_BINARY_DIR}/fixtures")
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    add_test(NAME ivf_cli_contracts COMMAND ${Python3_EXECUTABLE}
        "${CMAKE_CURRENT_SOURCE_DIR}/test_cli.py"
        --build $<TARGET_FILE:ivf_build> --driver $<TARGET_FILE:ivf_driver>
        --contracts $<TARGET_FILE:ivf_contract_tests>
        --scratch "${CMAKE_CURRENT_BINARY_DIR}/cli_fixtures")
endif()
