# Operational tools keep their historical names and build/tests output paths.
# Their definitions do not depend on enabling CTest registration.
foreach(tool ivf_build ivf_driver)
    add_executable(${tool} "${IVF_ROOT}/tests/${tool}.cpp")
    target_link_libraries(${tool} PRIVATE ivf_contracts)
    set_target_properties(${tool} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests")
endforeach()

# Native execution requires the actual Linux heap backend and opt-in routing.
if(TARGET ivf_index AND NOT MSVC)
    add_executable(ivf_cost_model "${IVF_ROOT}/tests/ivf_cost_model.cpp")
    target_link_libraries(ivf_cost_model PRIVATE ivf_cost_model_core ivf_lists BufANN)
    set_target_properties(ivf_cost_model PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests")
    add_library(ivf_cli_execution STATIC "${IVF_ROOT}/tests/ivf_cli_execution.cpp")
    target_link_libraries(ivf_cli_execution PUBLIC ivf_index Threads::Threads)
    foreach(tool ivf_build ivf_driver)
        target_compile_definitions(${tool} PRIVATE IVF_NATIVE_EXECUTION=1)
        target_link_libraries(${tool} PRIVATE ivf_cli_execution)
    endforeach()
endif()

if(TARGET ivf_router AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
    add_executable(ivf_router_diagnostics "${IVF_ROOT}/tests/ivf_router_diagnostics.cpp")
    target_link_libraries(ivf_router_diagnostics PRIVATE ivf_router Threads::Threads)
    set_target_properties(ivf_router_diagnostics PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests")
    add_executable(ivf_training_diagnostics ivf_training_diagnostics.cpp)
    target_link_libraries(ivf_training_diagnostics PRIVATE ivf_router Threads::Threads)
    set_target_properties(ivf_training_diagnostics PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests")
endif()

# Explicit resource smoke; historically built with the contract suite, but not
# registered with routine CTest or included in canonical benchmark metrics.
if(IVF_BUILD_CONTRACT_TESTS AND TARGET ivf_router)
    add_executable(ivf_centroid_scale_smoke ivf_centroid_scale_smoke.cpp)
    target_link_libraries(ivf_centroid_scale_smoke PRIVATE ivf_router)
endif()
