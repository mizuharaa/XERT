cmake_minimum_required(VERSION 3.20)
project(reach LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT MSVC OR NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
    message(FATAL_ERROR "Reach requires 64-bit MSVC and the Windows MuJoCo wheel")
endif()

set(Python_EXECUTABLE "${CMAKE_SOURCE_DIR}/.venv/Scripts/python.exe" CACHE FILEPATH "Checkout Python")
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
execute_process(COMMAND "${Python_EXECUTABLE}" -c "import mujoco, pathlib; assert mujoco.__version__ == '3.14.0'; print(pathlib.Path(mujoco.__file__).parent.as_posix())"
    OUTPUT_VARIABLE MUJOCO_DIR OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
execute_process(COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_DIR OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
find_package(pybind11 CONFIG REQUIRED)

# Generate an import library from the installed wheel; the DLL itself is only copied next to reach_trials.
find_program(DUMPBIN dumpbin REQUIRED)
find_program(MSVC_LIB lib REQUIRED)
execute_process(COMMAND "${DUMPBIN}" /exports "${MUJOCO_DIR}/mujoco.dll"
    OUTPUT_VARIABLE exports COMMAND_ERROR_IS_FATAL ANY)
string(REPLACE "\n" ";" export_lines "${exports}")
set(definition "LIBRARY mujoco.dll\nEXPORTS\n")
set(export_count 0)
foreach(line IN LISTS export_lines)
    if(line MATCHES "^[ \t]+[0-9]+[ \t]+[0-9A-Fa-f]+[ \t]+[0-9A-Fa-f]+[ \t]+([A-Za-z_][A-Za-z_0-9]*)")
        string(APPEND definition "${CMAKE_MATCH_1}\n")
        math(EXPR export_count "${export_count}+1")
    endif()
endforeach()
if(export_count EQUAL 0)
    message(FATAL_ERROR "No MuJoCo DLL exports found")
endif()
file(WRITE "${CMAKE_BINARY_DIR}/mujoco.def" "${definition}")
execute_process(COMMAND "${MSVC_LIB}" /nologo /machine:x64 "/def:${CMAKE_BINARY_DIR}/mujoco.def" "/out:${CMAKE_BINARY_DIR}/mujoco.lib"
    COMMAND_ERROR_IS_FATAL ANY)
add_library(mujoco SHARED IMPORTED)
find_path(MUJOCO_INCLUDE_DIR mujoco/mujoco.h PATHS "${MUJOCO_DIR}/include" "${MUJOCO_DIR}/include/mujoco" NO_DEFAULT_PATH REQUIRED)
set_target_properties(mujoco PROPERTIES IMPORTED_IMPLIB "${CMAKE_BINARY_DIR}/mujoco.lib"
    IMPORTED_LOCATION "${MUJOCO_DIR}/mujoco.dll" INTERFACE_INCLUDE_DIRECTORIES "${MUJOCO_INCLUDE_DIR}")

add_library(reach_core STATIC cpp/src/env.cpp cpp/src/ik.cpp)
target_include_directories(reach_core PUBLIC cpp/include)
target_link_libraries(reach_core PUBLIC mujoco)
# Keep NumPy's separate operations, including float32 quantisation, unfused.
target_compile_options(reach_core PRIVATE /W4 /fp:strict)
pybind11_add_module(_reach_core cpp/src/bindings.cpp)
target_link_libraries(_reach_core PRIVATE reach_core)
# Direct module imports must let Python's mujoco package load the wheel DLL first.
target_link_libraries(_reach_core PRIVATE delayimp)
target_link_options(_reach_core PRIVATE /DELAYLOAD:mujoco.dll)
set_target_properties(_reach_core PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/xle_workbench"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/xle_workbench")
foreach(config Debug Release RelWithDebInfo MinSizeRel)
    string(TOUPPER "${config}" upper_config)
    set_target_properties(_reach_core PROPERTIES LIBRARY_OUTPUT_DIRECTORY_${upper_config} "${CMAKE_SOURCE_DIR}/xle_workbench"
        RUNTIME_OUTPUT_DIRECTORY_${upper_config} "${CMAKE_SOURCE_DIR}/xle_workbench")
endforeach()
add_executable(reach_trials cpp/src/trials.cpp)
target_link_libraries(reach_trials PRIVATE reach_core)
add_executable(reach_tests cpp/tests/test_core.cpp)
target_link_libraries(reach_tests PRIVATE reach_core)
# The standalone executables load the same wheel DLL from their own folder.
add_custom_command(TARGET reach_trials POST_BUILD
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${MUJOCO_DIR}/mujoco.dll" "$<TARGET_FILE_DIR:reach_trials>")
enable_testing()
add_test(NAME export_bench COMMAND "${Python_EXECUTABLE}" -m xle_workbench.cli export-bench
    --model "${CMAKE_BINARY_DIR}/bench.xml" --positions "${CMAKE_BINARY_DIR}/positions.csv" --seeds 0..8)
set_tests_properties(export_bench PROPERTIES WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}" FIXTURES_SETUP bench)
add_test(NAME reach_core COMMAND reach_tests "${CMAKE_BINARY_DIR}/bench.xml" "${CMAKE_BINARY_DIR}/positions.csv")
set_tests_properties(reach_core PROPERTIES FIXTURES_REQUIRED bench ENVIRONMENT "PATH=${MUJOCO_DIR}\;$ENV{PATH}")
