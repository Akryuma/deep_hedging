cmake_minimum_required(VERSION 4.2 FATAL_ERROR)
# cmake_policy(SET CMP0148 OLD)

set(CMAKE_CUDA_COMPILER "/usr/local/cuda-13.1/bin/nvcc" CACHE FILEPATH "Force CUDA compiler" FORCE)
set(CMAKE_CUDA_ARCHITECTURES "89" CACHE STRING "Force CUDA architecture" FORCE)
set(TORCH_CUDA_ARCH_LIST "8.9" CACHE STRING "Force PyTorch architecture" FORCE)

project(DeepHedgingCpp VERSION 1.0.0 LANGUAGES CXX CUDA)

# PyTorch 3.0 -> C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -Wall -Wextra")


find_package(Torch REQUIRED)
# find_package(Python 3.10 COMPONENTS Interpreter Development REQUIRED)
# find_package(pybind11 REQUIRED)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(deep_hedging_test
    src/main.cpp
    src/simulation.cpp
    src/network.cpp)

target_link_libraries(deep_hedging_test 
    "-Wl,--no-as-needed" 
    "${TORCH_LIBRARIES}" 
    "-Wl,--as-needed"
)

add_executable(deep_hedging_eval
    src/evaluate.cpp
    src/simulation.cpp
    src/network.cpp
)

target_link_libraries(deep_hedging_eval
    "-Wl,--no-as-needed"
    "${TORCH_LIBRARIES}"
    "-Wl,--as-needed"
)
# if (MSVC)
#     target_compile_options(deep_hedging_test PRIVATE /O2)
# else()
#     target_compile_options(deep_hedging_test PRIVATE -O3 -Wall -Wextra)
# endif()

# # --- Module Python Bindings ---
# pybind11_add_module(heston_mc 
#     src/bindings.cpp 
#     ${ENGINE_SOURCES}
# )
# target_link_libraries(heston_mc PRIVATE pybind11::module)
# target_include_directories(heston_mc PRIVATE include)
# target_compile_options(heston_mc PRIVATE -O3 -march=native)
