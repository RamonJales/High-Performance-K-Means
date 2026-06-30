cmake_minimum_required(VERSION 3.20)

# Initializes the project, enabling C++
project(High-Performance-K-Means VERSION 1.0 LANGUAGES CXX)

# Sets the C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Exports compile commands for Neovim/LSP integration
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---------------------------------------------------------
# 1. Dependency Search (HPC)
# ---------------------------------------------------------
find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)

# ---------------------------------------------------------
# 2. Target Core (CPU) - Business Logic & Utilities
# ---------------------------------------------------------
add_library(kmeans_core STATIC
    src/core/kmeans_core.cpp
    src/core/utils.cpp
)

# Instructs the compiler where to find the .hpp header files
target_include_directories(kmeans_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(kmeans_core PUBLIC OpenMP::OpenMP_CXX)

add_executable(kmeans_seq src/sequential/main.cpp)

target_link_libraries(kmeans_seq PRIVATE kmeans_core)

# ---------------------------------------------------------
# 2b. Target Core (GPU - OpenMP Offloading)
# ---------------------------------------------------------
add_library(kmeans_core_gpu STATIC
    src/core/kmeans_core_gpu.cpp
    src/core/utils.cpp
)

target_include_directories(kmeans_core_gpu PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(kmeans_gpu src/openmp_gpu/main.cpp)

target_link_libraries(kmeans_gpu PRIVATE kmeans_core_gpu)

# find_package(OpenMP) does not recognize NVHPC's GPU offloading flag
# (-mp=gpu); it only handles standard CPU OpenMP flags such as
# (-fopenmp). Therefore, the offloading flags are added manually,
# and only when the selected compiler is NVHPC (nvc++).
if(CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC")
    set(GPU_OFFLOAD_FLAGS -mp=gpu -gpu=cc70) 
    target_compile_options(kmeans_core_gpu PUBLIC ${GPU_OFFLOAD_FLAGS})
    target_compile_options(kmeans_gpu PUBLIC ${GPU_OFFLOAD_FLAGS})
    target_link_options(kmeans_core_gpu PUBLIC ${GPU_OFFLOAD_FLAGS})
    target_link_options(kmeans_gpu PUBLIC ${GPU_OFFLOAD_FLAGS})
else()
    message(WARNING "Compilador atual (${CMAKE_CXX_COMPILER_ID}) nao e NVHPC: kmeans_gpu vai compilar, mas sem offload real para GPU.")
endif()

# ---------------------------------------------------------
# 2c. Target Core (MPI + OpenMP, CPU)
# ---------------------------------------------------------
add_library(kmeans_core_mpi STATIC
    src/core/kmeans_core_mpi.cpp
    src/core/utils.cpp
)

target_include_directories(kmeans_core_mpi PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(kmeans_core_mpi PUBLIC OpenMP::OpenMP_CXX MPI::MPI_CXX)

add_executable(kmeans_mpi_omp src/mpi_openmp/main.cpp)

target_link_libraries(kmeans_mpi_omp PRIVATE kmeans_core_mpi)

# ---------------------------------------------------------
# 2d. Target Core (CUDA - NVIDIA Native GPU)
# ---------------------------------------------------------
include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)

    add_library(kmeans_core_cuda STATIC
        src/core/kmeans_core_cuda.cu
        src/core/utils.cpp
    )

    target_include_directories(kmeans_core_cuda PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
    set_target_properties(kmeans_core_cuda PROPERTIES CUDA_STANDARD 17)
    set_target_properties(kmeans_core_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    set_target_properties(kmeans_core_cuda PROPERTIES CUDA_ARCHITECTURES 70)

    add_executable(kmeans_cuda src/cuda/main.cu)
    target_link_libraries(kmeans_cuda PRIVATE kmeans_core_cuda)
    set_target_properties(kmeans_cuda PROPERTIES CUDA_STANDARD 17)
    set_target_properties(kmeans_cuda PROPERTIES CUDA_ARCHITECTURES 70)
else()
    message(WARNING "Nenhum compilador CUDA (nvcc) encontrado: kmeans_cuda nao sera compilado.")
endif()

# ---------------------------------------------------------
# 3. Unit Tests (Google Test)
# ---------------------------------------------------------
enable_testing()

include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        release-1.12.1
)
FetchContent_MakeAvailable(googletest)

add_executable(run_tests
    tests/test_kmeans_core.cpp
    tests/test_utils.cpp
    tests/first_test.cpp
)

target_link_libraries(run_tests
    PRIVATE
    gtest_main
    kmeans_core
)

include(GoogleTest)
gtest_discover_tests(run_tests)