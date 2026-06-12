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
# 2. Target Core (Business Logic & Utilities)
# ---------------------------------------------------------
add_library(kmeans_core STATIC
    src/core/kmeans_core.cpp
    src/core/utils.cpp
)

# Instructs the compiler where to find the .hpp header files
target_include_directories(kmeans_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(kmeans_seq src/sequential/main.cpp)

target_link_libraries(kmeans_seq PRIVATE kmeans_core)

# ---------------------------------------------------------
# 3. Unit Tests (Google Test)
# ---------------------------------------------------------
# Enables CMake testing infrastructure (CTest)
enable_testing()

# Downloads Google Test automatically from GitHub
include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        release-1.12.1
)
FetchContent_MakeAvailable(googletest)

# Creates the test executable
add_executable(run_tests
    tests/test_kmeans_core.cpp
    tests/test_utils.cpp
    tests/first_test.cpp
)

# Links GTest AND our core logic (kmeans_core) to the test executable
target_link_libraries(run_tests 
    PRIVATE 
    gtest_main 
    kmeans_core
)

# Registers the executable for the 'ctest' command
include(GoogleTest)
gtest_discover_tests(run_tests)
