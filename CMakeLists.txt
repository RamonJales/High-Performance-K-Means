cmake_minimum_required(VERSION 3.20)

# Inicializa o projeto habilitando C++ e CUDA
project(High-Performance-K-Means VERSION 1.0 LANGUAGES CXX)

# Configura o padrão do C++ para C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Exporta o arquivo para o Neovim/LSP entender o código
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---------------------------------------------------------
# 1. Busca de Dependências (HPC)
# ---------------------------------------------------------
find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)

# ---------------------------------------------------------
# 4. Testes Unitários (Google Test)
# ---------------------------------------------------------
# Habilita a infraestrutura de testes do CMake (CTest)
enable_testing()

# Baixa o Google Test automaticamente do GitHub
include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        release-1.12.1
)
FetchContent_MakeAvailable(googletest)

# Cria o executável de testes
add_executable(run_tests tests/first_test.cpp)

# Linka o GTest e a NOSSA lógica de negócio (kmeans_core) ao teste
target_link_libraries(run_tests 
    PRIVATE 
    gtest_main 
)

# Registra o executável para o comando 'ctest'
include(GoogleTest)
gtest_discover_tests(run_tests)
