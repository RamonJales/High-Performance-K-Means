# High-Performance-K-Means
Parallel implementation and performance analysis of the K-Means clustering algorithm using MPI + OpenMP, OpenMP GPU, and CUDA. The project evaluates execution time, speedup, efficiency, and scalability across CPU and GPU architectures in high-performance computing environments.

## Dataset
This project uses the [Mall Customers Dataset from Kaggle](https://www.kaggle.com/datasets/simtoor/mall-customers?utm_source=chatgpt.com). The dataset contains customer information such as age, income, and spending score, making it suitable for clustering analysis with K-Means.

## Project Objectives
- Implement a sequential baseline version of K-Means
- Develop parallel versions using:
  - MPI + OpenMP
  - OpenMP GPU Offloading
  - CUDA
- Compare CPU and GPU performance
  - Analyze:
  - Execution time
  - Speedup
  - Efficiency
  - Strong scalability
  - Weak scalability
- Evaluate communication overhead and load balancing

## Technologies
  - C / C++
  - MPI
  - OpenMP
  - CUDA

## Project Structure
```
High-Performance-K-Means/
├── .clang-tidy                # Code quality rules
├── .gitignore                 
├── CMakeLists.txt             # Main build configuration file
├── README.md                  
├── data/                      # Kaggle dataset
├── include/                   # Header files (Shared core)
│   ├── kmeans_core.hpp        # Data structures (Points, Centroids, etc) and math algorithms
│   └── utils.hpp              # Utility functions (CSV parser, time measurement)
├── results/                   # GENERATED RESULTS (Keep out of version control)
│   ├── performance_logs/      # Execution times, speedup, and efficiency metrics
│   └── clustered_data/        # Output CSVs with the final clustered points
├── scripts/
│   ├── env_setup.sh           # Script to load environment modules (GCC, OpenMPI, CUDA)
│   └── run_experiments.sh     # SLURM/Bash script to automate performance tests
│   └── local_test.sh          # for local tests
├──src/
│   ├── core/                  # Core logic implementation
│   │   ├── kmeans_core.cpp    # Euclidean distance, centroid recalculation logic, etc.
│   │   └── utils.cpp          # CSV parsing, I/O implementation, etc.
│   ├── sequential/            # Baseline implementation
│   │   └── main.cpp           # Pure sequential K-means
│   ├── mpi_openmp/            # Distributed + Shared memory implementation
│   │   └── main.cpp           # K-means using MPI and OpenMP pragmas
│   ├── openmp_gpu/            # GPU Offloading implementation
│   │   └── main.cpp           # K-means with OpenMP target offload
│   └── cuda/                  # Native GPU implementation
│       └── main.cu            # K-means written in CUDA C++
├── tests/                     # <--- NOVA PASTA
│   └── first_test.cpp         # Arquivo com os testes unitários
└── ...
```

## How to Run

### 1. Environment Setup

Before building, ensure the required compilers and tools are available.

- **On NPAD (Cluster):** Load the necessary modules:
  ```bash
  module load gcc/11.2.0 openmpi/4.1.1 cmake/3.20.0 cuda
  ```

- **Local Machine:** Ensure `build-essential`, `libopenmpi-dev`, and `cmake` are installed. (Note: GPU implementations require a compatible NVIDIA CUDA Toolkit and hardware.)

  ```
  sudo apt update
  sudo apt install build-essential libopenmpi-dev cmake
  ```

### 2. Building the Project

Run the following commands from the project root to generate the build system and compile all targets:

```bash
cmake -B build
cmake --build build
```

### 3. Running Unit Tests

To verify the mathematical correctness of the core algorithms:

```bash
cd build
ctest --output-on-failure
```

## Team:
  - Ramon Cândido Jales de Barros
  - Victor Aguiar Gomes
  - Gabrielle de Vasconcelos Borja 
  - Jeremias Pinheiro de Araújo Andrade
  - Mariana Timbó de Oliveira

## Professors
- Carla dos Santos Santana
  - Department of Computer Engineering and Automation
- Samuel Xavier de Souza
  - Department of Computer Engineering and Automation

## Academic Context
- This project was developed for the course:
- IMD1116 — High Performance Computing
- Federal University of Rio Grande do Norte (UFRN)
