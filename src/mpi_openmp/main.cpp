#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <algorithm>
#include <mpi.h>
#include "kmeans_core_mpi.hpp"
#include "utils.hpp"

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0, world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // 1. Parse Command Line Arguments
    if (cmd_option_exists(argv, argv + argc, "-h") || argc == 1) {
        if (rank == 0) {
            std::cout << "Usage: mpirun -np <P> ./kmeans_mpi_omp -input <file.csv> -k <num_clusters> [-max_iter <iters>]\n";
        }
        MPI_Finalize();
        return 0;
    }

    std::string input_file = get_cmd_option(argv, argv + argc, "-input");
    std::string k_str = get_cmd_option(argv, argv + argc, "-k");
    std::string iter_str = get_cmd_option(argv, argv + argc, "-max_iter");

    if (input_file.empty() || k_str.empty()) {
        if (rank == 0) {
            std::cerr << "Error: -input and -k are required arguments.\n";
        }
        MPI_Finalize();
        return 1;
    }

    int k = std::stoi(k_str);
    int max_iter = iter_str.empty() ? 100 : std::stoi(iter_str); // Default max iterations: 100

    // For the Mall Customers Dataset, we want to cluster based on:
    // Annual Income (index 3) and Spending Score (index 4)
    std::vector<int> target_columns = {3, 4};
    int num_features = static_cast<int>(target_columns.size());

    // 2. Load the Dataset (only rank 0 reads from disk)
    Points_Data_GPU full;   // rank 0 only: the complete dataset (SoA)
    int global_n = 0;
    if (rank == 0) {
        try {
            std::cout << "Loading dataset: " << input_file << "...\n";
            full = gpu_read_csv(input_file, target_columns);
            global_n = full.n;

            if (global_n == 0) {
                std::cerr << "Error: Dataset is empty.\n";
                global_n = -1; // signal error to the other ranks below
            } else if (k > global_n) {
                std::cerr << "Error: K cannot be greater than the number of points.\n";
                global_n = -1;
            } else if (k <= 0) {
                std::cerr << "Error: K must be greater than 0.\n";
                global_n = -1;
            }
        } catch (const std::exception& e) {
            std::cerr << "Exception caught: " << e.what() << "\n";
            global_n = -1;
        }
    }

    // Share the global point count (and error status) with every rank.
    MPI_Bcast(&global_n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (global_n <= 0) { // an error occurred on rank 0
        MPI_Finalize();
        return 1;
    }

    // 3. Distribute the points across ranks (MPI_Scatterv).
    // Points are split as evenly as possible; the first (n % size) ranks
    // receive one extra point. Counts/displacements are expressed in doubles.
    std::vector<int> send_counts(world_size, 0);
    std::vector<int> displs(world_size, 0);
    int base = global_n / world_size;
    int rem = global_n % world_size;
    int offset_points = 0;
    for (int r = 0; r < world_size; ++r) {
        int pts = base + (r < rem ? 1 : 0);
        send_counts[r] = pts * num_features;
        displs[r] = offset_points * num_features;
        offset_points += pts;
    }

    // Build this rank's LOCAL data slice using the shared SoA container.
    Points_Data_GPU data;
    data.k = k;
    data.num_features = num_features;
    data.n = send_counts[rank] / num_features;
    data.points.resize(static_cast<size_t>(data.n) * num_features);
    data.cluster_ids.assign(data.n, -1);
    data.min_distances.assign(data.n, 0.0);

    MPI_Scatterv(rank == 0 ? full.points.data() : nullptr,
                 send_counts.data(), displs.data(), MPI_DOUBLE,
                 data.points.data(), send_counts[rank], MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    // 4. Initialize Centroids
    data.centroids.resize(static_cast<size_t>(k) * num_features);
    if (rank == 0) {
        std::cout << "Initializing " << k << " centroids...\n";
        std::mt19937 rng(42);
        std::uniform_int_distribution<int> uni(0, global_n - 1);
        for (int i = 0; i < k; ++i) {
            int point_idx = uni(rng);
            std::copy(
                full.points.begin() + point_idx * num_features,
                full.points.begin() + (point_idx + 1) * num_features,
                data.centroids.begin() + i * num_features
            );
        }
    }
    MPI_Bcast(data.centroids.data(), k * num_features, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // 5. Run K-Means Loop and Measure Execution Time
    if (rank == 0) {
        std::cout << "Starting MPI + OpenMP K-Means loop...\n";
    }
    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    int iterations = 0;
    bool changed = true;
    while (changed && iterations < max_iter) {
        // Step A: Assign points to the nearest centroid (OpenMP + MPI reduce)
        changed = assign_clusters_mpi(data);

        // Step B: Update centroids based on new assignments (OpenMP + MPI reduce)
        if (changed) {
            update_centroids_mpi(data);
        }
        iterations++;
    }

    double local_elapsed = MPI_Wtime() - start_time;
    double elapsed = 0.0; // report the slowest rank as the wall-clock cost
    MPI_Reduce(&local_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // 6. Output Performance Metrics
    if (rank == 0) {
        print_execution_summary("MPI + OpenMP", global_n, iterations, elapsed);
    }

    MPI_Finalize();
    return 0;
}
