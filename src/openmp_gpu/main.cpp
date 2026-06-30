#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <random>
#include <algorithm>
#include "kmeans_core_gpu.hpp"
#include "utils.hpp"

int main(int argc, char** argv) {
    // 1. Parse Command Line Arguments
    if (cmd_option_exists(argv, argv + argc, "-h") || argc == 1) {
        std::cout << "Usage: ./kmeans_gpu -input <file.csv> -k <num_clusters> [-max_iter <iters>]\n";
        return 0;
    }

    std::string input_file = get_cmd_option(argv, argv + argc, "-input");
    std::string k_str = get_cmd_option(argv, argv + argc, "-k");
    std::string iter_str = get_cmd_option(argv, argv + argc, "-max_iter");

    if (input_file.empty() || k_str.empty()) {
        std::cerr << "Error: -input and -k are required arguments.\n";
        return 1;
    }

    int k = std::stoi(k_str);
    int max_iter = iter_str.empty() ? 100 : std::stoi(iter_str); // Default max iterations: 100

    // For the Mall Customers Dataset, we want to cluster based on:
    // Annual Income (index 3) and Spending Score (index 4)
    std::vector<int> target_columns = {3, 4};

    try {
        // 2. Load the Dataset
        std::cout << "Loading dataset: " << input_file << "...\n";
        Points_Data_GPU data = gpu_read_csv(input_file, target_columns);

        if (data.n == 0) {
            std::cerr << "Error: Dataset is empty.\n";
            return 1;
        }
        if (k > data.n) {
            std::cerr << "Error: K cannot be greater than the number of points.\n";
            return 1;
        }

        data.k = k;
        int num_features = data.num_features;

        // 3. Initialize Centroids
        std::cout << "Initializing " << k << " centroids...\n";
        data.centroids.resize(k * num_features);

        std::mt19937 rng(42);
        std::uniform_int_distribution<int> uni(0, data.n - 1);
        for (int i = 0; i < k; ++i) {
            int point_idx = uni(rng);
            std::copy(
                data.points.begin() + point_idx * num_features,
                data.points.begin() + (point_idx + 1) * num_features,
                data.centroids.begin() + i * num_features
            );
        }

        // 4. Run K-Means Loop and Measure Execution Time
        std::cout << "Starting GPU K-Means loop...\n";
        auto start_time = std::chrono::high_resolution_clock::now();

        int iterations = 0;
        bool changed = true;

        const double* points_dev = data.points.data();
        const long points_len = static_cast<long>(data.n) * num_features;

        #pragma omp target data map(to: points_dev[0:points_len])
        {
            while (changed && iterations < max_iter) {
                // Step A: Assign points to the nearest centroid (kernel in GPU)
                changed = assign_clusters_gpu(data);

                // Step B: Update centroids based on new assignments (kernel in GPU)
                if (changed) {
                    update_centroids_gpu(data);
                }
                iterations++;
            }
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_time - start_time;

        // 5. Output Performance Metrics
        print_execution_summary("OpenMP-GPU", data.n, iterations, elapsed.count());

    } catch (const std::exception& e) {
        std::cerr << "Exception caught: " << e.what() << "\n";
        return 1;
    }

    return 0;
}