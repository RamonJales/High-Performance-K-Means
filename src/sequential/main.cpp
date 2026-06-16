#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <random>
#include <algorithm>
#include "kmeans_core.hpp"
#include "utils.hpp"

int main(int argc, char** argv) {
    // 1. Parse Command Line Arguments
    if (cmd_option_exists(argv, argv + argc, "-h") || argc == 1) {
        std::cout << "Usage: ./kmeans_seq -input <file.csv> -k <num_clusters> [-max_iter <iters>]\n";
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
        std::vector<Point> points = read_csv_preprocessed(input_file);
        
        if (points.empty()) {
            std::cerr << "Error: Dataset is empty.\n";
            return 1;
        }
        if (k > points.size()) {
            std::cerr << "Error: K cannot be greater than the number of points.\n";
            return 1;
        }

        int num_features = points[0].features.size();

        // 3. Initialize Centroids
        std::cout << "Initializing " << k << " centroids...\n";
        std::vector<std::vector<double>> centroids(k);
        
        // We use a fixed seed (42) for reproducibility. 
        // This ensures the Sequential and Parallel versions start from the exact same centroids.
        std::mt19937 rng(42); 
        std::uniform_int_distribution<int> uni(0, points.size() - 1);

        for (int i = 0; i < k; ++i) {
            centroids[i] = points[uni(rng)].features;
        }

        // 4. Run K-Means Loop and Measure Execution Time
        std::cout << "Starting Sequential K-Means loop...\n";
        auto start_time = std::chrono::high_resolution_clock::now();

        int iterations = 0;
        bool changed = true;

        while (changed && iterations < max_iter) {
            // Step A: Assign points to the nearest centroid
            changed = assign_clusters(points, centroids);

            // Step B: Update centroids based on new assignments
            if (changed) {
                centroids = update_centroids(points, k, num_features);
            }
            
            iterations++;
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_time - start_time;

        // 5. Output Performance Metrics
        print_execution_summary("Sequential", points.size(), iterations, elapsed.count());

    } catch (const std::exception& e) {
        std::cerr << "Exception caught: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
