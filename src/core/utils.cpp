#include "utils.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <kmeans_core_gpu.hpp>

std::vector<Point> read_csv(const std::string& filename, const std::vector<int>& feature_columns) {
    std::vector<Point> points;
    std::ifstream file(filename);

    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    std::string line;
    // Skip header
    if (std::getline(file, line)) {
        // Read data
        int point_id = 0;
        while (std::getline(file, line)) {
            std::stringstream ss(line);
            std::string cell;
            std::vector<std::string> row;

            while (std::getline(ss, cell, ',')) {
                row.push_back(cell);
            }

            std::vector<double> features;
            for (int col_idx : feature_columns) {
                if (col_idx < row.size()) {
                    features.push_back(std::stod(row[col_idx]));
                } else {
                    throw std::out_of_range("Column index out of bounds in CSV.");
                }
            }

            points.emplace_back(point_id++, features);
        }
    }

    file.close();
    return points;
}

bool cmd_option_exists(char** begin, char** end, const std::string& option) {
    return std::find(begin, end, option) != end;
}

std::string get_cmd_option(char** begin, char** end, const std::string& option) {
    char** itr = std::find(begin, end, option);
    if (itr != end && ++itr != end) {
        return std::string(*itr);
    }
    return "";
}

void print_execution_summary(const std::string& version_name, size_t dataset_size, int iterations, double elapsed_time) {
    std::cout << "======================================\n";
    std::cout << " Execution Summary (" << version_name << ")\n";
    std::cout << "======================================\n";
    std::cout << " Dataset Size:  " << dataset_size << " points\n";
    std::cout << " Status:        Converged in " << iterations << " iterations\n";
    std::cout << " Elapsed Time:  " << elapsed_time << " seconds\n";
    std::cout << "======================================\n";
}

Points_Data_GPU gpu_read_csv(const std::string& filename, const std::vector<int>& feature_columns) {
    Points_Data_GPU data;
    data.num_features = static_cast<int>(feature_columns.size());
 
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }
 
    std::string line;
    // Skip header
    if (std::getline(file, line)) {
        while (std::getline(file, line)) {
            std::stringstream ss(line);
            std::string cell;
            std::vector<std::string> row;
            while (std::getline(ss, cell, ',')) {
                row.push_back(cell);
            }
 
            for (int col_idx : feature_columns) {
                if (col_idx < static_cast<int>(row.size())) {
                    data.points.push_back(std::stod(row[col_idx]));
                } else {
                    throw std::out_of_range("Column index out of bounds in CSV.");
                }
            }
            data.n++;
        }
    }
    file.close();
 
    data.cluster_ids.assign(data.n, -1);
    data.min_distances.assign(data.n, 0.0);
 
    return data;
}
