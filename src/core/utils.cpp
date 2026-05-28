#include "utils.hpp"
#include <fstream>
#include <sstream>
#include <iostream>

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
