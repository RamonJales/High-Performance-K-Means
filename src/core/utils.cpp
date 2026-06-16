#include "utils.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <limits>
#include <cctype>
#include <stdexcept>

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

namespace {
std::string trim(const std::string& value) {
    size_t start = value.find_first_not_of(" \t\r\n");

    if (start == std::string::npos) {
        return "";
    }

    size_t end = value.find_last_not_of(" \t\r\n");
    return value.substr(start, end - start + 1);
}


double gender_to_binary(const std::string& gender) {
    if (gender == "Male") {
        return 0.0;
    }

    if (gender == "Female") {
        return 1.0;
    }
    throw std::invalid_argument("Unknown gender value: " + gender);
}

double min_max_normalize(double value, double min_value, double max_value) {
    if (max_value == min_value) {
        return 0.0;
    }

    return (value - min_value) / (max_value - min_value);
}
}

std::vector<Point> read_csv_preprocessed(const std::string& filename) {
    std::ifstream file(filename);

    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    struct RawRow {
        double gender;
        double age;
        double income;
        double spending_score;
    };

    std::vector<RawRow> raw_rows;

    double min_age = std::numeric_limits<double>::max();
    double max_age = std::numeric_limits<double>::lowest();

    double min_income = std::numeric_limits<double>::max();
    double max_income = std::numeric_limits<double>::lowest();

    double min_score = std::numeric_limits<double>::max();
    double max_score = std::numeric_limits<double>::lowest();

    std::string line;
    std::getline(file, line);

    while (std::getline(file, line)) {
        if (trim(line).empty()) {
            continue;
        }

        std::stringstream ss(line);
        std::string cell;
        std::vector<std::string> row;

        while (std::getline(ss, cell, ',')) {
            row.push_back(trim(cell));
        }

        if (row.size() < 5) {
            throw std::runtime_error("Invalid CSV row. Expected at least 5 columns.");
        }

        RawRow current {
            gender_to_binary(row[1]),
            std::stod(row[2]),
            std::stod(row[3]),
            std::stod(row[4])
        };

        raw_rows.push_back(current);

        min_age = std::min(min_age, current.age);
        max_age = std::max(max_age, current.age);

        min_income = std::min(min_income, current.income);
        max_income = std::max(max_income, current.income);

        min_score = std::min(min_score, current.spending_score);
        max_score = std::max(max_score, current.spending_score);
    }

    std::vector<Point> points;
    points.reserve(raw_rows.size());

    int point_id = 0;

    for (const auto& row : raw_rows) {
        std::vector<double> features = {
            row.gender,
            min_max_normalize(row.age, min_age, max_age),
            min_max_normalize(row.income, min_income, max_income),
            min_max_normalize(row.spending_score, min_score, max_score)
        };

        points.emplace_back(point_id++, features);
    }

    return points;
}