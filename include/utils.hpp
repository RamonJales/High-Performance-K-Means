#ifndef UTILS_HPP
#define UTILS_HPP

#include "kmeans_core.hpp"
#include <string>
#include <vector>

/**
 * @brief Parses a CSV file and loads specific columns into Point objects.
 * * This utility function reads a dataset from a CSV file, skipping the first line (header).
 * It extracts only the columns specified by their zero-based indices.
 * * @param filename The relative or absolute path to the CSV file.
 * @param feature_columns A vector of zero-based column indices to extract as features.
 * For example, to extract columns 3 and 4, pass {3, 4}.
 * @return std::vector<Point> A vector containing the parsed Point objects.
 * @throws std::runtime_error If the file cannot be opened.
 * @throws std::out_of_range If a requested column index does not exist in the CSV row.
 */
std::vector<Point> read_csv(const std::string& filename, const std::vector<int>& feature_columns);

#endif // UTILS_HPP
