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

/**
 * @brief Checks if a specific command-line option (flag) exists.
 * * @param begin Pointer to the beginning of the arguments array (argv).
 * @param end Pointer to the end of the arguments array (argv + argc).
 * @param option The string representing the option flag (e.g., "-h", "-k").
 * @return true If the option exists in the provided arguments.
 * @return false If the option does not exist.
 */
bool cmd_option_exists(char** begin, char** end, const std::string& option);

/**
 * @brief Retrieves the value associated with a command-line option.
 * * @param begin Pointer to the beginning of the arguments array (argv).
 * @param end Pointer to the end of the arguments array (argv + argc).
 * @param option The string representing the option flag (e.g., "-input").
 * @return std::string The value immediately following the option flag. Returns an empty string if the flag is not found or if it is the last argument without a subsequent value.
 */
std::string get_cmd_option(char** begin, char** end, const std::string& option);

/**
 * @brief Prints a standardized execution summary for the K-Means algorithm.
 * * This function outputs performance metrics including dataset size, iterations until
 * convergence, and total elapsed time in a clean, consistent format.
 * * @param version_name The name of the implementation version (e.g., "Sequential", "MPI + OpenMP", "CUDA").
 * @param dataset_size The total number of points processed.
 * @param iterations The number of iterations completed until convergence or limit.
 * @param elapsed_time The execution time in seconds.
 */
void print_execution_summary(const std::string& version_name, size_t dataset_size, int iterations, double elapsed_time);

#endif // UTILS_HPP
