#include <gtest/gtest.h>
#include "utils.hpp"
#include <fstream>
#include <cstdio> // For std::remove

// ==========================================
// TEST SUITE: CSV Parser
// ==========================================

class UtilsTest : public ::testing::Test {
protected:
    const std::string test_filename = "temp_test_data.csv";

    // Setup: Creates a dummy CSV file before each test
    void SetUp() override {
        std::ofstream file(test_filename);
        file << "CustomerID,Age,Annual Income (k$),Spending Score (1-100)\n";
        file << "1,19,15,39\n";
        file << "2,21,15,81\n";
        file << "3,20,16,6\n";
        file.close();
    }

    // Teardown: Deletes the dummy CSV file after each test
    void TearDown() override {
        std::remove(test_filename.c_str());
    }
};

TEST_F(UtilsTest, ReadCSVHappyPath) {
    // We want to extract columns 2 (Income) and 3 (Spending Score) -> 0-based index
    std::vector<int> target_columns = {2, 3};
    
    std::vector<Point> points = read_csv(test_filename, target_columns);

    // Verify correct number of points loaded
    EXPECT_EQ(points.size(), 3);

    // Verify data of the first point (Income: 15, Score: 39)
    EXPECT_DOUBLE_EQ(points[0].features[0], 15.0);
    EXPECT_DOUBLE_EQ(points[0].features[1], 39.0);

    // Verify data of the last point (Income: 16, Score: 6)
    EXPECT_DOUBLE_EQ(points[2].features[0], 16.0);
    EXPECT_DOUBLE_EQ(points[2].features[1], 6.0);
}

TEST_F(UtilsTest, ReadCSVFileNotFound) {
    std::vector<int> target_columns = {1, 2};
    // Expect a runtime_error when the file does not exist
    EXPECT_THROW(read_csv("non_existent_file.csv", target_columns), std::runtime_error);
}

TEST_F(UtilsTest, ReadCSVColumnOutOfBounds) {
    // Requesting column index 10, but the CSV only has 4 columns (indices 0 to 3)
    std::vector<int> target_columns = {1, 10};
    
    // Expect an out_of_range exception
    EXPECT_THROW(read_csv(test_filename, target_columns), std::out_of_range);
}

TEST(CLIParserTest, OptionExistsHappyPath) {
    const char* args[] = {"./kmeans_seq", "-k", "5", "-input", "data.csv"};
    int argc = 5;
    char** argv = const_cast<char**>(args); 

    EXPECT_TRUE(cmd_option_exists(argv, argv + argc, "-k"));
    EXPECT_TRUE(cmd_option_exists(argv, argv + argc, "-input"));
}

TEST(CLIParserTest, OptionDoesNotExist) {
    const char* args[] = {"./kmeans_seq", "-k", "5"};
    int argc = 3;
    char** argv = const_cast<char**>(args);

    EXPECT_FALSE(cmd_option_exists(argv, argv + argc, "-h"));
    EXPECT_FALSE(cmd_option_exists(argv, argv + argc, "-input"));
}

TEST(CLIParserTest, GetOptionValueHappyPath) {
    const char* args[] = {"./kmeans_seq", "-k", "5", "-input", "data.csv"};
    int argc = 5;
    char** argv = const_cast<char**>(args);

    EXPECT_EQ(get_cmd_option(argv, argv + argc, "-k"), "5");
    EXPECT_EQ(get_cmd_option(argv, argv + argc, "-input"), "data.csv");
}

TEST(CLIParserTest, GetOptionValueMissingValue) {
    const char* args[] = {"./kmeans_seq", "-k"};
    int argc = 2;
    char** argv = const_cast<char**>(args);

    EXPECT_EQ(get_cmd_option(argv, argv + argc, "-k"), "");
}

TEST(CLIParserTest, GetOptionValueNotFound) {
    const char* args[] = {"./kmeans_seq", "-k", "5"};
    int argc = 3;
    char** argv = const_cast<char**>(args);

    EXPECT_EQ(get_cmd_option(argv, argv + argc, "-input"), "");
}
