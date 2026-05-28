#include <gtest/gtest.h>
#include "kmeans_core.hpp"

// ==========================================
// TEST SUITE: Euclidean Distance
// ==========================================

TEST(KMeansMathTest, EuclideanDistanceHappyPath) {
    std::vector<double> p1 = {0.0, 0.0};
    std::vector<double> p2 = {3.0, 4.0};
    EXPECT_DOUBLE_EQ(calculate_euclidean_distance(p1, p2), 5.0);
}

TEST(KMeansMathTest, EuclideanDistanceSamePoint) {
    std::vector<double> p1 = {7.5, 2.1};
    EXPECT_DOUBLE_EQ(calculate_euclidean_distance(p1, p1), 0.0);
}

TEST(KMeansMathTest, EuclideanDistanceDimensionMismatch) {
    std::vector<double> p1 = {1.0, 2.0};
    std::vector<double> p2 = {1.0, 2.0, 3.0};
    // Expect the function to throw an invalid_argument exception
    EXPECT_THROW(calculate_euclidean_distance(p1, p2), std::invalid_argument);
}

// ==========================================
// TEST SUITE: Cluster Assignment
// ==========================================

TEST(KMeansLogicTest, AssignClustersHappyPath) {
    std::vector<Point> points = {
        Point(0, {1.0, 1.0}),
        Point(1, {10.0, 10.0})
    };
    std::vector<std::vector<double>> centroids = {
        {0.0, 0.0},   // Centroid 0 (closer to Point 0)
        {11.0, 11.0}  // Centroid 1 (closer to Point 1)
    };

    bool changed = assign_clusters(points, centroids);

    EXPECT_TRUE(changed);
    EXPECT_EQ(points[0].cluster_id, 0);
    EXPECT_EQ(points[1].cluster_id, 1);
}

TEST(KMeansLogicTest, AssignClustersEmptyCentroids) {
    std::vector<Point> points = { Point(0, {1.0, 1.0}) };
    std::vector<std::vector<double>> empty_centroids;

    EXPECT_THROW(assign_clusters(points, empty_centroids), std::invalid_argument);
}

// ==========================================
// TEST SUITE: Update Centroids
// ==========================================

TEST(KMeansLogicTest, UpdateCentroidsHappyPath) {
    std::vector<Point> points = {
        Point(0, {2.0, 2.0}),
        Point(1, {4.0, 4.0}),
        Point(2, {10.0, 10.0})
    };
    
    // Assign clusters manually for the test
    points[0].cluster_id = 0;
    points[1].cluster_id = 0;
    points[2].cluster_id = 1;

    int k = 2;
    int num_features = 2;
    auto new_centroids = update_centroids(points, k, num_features);

    // Centroid 0 should be mean of (2,2) and (4,4) -> (3,3)
    EXPECT_DOUBLE_EQ(new_centroids[0][0], 3.0);
    EXPECT_DOUBLE_EQ(new_centroids[0][1], 3.0);

    // Centroid 1 should be mean of (10,10) -> (10,10)
    EXPECT_DOUBLE_EQ(new_centroids[1][0], 10.0);
    EXPECT_DOUBLE_EQ(new_centroids[1][1], 10.0);
}

TEST(KMeansLogicTest, UpdateCentroidsInvalidK) {
    std::vector<Point> points = { Point(0, {1.0, 1.0}) };
    EXPECT_THROW(update_centroids(points, 0, 2), std::invalid_argument);
    EXPECT_THROW(update_centroids(points, -1, 2), std::invalid_argument);
}
