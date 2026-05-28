#ifndef KMEANS_CORE_HPP
#define KMEANS_CORE_HPP

#include <vector>
#include <stdexcept>

/**
 * @brief Represents a single data point in the dataset.
 * * This structure holds the unique identifier, the mathematical features (coordinates)
 * in an N-dimensional space, and the current clustering state of the point.
 */
struct Point {
    int id;                         /**< Unique identifier for the point */
    std::vector<double> features;   /**< N-dimensional coordinates of the point */
    int cluster_id;                 /**< ID of the cluster this point is assigned to (-1 if unassigned) */
    double min_distance;            /**< Distance to the nearest centroid */

    /**
     * @brief Constructs a new Point object.
     * * @param id Unique identifier.
     * @param features Vector containing the N-dimensional features.
     */
    Point(int id, const std::vector<double>& features) 
        : id(id), features(features), cluster_id(-1), min_distance(__DBL_MAX__) {}
};

/**
 * @brief Calculates the Euclidean distance between two points in N-dimensional space.
 * * @param p1 Vector representing the features of the first point.
 * @param p2 Vector representing the features of the second point.
 * @return double The calculated Euclidean distance.
 * @throws std::invalid_argument If the dimensions of p1 and p2 do not match.
 */
double calculate_euclidean_distance(const std::vector<double>& p1, const std::vector<double>& p2);

/**
 * @brief Assigns each point to the nearest centroid based on Euclidean distance.
 * * This function iterates through all points, calculates their distance to every 
 * available centroid, and updates the point's `cluster_id` to the closest one.
 * * @param points Reference to the vector of points to be clustered.
 * @param centroids Vector containing the coordinates of the current centroids.
 * @return true If at least one point changed its cluster assignment.
 * @return false If no points changed clusters (convergence condition).
 * @throws std::invalid_argument If the centroids vector is empty.
 */
bool assign_clusters(std::vector<Point>& points, const std::vector<std::vector<double>>& centroids);

/**
 * @brief Recalculates the positions of the centroids based on current point assignments.
 * * The new position of a centroid is the arithmetic mean of all points currently
 * assigned to its cluster.
 * * @param points Vector of all points in the dataset.
 * @param k The number of clusters.
 * @param num_features The number of dimensions (features) each point has.
 * @return std::vector<std::vector<double>> A new vector containing the updated centroid coordinates.
 * @throws std::invalid_argument If k is less than or equal to 0.
 */
std::vector<std::vector<double>> update_centroids(const std::vector<Point>& points, int k, int num_features);

#endif // KMEANS_CORE_HPP
