#include "kmeans_core.hpp"
#include <cmath>
#include <limits>
#include <stdexcept>

double calculate_euclidean_distance(const std::vector<double>& p1, const std::vector<double>& p2) {
    if (p1.size() != p2.size()) {
        throw std::invalid_argument("Points must have the same number of dimensions.");
    }

    double sum = 0.0;
    for (size_t i = 0; i < p1.size(); ++i) {
        double diff = p1[i] - p2[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

bool assign_clusters(std::vector<Point>& points, const std::vector<std::vector<double>>& centroids) {
    if (centroids.empty()) {
        throw std::invalid_argument("Centroids vector cannot be empty.");
    }

    bool changed = false;

    for (auto& point : points) {
        double min_dist = std::numeric_limits<double>::max();
        int best_cluster = -1;

        for (size_t i = 0; i < centroids.size(); ++i) {
            double dist = calculate_euclidean_distance(point.features, centroids[i]);
            if (dist < min_dist) {
                min_dist = dist;
                best_cluster = static_cast<int>(i);
            }
        }

        if (point.cluster_id != best_cluster) {
            point.cluster_id = best_cluster;
            point.min_distance = min_dist;
            changed = true;
        }
    }

    return changed;
}

std::vector<std::vector<double>> update_centroids(const std::vector<Point>& points, int k, int num_features) {
    if (k <= 0) {
        throw std::invalid_argument("Number of clusters (K) must be greater than 0.");
    }

    std::vector<std::vector<double>> new_centroids(k, std::vector<double>(num_features, 0.0));
    std::vector<int> counts(k, 0);

    // Sum all coordinates for each cluster
    for (const auto& point : points) {
        if (point.cluster_id >= 0 && point.cluster_id < k) {
            for (int i = 0; i < num_features; ++i) {
                new_centroids[point.cluster_id][i] += point.features[i];
            }
            counts[point.cluster_id]++;
        }
    }

    // Calculate the mean
    for (int i = 0; i < k; ++i) {
        if (counts[i] > 0) {
            for (int j = 0; j < num_features; ++j) {
                new_centroids[i][j] /= counts[i];
            }
        }
    }

    return new_centroids;
}
