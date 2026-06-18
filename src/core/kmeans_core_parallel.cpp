#include "kmeans_core.hpp"
#include <cmath>
#include <limits>
#include <stdexcept>
#include <omp.h>

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

bool assign_clusters_parallel(std::vector<Point>& points, const std::vector<double> centroids_flat) {
    if (centroids.empty()) {
        throw std::invalid_argument("Centroids vector cannot be empty.");
    }

    int changed_flag = 0;                              
    const int n = static_cast<int>(points.size());     
    const int k = static_cast<int>(centroids.size()); 

    #pragma omp parallel for schedule(static) reduction(|:changed_flag)
    for (int idx = 0; idx < n; ++idx) {
        Point& point = points[idx];
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
            changed_flag = 1;
        }
    }

    return changed_flag != 0;
}

std::vector<std::vector<double>> update_centroids(const std::vector<Point>& points, int k, int num_features) {
    if (k <= 0) {
        throw std::invalid_argument("Number of clusters (K) must be greater than 0.");
    }

    std::vector<std::vector<double>> new_centroids(k, std::vector<double>(num_features, 0.0));
    std::vector<int> counts(k, 0);
    const int n = static_cast<int>(points.size());     
 
    #pragma omp parallel                                 
    {
    std::vector<std::vector<double>> local_sums(k, std::vector<double>(num_features, 0.0));
    std::vector<int> local_counts(k, 0);

     // Sum all coordinates for each cluster
            #pragma omp for schedule(static)                 
        for (int idx = 0; idx < n; ++idx) {              
            const Point& point = points[idx];           
            if (point.cluster_id >= 0 && point.cluster_id < k) {
                for (int j = 0; j < num_features; ++j) {
                    local_sums[point.cluster_id][j] += point.features[j];  
                }
                local_counts[point.cluster_id]++;        
            }
        }
 
        #pragma omp critical                             
        {
            for (int c = 0; c < k; ++c) {
                counts[c] += local_counts[c];
                for (int j = 0; j < num_features; ++j) {
                    new_centroids[c][j] += local_sums[c][j];
                }
            }
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
