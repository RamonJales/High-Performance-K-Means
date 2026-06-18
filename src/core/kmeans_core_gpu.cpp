#include "kmeans_core_gpu.hpp"
#include <cmath>
#include <limits>
#include <stdexcept>
#include <omp.h>


#pragma omp declare target
double calculate_euclidean_distance_gpu(const double* p1, const double* p2, int dim) {
    double sum = 0.0;
    for (int i = 0; i < dim; ++i) {
        double diff = p1[i] - p2[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}
#pragma omp end declare target

bool assign_clusters_gpu(Points_Data_GPU& data) {
    if (data.k <= 0) {
        throw std::invalid_argument("Invalid value: the number of clusters (K) must be greater than zero.");
    }

    const int n = data.n;
    const int k = data.k;
    const int dim = data.num_features;
    int changed_flag = 0;

    double* points = data.points.data();
    double* centroids = data.centroids.data();
    int* cluster_ids = data.cluster_ids.data();
    double* min_distances = data.min_distances.data();

    #pragma omp target teams distribute parallel for \
        map(to: points[0:n*dim], centroids[0:k*dim]) \
        map(tofrom: cluster_ids[0:n], min_distances[0:n]) \
        reduction(|:changed_flag)
    for (int idx = 0; idx < n; ++idx) {
        double min_dist = std::numeric_limits<double>::max();
        int best_cluster = -1;

        for (int c = 0; c < k; ++c) {
            double dist = calculate_euclidean_distance_gpu(&points[idx * dim], &centroids[c * dim], dim);
            if (dist < min_dist) {
                min_dist = dist;
                best_cluster = c;
            }
        }

        if (cluster_ids[idx] != best_cluster) {
            cluster_ids[idx] = best_cluster;
            min_distances[idx] = min_dist;
            changed_flag = 1;
        }
    }

    return changed_flag != 0;
}

void update_centroids_gpu(Points_Data_GPU& data) {
    if (data.k <= 0) {
        throw std::invalid_argument("Invalid value: the number of clusters (K) must be greater than zero.");
    }

    const int n = data.n;
    const int k = data.k;
    const int dim = data.num_features;

    std::vector<double> sums(k * dim, 0.0);
    std::vector<int> counts(k, 0);

    double* points = data.points.data();
    int* cluster_ids = data.cluster_ids.data();
    double* sums_ptr = sums.data();
    int* counts_ptr = counts.data();
    
    #pragma omp target teams distribute parallel for \
        map(to: points[0:n*dim], cluster_ids[0:n]) \
        map(tofrom: sums_ptr[0:k*dim], counts_ptr[0:k])
    for (int idx = 0; idx < n; ++idx) {
        int c = cluster_ids[idx];
        if (c >= 0 && c < k) {
            #pragma omp atomic
            counts_ptr[c]++;

            for (int j = 0; j < dim; ++j) {
                #pragma omp atomic
                sums_ptr[c * dim + j] += points[idx * dim + j];
            }
        }
    }

    for (int i = 0; i < k; ++i) {
        if (counts[i] > 0) {
            for (int j = 0; j < dim; ++j) {
                data.centroids[i * dim + j] = sums[i * dim + j] / counts[i];
            }
        }
    }
}    
       
