#include "kmeans_core_mpi.hpp"
#include <mpi.h>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>
#include <omp.h>


double calculate_euclidean_distance_mpi(const double* p1, const double* p2, int dim) {
    double sum = 0.0;
    for (int i = 0; i < dim; ++i) {
        double diff = p1[i] - p2[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

bool assign_clusters_mpi(Points_Data_GPU& data) {
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

    #pragma omp parallel for schedule(static) reduction(|:changed_flag)
    for (int idx = 0; idx < n; ++idx) {
        double min_dist = std::numeric_limits<double>::max();
        int best_cluster = -1;

        for (int c = 0; c < k; ++c) {
            double dist = calculate_euclidean_distance_mpi(&points[idx * dim], &centroids[c * dim], dim);
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

    int global_changed = 0;
    MPI_Allreduce(&changed_flag, &global_changed, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);

    return global_changed != 0;
}

void update_centroids_mpi(Points_Data_GPU& data) {
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
    const int kdim = k * dim;

    #pragma omp parallel for schedule(static) \
        reduction(+:sums_ptr[:kdim]) reduction(+:counts_ptr[:k])
    for (int idx = 0; idx < n; ++idx) {
        int c = cluster_ids[idx];
        if (c >= 0 && c < k) {
            for (int j = 0; j < dim; ++j) {
                sums_ptr[c * dim + j] += points[idx * dim + j];
            }
            counts_ptr[c]++;
        }
    }

    std::vector<double> global_sums(k * dim, 0.0);
    std::vector<int> global_counts(k, 0);
    MPI_Allreduce(sums_ptr, global_sums.data(), kdim, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(counts_ptr, global_counts.data(), k, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    for (int i = 0; i < k; ++i) {
        if (global_counts[i] > 0) {
            for (int j = 0; j < dim; ++j) {
                data.centroids[i * dim + j] = global_sums[i * dim + j] / global_counts[i];
            }
        }
    }
}
