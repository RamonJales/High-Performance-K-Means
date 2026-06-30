#ifndef KMEANS_CORE_CUDA_HPP
#define KMEANS_CORE_CUDA_HPP

// kmeans_core_gpu.hpp is included to reuse the Points_Data_GPU structure for CUDA operations
#include "kmeans_core_gpu.hpp"

struct KMeansCudaContext {
    double* d_points        = nullptr; // n * dim   (immutable, uploaded once)
    double* d_centroids     = nullptr; // k * dim   (updated in place on device)
    int*    d_cluster_ids   = nullptr; // n         (updated in place on device)
    double* d_min_distances = nullptr; // n
    int*    d_changed_flag  = nullptr; // 1
    double* d_sums          = nullptr; // k * dim   (scratch for centroid update)
    int*    d_counts        = nullptr; // k         (scratch for centroid update)

    int n = 0;
    int k = 0;
    int dim = 0;
};

// Allocates the device buffers and uploads the immutable points plus the initial
// centroids and cluster assignments. Call once, before the loop.
void kmeans_cuda_init(KMeansCudaContext& ctx, const Points_Data_GPU& data);

// One assignment step (CUDA kernel). Returns true if any point changed cluster.
bool assign_clusters_cuda(KMeansCudaContext& ctx);

// One centroid-update step (CUDA kernels, in place on the device).
void update_centroids_cuda(KMeansCudaContext& ctx);

// Copies the final centroids and assignments back to host and frees the device
// buffers. Call once, after the loop.
void kmeans_cuda_finalize(KMeansCudaContext& ctx, Points_Data_GPU& data);

#endif