#include "kmeans_core_cuda.hpp"
#include <cuda_runtime.h>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include <cfloat>

// Wraps a CUDA runtime call and throws on failure, so errors surface as
// exceptions (like the other versions) instead of silently corrupting results.
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t _err = (call);                                             \
        if (_err != cudaSuccess) {                                             \
            throw std::runtime_error(std::string("CUDA error at ") + __FILE__  \
                + ":" + std::to_string(__LINE__) + " - "                       \
                + cudaGetErrorString(_err));                                   \
        }                                                                      \
    } while (0)

// Checks for asynchronous kernel-launch errors and waits for completion.
#define CUDA_CHECK_KERNEL()                                                    \
    do {                                                                       \
        CUDA_CHECK(cudaGetLastError());                                        \
        CUDA_CHECK(cudaDeviceSynchronize());                                   \
    } while (0)

__device__
double calculate_euclidean_distance_cuda(const double* p1, const double* p2, int dim) {
    double sum = 0.0;
    for (int i = 0; i < dim; i++) {
        double diff = p1[i] - p2[i];
        sum += diff * diff;
    }
    return sqrt(sum);
}

__global__
void assign_clusters_kernel(const double* points, const double* centroids, int* cluster_ids, double* min_distances, int* changed_flag, int n, int k, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n)
        return;

    double min_dist = DBL_MAX;
    int best_cluster = -1;

    for (int c = 0; c < k; c++) {
        double dist = calculate_euclidean_distance_cuda(&points[idx * dim], &centroids[c * dim], dim);
        if (dist < min_dist) {
            min_dist = dist;
            best_cluster = c;
        }
    }

    if (cluster_ids[idx] != best_cluster) {
        cluster_ids[idx] = best_cluster;
        min_distances[idx] = min_dist;
        atomicExch(changed_flag, 1);
    }
}

__global__
void update_centroids_kernel(const double* points, const int* cluster_ids, double* global_sums, int* global_counts, int n, int k, int dim) {
    extern __shared__ double local_sums[];
    int* local_counts = (int*)&local_sums[k * dim];
    int tid = threadIdx.x;

    // Shared Memory
    for (int i = tid; i < k * dim; i += blockDim.x) {
        local_sums[i] = 0.0;
    }

    for (int i = tid; i < k; i += blockDim.x) {
        local_counts[i] = 0;
    }

    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        int cluster = cluster_ids[idx];
        if (cluster >= 0 && cluster < k) {
            atomicAdd(&local_counts[cluster], 1);
            for (int j = 0; j < dim; j++) {
                atomicAdd(&local_sums[cluster * dim + j], points[idx * dim + j]);
            }
        }
    }

    __syncthreads();

    for (int i = tid; i < k * dim; i += blockDim.x) {
        atomicAdd(&global_sums[i], local_sums[i]);
    }

    for (int i = tid; i < k; i += blockDim.x) {
        atomicAdd(&global_counts[i], local_counts[i]);
    }
}

__global__
void finalize_centroids_kernel(double* centroids, const double* sums, const int* counts, int k, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = k * dim;
    if (idx >= total)
        return;

    int cluster = idx / dim;

    // Empty clusters keep their previous centroid (consistent with the OpenMP-GPU
    // and MPI versions), so only update when the cluster received points.
    if (counts[cluster] > 0) {
        centroids[idx] = sums[idx] / counts[cluster];
    }
}

void kmeans_cuda_init(KMeansCudaContext& ctx, const Points_Data_GPU& data) {
    if (data.k <= 0) {
        throw std::invalid_argument(
            "Invalid value: the number of clusters (K) must be greater than zero.");
    }

    ctx.n = data.n;
    ctx.k = data.k;
    ctx.dim = data.num_features;

    const size_t n = static_cast<size_t>(ctx.n);
    const size_t k = static_cast<size_t>(ctx.k);
    const size_t dim = static_cast<size_t>(ctx.dim);

    CUDA_CHECK(cudaMalloc(&ctx.d_points, n * dim * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&ctx.d_centroids, k * dim * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&ctx.d_cluster_ids, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ctx.d_min_distances, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&ctx.d_changed_flag, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ctx.d_sums, k * dim * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&ctx.d_counts, k * sizeof(int)));

    // Upload once: the immutable points, the initial centroids, and the initial
    // (all -1) cluster assignments. From here on everything stays on the device.
    CUDA_CHECK(cudaMemcpy(ctx.d_points, data.points.data(), n * dim * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ctx.d_centroids, data.centroids.data(), k * dim * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ctx.d_cluster_ids, data.cluster_ids.data(), n * sizeof(int), cudaMemcpyHostToDevice));
}

bool assign_clusters_cuda(KMeansCudaContext& ctx) {
    const int n = ctx.n;
    const int k = ctx.k;
    const int dim = ctx.dim;

    CUDA_CHECK(cudaMemset(ctx.d_changed_flag, 0, sizeof(int)));

    const int threads_per_block = 256;
    const int blocks = (n + threads_per_block - 1) / threads_per_block;

    assign_clusters_kernel<<<blocks, threads_per_block>>>(
        ctx.d_points, ctx.d_centroids, ctx.d_cluster_ids, ctx.d_min_distances,
        ctx.d_changed_flag, n, k, dim);
    CUDA_CHECK_KERNEL();

    // Only the small flag travels back to the host each iteration; the updated
    // assignments stay resident on the device for the next steps.
    int changed_flag = 0;
    CUDA_CHECK(cudaMemcpy(&changed_flag, ctx.d_changed_flag, sizeof(int), cudaMemcpyDeviceToHost));

    return changed_flag != 0;
}

void update_centroids_cuda(KMeansCudaContext& ctx) {
    const int n = ctx.n;
    const int k = ctx.k;
    const int dim = ctx.dim;

    CUDA_CHECK(cudaMemset(ctx.d_sums, 0, static_cast<size_t>(k) * dim * sizeof(double)));
    CUDA_CHECK(cudaMemset(ctx.d_counts, 0, static_cast<size_t>(k) * sizeof(int)));

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    size_t shared_mem_size = static_cast<size_t>(k) * dim * sizeof(double)
                           + static_cast<size_t>(k) * sizeof(int);
    update_centroids_kernel<<<blocks, threads, shared_mem_size>>>(
        ctx.d_points, ctx.d_cluster_ids, ctx.d_sums, ctx.d_counts, n, k, dim);
    CUDA_CHECK_KERNEL();

    const int centroid_threads = 256;
    const int centroid_blocks = ((k * dim) + centroid_threads - 1) / centroid_threads;

    finalize_centroids_kernel<<<centroid_blocks, centroid_threads>>>(
        ctx.d_centroids, ctx.d_sums, ctx.d_counts, k, dim);
    CUDA_CHECK_KERNEL();
}

void kmeans_cuda_finalize(KMeansCudaContext& ctx, Points_Data_GPU& data) {
    const size_t n = static_cast<size_t>(ctx.n);
    const size_t k = static_cast<size_t>(ctx.k);
    const size_t dim = static_cast<size_t>(ctx.dim);

    CUDA_CHECK(cudaMemcpy(data.centroids.data(), ctx.d_centroids, k * dim * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(data.cluster_ids.data(), ctx.d_cluster_ids, n * sizeof(int), cudaMemcpyDeviceToHost));

    cudaFree(ctx.d_points);
    cudaFree(ctx.d_centroids);
    cudaFree(ctx.d_cluster_ids);
    cudaFree(ctx.d_min_distances);
    cudaFree(ctx.d_changed_flag);
    cudaFree(ctx.d_sums);
    cudaFree(ctx.d_counts);

    ctx = KMeansCudaContext{};
}