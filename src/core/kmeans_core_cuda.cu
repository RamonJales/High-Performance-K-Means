#include "kmeans_core_cuda.hpp"
#include <cuda_runtime.h>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>
#include <cfloat>

__device__ 
double calculate_euclidean_distance_cuda( const double* p1, const double* p2, int dim){
    double sum = 0.0;
    for(int i = 0; i < dim; i++) {
        double diff = p1[i] - p2[i];
        sum += diff * diff;
    }
    return sqrt(sum);
}

__global__
void assign_clusters_kernel( const double* points, const double* centroids, int* cluster_ids, double* min_distances, int* changed_flag, int n, int k, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= n)
        return;

    double min_dist = DBL_MAX;
    int best_cluster = -1;

    for(int c = 0; c < k; c++){
        double dist = calculate_euclidean_distance_cuda( &points[idx * dim], &centroids[c * dim], dim);
        if(dist < min_dist) {
            min_dist = dist;
            best_cluster = c;
        }
    }

    if(cluster_ids[idx] != best_cluster) {
        cluster_ids[idx] = best_cluster;
        min_distances[idx] = min_dist;
        atomicExch(changed_flag, 1); //
    }
}

bool assign_clusters_cuda(Points_Data_GPU& data) {
    if (data.k <= 0) {
        throw std::invalid_argument(
            "Invalid value: the number of clusters (K) must be greater than zero.");
    }

    const int n = data.n;
    const int k = data.k;
    const int dim = data.num_features;
    int changed_flag = 0;
    double* d_points;
    double* d_centroids;
    double* d_min_distances;
    int* d_cluster_ids;
    int* d_changed_flag;

    // cudaMalloc
    cudaMalloc(&d_points, n * dim * sizeof(double));
    cudaMalloc(&d_centroids, k * dim * sizeof(double));
    cudaMalloc(&d_cluster_ids, n * sizeof(int));
    cudaMalloc(&d_min_distances, n * sizeof(double));
    cudaMalloc(&d_changed_flag, sizeof(int));

    cudaMemcpy(d_points, data.points.data(), n * dim * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_centroids, data.centroids.data(), k * dim * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cluster_ids, data.cluster_ids.data(), n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_min_distances, data.min_distances.data(), n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_changed_flag, &changed_flag, sizeof(int), cudaMemcpyHostToDevice);

    // kernel<<<blocks,threads>>>
    const int threads_per_block = 256;
    const int blocks = (n + threads_per_block - 1) / threads_per_block;

    assign_clusters_kernel<<<blocks, threads_per_block>>>( d_points, d_centroids, d_cluster_ids, d_min_distances, d_changed_flag, n, k, dim );

    cudaDeviceSynchronize();

    cudaMemcpy(data.cluster_ids.data(), d_cluster_ids, n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(data.min_distances.data(), d_min_distances, n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&changed_flag, d_changed_flag, sizeof(int), cudaMemcpyDeviceToHost);
    
    // cudaFree
    cudaFree(d_points);
    cudaFree(d_centroids);
    cudaFree(d_cluster_ids);
    cudaFree(d_min_distances);
    cudaFree(d_changed_flag);

    return changed_flag != 0;
}

__global__
void update_centroids_kernel(const double* points, const int* cluster_ids, double* global_sums, int* global_counts, int n, int k, int dim) {
    extern __shared__ double local_sums[];
    int* local_counts =
        (int*)&local_sums[k * dim];
    int tid = threadIdx.x;

    //Shared Memory
    for(int i = tid; i < k * dim; i += blockDim.x) {
        local_sums[i] = 0.0;
    }

    for(int i = tid; i < k; i += blockDim.x) {
        local_counts[i] = 0;
    }

    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < n) {
        int cluster = cluster_ids[idx];
        if(cluster >= 0 && cluster < k) {
            atomicAdd(&local_counts[cluster], 1);
            for(int j = 0; j < dim; j++) {
                atomicAdd(&local_sums[cluster * dim + j], points[idx * dim + j]);
            }
        }
    }

    __syncthreads();

    for(int i = tid; i < k * dim; i += blockDim.x){
        atomicAdd(&global_sums[i], local_sums[i]);
    }

    for(int i = tid; i < k; i += blockDim.x){
        atomicAdd(&global_counts[i], local_counts[i]);
    }
}

__global__
void finalize_centroids_kernel( double* centroids, const double* sums, const int* counts, int k, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = k * dim;
    if(idx >= total)
        return;

    int cluster = idx / dim;
    int feature = idx % dim;

    if(counts[cluster] > 0){
        centroids[idx] = sums[idx] / counts[cluster];
    }
}

void update_centroids_cuda(Points_Data_GPU& data){
    const int n = data.n;
    const int k = data.k;
    const int dim = data.num_features;
    double* d_points;
    double* d_centroids;
    int* d_cluster_ids;
    double* d_sums;
    int* d_counts;

    cudaMalloc(&d_points, n * dim * sizeof(double));
    cudaMalloc(&d_centroids, k * dim * sizeof(double));
    cudaMalloc(&d_cluster_ids, n * sizeof(int));
    cudaMalloc(&d_sums, k * dim * sizeof(double));
    cudaMalloc(&d_counts, k * sizeof(int));

    cudaMemcpy(d_points, data.points.data(), n * dim * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy( d_centroids, data.centroids.data(), k * dim * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cluster_ids, data.cluster_ids.data(), n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_sums, 0, k * dim * sizeof(double));
    cudaMemset(d_counts, 0, k * sizeof(int));

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    
    size_t shared_mem_size = (k * dim * sizeof(double)) + (k * sizeof(int));
    update_centroids_kernel<<< blocks, threads, shared_mem_size >>>(d_points, d_cluster_ids, d_sums, d_counts, n, k, dim);
    cudaDeviceSynchronize();

    const int centroid_threads = 256;
    const int centroid_blocks = ((k * dim) + centroid_threads - 1) / centroid_threads;

    finalize_centroids_kernel<<< centroid_blocks, centroid_threads >>>( d_centroids, d_sums, d_counts, k, dim);
    
    cudaDeviceSynchronize();

    cudaMemcpy( data.centroids.data(), d_centroids, k * dim * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_points);
    cudaFree(d_centroids);
    cudaFree(d_cluster_ids);
    cudaFree(d_sums);
    cudaFree(d_counts);
}

