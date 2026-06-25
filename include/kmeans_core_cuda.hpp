#ifndef KMEANS_CORE_CUDA_HPP
#define KMEANS_CORE_CUDA_HPP

// points_data_gpu.hpp is included to reuse the Points_Data_GPU structure for CUDA operations
#include "kmeans_core_gpu.hpp"

bool assign_clusters_cuda(Points_Data_GPU& data);

void update_centroids_cuda(Points_Data_GPU& data);

#endif