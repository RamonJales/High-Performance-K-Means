
#ifndef KMEANS_GPU_HPP
#define KMEANS_GPU_HPP
 
#include <vector>
 
// SoA layout (Structure of Arrays)
struct Points_Data_GPU {
    std::vector<double> points;        // n * num_features
    std::vector<double> centroids;     // k * num_features
    std::vector<int> cluster_ids;      // n (initialize with -1)
    std::vector<double> min_distances; // n
 
    int n = 0;
    int k = 0;
    int num_features = 0;
};
 
bool assign_clusters_gpu(Points_Data_GPU& data);
void update_centroids_gpu(Points_Data_GPU& data);
 
#endif
