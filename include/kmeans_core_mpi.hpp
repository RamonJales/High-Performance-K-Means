#ifndef KMEANS_CORE_MPI_HPP
#define KMEANS_CORE_MPI_HPP

// Reuse the project's SoA container (Structure of Arrays). It is the same
// layout the GPU version uses (flat points/centroids), which is exactly what
// MPI_Scatterv and MPI_Allreduce need. Each rank holds only its LOCAL slice of
// points; the centroids are replicated (identical) on every rank.
#include "kmeans_core_gpu.hpp"

// Assigns each LOCAL point to the nearest centroid using OpenMP threads, then
// OR-reduces the "changed" flag across all MPI ranks.
//
// @param data Local slice of points + replicated centroids (SoA layout).
// @return true if at least one point changed cluster on ANY rank
//         (false signals global convergence).
bool assign_clusters_mpi(Points_Data_GPU& data);

// Recomputes the centroids from the GLOBAL point assignments. Each rank sums
// its local points per cluster with OpenMP, the partial sums/counts are
// combined across ranks with MPI_Allreduce, and every rank recomputes the same
// (identical) centroids in place.
//
// @param data Local slice of points + replicated centroids (SoA layout).
void update_centroids_mpi(Points_Data_GPU& data);

#endif
