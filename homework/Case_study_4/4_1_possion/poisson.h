#pragma once
#include <vector>
#include <Eigen/Sparse>

// Generate the right-hand side vector b for -Δu = f
std::vector<double> generate_rhs(int N, double h);

// Build the sparse matrix A using the 5-point finite difference stencil
Eigen::SparseMatrix<double> build_poisson_matrix(int N, double h);
