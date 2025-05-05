#pragma once
#include <Eigen/Sparse>
#include <Eigen/Core>

// Build the 2D Poisson matrix A of size N×N (five-point stencil) with mesh width h.
// The matrix corresponds to -(1/h²) Δ on the interior.
Eigen::SparseMatrix<double> build_poisson_matrix(int N, double h);

// Generate the right-hand side vector b of length N²
// for f(x,y) = 2π² sin(π x) sin(π y) on the unit square.
Eigen::VectorXd generate_rhs(int N, double h);
