#ifndef MG_H
#define MG_H

#include <Eigen/Sparse>
#include <vector>

using SpMat = Eigen::SparseMatrix<double>;
using Vec   = Eigen::VectorXd;

// Build the 2D Poisson matrix of size N×N with mesh spacing h
SpMat build_poisson_matrix(int N, double h);

// (Removed generate_rhs declaration here – it belongs in poisson.h)

// One sweep of weighted Jacobi (here implemented as Gauss–Seidel)
void smooth(const SpMat& A, const Vec& b, Vec& x,
            double omega, int nu);

// Full-weighting restriction: fine → coarse
Vec restrict_full(const Vec& fine, int N_fine);

// Bilinear prolongation: coarse → fine
Vec prolong_bilinear(const Vec& coarse, int N_fine);

// Direct solve on the coarsest grid
Vec coarse_solve(const SpMat& A, const Vec& b);

// Recursive V-cycle from level `level` up to `lmax`
void Vcycle(const std::vector<SpMat>& A_levels,
            std::vector<Vec>&          x_levels,
            const std::vector<Vec>&    b_levels,
            double omega, int nu,
            int level, int lmax);

#endif // MG_H
