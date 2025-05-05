#ifndef MG_H
#define MG_H

#include <Eigen/Sparse>
#include <vector>
#include "../4_1_possion/poisson.h"   // pull in build_poisson_matrix & generate_rhs

using SpMat = Eigen::SparseMatrix<double>;
using Vec   = Eigen::VectorXd;

// Build the 2D Poisson matrix of size N×N with mesh width h
//SpMat build_poisson_matrix(int N, double h);

// Build the right-hand side f(x,y)=2π²·sin(πx)·sin(πy)
//Vec   generate_rhs(int N, double h);
Vec   rhs_vec(int N, double h);

// One sweep of weighted Jacobi (here Gauss–Seidel) smoothing
void  smooth(const SpMat& A, const Vec& b, Vec& x,
             double omega, int nu);

// Full-weighting restriction: fine → coarse
Vec   restrict_full(const Vec& fine, int N_fine);

// Bilinear prolongation: coarse → fine
Vec   prolong_bilinear(const Vec& coarse, int N_fine);

// Direct solve on the coarsest grid
Vec   coarse_solve(const SpMat& A, const Vec& b);

// Recursive V-cycle from level `level` up to `lmax`
void  Vcycle(const std::vector<SpMat>& A_levels,
             std::vector<Vec>&          x_levels,
             const std::vector<Vec>&    b_levels,
             double omega, int nu,
             int level, int lmax);

#endif // MG_H
