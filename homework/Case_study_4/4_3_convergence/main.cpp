// 4_3_convergence/main.cpp
// ------------------------------------------------------------
// Convergence study for multigrid (MG) – Case‑study 4.3
//
//  (1) For a fixed fine grid (N = 128) vary the maximum MG
//      level  lmax = 2 … 5 and stop when ||r|| < 1e‑7.
//  (2) Compare two‑level vs. “max‑level” MG for N = 16 … 256.
// ------------------------------------------------------------

#include <iostream>
#include <vector>
#include <chrono>
#include <Eigen/Core>
#include "poisson.h"   // build_poisson_matrix, generate_rhs
#include "mg.h"        // Vcycle + helpers

using Vec   = Eigen::VectorXd;
using SpMat = Eigen::SparseMatrix<double>;

struct Result {
    int    N;          // grid points per dimension
    int    lmax;       // deepest level used
    int    iters;      // V‑cycle iterations
    double residual;   // final ||r||
    double time_sec;   // runtime in seconds
};

//--------------------------------------------------------------
// run_one : solve on a single finest‑grid size / hierarchy
//--------------------------------------------------------------
Result run_one(int N, int lmax, double tol,
               double omega = 2.0/3.0, int nu = 3)
{
    /* ---------- build multigrid hierarchy ---------- */
    const int levels = lmax + 1;
    std::vector<SpMat> A(levels);
    std::vector<Vec>   b(levels), x(levels);

    int Ncur = N;
    for (int lvl = 0; lvl <= lmax; ++lvl) {
        const double h = 1.0 / (Ncur + 1);
        const int    sz = Ncur * Ncur;

        A[lvl] = build_poisson_matrix(Ncur, h);
        b[lvl] = generate_rhs(Ncur, h);
        x[lvl] = Vec::Zero(sz);

        Ncur /= 2;          // next coarser grid
    }

    /* ---------- iterate V‑cycles ---------- */
    int    iter = 0;
    double res  = (b[0] - A[0] * x[0]).norm();

    const auto t0 = std::chrono::high_resolution_clock::now();
    while (res > tol) {
        Vcycle(A, x, b, omega, nu, /*level=*/0, lmax);
        ++iter;
        res = (b[0] - A[0] * x[0]).norm();
    }
    const auto t1 = std::chrono::high_resolution_clock::now();
    const double elapsed = std::chrono::duration<double>(t1 - t0).count();

    return { N, lmax, iter, res, elapsed };
}

//--------------------------------------------------------------
// main : perform the two required experiments
//--------------------------------------------------------------
int main()
{
    const double tol   = 1e-7;

    std::cout << "#  N  scheme  lmax  iters  residual  time[s]\n";

    /* -------- Experiment (2) : 2‑level vs max‑level -------- */
    for (int N : {16, 32, 64, 128, 256}) {

        /* two‑level : lmax = 1 (coarse grid  N/2) */
        Result two  = run_one(N, /*lmax=*/1, tol);

        /* max‑level : coarsest grid size = 8  ->  lmax = log2(N/8)+1 */
        int lmax_full = 1;          // start with 2 levels
        int tmp = N;
        while (tmp > 8) { tmp >>= 1; ++lmax_full; }

        Result full = run_one(N, lmax_full, tol);

        /* ----------- print formatted table ----------- */
        std::cout << two.N << "  2‑level  "
                  << two.lmax    << "  "
                  << two.iters   << "  "
                  << two.residual<< "  "
                  << two.time_sec<< '\n';

        std::cout << full.N << "  max‑level  "
                  << full.lmax    << "  "
                  << full.iters   << "  "
                  << full.residual<< "  "
                  << full.time_sec<< '\n';
    }

    return 0;
}
