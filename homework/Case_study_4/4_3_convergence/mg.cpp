// mg.cpp — recursive V-cycle implementation (no run_experiment, no RHS generation)

#include "mg.h"
#include <Eigen/SparseLU>

//----------------------------------------------------------------------------
// 1) Gauss–Seidel smoothing (in place)
void smooth(const SpMat& A,
            const Vec&  b,
            Vec&        x,
            double /*omega*/,
            int         nu)
{
    int n = A.rows();
    const int* outer = A.outerIndexPtr();
    const int* inner = A.innerIndexPtr();
    const double* val = A.valuePtr();

    for(int sweep = 0; sweep < nu; ++sweep)
      for(int i = 0; i < n; ++i){
        double diag = 0.0, sigma = 0.0;
        for(int k = outer[i]; k < outer[i+1]; ++k){
          int    j = inner[k];
          double a = val[k];
          if(j == i) diag = a;
          else       sigma += a * x[j];
        }
        x[i] = (b[i] - sigma) / diag;
      }
}

//----------------------------------------------------------------------------
// 2) Full-weighting restriction: fine → coarse
Vec restrict_full(const Vec& fine, int N_fine) {
    int N_coarse = N_fine/2;
    Vec coarse = Vec::Zero(N_coarse*N_coarse);
    for(int jc=1; jc<N_coarse-1; ++jc)
    for(int ic=1; ic<N_coarse-1; ++ic){
      int ci = jc*N_coarse + ic;
      int fi = (2*jc)*N_fine + (2*ic);
      coarse[ci] =
        0.25  * fine[fi] +
        0.125 * (fine[fi-1] + fine[fi+1]
               + fine[fi-N_fine] + fine[fi+N_fine]) +
        0.0625* (fine[fi-N_fine-1] + fine[fi-N_fine+1]
               + fine[fi+N_fine-1] + fine[fi+N_fine+1]);
    }
    return coarse;
}

//----------------------------------------------------------------------------
// 3) Bilinear prolongation: coarse → fine
Vec prolong_bilinear(const Vec& coarse, int N_fine) {
    int N_coarse = N_fine/2;
    Vec fine = Vec::Zero(N_fine*N_fine);
    for(int jc=0; jc<N_coarse; ++jc)
    for(int ic=0; ic<N_coarse; ++ic){
      int ci = jc*N_coarse + ic;
      int fi = (2*jc)*N_fine + (2*ic);
      fine[fi] += coarse[ci];
      if(ic+1 < N_fine)   fine[fi+1]          += 0.5*coarse[ci];
      if(jc+1 < N_fine)   fine[fi+N_fine]     += 0.5*coarse[ci];
      if(ic+1<N_fine && jc+1<N_fine)
                          fine[fi+N_fine+1] += 0.25*coarse[ci];
    }
    return fine;
}

//----------------------------------------------------------------------------
// 4) Direct solve at coarsest level
Vec coarse_solve(const SpMat& A, const Vec& b) {
    Eigen::SparseLU<SpMat> solver;
    solver.analyzePattern(A);
    solver.factorize(A);
    return solver.solve(b);
}

//----------------------------------------------------------------------------
// 5) Recursive V-cycle
void Vcycle(const std::vector<SpMat>& A_levels,
            std::vector<Vec>&          x_levels,
            const std::vector<Vec>&    b_levels,
            double omega, int nu,
            int level, int lmax)
{
    const SpMat& A = A_levels[level];
    Vec&         x = x_levels[level];
    const Vec&   b = b_levels[level];

    // pre-smoothing
    smooth(A, b, x, omega, nu);

    // compute residual
    Vec r = b - A*x;

    if(level+1 == lmax){
      // coarsest: restrict→solve→prolong
      Vec r_coarse = restrict_full(r, (int)std::sqrt(b.size()));
      Vec e_coarse = coarse_solve(A_levels[lmax], r_coarse);
      Vec e_fine   = prolong_bilinear(e_coarse, (int)std::sqrt(b.size()));
      x += e_fine;
    } else {
      // V-cycle recursion
      Vec r_coarse = restrict_full(r, (int)std::sqrt(b.size()));
      x_levels[level+1] = Vec::Zero(r_coarse.size());
      auto b_new = b_levels;
      b_new[level+1] = r_coarse;
      Vcycle(A_levels, x_levels, b_new, omega, nu, level+1, lmax);
      Vec e_fine = prolong_bilinear(x_levels[level+1], (int)std::sqrt(b.size()));
      x += e_fine;
    }

    // post-smoothing
    smooth(A, b, x, omega, nu);
}
