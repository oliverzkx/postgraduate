#include "mg.h"
#include <Eigen/SparseLU>
#include <cmath>
#include <vector>
#include <iostream>
#include "../4_1_possion/poisson.h"

//----------------------------------------------------------------------------
// 1) Build 2D Poisson matrix on N×N grid with mesh width h
//----------------------------------------------------------------------------
// SpMat build_poisson_matrix(int N, double h) {
//     typedef Eigen::Triplet<double> T;
//     std::vector<T> coef;
//     coef.reserve(N*N*5);

//     double invh2 = 1.0/(h*h);
//     for(int j=0;j<N;++j){
//       for(int i=0;i<N;++i){
//         int row = j*N + i;
//         coef.emplace_back(row,row,4.0*invh2);
//         if(i>0)   coef.emplace_back(row,row-1,-1.0*invh2);
//         if(i<N-1)coef.emplace_back(row,row+1,-1.0*invh2);
//         if(j>0)   coef.emplace_back(row,row-N,-1.0*invh2);
//         if(j<N-1)coef.emplace_back(row,row+N,-1.0*invh2);
//       }
//     }

//     SpMat A(N*N,N*N);
//     A.setFromTriplets(coef.begin(),coef.end());
//     return A;
// }

//----------------------------------------------------------------------------
// 2) RHS f(x,y)=2π²·sin(πx)·sin(πy) on interior points
//----------------------------------------------------------------------------
//Vec generate_rhs(int N, double h) {
Vec rhs_vec(int N, double h) {
    Vec b(N*N);
    for(int j=0;j<N;++j){
      double y=(j+1)*h;
      for(int i=0;i<N;++i){
        double x=(i+1)*h;
        b[j*N+i] = 2.0*M_PI*M_PI * std::sin(M_PI*x)*std::sin(M_PI*y);
      }
    }
    return b;
}

//----------------------------------------------------------------------------
// 3) “Smoothing” sweep: here simple Gauss–Seidel
//----------------------------------------------------------------------------
void smooth(const SpMat& A, const Vec& b, Vec& x,
            double /*omega*/, int nu)
{
    int n = A.rows();
    const int* outer = A.outerIndexPtr();
    const int* inner = A.innerIndexPtr();
    const double* val = A.valuePtr();

    for(int sweep=0;sweep<nu;++sweep){
      for(int i=0;i<n;++i){
        double sigma=0, diag=0;
        for(int k=outer[i];k<outer[i+1];++k){
          int j=inner[k]; double a=val[k];
          if(j==i) diag=a;
          else      sigma+=a*x[j];
        }
        x[i] = (b[i]-sigma)/diag;
      }
    }
}

//----------------------------------------------------------------------------
// 4) Restriction (full weighting): fine→coarse
//----------------------------------------------------------------------------
Vec restrict_full(const Vec& fine, int N_fine){
    int N_coarse = N_fine/2;
    Vec coarse = Vec::Zero(N_coarse*N_coarse);

    for(int jc=1; jc<N_coarse-1; ++jc){
      for(int ic=1; ic<N_coarse-1; ++ic){
        int ci = jc*N_coarse + ic;
        int fi = 2*jc*N_fine + 2*ic;
        coarse[ci] =
          0.25* fine[fi]
        + 0.125*(fine[fi-1]+fine[fi+1]
               +fine[fi-N_fine]+fine[fi+N_fine])
        + 0.0625*(fine[fi-N_fine-1]+fine[fi-N_fine+1]
                +fine[fi+N_fine-1]+fine[fi+N_fine+1]);
      }
    }
    return coarse;
}

//----------------------------------------------------------------------------
// 5) Prolongation (bilinear): coarse→fine
//----------------------------------------------------------------------------
Vec prolong_bilinear(const Vec& coarse, int N_fine){
    int N_coarse=N_fine/2;
    Vec fine=Vec::Zero(N_fine*N_fine);

    for(int jc=0;jc<N_coarse;++jc){
      for(int ic=0;ic<N_coarse;++ic){
        int ci=jc*N_coarse+ic;
        int fi=2*jc*N_fine + 2*ic;
        fine[fi] += coarse[ci];                    // center
        if(ic+1<N_fine)      fine[fi+1]      += 0.5*coarse[ci];
        if(jc+1<N_fine)      fine[fi+N_fine] += 0.5*coarse[ci];
        if(ic+1<N_fine && jc+1<N_fine)
                              fine[fi+N_fine+1]+=0.25*coarse[ci];
      }
    }
    return fine;
}

//----------------------------------------------------------------------------
// 6) Direct solve on coarsest grid with SparseLU
//----------------------------------------------------------------------------
Vec coarse_solve(const SpMat& A, const Vec& b){
    Eigen::SparseLU<SpMat> solver;
    solver.analyzePattern(A);
    solver.factorize(A);
    return solver.solve(b);
}

//----------------------------------------------------------------------------
// 7) Recursive V-cycle
//----------------------------------------------------------------------------
void Vcycle(
    const std::vector<SpMat>& A_levels,
    std::vector<Vec>&          x_levels,
    const std::vector<Vec>&    b_levels,
    double omega, int nu,
    int level, int lmax
){
    const SpMat& A = A_levels[level];
    Vec& x         = x_levels[level];
    const Vec& b   = b_levels[level];

    // Pre-smooth
    smooth(A,b,x,omega,nu);

    // Compute residual
    Vec r = b - A*x;

    if(level+1==lmax){
      // Coarsest: restrict→solve exactly→prolong
      Vec rc = restrict_full(r, std::sqrt(b.size()));
      Vec ec = coarse_solve(A_levels[lmax], rc);
      Vec ef = prolong_bilinear(ec, std::sqrt(b.size()));
      x += ef;
    } else {
      // Recursive call
      Vec rc     = restrict_full(r, std::sqrt(b.size()));
      x_levels[level+1] = Vec::Zero(rc.size());
      std::vector<Vec> b2 = b_levels;
      b2[level+1] = rc;

      Vcycle(A_levels, x_levels, b2, omega, nu, level+1, lmax);

      Vec ef = prolong_bilinear(x_levels[level+1], std::sqrt(b.size()));
      x += ef;
    }

    // Post-smooth
    smooth(A,b,x,omega,nu);
}
