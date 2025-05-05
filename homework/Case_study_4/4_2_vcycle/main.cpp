#include <iostream>
#include <vector>
#include "mg.h"

int main(){
    // Problem setup
    int N      = 64;             // fine grid points per direction
    int lmax   = 2;              // number of levels (64→32→16)
    double h   = 1.0/(N+1);
    double omega = 2.0/3.0;      // Jacobi weight (unused in GS)
    int nu       = 3;            // pre/post smoothing sweeps

    // Build hierarchy
    std::vector<SpMat> A_levels;
    std::vector<Vec>   x_levels(lmax+1), b_levels(lmax+1);

    int Ncur = N;
    for(int l=0; l<=lmax; ++l){
        // build matrix and RHS on level l
        A_levels.push_back(build_poisson_matrix(Ncur,h));
        b_levels[l] = rhs_vec(Ncur,h);
        x_levels[l] = Vec::Zero(Ncur*Ncur);
        Ncur /= 2;
    }

    // Compute and print initial residual
    Vec& x0 = x_levels[0];
    Vec& b0 = b_levels[0];
    SpMat& A0 = A_levels[0];

    double r0 = (b0 - A0*x0).norm();
    std::cout << "Start residual: " << r0 << "\n";

    // Apply exactly one V-cycle
    Vcycle(A_levels, x_levels, b_levels, omega, nu, 0, lmax);

    double r1 = (b0 - A0*x0).norm();
    std::cout << "After one V-cycle: residual = " << r1 << "\n";

    return 0;
}
