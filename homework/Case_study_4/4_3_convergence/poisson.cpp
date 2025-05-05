#include "poisson.h"
#include <vector>
#include <cmath>

using Triplet = Eigen::Triplet<double>;

Eigen::SparseMatrix<double> build_poisson_matrix(int N, double h) {
    int M = N*N;
    std::vector<Triplet> tri;
    tri.reserve(5*M);

    double invh2 = 1.0/(h*h);

    for(int j=0; j<N; ++j) {
        for(int i=0; i<N; ++i) {
            int row = j*N + i;
            // diagonal
            tri.emplace_back(row, row, 4.0 * invh2);
            // left
            if(i>0)   tri.emplace_back(row, row-1, -1.0 * invh2);
            // right
            if(i<N-1) tri.emplace_back(row, row+1, -1.0 * invh2);
            // down
            if(j>0)   tri.emplace_back(row, row-N, -1.0 * invh2);
            // up
            if(j<N-1) tri.emplace_back(row, row+N, -1.0 * invh2);
        }
    }

    Eigen::SparseMatrix<double> A(M,M);
    A.setFromTriplets(tri.begin(), tri.end());
    return A;
}

Eigen::VectorXd generate_rhs(int N, double h) {
    int M = N*N;
    Eigen::VectorXd b(M);
    for(int j=0; j<N; ++j) {
        double y = (j+1)*h;
        for(int i=0; i<N; ++i) {
            double x = (i+1)*h;
            b[j*N + i] = 2.0 * M_PI * M_PI
                       * std::sin(M_PI * x)
                       * std::sin(M_PI * y);
        }
    }
    return b;
}
