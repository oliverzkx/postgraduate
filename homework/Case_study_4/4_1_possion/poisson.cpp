#include "poisson.h"
#include <cmath>
#include <Eigen/Sparse>

// Generate right-hand side vector b, using f(x, y) = 2π² sin(πx) sin(πy)
std::vector<double> generate_rhs(int N, double h) {
    std::vector<double> b(N * N);
    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < N; ++i) {
            double x = (i + 1) * h; // x coordinate
            double y = (j + 1) * h; // y coordinate
            b[j * N + i] = 2 * M_PI * M_PI * sin(M_PI * x) * sin(M_PI * y);
        }
    }
    return b;
}

// Build sparse matrix A using 5-point stencil on a 2D grid
Eigen::SparseMatrix<double> build_poisson_matrix(int N, double h) {
    typedef Eigen::Triplet<double> T;
    std::vector<T> triplets;
    int size = N * N;

    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < N; ++i) {
            int idx = j * N + i;

            // Center point
            triplets.emplace_back(idx, idx, 4.0);

            // Left neighbor
            if (i > 0)       triplets.emplace_back(idx, idx - 1, -1.0);

            // Right neighbor
            if (i < N - 1)   triplets.emplace_back(idx, idx + 1, -1.0);

            // Down neighbor
            if (j > 0)       triplets.emplace_back(idx, idx - N, -1.0);

            // Up neighbor
            if (j < N - 1)   triplets.emplace_back(idx, idx + N, -1.0);
        }
    }

    Eigen::SparseMatrix<double> A(size, size);
    A.setFromTriplets(triplets.begin(), triplets.end());
    A *= 1.0 / (h * h); // Scale by 1/h²

    return A;
}
