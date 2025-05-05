#include <iostream>
#include "poisson.h"

int main() {
    int N = 64;                // Number of interior points per direction
    double h = 1.0 / (N + 1);  // Grid spacing

    // Generate matrix A and right-hand side b
    auto A = build_poisson_matrix(N, h);
    auto b = generate_rhs(N, h);

    // Print information
    std::cout << "Matrix A size: " << A.rows() << " x " << A.cols() << std::endl;
    std::cout << "Vector b size: " << b.size() << std::endl;
    std::cout << "First value of b: " << b[0] << std::endl;

    return 0;
}
