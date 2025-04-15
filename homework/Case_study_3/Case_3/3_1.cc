/**
 * @file 3_1.cc
 * @brief Finite difference discretization of 2D Poisson equation on unit square.
 * 
 * @author Kaixaing Zou
 * @date April 2025
 *
 * This code sets up the linear system Ay = b for solving the Poisson equation
 * -Δu = f on a 2D unit square using 5-point finite difference stencil.
 * Dirichlet boundary condition u = 0 is assumed on the entire boundary.
 */

 #include <iostream>
 #include <vector>
 #include <cmath>
 
 const double PI = 3.14159265358979323846;
 
 /**
  * @brief Right-hand side function f(x, y) = 2π² sin(πx) sin(πy)
  * 
  * @param x x-coordinate
  * @param y y-coordinate
  * @return double value of f(x, y)
  */
 double f(double x, double y) {
     return 2 * PI * PI * sin(PI * x) * sin(PI * y);
 }
 
 /**
  * @brief Convert 2D grid indices to 1D index for linear system
  * 
  * @param i grid row index
  * @param j grid column index
  * @param N number of interior grid points per direction
  * @return int corresponding 1D index
  */
 int idx(int i, int j, int N) {
     return i * N + j;
 }
 
 /**
  * @brief Main routine: assemble matrix A and right-hand side vector b
  * 
  * The system corresponds to discretizing the Poisson problem on a uniform
  * (N+2)x(N+2) grid (including boundaries), with N^2 interior unknowns.
  */
 int main() {
     int N = 4;                    ///< Number of interior grid points in one dimension
     double h = 1.0 / (N + 1);     ///< Grid spacing
     int size = N * N;             ///< Total number of unknowns
 
     // Dense matrix A (size x size), initialized to 0
     std::vector<std::vector<double>> A(size, std::vector<double>(size, 0.0));
     
     // Right-hand side vector b
     std::vector<double> b(size, 0.0);
 
     // Loop over interior grid points
     for (int i = 0; i < N; ++i) {
         double x = (i + 1) * h;
         for (int j = 0; j < N; ++j) {
             double y = (j + 1) * h;
             int k = idx(i, j, N);
 
             // Compute b[k] using function f(x, y)
             b[k] = f(x, y);
 
             // 5-point stencil assembly for A
             A[k][k] = 4.0;
             if (i > 0)       A[k][idx(i - 1, j, N)] = -1.0;
             if (i < N - 1)   A[k][idx(i + 1, j, N)] = -1.0;
             if (j > 0)       A[k][idx(i, j - 1, N)] = -1.0;
             if (j < N - 1)   A[k][idx(i, j + 1, N)] = -1.0;
 
             // Scale right-hand side with h^2
             b[k] *= h * h;
         }
     }
 
     // Print matrix A
     std::cout << "Matrix A:\n";
     for (const auto& row : A) {
         for (double val : row)
             std::cout << val << "\t";
         std::cout << "\n";
     }
 
     // Print vector b
     std::cout << "\nVector b:\n";
     for (double val : b)
         std::cout << val << "\n";
 
     return 0;
 }
 