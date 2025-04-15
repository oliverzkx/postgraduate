/**
 * @file 3_2.cc
 * @brief Solve the 2D Poisson problem using the 5-point finite difference 
 *        discretization and the Conjugate Gradient (CG) method.
 *
 * This program solves the following problem on the unit square (0,1)×(0,1):
 *
 *      -Δu = f(x,y)   in (0,1)×(0,1)
 *         u = 0      on the boundary,
 *
 * where f(x,y) = 2π² sin(πx) sin(πy). The finite difference discretization 
 * leads to a symmetric positive definite linear system A * x = b.
 *
 * The program is designed to run for several grid resolutions (number of 
 * interior points per dimension, e.g., N = 8, 16, 32, 64, 128, 256), and for each,
 * it tabulates the number of CG iterations and the time to solution.
 *
 * In addition, a one-dimensional profile of the computed solution along a horizontal
 * line (at y = 0.5) is extracted and written to a CSV file (u_profile.csv) so that it 
 * can be plotted later.
 *
 * The initial guess for the solution is perturbed by a small random amount (instead 
 * of starting at zero) to avoid any special alignment with the eigenvectors of A.
 *
 * @author Kaixiang Zou
 * @date 2025-04-13
 */

 #include <iostream>
 #include <vector>
 #include <cmath>
 #include <cstdlib>
 #include <ctime>
 #include <iomanip>
 #include <chrono>    // for timing
 #include <fstream>   // for file output
 
 /**
  * @brief Maps two-dimensional indices (i, j) to a one-dimensional index.
  * 
  * @param i The row index.
  * @param j The column index.
  * @param N The number of interior grid points per dimension.
  * @return int The corresponding one-dimensional index (in row-major order).
  */
 inline int idx2D(int i, int j, int N) {
     return i * N + j;
 }
 
 /**
  * @brief Applies the five-point finite difference matrix A to a vector.
  * 
  * The discrete Laplacian with Dirichlet boundary conditions is given by:
  * 
  *   A(i,j)(i,j)         = 4,
  *   A(i,j)(i-1,j), (i+1,j), (i,j-1), (i,j+1) = -1,
  *
  * with contributions from points outside the domain omitted.
  *
  * @param x The input vector.
  * @param r The output vector to store the result A * x.
  * @param N The number of interior grid points per dimension.
  */
 void ApplyA(const std::vector<double>& x, std::vector<double>& r, int N) {
     for (int i = 0; i < N; i++) {
         for (int j = 0; j < N; j++) {
             int index = idx2D(i, j, N);
             double value = 4.0 * x[index];
             if (j > 0)         value -= x[idx2D(i, j - 1, N)];
             if (j < N - 1)     value -= x[idx2D(i, j + 1, N)];
             if (i > 0)         value -= x[idx2D(i - 1, j, N)];
             if (i < N - 1)     value -= x[idx2D(i + 1, j, N)];
             r[index] = value;
         }
     }
 }
 
 /**
  * @brief Computes the dot product of two vectors.
  * 
  * @param u The first vector.
  * @param v The second vector.
  * @return double The dot product u·v.
  */
 double dot(const std::vector<double>& u, const std::vector<double>& v) {
     double sum = 0.0;
     for (size_t i = 0; i < u.size(); i++) {
         sum += u[i] * v[i];
     }
     return sum;
 }
 
 /**
  * @brief Computes the L2 (Euclidean) norm of a vector.
  * 
  * @param v The input vector.
  * @return double The L2 norm of v.
  */
 double normL2(const std::vector<double>& v) {
     return std::sqrt(dot(v, v));
 }
 
 /**
  * @brief Solves the discretized 2D Poisson problem using the Conjugate Gradient (CG) method.
  * 
  * The system A * x = b is set up by discretizing the 2D Poisson equation on the unit square
  * with a 5-point finite difference scheme. The right-hand side is given by:
  * 
  *    f(x,y) = 2π² sin(πx) sin(πy),
  *
  * scaled by h², where h = 1/(N+1). The function performs the CG iterations until the relative 
  * residual is below the specified tolerance. The function also measures the time taken by the CG loop.
  *
  * @param N The number of interior grid points per dimension.
  * @param tol The tolerance for the relative residual.
  * @param cgIterations (Output) The number of CG iterations performed.
  * @param timeElapsed (Output) The elapsed time (in seconds) for the CG iterations.
  * @return std::vector<double> The computed solution vector.
  */
 std::vector<double> solvePoissonCG(int N, double tol, int &cgIterations, double &timeElapsed) {
     double h = 1.0 / (N + 1);
     int size = N * N;
     
     // Assemble the right-hand side vector b.
     // f(x,y) = 2π² sin(πx) sin(πy) is scaled by h².
     std::vector<double> b(size, 0.0);
     for (int i = 0; i < N; i++) {
         double x_coord = (i + 1) * h;
         for (int j = 0; j < N; j++) {
             double y_coord = (j + 1) * h;
             b[idx2D(i, j, N)] = 2.0 * M_PI * M_PI * std::sin(M_PI * x_coord) * std::sin(M_PI * y_coord) * (h * h);
         }
     }
     
     // Initialize the solution vector x with a small random perturbation.
     std::vector<double> x(size, 0.0);
     for (int k = 0; k < size; k++) {
         // Random perturbation in the range [-1e-3, 1e-3]
         x[k] = 1e-3 * ((double)rand() / RAND_MAX * 2.0 - 1.0);
     }
     
     // Compute the initial residual r = b - A * x.
     std::vector<double> r(size, 0.0);
     std::vector<double> Ax(size, 0.0);
     ApplyA(x, Ax, N);
     for (int k = 0; k < size; k++) {
         r[k] = b[k] - Ax[k];
     }
     
     // Initialize the CG search direction and allocate a temporary vector Ap.
     std::vector<double> p = r;
     std::vector<double> Ap(size, 0.0);
     
     double rr_old = dot(r, r);
     double rr_new = 0.0;
     double alpha, beta;
     double normb = normL2(b);
     int iter = 0;
     const int maxIter = 100000;
     
     // Start timer for the CG iterations.
     auto start = std::chrono::high_resolution_clock::now();
     
     // Main Conjugate Gradient loop.
     for (iter = 0; iter < maxIter; iter++) {
         ApplyA(p, Ap, N);
         double pAp = dot(p, Ap);
         if (std::fabs(pAp) < 1e-16) {
             std::cout << "pAp is nearly zero, stopping iteration." << std::endl;
             break;
         }
         alpha = rr_old / pAp;
         for (int k = 0; k < size; k++) {
             x[k] += alpha * p[k];
             r[k] -= alpha * Ap[k];
         }
         rr_new = dot(r, r);
         double relRes = std::sqrt(rr_new) / ((normb > 1e-14) ? normb : 1.0);
         if (relRes < tol)
             break;
         beta = rr_new / rr_old;
         for (int k = 0; k < size; k++) {
             p[k] = r[k] + beta * p[k];
         }
         rr_old = rr_new;
     }
     
     // Stop timer and compute the elapsed time (in seconds).
     auto end = std::chrono::high_resolution_clock::now();
     timeElapsed = std::chrono::duration<double>(end - start).count();
     
     cgIterations = iter;
     return x;
 }
 
 /**
  * @brief Main function.
  * 
  * Iterates over a set of grid sizes to solve the Poisson problem, tabulating the number
  * of CG iterations and time to solution for each case. Also extracts a one-dimensional 
  * profile of the computed solution along y = 0.5 and writes it to a CSV file for plotting.
  *
  * @return int Exit status.
  */
 int main() {
     // Seed the random number generator.
     srand(static_cast<unsigned int>(time(NULL)));
     
     double tol = 1e-8;
     std::vector<int> gridSizes = {8, 16, 32, 64, 128, 256};
     
     // Print header for the performance table.
     std::cout << "   N       CG_iter       Time (sec)" << std::endl;
     std::cout << "-------------------------------------" << std::endl;
     
     // Open a CSV file to save performance data.
     std::ofstream tableFile("cg_performance.csv");
     tableFile << "N,CG_iter,Time_sec\n";
     
     std::vector<double> solutionForPlot; // To hold the solution for one selected grid size.
     int gridToPlot = 128; // Select one grid size for plotting the 1D profile.
     
     // Loop over each grid size.
     for (int N : gridSizes) {
         int cgIterations = 0;
         double timeElapsed = 0.0;
         std::vector<double> u = solvePoissonCG(N, tol, cgIterations, timeElapsed);
         std::cout << std::setw(5) << N 
                   << std::setw(15) << cgIterations 
                   << std::setw(15) << std::fixed << std::setprecision(6) << timeElapsed
                   << std::endl;
         tableFile << N << "," << cgIterations << "," << timeElapsed << "\n";
         
         // Save the solution for the grid selected for plotting.
         if (N == gridToPlot)
             solutionForPlot = u;
     }
     tableFile.close();
     
     // --- Extract and Save the 1D Profile of u(x) ---
     // For a grid of size N x N, extract the solution along the mid-line (y = 0.5).
     if (!solutionForPlot.empty()) {
         int N = gridToPlot;
         double h = 1.0 / (N + 1);
         int midRow = N / 2;  // Choose the middle row.
         std::vector<double> x_vals;
         std::vector<double> u_profile;
         for (int j = 0; j < N; j++) {
             double x_coord = (j + 1) * h;
             double u_val = solutionForPlot[idx2D(midRow, j, N)];
             x_vals.push_back(x_coord);
             u_profile.push_back(u_val);
         }
         
         // Write the 1D solution profile to a CSV file.
         std::ofstream profileFile("u_profile.csv");
         profileFile << "x,u(x)\n";
         for (size_t i = 0; i < x_vals.size(); i++) {
             profileFile << x_vals[i] << "," << u_profile[i] << "\n";
         }
         profileFile.close();
         
         std::cout << "A 1D profile of u(x) at y=0.5 has been saved to u_profile.csv." << std::endl;
     }
     
     return 0;
 }
 