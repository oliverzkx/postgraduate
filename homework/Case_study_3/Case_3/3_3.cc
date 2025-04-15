/**
 * @file 3_3.cc
 * @brief Solve the linear system A*y = b using the Conjugate Gradient (CG) method,
 *        where A is defined by A(i,j) = N - |i - j|. The right-hand side is chosen 
 *        as b = A*ones so that the exact solution is the ones vector.
 *
 * The stopping criterion is:
 *     ||r_k|| < max(reltol * ||r0||, abstol),
 * with reltol = 1e-8 and abstol = 1e-12.
 *
 * Residual norms per iteration are saved to "cg_convergence.csv" for plotting.
 *
 * A simple power iteration estimates the largest eigenvalue; a dummy value is returned
 * for the smallest eigenvalue.
 *
 * @author
 *   Kaixiang Zou
 * @date
 *   2025-04-13
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <chrono>
#include <fstream>
#include <iomanip>

/**
 * @brief Applies the matrix A to the vector x.
 *
 * The matrix A is defined by:
 *    A(i,j) = N - |i - j|, for i,j = 0,...,N-1.
 *
 * @param x The input vector (size N).
 * @param y The output vector (size N) which will hold y = A*x.
 * @param N The dimension of the matrix.
 */
void ApplyA(const std::vector<double>& x, std::vector<double>& y, int N) {
    for (int i = 0; i < N; i++) {
        double sum = 0.0;
        for (int j = 0; j < N; j++) {
            sum += (N - std::abs(i - j)) * x[j];
        }
        y[i] = sum;
    }
}

/**
 * @brief Computes the dot product of two vectors of length N.
 *
 * @param a The first vector.
 * @param b The second vector.
 * @param N The length of the vectors.
 * @return double The dot product.
 */
double dot(const std::vector<double>& a, const std::vector<double>& b, int N) {
    double s = 0.0;
    for (int i = 0; i < N; i++) {
        s += a[i] * b[i];
    }
    return s;
}

/**
 * @brief Computes the L2 norm of a vector of length N.
 *
 * @param a The input vector.
 * @param N The length of the vector.
 * @return double The L2 norm.
 */
double normL2(const std::vector<double>& a, int N) {
    return std::sqrt(dot(a, a, N));
}

/**
 * @brief Solves the linear system A*y = b using the Conjugate Gradient method.
 *
 * The right-hand side is defined as b = A*ones, where ones is the vector of all ones,
 * so that the exact solution is the ones vector.
 *
 * The residual norm is recorded in the vector residuals at each iteration.
 *
 * @param N The system size.
 * @param tol The relative tolerance.
 * @param abstol The absolute tolerance.
 * @param maxIter Maximum allowed iterations.
 * @param residuals Output vector to store the residual norm at each iteration.
 * @return std::vector<double> The computed solution vector y.
 */
std::vector<double> CG_solver(int N, double tol, double abstol, int maxIter, std::vector<double>& residuals) {
    std::vector<double> y(N, 0.0); // initial guess zero
    // b = A * ones, where ones is the all-one vector.
    std::vector<double> b(N, 0.0);
    std::vector<double> ones(N, 1.0);
    ApplyA(ones, b, N);

    // initial residual r = b - A*y; initial guess is zero -> r = b.
    std::vector<double> r = b;
    std::vector<double> p = r;
    std::vector<double> Ap(N, 0.0);

    double rnorm0 = normL2(r, N);
    residuals.push_back(rnorm0);

    int iter = 0;
    for (iter = 0; iter < maxIter; iter++) {
        double r_old_sq = dot(r, r, N);
        ApplyA(p, Ap, N);
        double alpha = r_old_sq / dot(p, Ap, N);

        // Update solution and residual.
        for (int i = 0; i < N; i++) {
            y[i] += alpha * p[i];
            r[i] -= alpha * Ap[i];
        }

        double rnorm = normL2(r, N);
        residuals.push_back(rnorm);

        // Check stopping criterion.
        if (rnorm < std::max(tol * rnorm0, abstol))
            break;

        double r_new_sq = dot(r, r, N);
        double beta = r_new_sq / r_old_sq;

        // Update search direction.
        for (int i = 0; i < N; i++) {
            p[i] = r[i] + beta * p[i];
        }
    }

    std::cout << "CG converged in " << iter << " iterations." << std::endl;
    return y;
}

/**
 * @brief Estimates the largest eigenvalue of A using power iteration.
 *
 * @param N The dimension of the matrix.
 * @param numIter The number of power iterations (default 1000).
 * @return double The estimated largest eigenvalue.
 */
double estimateMaxEigenvalue(int N, int numIter = 1000) {
    std::vector<double> x(N, 1.0);
    std::vector<double> Ax(N, 0.0);
    double lambda = 0.0;
    for (int k = 0; k < numIter; k++) {
        ApplyA(x, Ax, N);
        lambda = normL2(Ax, N);
        double normAx = normL2(Ax, N);
        for (int i = 0; i < N; i++) {
            x[i] = Ax[i] / normAx;
        }
    }
    return lambda;
}

/**
 * @brief Dummy function to estimate the smallest eigenvalue.
 *
 * For large N, an efficient method is needed. Here, a dummy value is returned.
 *
 * @param N The dimension of the matrix.
 * @return double The estimated smallest eigenvalue.
 */
double estimateMinEigenvalue(int N) {
    if (N <= 1000) {
        return 1.0; // dummy value for demonstration
    } else {
        std::cout << "Minimum eigenvalue estimation not implemented for large N." << std::endl;
        return 1.0;
    }
}

/**
 * @brief Main function.
 *
 * Constructs and solves the linear system A*y = b using the CG method for a given N.
 * Estimates eigenvalues, computes the condition number and theoretical convergence rate,
 * records residuals, and writes the convergence data to a CSV file.
 *
 * @return int Exit status.
 */
int main() {
    int N = 1000;  // Modify as needed (e.g., 100, 1000, 10000)
    double tol = 1e-8;
    double abstol = 1e-12;
    int maxIter = 10000;

    double lambda_max = estimateMaxEigenvalue(N);
    double lambda_min = estimateMinEigenvalue(N);
    double kappa = (lambda_min != 0.0) ? lambda_max / lambda_min : 0.0;
    double theoretical_rate = (std::sqrt(kappa) - 1) / (std::sqrt(kappa) + 1);

    std::cout << "Estimated lambda_max = " << lambda_max << "\n";
    std::cout << "Estimated lambda_min = " << lambda_min << "\n";
    std::cout << "Estimated condition number (kappa) = " << kappa << "\n";
    std::cout << "Theoretical convergence rate = " << theoretical_rate << "\n";

    std::vector<double> residuals;
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<double> y = CG_solver(N, tol, abstol, maxIter, residuals);
    auto end = std::chrono::high_resolution_clock::now();
    double timeElapsed = std::chrono::duration<double>(end - start).count();
    std::cout << "Time to solution: " << timeElapsed << " seconds." << std::endl;

    // Write convergence data to CSV.
    std::ofstream resFile("cg_convergence.csv");
    resFile << "Iteration,ResidualNorm\n";
    for (size_t i = 0; i < residuals.size(); i++) {
        resFile << i << "," << std::scientific << residuals[i] << "\n";
    }
    resFile.close();
    std::cout << "Residual convergence data saved to cg_convergence.csv" << std::endl;

    // Compute the sum of absolute errors relative to the exact solution (ones vector).
    double error = 0.0;
    for (int i = 0; i < N; i++) {
        error += std::fabs(y[i] - 1.0);
    }
    std::cout << "Sum of absolute errors relative to the exact solution: " << error << "\n";

    return 0;
}
