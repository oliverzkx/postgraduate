#!/usr/bin/env python3
"""
analyze_cg_convergence.py

This script reads the CSV file "cg_convergence.csv" which contains:
    Iteration,ResidualNorm
generated by the CG solver in your third assignment problem.

It then:
1. Prints some basic statistics about the convergence (final residual, min residual, etc.).
2. Plots the residual norm vs. iteration on a semilog scale.
3. Saves the plot as "cg_convergence.png".

Usage:
    python3 analyze_cg_convergence.py
"""

import pandas as pd
import matplotlib.pyplot as plt

def main():
    # 1. Read the CSV file
    csv_file = "cg_convergence.csv"
    try:
        data = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: file '{csv_file}' not found.")
        return

    # 2. Check columns
    required_columns = {"Iteration", "ResidualNorm"}
    if not required_columns.issubset(data.columns):
        print("Error: CSV must have columns 'Iteration' and 'ResidualNorm'.")
        return

    # 3. Extract iteration and residual data
    iteration = data["Iteration"].to_numpy()
    residuals = data["ResidualNorm"].to_numpy()

    # 4. Print some basic statistics
    final_residual = residuals[-1] if len(residuals) > 0 else None
    min_residual = residuals.min() if len(residuals) > 0 else None
    max_residual = residuals.max() if len(residuals) > 0 else None

    print(f"Number of data points (iterations recorded): {len(residuals)}")
    print(f"Initial residual: {residuals[0]:.6e}" if len(residuals) > 0 else "No data.")
    print(f"Final residual:   {final_residual:.6e}" if final_residual is not None else "No data.")
    print(f"Minimum residual: {min_residual:.6e}" if min_residual is not None else "No data.")
    print(f"Maximum residual: {max_residual:.6e}" if max_residual is not None else "No data.")

    # 5. Plot the convergence curve on a semilog scale
    plt.figure(figsize=(8, 6))
    plt.semilogy(iteration, residuals, marker='o', linestyle='-', color='blue', label='Residual Norm')
    plt.title("CG Convergence Analysis", fontsize=16)
    plt.xlabel("Iteration", fontsize=14)
    plt.ylabel("Residual Norm", fontsize=14)
    plt.grid(True, which='both', linestyle='--')
    plt.legend()
    plt.tight_layout()

    # 6. Save the plot as "cg_convergence.png"
    output_image = "cg_convergence.png"
    plt.savefig(output_image, dpi=300)
    print(f"Plot saved as '{output_image}'.")

    # (Optional) If you want to display the plot interactively, uncomment below:
    # plt.show()

if __name__ == "__main__":
    main()
