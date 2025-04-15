#!/usr/bin/env python3
"""
@file plot_u_profile.py
@brief Plot the 1D profile of u(x) obtained from u_profile.csv.

This script reads the CSV file "u_profile.csv" (produced by the C++ program)
and plots u(x) along y=0.5 using matplotlib.

Because we might be running in a headless environment, the figure is saved
to a PNG file instead of trying to display it interactively.
"""

import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file.
data = pd.read_csv('u_profile.csv')

# Create the plot.
plt.figure(figsize=(8, 6))
plt.plot(data['x'], data['u(x)'], marker='o', linestyle='-', color='blue')
plt.xlabel('x', fontsize=14)
plt.ylabel('u(x) at y=0.5', fontsize=14)
plt.title('1D Profile of u(x) along y=0.5', fontsize=16)
plt.grid(True)
plt.tight_layout()

# Save the figure to a PNG file with 300 dpi resolution.
plt.savefig('u_profile.png', dpi=300)

print("Plot saved as 'u_profile.png'.")
