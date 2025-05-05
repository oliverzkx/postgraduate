# Case Study 4 – Multigrid Method (MAP55672, 2024‑25)

This repository contains my complete solution to Case Study 4, which implements,
tests, and analyses a recursive V‑cycle multigrid (MG) solver for the 2‑D
Poisson problem on a unit square.

---

## Directory layout

4_1_possion/        Poisson discretisation (five‑point stencil)
4_2_vcycle/         Serial V‑cycle multigrid implementation + unit test
4_3_convergence/    Convergence study driver, results, plotting script
README.md           This file

---

## Quick build & test

```bash
# 4.1 – build Poisson test
cd 4_1_possion
make               # produces ./poisson_test
./poisson_test

# 4.2 – build V‑cycle unit test
cd ../4_2_vcycle
make               # produces ./vcycle_test
./vcycle_test

# 4.3 – run convergence experiments
cd ../4_3_convergence
make               # produces ./mg_conv
./mg_conv          # writes results.csv and prints summary
python3 plot_mg.py # optional: generate iteration‑plots
```

Eigen 3 (header‑only) must be available, e.g. `/usr/include/eigen3`.

## Key results 

|    N | l<sub>max</sub> | V‑cycles | final residual |                       runtime (s) |
| ---: | --------------: | -------: | -------------: | --------------------------------: |
|   16 |               1 |       23 |    5.77 × 10⁻⁸ |                            0.0013 |
|   16 |               1 |       23 |    5.77 × 10⁻⁸ | 0.0010  (re‑run, stability check) |
|   32 |               1 |       26 |    6.04 × 10⁻⁸ |                            0.0085 |
|   32 |               2 |       47 |    6.61 × 10⁻⁸ |                            0.0046 |
|   64 |               1 |       30 |    6.52 × 10⁻⁸ |                             0.055 |
|   64 |               3 |       91 |    9.83 × 10⁻⁸ |                             0.030 |
|  128 |               1 |       33 |    6.50 × 10⁻⁸ |                              0.29 |
|  128 |               4 |      177 |    9.20 × 10⁻⁸ |                              0.23 |
|  256 |               1 |       36 |    4.57 × 10⁻⁸ |                              1.67 |
|  256 |               5 |      339 |    9.43 × 10⁻⁸ |                              1.77 |

**Best practice** – For grids up to *N = 256* a 2‑ or 3‑level V‑cycle
(coarsest grid ≈ 16×16) minimises wall‑clock time while still meeting the
10⁻⁷ residual tolerance.

## Build details

- **Compiler** : g++‑11 (`-O2 -std=c++14`)
- **Dependencies** : Eigen 3 only (no external libraries)
- **Makefiles** : one per sub‑directory, no CMake required
- **Tested on** : Ubuntu 22.04 and WSL Ubuntu 20.04