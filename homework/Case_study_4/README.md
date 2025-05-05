# Case Study 4 – Multigrid Poisson Solver  

*High‑Performance Computing · 2025 Spring*

## Directory structure

Case_study_4/
 ├── 4_1_possion/        # Task 4.1 – assemble Ax = b for −Δu=f
 │   ├── main.cpp        # quick functional test: computes ‖Ax−b‖
 │   ├── poisson.{h,cpp} # 5‑point stencil + RHS generator
 │   ├── Makefile
 │   └── …
 ├── 4_2_vcycle/         # Task 4.2 – recursive V‑cycle implementation
 │   ├── main.cpp        # single‑cycle demo + residual print‑out
 │   ├── mg.{h,cpp}      # multigrid core (smooth/restrict/prolong/solve)
 │   ├── Makefile
 │   └── vcycle_test     # binary after build
 ├── 4_3_convergence/    # Task 4.3 – convergence & performance study
 │   ├── main.cpp        # produces tables for sections 4.3‑1 & 4.3‑2
 │   ├── poisson.{h,cpp} # local copy (avoids include clashes)
 │   ├── mg.{h,cpp}      # identical to 4_2_vcycle/mg.*
 │   ├── Makefile
 │   └── report.md       # lab‑report template with results & discussion
 └── README.md           # this file

*All code is C++14 and depends only on [Eigen 3] for linear algebra.*

---

## Quick start

### 1 · Build everything

```bash
# from Case_study_4/
make -C 4_1_possion
make -C 4_2_vcycle
make -C 4_3_convergence
```

> Each sub‑directory has an independent Makefile, so you can also call
>  `make` from inside the folder you want to test.

### 2 · Run the demos

| Task | Command                                          | What happens                                                 |
| ---- | ------------------------------------------------ | ------------------------------------------------------------ |
| 4.1  | `./4_1_possion/poisson_test` *(built by `make`)* | Prints ‖Ax − b‖ to verify assembly                           |
| 4.2  | `./4_2_vcycle/vcycle_test`                       | Performs **one** V‑cycle on a 64 × 64 grid and reports the new residual |
| 4.3  | `./4_3_convergence/convergence_runner`           | Generates Tables 4.3‑1 & 4.3‑2 (iteration, residual, runtime) |



Typical output for Task 4.3:

```
#  N  scheme  lmax  iters  residual  time[s]
16  2‑level  1  23  5.77e‑08  0.0012
...
256  max‑level  6  411  9.75e‑08  2.17
```

------

## Files you should hand in

| File / folder                      | Purpose                              |
| ---------------------------------- | ------------------------------------ |
| `4_1_possion/` *(full folder)*     | Source + Makefile for Task 4.1       |
| `4_2_vcycle/` *(full folder)*      | Source + Makefile for Task 4.2       |
| `4_3_convergence/` *(full folder)* | Source + Makefile + **`report.md`**  |
| `README.md`                        | build & run instructions (this file) |

------



## Notes & troubleshooting

- The code uses **in‑place Gauss–Seidel** as weighted Jacobi (ω = 2/3).
   Adjust `omega` / `nu` in `mg.cpp` if you want different smoothers.

- Eigen is included system‑wide (`-I/usr/include/eigen3`).
   If Eigen lives elsewhere, change the include path in every Makefile:

  ```
  CXXFLAGS = -O2 -std=c++14 -I/path/to/eigen
  ```

- All grids are interior points only (Dirichlet BCs are implicit).
   Grid size *N* means *N*×*N* unknowns.