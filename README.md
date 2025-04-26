
# Linear Equation Solver

## Project Overview
This project demonstrates the implementation and comparison of various numerical methods for solving systems of linear equations in Python. It focuses on both **iterative** and **direct** methods, analyzes their convergence, residual norms, and computational efficiency.

## Implemented Methods
- **Jacobi Method** (iterative)
- **Gauss-Seidel Method** (iterative)
- **LU Decomposition** (direct)
- **Optimized LU Decomposition** for banded matrices (direct)
- **Reference Functions**:
  - `scipy.linalg.lu()`
  - `numpy.linalg.solve()`

## Problem Description
Systems of linear equations are represented in matrix form as:

\[
Ax = b
\]

where:
- \( A \) — coefficient matrix
- \( x \) — vector of unknowns
- \( b \) — right-hand side vector

The project operates on two types of matrices:
- **Matrix A**: 1299×1299, symmetric, diagonally dominant, well-conditioned, positive definite (band width = 2).
- **Matrix C**: 1299×1299, *not* diagonally dominant, *not* positive definite, poorly conditioned.

The right-hand side vector \( b \) is generated using a sine function.

## Numerical Methods

### Iterative Methods
- **Jacobi Method**: Updates all unknowns simultaneously based on values from the previous iteration.
- **Gauss-Seidel Method**: Updates unknowns sequentially within the same iteration, using already updated values.

Both methods check the convergence based on the **residual norm**:

\[
r^{(k)} = Ax^{(k)} - b
\]

where convergence is achieved when:

\[
\|r^{(k)}\| < \varepsilon
\]

with \(\varepsilon = 10^{-9}\).

### Direct Methods
- **LU Decomposition**: Decomposes matrix \( A \) into lower \( L \) and upper \( U \) triangular matrices.
- **Optimized LU for Banded Matrices**: Leverages the band structure to reduce complexity from \( O(N^3) \) to \( O(N) \).

Additionally, built-in library functions were tested:
- `scipy.linalg.lu()`
- `numpy.linalg.solve()`

## Project Structure
- `main.py` — Main script running simulations, solving tasks B–E.
- `jacobi.py`, `gauss_seidel.py` — Implementations of iterative methods.
- `lu.py` — LU decomposition (standard and optimized versions).
- `helpers.py` — Matrix generation, solving helpers.
- `parameters.py` — Problem parameters.
- `plotting.py` — Visualization of results.

## Performance Analysis
- **Iterative methods** converge successfully for matrix A but **fail** for matrix C due to the lack of diagonal dominance.
- **Direct methods** provide highly accurate solutions for both matrices, with residual norms close to \(10^{-15}\).
- **Optimized LU decomposition** significantly outperforms the classic version for large matrices.

Simulations were performed for matrix sizes ranging from **N = 100** to **N = 3500** to analyze scalability.

## Results Highlights
| Method | Residual Norm | Execution Time (Matrix A, N=1299) |
|:------:|:-------------:|:---------------------------------:|
| Jacobi | \(9.72 \times 10^{-10}\) | 0.159 s |
| Gauss-Seidel | \(9.64 \times 10^{-10}\) | 0.222 s |
| LU Decomposition (classic) | \(2.59 \times 10^{-15}\) | 2.69 s |
| LU Optimized | \(2.59 \times 10^{-15}\) | 0.023 s |
| `scipy.linalg.lu()` | \(2.58 \times 10^{-15}\) | 0.050 s |
| `numpy.linalg.solve()` | \(3.34 \times 10^{-15}\) | 0.040 s |

## How to Run
1. Install required libraries:
   ```bash
   pip install requirements.txt
   ```
2. Run the main script:
   ```bash
   python main.py
   ```
3. Results and plots will be generated and saved in the `/data/` directory.

## Conclusion
The project shows that:
- Iterative methods are efficient for large, well-conditioned matrices but require proper matrix properties (e.g., diagonal dominance).
- Direct methods guarantee high precision but may become computationally expensive for very large systems.
- Optimizations based on matrix structure (banded matrices) drastically improve performance.
