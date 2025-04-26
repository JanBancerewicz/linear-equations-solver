
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

## Report

A detailed description of algorithms, results interpretation, and charts are available in [Linear_equation_solver.pdf](./Linear_equation_solver.pdf) (in Polish).


## Project Structure
- `main.py` — Main script running simulations, solving tasks B–E.
- `jacobi.py`, `gauss_seidel.py` — Implementations of iterative methods.
- `lu.py` — LU decomposition (standard and optimized versions).
- `helpers.py` — Matrix generation, solving helpers.
- `parameters.py` — Program parameters.
- `plotting.py` — Methods for data visualization.

## Performance Analysis
- **Iterative methods** converge successfully for matrix A but **fail** for matrix C due to the lack of diagonal dominance.
- **Direct methods** provide highly accurate solutions for both matrices.
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
