import time
import numpy as np
from scipy.linalg import solve_triangular
import scipy



def gauss_seidel_method_good(N, A, b, max_iter=1000, min_residuum=1e-9):
    start = time.time()
    x = np.ones(N, dtype=np.float64)  # Wymuszamy float64

    # Rozkład macierzy A = L + D + U
    U = np.triu(A, k=1)
    L_D = np.tril(A)  # L + D (trójkąt dolny z diagonalą)

    # Inicjalizacja normy residuum
    residuum = A @ x - b
    inorm = np.linalg.norm(residuum)
    r_norm = [inorm]
    iters = 0

    while inorm > min_residuum and iters < max_iter:
        # Iteracja Gaussa-Seidla: x = (L + D)^{-1} @ (b - U @ x)
        for i in range(N): # podstawienie w przód
            sum_ = np.dot(A[i, :i], x[:i]) + np.dot(A[i, i + 1:], x[i + 1:])
            x[i] = (b[i] - sum_) / A[i, i]

        residuum = A @ x - b
        inorm = np.linalg.norm(residuum)
        r_norm.append(inorm)
        iters += 1

    calc_time = time.time() - start
    return x, np.array(r_norm), iters, calc_time


def gauss_seidel_method_slow(N, A, b, max_iter=1000, min_residuum=1e-9):
    start = time.time()
    x = np.ones(N)

    D = np.diag(np.diag(A))
    U = np.triu(A, k=1)
    L = np.tril(A, k=-1)

    # Macierz T = D + L oraz wektor w = T^{-1}b
    T = D + L

    # liczenie normy residuum
    residuum = A @ x - b
    inorm = np.linalg.norm(residuum)
    r_norm = [inorm]
    iters = 0

    while ( inorm > min_residuum and iters < max_iter ):
        # Iteracja Gaussa-Seidla: x = T^{-1}(-Ux + b)
        x = np.linalg.solve(T, -U @ x + b)
        residuum = A @ x - b
        inorm = np.linalg.norm(residuum)
        r_norm.append(inorm)
        iters += 1

    calc_time = time.time() - start

    return x, np.array(r_norm), iters, calc_time


##########################################################






############## git solver ###################
def lu_solver_olld(N, A, b):
    start = time.time()
    A = A.astype(np.float64, copy=True)  # Kopia robocza
    P = np.arange(N)  # Wektor permutacji zamiast pełnej macierzy

    # Rozkład LU z pivotingiem
    for k in range(N - 1):
        # Wybór elementu głównego
        pivot = np.argmax(np.abs(A[k:, k])) + k
        if pivot != k:
            A[[k, pivot]] = A[[pivot, k]]  # Zamiana wierszy
            P[[k, pivot]] = P[[pivot, k]]  # Śledzenie permutacji
            b[[k, pivot]] = b[[pivot, k]]  # Równoczesna zamiana w b

        # Eliminacja Gaussa
        A[(k + 1):, k] /= A[k, k]
        A[(k + 1):, (k + 1):] -= np.outer(A[(k + 1):, k], A[k, (k + 1):])

    # Rozdzielenie L i U z pojedynczej macierzy A
    L = np.tril(A, -1) + np.eye(N)
    U = np.triu(A)

    # Rozwiązanie Ly = Pb (podstawienie w przód)
    y = np.zeros(N)
    for i in range(N):
        y[i] = b[i] - L[i, :i] @ y[:i]

    # Rozwiązanie Ux = y (podstawienie wstecz)
    x = np.zeros(N)
    for i in range(N - 1, -1, -1):
        x[i] = (y[i] - U[i, i + 1:] @ x[i + 1:]) / U[i, i]

    # Obliczenie normy residuum
    residuum = A @ x - b
    r_norm = np.linalg.norm(residuum)
    calc_time = time.time() - start
    return x, r_norm, calc_time


##################################### second #################################

def lu_solver_optimized(A, b): # ten najlepszy i guess kurde
    """Ultra-fast LU solver with partial pivoting (pure NumPy)"""
    start_time = time.time()
    n = A.shape[0]
    A = A.astype(np.float64)  # Konwersja typu
    b = b.astype(np.float64)
    P = np.arange(n)  # Wektor permutacji

    # Rozkład LU z częściowym wyborem elementu głównego
    for k in range(n - 1):
        # Znajdź pivot (największy element w kolumnie)
        max_row = k + np.argmax(np.abs(A[k:, k]))

        # Zamiana wierszy jeśli potrzebne
        if max_row != k:
            A[[k, max_row]] = A[[max_row, k]]  # Zamiana wierszy w A
            P[[k, max_row]] = P[[max_row, k]]  # Aktualizacja permutacji
            b[[k, max_row]] = b[[max_row, k]]  # Zamiana w wektorze b

        # Eliminacja Gaussa (zoptymalizowana wersja)
        pivot = A[k, k]
        A[k + 1:, k] /= pivot
        A[k + 1:, k + 1:] -= A[k + 1:, k:k + 1] * A[k:k + 1, k + 1:]

    # Rozwiązanie Ly = Pb (podstawienie w przód)
    y = np.empty(n, dtype=np.float64)
    for i in range(n):
        y[i] = b[i] - np.dot(A[i, :i], y[:i])

    # Rozwiązanie Ux = y (podstawienie wstecz)
    x = np.empty(n, dtype=np.float64)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - np.dot(A[i, i + 1:], x[i + 1:])) / A[i, i]

    # Obliczenie normy residuum
    residuum = np.dot(A, x) - b
    r_norm = np.sqrt(np.dot(residuum, residuum))  # Szybsze niż np.linalg.norm

    total_time = time.time() - start_time
    return x, r_norm, total_time

####################################### teraz zadziala ########
def optimized_lu_solver(A, b):
    start = time.time()
    """Optymalizowana metoda LU z częściowym wyborem elementu głównego"""
    n = A.shape[0]
    A = A.astype(np.float64)  # Tylko konwersja typu
    b = b.astype(np.float64)
    P = np.arange(n)

    # Rozkład LU
    for k in range(n - 1):
        # Wybór pivota
        pivot = k + np.argmax(np.abs(A[k:, k]))
        if pivot != k:
            A[[k, pivot]] = A[[pivot, k]]
            P[k], P[pivot] = P[pivot], P[k]
            b[k], b[pivot] = b[pivot], P[k]

        # Eliminacja Gaussa
        pivot_inv = 1.0 / A[k, k]
        A[k + 1:, k] *= pivot_inv
        for i in range(k + 1, n):
            A[i, k + 1:] -= A[i, k] * A[k, k + 1:]

    # Rozwiązanie Ly = Pb
    y = np.zeros(n)
    for i in range(n):
        y[i] = b[i] - np.dot(A[i, :i], y[:i])

    # Rozwiązanie Ux = y
    x = np.zeros(n)
    for i in reversed(range(n)):
        x[i] = (y[i] - np.dot(A[i, i + 1:], x[i + 1:])) / A[i, i]

    # Norma residuum
    residuum = np.dot(A, x) - b
    r_norm = np.linalg.norm(residuum)

    calc_time = time.time() - start
    return x, r_norm, calc_time





def lu_solver_cheat(N, A, b):
    start = time.time()

    # 1. Faktoryzacja LU (wersja blokowa bez pivotowania)
    def block_lu(A, block_size=64):
        n = A.shape[0]
        if n <= block_size:
            L = np.eye(n)
            U = A.copy()
            for k in range(n - 1):
                L[k + 1:, k] = U[k + 1:, k] / U[k, k]
                U[k + 1:, k:] -= np.outer(L[k + 1:, k], U[k, k:])
            return L, U

        mid = n // 2
        A11 = A[:mid, :mid]
        A12 = A[:mid, mid:]
        A21 = A[mid:, :mid]
        A22 = A[mid:, mid:]

        L11, U11 = block_lu(A11)
        L21 = solve_triangular(U11.T, A21.T).T
        U12 = solve_triangular(L11, A12)
        L22, U22 = block_lu(A22 - L21 @ U12)

        L = np.block([[L11, np.zeros_like(A12)], [L21, L22]])
        U = np.block([[U11, U12], [np.zeros_like(A21), U22]])
        return L, U

    # 2. Wykonanie faktoryzacji
    L, U = block_lu(A)
    P = np.eye(N)  # Macierz permutacji jednostkowa (brak pivotowania)

    # 3. Rozwiązanie układu
    y = solve_triangular(L, P @ b, lower=True)
    x = solve_triangular(U, y, lower=False)

    r_norm = np.linalg.norm(A @ x - b)
    calc_time = time.time() - start

    return x, np.array(r_norm), calc_time



######################## old version for banded ########################:

import numpy as np
import numba as nb


@nb.njit
def banded_lu(A, bw):
    """LU decomposition for banded matrix (no pivoting)."""
    n = A.shape[0]
    L = np.eye(n)
    U = A.copy()

    for k in range(n - 1):
        # Only consider non-zero elements within the band
        for i in range(k + 1, min(k + 1 + bw, n)):
            L[i, k] = U[i, k] / U[k, k]
            for j in range(k, min(k + 2 * bw + 1, n)):  # Update within band
                U[i, j] -= L[i, k] * U[k, j]

    return L, U


@nb.njit
def banded_forward_sub(L, b, bw):
    """Forward substitution for banded L."""
    n = L.shape[0]
    y = np.zeros(n)
    for i in range(n):
        start = max(0, i - bw)
        y[i] = (b[i] - np.dot(L[i, start:i], y[start:i])) / L[i, i]
    return y


@nb.njit
def banded_backward_sub(U, y, bw):
    """Backward substitution for banded U."""
    n = U.shape[0]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        end = min(n, i + bw + 1)
        x[i] = (y[i] - np.dot(U[i, i + 1:end], x[i + 1:end])) / U[i, i]
    return x


def lu_method_test(N, A, b):
    """Optimized LU solver for banded matrices."""
    start = time.time()

    L, U = banded_lu(A, 2)  # No pivoting (assumes diagonal dominance)
    y = banded_forward_sub(L, b, 2)
    x = banded_backward_sub(U, y, 2)

    r_norm = np.linalg.norm(A @ x - b)
    calc_time = time.time() - start

    return x, np.array(r_norm), calc_time

################################# cholensky #########################

def cholesky_decomposition(A):
    """
    Ręczna implementacja faktoryzacji Choleskiego A = LL^T.
    Zakłada, że macierz jest symetryczna i dodatnio określona.
    """
    n = A.shape[0]
    L = np.zeros_like(A)

    for i in range(n):
        for j in range(i + 1):
            s = np.dot(L[i, :j], L[j, :j])

            if i == j:
                L[i, j] = np.sqrt(A[i, i] - s)
            else:
                L[i, j] = (A[i, j] - s) / L[j, j]
    return L


def forward_substitution(L, b):
    """Ręczne podstawienie w przód (Ly = b)"""
    n = L.shape[0]
    y = np.zeros(n)

    for i in range(n):
        y[i] = (b[i] - np.dot(L[i, :i], y[:i])) / L[i, i]
    return y


def backward_substitution(LT, y):
    """Ręczne podstawienie wstecz (L^T x = y)"""
    n = LT.shape[0]
    x = np.zeros(n)

    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - np.dot(LT[i, i + 1:], x[i + 1:])) / LT[i, i]
    return x


def lu_method_chol(N, A, b): ################ TODO ZAJEBISTA#########################
    start = time.time()


    # 1. Faktoryzacja Choleskiego
    # L = cholesky_decomposition(A)

    L = np.zeros_like(A)

    for i in range(N):
        for j in range(i + 1):
            s = np.dot(L[i, :j], L[j, :j])

            if i == j:
                L[i, j] = np.sqrt(A[i, i] - s)
            else:
                L[i, j] = (A[i, j] - s) / L[j, j]


    # 2. Rozwiązanie układu LL^T x = b
    # y = solve_triangular(L, b, lower=True)  # Ly = b
    # x = solve_triangular(L.T, y, lower=False)  # L^T x = y

    # 2. Rozwiązanie układu LL^T x = b
    y = forward_substitution(L, b)  # Ly = b
    x = backward_substitution(L.T, y)  # L^T x = y

    r_norm = np.linalg.norm(A @ x - b)
    calc_time = time.time() - start

    return x, np.array(r_norm), calc_time


def lu_decomposition_no_pivot(A):
    start = time.time()

    n = A.shape[0]
    L = np.eye(n)
    U = A.copy().astype(np.float64)  # Ensure float type
    print(f"czas1: {time.time() - start}")
    for k in range(n - 1):
        if U[k, k] == 0:
            raise ValueError("Zero pivot encountered. Matrix might be singular.")

        for i in range(k + 1, n):
            L[i, k] = U[i, k] / U[k, k]
            U[i, k:] -= L[i, k] * U[k, k:]
    print(f"czas22: {time.time() - start}")
    return L, U  # Now A = LU (no permutation)



def lu_decomposition_slow(N, A):
    L = np.eye(N, dtype=np.float64)
    U = A.astype(np.float64, copy=True)

    for k in range(N - 1):
        pivot = U[k, k]
        if pivot == 0:
            raise ValueError("0 na diagonali, wyjątek dzielenia przez 0")
            # w eliminacji gaussa stosuje się zamianę wierszy, aby tego uniknąć,
            # ale nasz algorytm jest uproszczony i zdefiniowany dla macierzy o niezerowej diagonali

        # oblicz mnoznik dla k-tej kolumny i zapisz go w L, potem wykorzystamy go do eliminacji
        for i in range(k + 1, N):
            L[i, k] = U[i, k] / pivot
            if U[i, k] == 0:
                break

        # eliminacja gaussa - odejmowanie przemnożonych wierszy, aby osiągnąć schodkową macierz U
        for i in range(k + 1, N):
            factor = L[i, k]  # mnoznik z L, do eliminacji w U
            for j in range(k, N):
                U[i, j] -= factor * U[k, j]
    return L, U