import time
import numpy as np
def gauss_seidel_method(N, A, b, max_iter=1000, min_residuum=1e-9):
    start = time.time()
    x = np.ones(N, dtype=np.float64)

    # wyznaczanie U i L+D
    U = np.triu(A, k=1)
    L_D = np.tril(A)  # L + D (trójkąt dolny z diagonalą)

    residuum = A @ x - b # liczenie residuum
    inorm = np.linalg.norm(residuum)
    r_norm = [inorm]
    iters = 1

    while inorm > min_residuum and iters < max_iter:
        # iteracja Gaussa-Seidla: x = (L + D)^{-1} @ (b - U @ x)
        right_side = b - U @ x  # prawa strona rownania
        x_new = np.zeros(N)

        for i in range(N): # podstawienie w przód - rozwiazanie układu równań
            x_new[i] = (right_side[i] - np.dot(L_D[i, :i], x_new[:i])) / L_D[i, i]
        x = x_new


        residuum = A @ x - b
        inorm = np.linalg.norm(residuum)
        r_norm.append(inorm)
        iters += 1

    calc_time = time.time() - start
    return x, np.array(r_norm), iters, calc_time