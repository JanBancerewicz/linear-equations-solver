import time
import numpy as np

def jacobi_method(N, A, b, max_iter=1000, min_residuum=1e-9):
    start = time.time()
    x = np.ones(N, dtype=np.float64)

    D = np.diag(np.diag(A))  # macierz diagonalna
    L = np.tril(A, k=-1)  # macierz trójkątna dolna
    U = np.triu(A, k=1)  # macierz trójkątna górna

    D_inv = np.diag(1.0 / np.diag(D))  # odwrotność D, czyli odwrotność każdego elementu na diagonali
    M = -D_inv @ (L + U)  # macierz iteracyjna
    w = D_inv @ b  # wektor iteracyjny

    residuum = A @ x - b
    inorm = np.linalg.norm(residuum)
    r_norm = [inorm]
    iters = 1

    while inorm > min_residuum and iters < max_iter:
        # iteracja Jacobiego: x = Mx + w
        x = M @ x + w
        residuum = A @ x - b
        inorm = np.linalg.norm(residuum)
        r_norm.append(inorm)
        iters += 1

    calc_time = time.time() - start
    return x, np.array(r_norm), iters, calc_time