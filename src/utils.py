import numpy as np

__all__ = [
    "euc_distance", "logeuc_distance",
    "EuclideanCentroid", "avg_log_euclidean", "AvgRiemann",
]


def _sym(A):
    return 0.5 * (A + A.T)


def _eig_spd(A, eps=1e-12):
    w, V = np.linalg.eigh(_sym(A))
    w = np.maximum(w, eps)
    return w, V


def _reconstruct(V, d):
    return V @ np.diag(d) @ V.T


def spd_le(A):
    w, V = _eig_spd(A)
    return _reconstruct(V, np.log(w))


def spd_exp(A):
    w, V = np.linalg.eigh(_sym(A))
    return _reconstruct(V, np.exp(w))


def spd_sqrt(A):
    w, V = _eig_spd(A)
    return _reconstruct(V, np.sqrt(w))


def spd_invsqrt(A):
    w, V = _eig_spd(A)
    return _reconstruct(V, 1.0 / np.sqrt(w))


def _weights(pesos, n):
    if pesos is None:
        w = np.ones(n, dtype=float) / n
    else:
        P = np.asarray(pesos)
        if P.ndim == 3:
            w = P.reshape(n, -1).mean(axis=1)
        elif P.ndim == 1:
            w = P.astype(float)
        else:
            w = np.ones(n, dtype=float)
        s = w.sum()
        w = w / s if s > 0 else np.ones(n) / n
    return w


def euc_distance(A, B):
    return float(np.linalg.norm(A - B, ord="fro"))


def logeuc_distance(A, B):
    LA, LB = spd_le(A), spd_le(B)
    return float(np.linalg.norm(LA - LB, ord="fro"))


def EuclideanCentroid(manifold, samples, expoent, pesos):
    X = np.asarray(samples)
    n = X.shape[0]
    w = _weights(pesos, n)
    M = np.tensordot(w, X, axes=1)
    w_eig, V = _eig_spd(M)
    return _reconstruct(V, w_eig)


def avg_log_euclidean(samples, pesos=None):
    X = np.asarray(samples)
    n = X.shape[0]
    w = _weights(pesos, n)
    L = np.array([spd_le(X[i]) for i in range(n)])
    Lm = np.tensordot(w, L, axes=1)
    return spd_exp(Lm)


def AvgRiemann(manifold, samples, expoent, pesos, max_iter=50, tol=1e-7):
    X = np.asarray(samples)
    n = X.shape[0]
    w = _weights(pesos, n)
    G = EuclideanCentroid(None, X, expoent, pesos)
    for _ in range(max_iter):
        G_half = spd_sqrt(G)
        G_invhalf = spd_invsqrt(G)
        S = np.zeros_like(G)
        for i in range(n):
            Y = G_invhalf @ X[i] @ G_invhalf
            S += w[i] * spd_le(Y)
        normS = np.linalg.norm(S, ord="fro")
        if normS < tol:
            break
        G = G_half @ spd_exp(S) @ G_half
        w_eig, V = _eig_spd(G)
        G = _reconstruct(V, w_eig)
    return G
