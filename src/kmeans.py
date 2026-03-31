import numpy as np
from pymanopt.manifolds import Euclidean
from pymanopt.manifolds.positive_definite import SymmetricPositiveDefinite as PositiveDefinite
import utils as U_spd


def _normalize_metric_name(name: str) -> str:
    n = str(name).strip().lower()
    if n in {"airm", "riemannian", "affine-invariant"}:
        return "riemannian"
    if n in {"spd_le", "logeuclidean", "log-euclidean", "le"}:
        return "logeuclidean"
    if n in {"no_spd", "euclidean", "euc", "euclid"}:
        return "euclidean"
    raise ValueError(f"Invalid metric alias: {name}")


class Point:
    def __init__(self, value, id_point, name=""):
        self.id_point = id_point
        self.id_cluster = -1
        self.value = value
        self.name = name
        self.weight = {}

    def getID(self):
        return self.id_point

    def getCluster(self):
        return self.id_cluster

    def setCluster(self, cid):
        self.id_cluster = int(cid)

    def getValue(self):
        return self.value

    def addValue(self, value):
        self.value = value

    def getName(self):
        return self.name

    def Setweights(self, w):
        if isinstance(w, dict):
            self.weight = {int(k): float(v) for k, v in w.items()}
        else:
            self.weight = {i: float(w[i]) for i in range(len(w))}

    def Getweights(self):
        return dict(self.weight)


class Cluster:
    def __init__(self, id_cluster, point):
        self.id_cluster = id_cluster
        self.central_value = np.array(point.getValue(), dtype=float)
        self.points = [point]
        self.mudou = False

    def addPoint(self, p):
        self.points.append(p)

    def removePoint(self, pid):
        for q in list(self.points):
            if q.getID() == pid:
                self.points.remove(q)
                return True
        return False

    def setMudou(self, v):
        self.mudou = bool(v)

    def getMudou(self):
        return self.mudou

    def getCentralValue(self):
        return self.central_value

    def setCentralValue(self, v):
        self.central_value = np.array(v, dtype=float)

    def getPoint(self, idx):
        return self.points[idx]

    def getPoints(self):
        return self.points

    def getTotalPoints(self):
        return len(self.points)

    def getID(self):
        return self.id_cluster


def _dist_metric(manifold, A, B, metric_type: str) -> float:
    if metric_type == "riemannian":
        try:
            return float(manifold.dist(A, B))
        except Exception:
            wA, VA = U_spd._eig_spd(A)
            wB, VB = U_spd._eig_spd(B)
            A = U_spd._reconstruct(VA, wA)
            B = U_spd._reconstruct(VB, wB)
            return float(manifold.dist(A, B))
    if metric_type == "euclidean":
        return U_spd.euc_distance(A, B)
    if metric_type == "logeuclidean":
        return U_spd.logeuc_distance(A, B)
    raise ValueError("Invalid metric")


class KMeansSPD:
    def __init__(self, n_claster, total_points, metric_type="riemannian",
                 index_centers=None, max_iterations=1000, dim_point=3, tol=1e-4):
        self.n_claster = int(n_claster)
        self.total_points = int(total_points)
        self.max_iterations = int(max_iterations)
        self.index_centers = index_centers
        self.clusters = []
        self.classes = {}
        self.metric_type = _normalize_metric_name(metric_type)
        self.dim = int(dim_point)
        self.tol = float(tol)
        if self.metric_type == "euclidean":
            self.manifold = Euclidean(self.dim, self.dim)
        else:
            self.manifold = PositiveDefinite(self.dim)

    def _init_centers_kpp(self, points):
        rng = np.random
        self.clusters = []
        self.classes = {}
        c0 = rng.randint(self.total_points)
        first = points[c0]
        self.clusters.append(Cluster(0, first))
        first.setCluster(0)
        self.classes[first.getID()] = 1
        min_d2 = np.full(self.total_points, np.inf, dtype=float)
        C0 = self.clusters[0].getCentralValue()
        for i, p in enumerate(points):
            d = _dist_metric(self.manifold, np.array(p.getValue(), dtype=float), C0, self.metric_type)
            min_d2[i] = d * d
        for k in range(1, self.n_claster):
            w = min_d2.copy()
            s = w.sum()
            if not np.isfinite(s) or s <= 0:
                idx = rng.randint(self.total_points)
            else:
                probs = w / s
                idx = rng.choice(self.total_points, p=probs)
            pcent = points[idx]
            self.clusters.append(Cluster(k, pcent))
            pcent.setCluster(k)
            self.classes[pcent.getID()] = k + 1
            Ck = self.clusters[k].getCentralValue()
            for i, p in enumerate(points):
                d = _dist_metric(self.manifold, np.array(p.getValue(), dtype=float), Ck, self.metric_type)
                d2 = d * d
                if d2 < min_d2[i]:
                    min_d2[i] = d2

    def _cluster_alpha(self, Xk, base_alpha, alpha_inf):
        if Xk.shape[0] <= 1:
            return float(base_alpha)
        if self.metric_type in {"riemannian", "logeuclidean"}:
            V = np.array([U_spd.spd_le(Xk[i]) for i in range(Xk.shape[0])], dtype=float)
        else:
            V = np.array(Xk, dtype=float)
        n = V.shape[0]
        d = V.shape[1] * V.shape[2]
        M = V.reshape(n, d)
        m = M.mean(axis=0)
        s = M.std(axis=0, ddof=1)
        s[s <= 0.0] = 1e-12
        Z = (M - m) / s
        skew = np.mean(Z ** 3.0, axis=0)
        kurt = np.mean(Z ** 4.0, axis=0)
        if np.max(np.abs(kurt)) > 3.0:
            return float(alpha_inf)
        x = np.abs(skew) + np.abs(kurt - 3.0)
        x_max = float(np.max(x))
        x_min = float(np.min(x))
        if x_max <= 1e-12:
            return float(base_alpha)
        return float(1.0 + (x_max - x_min) / x_max)

    def _compute_adaptive_alphas(self, X, labels, base_alpha, alpha_inf):
        K = self.n_claster
        alphas = np.full(K, float(base_alpha), dtype=float)
        for k in range(K):
            idx = np.where(labels == k)[0]
            if idx.size <= 1:
                continue
            Xk = X[idx]
            alphas[k] = self._cluster_alpha(Xk, base_alpha, alpha_inf)
        return alphas

    def run(self, points, expoent, index_centers=None, adaptive=False, base_alpha=None, alpha_inf=50.0):
        eps = 1e-12
        if base_alpha is None:
            base_alpha = float(expoent)
        if adaptive:
            alphas = np.full(self.n_claster, float(base_alpha), dtype=float)
        else:
            alpha = float(expoent)
        if (index_centers if index_centers is not None else self.index_centers) is None:
            self._init_centers_kpp(points)
        else:
            self.clusters = []
            self.classes = {}
            idxs = index_centers if index_centers is not None else self.index_centers
            for i in range(self.n_claster):
                idx = int(idxs[i])
                points[idx].setCluster(i)
                self.classes[points[idx].getID()] = i + 1
                self.clusters.append(Cluster(i, points[idx]))
        X = np.array([np.array(p.getValue(), dtype=float) for p in points], dtype=float)
        N = X.shape[0]
        it = 0
        labels = np.zeros(N, dtype=int)
        while True:
            it += 1
            C = np.array([cl.getCentralValue() for cl in self.clusters], dtype=float)
            K = C.shape[0]
            if self.metric_type == "euclidean":
                diff = X[:, None, :, :] - C[None, :, :, :]
                D = np.linalg.norm(diff, axis=(2, 3))
            elif self.metric_type == "logeuclidean":
                LX = np.array([U_spd.spd_le(X[i]) for i in range(N)])
                LC = np.array([U_spd.spd_le(C[k]) for k in range(K)])
                diff = LX[:, None, :, :] - LC[None, :, :, :]
                D = np.linalg.norm(diff, axis=(2, 3))
            else:
                D = np.zeros((N, K), dtype=float)
                for i in range(N):
                    Xi = X[i]
                    for k in range(K):
                        D[i, k] = _dist_metric(self.manifold, Xi, C[k], self.metric_type)
            if adaptive:
                S = np.zeros_like(D)
                for k in range(K):
                    ak = float(alphas[k])
                    S[:, k] = np.power(1.0 / np.power(D[:, k] + eps, 2.0), ak)
            else:
                S = np.power(1.0 / np.power(D + eps, 2.0), alpha)
            row_sums = S.sum(axis=1, keepdims=True)
            mask = (~np.isfinite(row_sums)) | (row_sums <= 0.0)
            if np.any(mask):
                mask_rows = mask.ravel()
                S[mask_rows, :] = 1.0
                row_sums[mask_rows, :] = float(self.n_claster)
            P = S / row_sums
            y = np.argmax(P, axis=1)
            labels = y.copy()
            for i, p in enumerate(points):
                p.setCluster(int(y[i]))
                p.Setweights(P[i])
                self.classes[p.getID()] = int(y[i]) + 1
            for cl in self.clusters:
                cl.points = []
            for i, p in enumerate(points):
                self.clusters[y[i]].addPoint(p)
            max_delta = 0.0
            col_sums = P.sum(axis=0, keepdims=True)
            col_sums[col_sums <= 0.0] = 1.0
            Pn = P / col_sums
            for k, cl in enumerate(self.clusters):
                old_center = cl.getCentralValue()
                w_k = Pn[:, k]
                if self.metric_type == "euclidean":
                    mean_sample = U_spd.EuclideanCentroid(self.manifold, X, base_alpha if adaptive else alpha, w_k)
                elif self.metric_type == "logeuclidean":
                    mean_sample = U_spd.avg_log_euclidean(X, w_k)
                elif self.metric_type == "riemannian":
                    mean_sample = U_spd.AvgRiemann(self.manifold, X, base_alpha if adaptive else alpha, w_k)
                else:
                    raise ValueError("Invalid metric")
                cl.setCentralValue(mean_sample)
                delta = float(np.linalg.norm(mean_sample - old_center, ord="fro"))
                if delta > max_delta:
                    max_delta = delta
                cl.setMudou(False)
            if adaptive:
                alphas = self._compute_adaptive_alphas(X, labels, base_alpha, alpha_inf)
            if max_delta < self.tol:
                #print(f"Converged by tol={self.tol:.1e} at iter={it}  max_centroid_shift={max_delta:.6e}", flush=True)
                break
            if it >= self.max_iterations:
                #print(f"Stopped at iter={it} (max_iterations={self.max_iterations})  last_max_shift={max_delta:.6e}", flush=True)
                break
        return self.classes, it

    def fit(self, points, expoent, index_centers=None, adaptive=False, base_alpha=None, alpha_inf=50.0):
        return self.run(points, expoent, index_centers=index_centers, adaptive=adaptive, base_alpha=base_alpha, alpha_inf=alpha_inf)


def KMeans(n_claster, total_points, metric_type="riemannian",
           index_centers=None, max_iterations=1000, dim_point=3, tol=1e-4):
    metric_norm = _normalize_metric_name(metric_type)
    return KMeansSPD(
        n_claster=n_claster,
        total_points=total_points,
        metric_type=metric_norm,
        index_centers=index_centers,
        max_iterations=max_iterations,
        dim_point=dim_point,
        tol=tol,
    )
