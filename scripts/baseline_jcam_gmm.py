import argparse
import os
from pathlib import Path 
import numpy as np
from dipy.io.image import load_nifti
import time
import itertools 
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def _sym(A):
    return 0.5 * (A + A.T)


def _eig_clamped(A, eps=1e-12):
    w, V = np.linalg.eigh(_sym(A))
    w = np.maximum(w, eps)
    return w, V


def spd_le(A):
    w, V = _eig_clamped(A)
    return V @ np.diag(np.log(w)) @ V.T


def spd_exp(A):
    w, V = np.linalg.eigh(_sym(A))
    return V @ np.diag(np.exp(w)) @ V.T


def spd_sqrt(A):
    w, V = _eig_clamped(A)
    return V @ np.diag(np.sqrt(w)) @ V.T


def spd_invsqrt(A):
    w, V = _eig_clamped(A)
    return V @ np.diag(w ** -0.5) @ V.T


def nearest_spd(A):
    A = _sym(A)
    w, V = np.linalg.eigh(A)
    w = np.maximum(w, 1e-12)
    return V @ np.diag(w) @ V.T


def pack_dti_to_spd(arr):
    if arr.ndim == 5 and arr.shape[-2:] == (3, 3):
        M = arr
    elif arr.ndim == 4 and arr.shape[-1] in (6, 9):
        if arr.shape[-1] == 6:
            Dxx, Dxy, Dxz, Dyy, Dyz, Dzz = [arr[..., i] for i in range(6)]
            M = np.stack(
                [
                    np.stack([Dxx, Dxy, Dxz], axis=-1),
                    np.stack([Dxy, Dyy, Dyz], axis=-1),
                    np.stack([Dxz, Dyz, Dzz], axis=-1),
                ],
                axis=-2,
            )
        else:
            M = arr.reshape(arr.shape[:3] + (3, 3))
            M = _sym(M)
    else:
        raise ValueError()
    return M


def extract_spd_from_mask(dti, mask):
    mask = mask.astype(bool)
    M = pack_dti_to_spd(dti)
    X = M[mask].reshape((-1, 3, 3)).astype(float)
    for i in range(X.shape[0]):
        X[i] = nearest_spd(X[i])
    return X, mask

def riemann_mean(X, W, max_iter=10, tol=1e-4, eta=0.5):
    X = np.asarray(X, float)
    W = np.asarray(W, float)
    s = float(W.sum())
    if s <= 0:
        W = np.ones(len(X)) / float(len(X))
    else:
        W = W / s
    G = nearest_spd(np.mean(X, axis=0))
    for _ in range(max_iter):
        Gh = spd_sqrt(G)
        Gi = spd_invsqrt(G)
        S = np.zeros_like(G)
        for i in range(len(X)):
            Y = Gi @ X[i] @ Gi
            S += W[i] * spd_le(Y)
        if np.linalg.norm(S, "fro") < tol:
            break
        G = Gh @ spd_exp(eta * S) @ Gh
        G = nearest_spd(G)
    return G


def gmm_riemann(X, K, max_iter, seed, tol_ll=1e-4, jitter=1e-4):
    rng = np.random.RandomState(seed)
    n = X.shape[0]
    d = X.shape[1]
    p = d * (d + 1) // 2
    idx_ut = np.triu_indices(d)
    idx0 = rng.choice(n, K, replace=False)
    centers = X[idx0].copy()
    covs = np.array([np.eye(p) for _ in range(K)], float)
    weights = np.full(K, 1.0 / K, float)
    prev_ll = -np.inf
    for it in range(max_iter):
        logp = np.zeros((n, K), float)
        for k in range(K):
            Mk = centers[k]
            Ck = covs[k]
            invC = np.linalg.inv(Ck)
            sgn, logdet = np.linalg.slogdet(Ck)
            Gi = spd_invsqrt(Mk)
            for i in range(n):
                A = Gi @ X[i] @ Gi
                v = spd_le(A)[idx_ut]
                v = np.clip(v, -8, 8)
                q = v @ invC @ v
                logp[i, k] = (
                    np.log(weights[k] + 1e-12)
                    - 0.5 * (p * np.log(2 * np.pi) + logdet + q)
                )
        m = np.max(logp, axis=1, keepdims=True)
        R = np.exp(logp - m)
        R = R / R.sum(axis=1, keepdims=True)
        ll = float(np.sum(m + np.log(np.sum(np.exp(logp - m), axis=1, keepdims=True))))
        print(f"[gmm_spd] iter={it}, ll={ll:.6f}")
        if ll - prev_ll < tol_ll:
            break
        prev_ll = ll
        Nk = R.sum(axis=0)
        weights = Nk / float(n)
        new_centers = []
        for k in range(K):
            if Nk[k] <= 1e-12:
                new_centers.append(centers[k])
            else:
                new_centers.append(riemann_mean(X, R[:, k]))
        new_centers = np.asarray(new_centers, float)
        for k in range(K):
            Mk = new_centers[k]
            Gi = spd_invsqrt(Mk)
            Vk = []
            for i in range(n):
                A = Gi @ X[i] @ Gi
                v = spd_le(A)[idx_ut]
                v = np.clip(v, -8, 8)
                Vk.append(np.sqrt(R[i, k]) * v)
            Vk = np.asarray(Vk, float)
            if Nk[k] > 1e-12:
                C = (Vk.T @ Vk) / Nk[k]
            else:
                C = np.eye(p)
            C += jitter * np.eye(p)
            covs[k] = C
        centers = new_centers
    return centers, covs, weights, it + 1

def gmm_predict_loglik_riemann(X, centers, covs, weights):
    X = np.asarray(X, float)
    n, d, _ = X.shape
    p = d * (d + 1) // 2
    idx_ut = np.triu_indices(d)
    K = centers.shape[0]
    logp = np.zeros((n, K), float)
    for k in range(K):
        Mk = centers[k]
        Ck = np.asarray(covs[k], float)
        invC = np.linalg.inv(Ck)
        sgn, logdet = np.linalg.slogdet(Ck)
        Gi = spd_invsqrt(Mk)
        for i in range(n):
            A = Gi @ X[i] @ Gi
            v = spd_le(A)[idx_ut]
            v = np.clip(v, -8, 8)
            q = float(v @ invC @ v)
            logp[i, k] = (
                np.log(weights[k] + 1e-12)
                - 0.5 * (p * np.log(2 * np.pi) + logdet + q)
            )
    m = np.max(logp, axis=1, keepdims=True)
    ll = float(np.sum(m + np.log(np.sum(np.exp(logp - m), axis=1, keepdims=True))))
    labels = np.argmax(logp, axis=1)
    return labels, ll


def confusion_flat(y_true, y_pred, n_classes):
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    cm = np.zeros((n_classes, n_classes), int)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < n_classes and 0 <= p < n_classes:
            cm[t, p] += 1
    return cm


def metrics_from_cm(cm, background_class=0):
    K = cm.shape[0]
    fg = [c for c in range(K) if c != background_class]
    ious = []
    Ps = []
    Rs = []
    Ds = []
    for k in fg:
        tp = cm[k, k]
        fp = cm[:, k].sum() - tp
        fn = cm[k, :].sum() - tp
        iou = float(tp / (tp + fp + fn)) if tp + fp + fn > 0 else 0.0
        p = float(tp / (tp + fp)) if tp + fp > 0 else 0.0
        r = float(tp / (tp + fn)) if tp + fn > 0 else 0.0
        d = float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn > 0 else 0.0
        Ps.append(p)
        Rs.append(r)
        Ds.append(d)
        ious.append(iou)
    iou_macro_fg = float(np.mean(ious)) if ious else 0.0
    precision_macro_fg = float(np.mean(Ps)) if Ps else 0.0
    recall_macro_fg = float(np.mean(Rs)) if Rs else 0.0
    dice_macro_fg = float(np.mean(Ds)) if Ds else 0.0
    acc = float(np.trace(cm) / cm.sum()) if cm.sum() > 0 else 0.0
    return {
        "iou_macro_fg": iou_macro_fg,
        "precision_macro_fg": precision_macro_fg,
        "recall_macro_fg": recall_macro_fg,
        "dice_macro_fg": dice_macro_fg,
        "accuracy": acc,
    }


def evaluate_clustering(y_true, y_pred, n_classes, background_class=0):
    best = None
    best_m = None
    for perm in itertools.permutations(range(n_classes)):
        mapping = np.asarray(perm, int)
        y_pred_mapped = mapping[y_pred]
        cm = confusion_flat(y_true, y_pred_mapped, n_classes)
        m = metrics_from_cm(cm, background_class)
        if best is None or m["iou_macro_fg"] > best:
            best = m["iou_macro_fg"]
            best_m = m
    return best_m


def resolve_single_file(data_dir: Path, explicit_name, patterns):
    if explicit_name:
        candidate = data_dir / explicit_name
        if candidate.exists():
            return candidate

    for pattern in patterns:
        matches = sorted(data_dir.glob(pattern))
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise RuntimeError(
                f"Ambiguous pattern in {data_dir}: {pattern} -> {[m.name for m in matches]}"
            )

    raise FileNotFoundError(
        f"No compatible file found in {data_dir} for patterns: {patterns}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", type=int, required=True)
    ap.add_argument("--data-dir", type=str, required=True)
    ap.add_argument("--seed", type=int, default=50)
    ap.add_argument("--max-iter", type=int, default=300)
    ap.add_argument("--restarts", type=int, default=10)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--dti-file", type=str, default=None)
    ap.add_argument("--mask-file", type=str, default=None)
    ap.add_argument("--gt-file", type=str, default=None)
    ap.add_argument("--metrics-out", type=str, default=None)
    ap.add_argument("--n-classes", type=int, default=None)
    ap.add_argument("--metric", type=str, required=True, choices=["riemannian"])
    args = ap.parse_args()

    data_dir = Path(args.data_dir)

    dti_path = resolve_single_file(
        data_dir,
        args.dti_file,
        ["*spd*_3x3.nii.gz"]
    )
    mask_path = resolve_single_file(
        data_dir,
        args.mask_file,
        ["*mask*.nii.gz"]
    )
    gt_path = resolve_single_file(
        data_dir,
        getattr(args, "gt_file", None),
        ["*gt*classes*.nii.gz"]
    )

    dti, _ = load_nifti(str(dti_path))
    mask, _ = load_nifti(str(mask_path))

    X_spd, mask_bool = extract_spd_from_mask(dti, mask)
    X = X_spd
    print(f"[gmm_spd] K={args.classes}, {args.metric}, seed={args.seed}")

    
    rng = np.random.RandomState(args.seed)
    best_ll = None
    best_labels = None
    best_iter = None
    best_seed = None
    t0 = time.perf_counter()

    for _ in range(args.restarts):
        sd = int(rng.randint(0, 10**9))
        centers, covs, weights, last_iter = gmm_riemann(X, args.classes, args.max_iter, sd)
        labels, ll = gmm_predict_loglik_riemann(X, centers, covs, weights)

        if best_ll is None or ll > best_ll:
            best_ll = ll
            best_labels = labels.copy()
            best_iter = last_iter
            best_seed = sd

    elapsed = float(time.perf_counter() - t0)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savetxt(args.out, best_labels.astype(int), fmt="%d", delimiter=",")

    gt, _ = load_nifti(str(gt_path))
    y_ref = gt.astype(int)

    y_pred_3d = np.zeros_like(gt, dtype=int)
    y_pred_3d[mask_bool] = best_labels.astype(int) + 1

    classes = args.classes
    method_slug = f"gmm_{args.metric}"
    folder_tag = f"k{classes}"
    seed = args.seed
 
    X, Y, Z = y_ref.shape
    xs, ys, zs = X // 2, Y // 2, Z // 2

    seg_colors = ["#FFFFFF", "#264653", "#E76F51", "#2A9D8F", "#F4A261", "#E9C46A"]
    cmap = ListedColormap(seg_colors)

    out_dir = Path(args.out).parent / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    # SAGITTAL
    ref = np.rot90(y_ref[xs, :, :])
    plt.figure(figsize=(3, 3)); plt.imshow(ref, cmap=cmap); plt.axis("off"); plt.tight_layout()
    plt.savefig(out_dir / f"{folder_tag}_reference_seed{seed}_sagittal_slice{xs}.png", dpi=300, bbox_inches="tight"); plt.close()

    pred = np.rot90(y_pred_3d[xs, :, :])
    plt.figure(figsize=(3, 3)); plt.imshow(pred, cmap=cmap); plt.axis("off"); plt.tight_layout()
    plt.savefig(out_dir / f"{folder_tag}_{method_slug}_seed{seed}_sagittal_slice{xs}.png", dpi=300, bbox_inches="tight"); plt.close()

    # CORONAL
    ref = np.rot90(y_ref[:, ys, :])
    plt.figure(figsize=(3, 3)); plt.imshow(ref, cmap=cmap); plt.axis("off"); plt.tight_layout()
    plt.savefig(out_dir / f"{folder_tag}_reference_seed{seed}_coronal_slice{ys}.png", dpi=300, bbox_inches="tight"); plt.close()

    pred = np.rot90(y_pred_3d[:, ys, :])
    plt.figure(figsize=(3, 3)); plt.imshow(pred, cmap=cmap); plt.axis("off"); plt.tight_layout()
    plt.savefig(out_dir / f"{folder_tag}_{method_slug}_seed{seed}_coronal_slice{ys}.png", dpi=300, bbox_inches="tight"); plt.close()

    # AXIAL
    ref = np.rot90(y_ref[:, :, zs])
    plt.figure(figsize=(3, 3)); plt.imshow(ref, cmap=cmap); plt.axis("off"); plt.tight_layout()
    plt.savefig(out_dir / f"{folder_tag}_reference_seed{seed}_axial_slice{zs}.png", dpi=300, bbox_inches="tight"); plt.close()

    pred = np.rot90(y_pred_3d[:, :, zs])
    plt.figure(figsize=(3, 3)); plt.imshow(pred, cmap=cmap); plt.axis("off"); plt.tight_layout()
    plt.savefig(out_dir / f"{folder_tag}_{method_slug}_seed{seed}_axial_slice{zs}.png", dpi=300, bbox_inches="tight"); plt.close()


    if args.metrics_out is not None:
        gt, _ = load_nifti(str(gt_path))
        y_true = gt[mask_bool].reshape(-1).astype(int)
        y_pred = best_labels.reshape(-1).astype(int)
        if args.n_classes is not None:
            n = args.n_classes
        else:
            n = int(max(y_true.max(), y_pred.max()) + 1)
        metrics = evaluate_clustering(y_true, y_pred, n, background_class=0)
        metrics["time_sec"] = elapsed
        os.makedirs(os.path.dirname(args.metrics_out), exist_ok=True)
        with open(args.metrics_out, "w") as f:
            f.write("IoU,Dice,Precision,Recall,Accuracy,Time(s),Seed,Iter,MaxIter,Restarts\n")
            f.write(
                "{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{},{},{},{}\n".format(
                    metrics["iou_macro_fg"],
                    metrics["dice_macro_fg"],
                    metrics["precision_macro_fg"],
                    metrics["recall_macro_fg"],
                    metrics["accuracy"],
                    metrics["time_sec"],
                    args.seed,
                    best_iter,
                    args.max_iter,
                    args.restarts,
                )
            )

        print(
            " IOU= {:.4f} | DICE= {:.4f} | PREC= {:.4f} | RECL= {:.4f} | ACC= {:.4f} | "
            " SEED= {} | MAXITER= {} | REST= {} | ITER= {} | TIME(s)= {:.4f} ".format(
                metrics["iou_macro_fg"],
                metrics["dice_macro_fg"],
                metrics["precision_macro_fg"],
                metrics["recall_macro_fg"],
                metrics["accuracy"],
                args.seed,
                args.max_iter,
                args.restarts,
                best_iter,
                metrics["time_sec"],
            )
        )


if __name__ == "__main__":
    main()
