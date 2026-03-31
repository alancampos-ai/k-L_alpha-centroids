import os, time, csv, argparse 
import numpy as np
from pathlib import Path
from scipy.optimize import linear_sum_assignment
from dipy.io.image import load_nifti, save_nifti
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from segment_dti import segmentation as segmentation_spd

from metrics import (
    confusion_flat, iou_from_cm, dice_from_cm,
    precision_from_cm, recall_from_cm, f1_from_cm, accuracy_from_cm
)
from multirun import run_multi_seed
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from iteration_convergence import add_iteration_and_convergence_to_csv


def _cm_fg(y_true, y_pred, K_fg: int):
    t = y_true.reshape(-1); p = y_pred.reshape(-1)
    cm = np.zeros((K_fg, K_fg), dtype=np.int64)
    m = (t > 0) & (p > 0)
    for ti, pi in zip(t[m], p[m]):
        if 1 <= ti <= K_fg and 1 <= pi <= K_fg:
            cm[ti-1, pi-1] += 1
    return cm


def _relabel_by_hungarian(y_true, y_pred, K_fg: int):
    cm = _cm_fg(y_true, y_pred, K_fg)
    if cm.sum() == 0:
        return y_pred
    r, c = linear_sum_assignment(-cm)
    y_new = y_pred.copy()
    for ri, cj in zip(r, c):
        y_new[y_pred == (cj+1)] = (ri+1)
    return y_new


def _save_three_planes(y_ref, y_pred, out_dir: Path, method_name: str, seed: int, classes: int, fs=9, alpha=None):


    method_slug = method_name.lower().replace(" ", "_"); folder_tag = f"k{classes}"
    X, Y, Z = y_ref.shape; xs, ys, zs = X // 2, Y // 2, Z // 2

    seg_colors = ["#FFFFFF", "#FFB703", "#219EBC", "#8ECAE6", "#FB8500", "#023047"]

    cmap = ListedColormap(seg_colors)

    # SAGITTAL
    ref = np.rot90(y_ref[xs, :, :]); plt.figure(figsize=(3, 3)); plt.imshow(ref, cmap=cmap); plt.axis("off"); plt.tight_layout(); 
    plt.savefig(out_dir / f"{folder_tag}_reference_alpha_{alpha:.3f}_seed{seed}_sagittal_slice{xs}.png", dpi=300, bbox_inches="tight"); plt.close()
    pred = np.rot90(y_pred[xs, :, :]); plt.figure(figsize=(3, 3)); plt.imshow(pred, cmap=cmap); plt.axis("off"); plt.tight_layout(); 
    plt.savefig(out_dir / f"{folder_tag}_{method_slug}_alpha_{alpha:.3f}_seed{seed}_sagittal_slice{xs}.png", dpi=300, bbox_inches="tight"); plt.close()

    # CORONAL
    ref = np.rot90(y_ref[:, ys, :]); plt.figure(figsize=(3, 3)); plt.imshow(ref, cmap=cmap); plt.axis("off"); plt.tight_layout(); 
    plt.savefig(out_dir / f"{folder_tag}_reference_alpha_{alpha:.3f}_seed{seed}_coronal_slice{ys}.png", dpi=300, bbox_inches="tight"); plt.close()
    pred = np.rot90(y_pred[:, ys, :]); plt.figure(figsize=(3, 3)); plt.imshow(pred, cmap=cmap); plt.axis("off"); plt.tight_layout(); 
    plt.savefig(out_dir / f"{folder_tag}_{method_slug}_alpha_{alpha:.3f}_seed{seed}_coronal_slice{ys}.png", dpi=300, bbox_inches="tight"); plt.close()

    # AXIAL
    ref = np.rot90(y_ref[:, :, zs]); plt.figure(figsize=(3, 3)); plt.imshow(ref, cmap=cmap); plt.axis("off"); plt.tight_layout(); 
    plt.savefig(out_dir / f"{folder_tag}_reference_alpha_{alpha:.3f}_seed{seed}_axial_slice{zs}.png", dpi=300, bbox_inches="tight"); plt.close()
    pred = np.rot90(y_pred[:, :, zs]); plt.figure(figsize=(3, 3)); plt.imshow(pred, cmap=cmap); plt.axis("off"); plt.tight_layout(); 
    plt.savefig(out_dir / f"{folder_tag}_{method_slug}_alpha_{alpha:.3f}_seed{seed}_axial_slice{zs}.png", dpi=300, bbox_inches="tight"); plt.close()


def _mean_std_ci(x):
    x = np.asarray(x, dtype=float)
    n = x.size
    m = float(np.mean(x)) if n else float("nan")
    s = float(np.std(x, ddof=1)) if n > 1 else 0.0
    ci = 1.96 * s / np.sqrt(n) if n > 1 else 0.0
    return m, s, ci, n


def _cleanup_alpha_seed_csv(out_dir: Path):
    for p in out_dir.glob("*_alpha.csv"):
        try:
            p.unlink()
        except FileNotFoundError:
            pass
    for p in out_dir.glob("*_seeds.csv"):
        try:
            p.unlink()
        except FileNotFoundError:
            pass


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
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    ap = argparse.ArgumentParser(description="euclidean, riemannian, logeuclidean; SPD aliases.")
    ap.add_argument(
        "--metric",
        choices=[
            "riemannian", "euclidean", "logeuclidean",
            "no_spd", "spd_le", "airm"
        ],
        default="riemannian"
    )
    ap.add_argument("--classes", type=int, default=3)
    ap.add_argument("--a-min", type=float, default=1.0)
    ap.add_argument("--a-max", type=float, default=2.0)
    ap.add_argument("--a-step", type=float, default=0.01)
    ap.add_argument("--max-iter", type=int, default=300)
    ap.add_argument("--restarts", type=int, default=10)
    ap.add_argument("--save-best", action="store_true")
    ap.add_argument("--data-dir", type=str, required=True)    
    ap.add_argument("--gt-pattern", default=None)
    ap.add_argument("--dti-file", default=None)
    ap.add_argument("--mask-file", default=None)    
    ap.add_argument("--seed", type=int, default=50)
    ap.add_argument("--iou-scheme", choices=["macro-fg", "macro-all", "micro-all-legacy"], default="macro-fg")
    ap.add_argument("--multi-seed", type=int, default=1)
    ap.add_argument("--seed-list", default="")
    args = ap.parse_args()
    ROOT = Path(__file__).resolve().parents[1]
    DATA = (ROOT / args.data_dir).resolve()
    run_name_map = {
        "riemannian": "airm", "euclidean": "no_spd", "logeuclidean": "spd",
        "no_spd": "no_spd", "spd_le": "spd", "airm": "airm",
    }
    metric_label = args.metric
    run_name = run_name_map[metric_label]
    if metric_label == "no_spd":
        args.metric = "euclidean"
    elif metric_label == "spd_le":
        args.metric = "logeuclidean"
    elif metric_label == "airm":
        args.metric = "riemannian"

    segmentation_fn = segmentation_spd

    OUT = (ROOT / "results" / f"k{args.classes}" / run_name).resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "Out").mkdir(parents=True, exist_ok=True)
    if args.seed_list:
        seeds_list = [int(s) for s in args.seed_list.split(",") if s.strip() != ""]
        for sd in seeds_list:
            run_multi_seed(
                script_path=Path(__file__).resolve(),
                metric=metric_label,
                classes=args.classes,
                a_min=args.a_min, a_max=args.a_max, a_step=args.a_step,
                max_iter=args.max_iter, restarts=args.restarts,
                data_dir=args.data_dir, dti_file=args.dti_file, mask_file=args.mask_file, gt_pattern=args.gt_pattern,
                seed_base=int(sd), seeds=1,
                iou_scheme=args.iou_scheme,
                save_best=bool(args.save_best)
            )
        agg_path = OUT / f"result_k{args.classes}_{run_name}_aggregate_seeds.csv"
        base_csv = OUT / f"result_k{args.classes}_{run_name}.csv"
        if base_csv.exists():
            rows = []
            with open(base_csv, "r") as f:
                rd = csv.DictReader(f)
                for r in rd:
                    rows.append(r)
            if rows:
                alphas = sorted(set(float(r["alpha"]) for r in rows))
                cols = ["Precision", "Recall", "Accuracy", "IoU", "Dice", "Time(s)"]
                with open(agg_path, "w", newline="") as fo:
                    wr = csv.writer(fo)
                    header = ["alpha"]
                    for c in cols:
                        header += [f"{c}_mean", f"{c}_std", f"{c}_ci95", f"{c}_n"]
                    wr.writerow(header)
                    for a in alphas:
                        sel = [r for r in rows if abs(float(r["alpha"]) - a) < 1e-9]
                        stats = []
                        for c in cols:
                            arr = [float(r[c]) for r in sel]
                            m, s, ci, n = _mean_std_ci(arr)
                            stats.extend([f"{m:.6f}", f"{s:.6f}", f"{ci:.6f}", str(n)])
                        wr.writerow([f"{a:.6f}"] + stats)
        _cleanup_alpha_seed_csv(OUT)
        return
    if args.multi_seed > 1:
        run_multi_seed(
            script_path=Path(__file__).resolve(),
            metric=metric_label,
            classes=args.classes,
            a_min=args.a_min, a_max=args.a_max, a_step=args.a_step,
            max_iter=args.max_iter, restarts=args.restarts,
            data_dir=args.data_dir, dti_file=args.dti_file, mask_file=args.mask_file, gt_pattern=args.gt_pattern,
            seed_base=int(args.seed), seeds=int(args.multi_seed),
            iou_scheme=args.iou_scheme,
            save_best=bool(args.save_best)
        )
        agg_path = OUT / f"result_k{args.classes}_{run_name}_aggregate_seeds.csv"
        base_csv = OUT / f"result_k{args.classes}_{run_name}.csv"
        if base_csv.exists():
            rows = []
            with open(base_csv, "r") as f:
                rd = csv.DictReader(f)
                for r in rd:
                    rows.append(r)
            if rows:
                alphas = sorted(set(float(r["alpha"]) for r in rows))
                cols = ["Precision", "Recall", "Accuracy", "IoU", "Dice", "Time(s)"]
                with open(agg_path, "w", newline="") as fo:
                    wr = csv.writer(fo)
                    header = ["alpha"]
                    for c in cols:
                        header += [f"{c}_mean", f"{c}_std", f"{c}_ci95", f"{c}_n"]
                    wr.writerow(header)
                    for a in alphas:
                        sel = [r for r in rows if abs(float(r["alpha"]) - a) < 1e-9]
                        stats = []
                        for c in cols:
                            arr = [float(r[c]) for r in sel]
                            m, s, ci, n = _mean_std_ci(arr)
                            stats.extend([f"{m:.6f}", f"{s:.6f}", f"{ci:.6f}", str(n)])
                        wr.writerow([f"{a:.6f}"] + stats)
        _cleanup_alpha_seed_csv(OUT)
        return
        
        
    dti_path = resolve_single_file(
        DATA,
        args.dti_file,
        ["*spd*_3x3.nii.gz"]
    )
    mask_path = resolve_single_file(
        DATA,
        args.mask_file,
        ["*mask*.nii.gz"]
    )
    gt_path = resolve_single_file(
        DATA,
        args.gt_pattern.format(n=args.classes) if args.gt_pattern else None,
        ["*gt*classes*.nii.gz"]
    )

    dti, affine = load_nifti(str(dti_path))
    mask, _ = load_nifti(str(mask_path)); mask = mask.astype(bool)
    y_true, _ = load_nifti(str(gt_path))
        
    
    if dti.ndim < 3 or mask.ndim != 3 or y_true.ndim != 3:
        raise ValueError("Invalid dimensionality.")
    if dti.shape[:3] != mask.shape or mask.shape != y_true.shape:
        raise ValueError("Shape mismatch among DTI, mask and ground truth.")
    csv_path = OUT / f"result_k{args.classes}_{run_name}.csv"
    exists = csv_path.exists()
    alphas = np.arange(args.a_min, args.a_max + 1e-12, args.a_step)
    add_iteration_and_convergence_to_csv(OUT, alphas, dti, mask, y_true, args, _relabel_by_hungarian)
    with open(csv_path, "a", newline="") as f:
        wr = csv.writer(f)
        if not exists:
            wr.writerow([
                "alpha",
                "IoU", "Dice", "Precision", "Recall", "Accuracy",
                "Time(s)", "seed", "Iter", "pred_path"
            ])
        per_restart_csv = OUT / f"result_k{args.classes}_{run_name}_per_restart.csv"
        per_exists = per_restart_csv.exists()
        per_f = open(per_restart_csv, "a", newline="")
        per_wr = csv.writer(per_f)
        if not per_exists:
            per_wr.writerow([
                "alpha", "seed",
                "IoU", "Dice", "Precision", "Recall", "Accuracy",
                "Time(s)"
            ])
        agg_alpha_csv = OUT / f"result_k{args.classes}_{run_name}_aggregate_alpha.csv"
        agg_alpha_rows = []
        best_global = -1.0
        best_global_alpha = None
        best_global_pred = None
        best_global_seed = None
        alphas = np.arange(args.a_min, args.a_max + 1e-12, args.a_step)
        Kcm = args.classes + 1
        summary_rows = []
        for a in alphas:
            t0 = time.time()
            best_local = -1.0
            best_seed = None
            best_pred = None
            best_prec = None
            best_rec = None
            best_acc = None
            best_iou = None
            best_dice = None
            best_iter_local = None
            first_pred = None
            first_metrics = None
            m_prec = []; m_rec = []; m_f1fg = []; m_f1mi = []; m_acc = []; m_ioufg = []; m_iouma = []; m_ioumi = []; m_dice = []
            for r in range(args.restarts):
                np.random.seed(args.seed + r)
                r0 = time.time()
                y_pred_raw, iter_real  = segmentation_fn(
                    dti, n_claster=args.classes, mask=mask,
                    metric_type=args.metric, expoent=float(a),
                    max_iterations=args.max_iter, dim_point=3
                )
                y_pred = _relabel_by_hungarian(y_true, y_pred_raw, args.classes)
                cm = confusion_flat(y_true.astype(np.int32), y_pred.astype(np.int32), n_classes=Kcm)
                iou_fg = iou_from_cm(cm, ignore_background=True, background_class=0, mode="macro")
                iou_ma = iou_from_cm(cm, ignore_background=False, background_class=0, mode="macro")
                iou_mi = iou_from_cm(cm, ignore_background=False, background_class=0, mode="micro")
                score = {"macro-fg": iou_fg, "macro-all": iou_ma, "micro-all-legacy": iou_mi}[args.iou_scheme]
                prec_fg = precision_from_cm(cm, ignore_background=True, background_class=0, mode="macro")
                rec_fg = recall_from_cm(cm, ignore_background=True, background_class=0, mode="macro")
                f1_fg = f1_from_cm(cm, ignore_background=True, background_class=0, mode="macro")
                f1_mi = f1_from_cm(cm, ignore_background=False, background_class=0, mode="micro")
                acc = accuracy_from_cm(cm)
                dice_fg = dice_from_cm(cm, ignore_background=True, background_class=0, mode="macro")
                m_prec.append(prec_fg); m_rec.append(rec_fg); m_f1fg.append(f1_fg); m_f1mi.append(f1_mi)
                m_acc.append(acc); m_ioufg.append(iou_fg); m_iouma.append(iou_ma); m_ioumi.append(iou_mi); m_dice.append(dice_fg)
                dt_r = time.time() - r0
                per_wr.writerow([
                    f"{a:.6f}", str(args.seed + r),
                    f"{iou_fg:.6f}", f"{dice_fg:.6f}", f"{prec_fg:.6f}", f"{rec_fg:.6f}", f"{acc:.6f}", f"{dt_r:.4f}"
                ])
                if score > best_local:
                    best_local = score
                    best_seed = int(args.seed + r)
                    best_pred = y_pred
                    best_prec = prec_fg
                    best_rec = rec_fg
                    best_acc = acc
                    best_iou = iou_fg
                    best_dice = dice_fg
                    best_iter_local = int(iter_real)
                    if r == 0:
                        first_pred = y_pred
                        first_metrics = (prec_fg, rec_fg, f1_fg, f1_mi, acc, iou_fg, iou_ma, iou_mi, dice_fg)
                elif r == 0:
                    first_pred = y_pred
                    first_metrics = (prec_fg, rec_fg, f1_fg, f1_mi, acc, iou_fg, iou_ma, iou_mi, dice_fg)
            dt = time.time() - t0
            pred_path = OUT / "Out" / f"gkac_k{args.classes}_{run_name}_alpha_{a:.3f}_seed{args.seed}.nii.gz"
            save_nifti(str(pred_path), best_pred.astype(np.int16), affine)
            wr.writerow([
                f"{a:.6f}",
                f"{best_iou:.6f}", f"{best_dice:.6f}", f"{best_prec:.6f}", f"{best_rec:.6f}", f"{best_acc:.6f}",
                f"{dt:.4f}", str(best_seed), str(best_iter_local), str(pred_path)
            ])
            summary_rows.append((a, best_iou, best_dice, best_prec, best_rec, best_acc, dt, best_iter_local))
            mp, sp, cp, np_ = _mean_std_ci(m_prec)
            mr, sr, cr, nr = _mean_std_ci(m_rec)
            mfg, sfg, cfg, nfg = _mean_std_ci(m_f1fg)
            mmi, smi, cmi, nmi = _mean_std_ci(m_f1mi)
            ma, sa, ca, na = _mean_std_ci(m_acc)
            mif, sif, cif, nif = _mean_std_ci(m_ioufg)
            mia, sia, cia, nia = _mean_std_ci(m_iouma)
            mim, sim, cim, nim = _mean_std_ci(m_ioumi)
            md, sd, cd, nd = _mean_std_ci(m_dice)
            agg_alpha_rows.append([
                f"{a:.6f}",
                f"{mp:.6f}", f"{sp:.6f}", f"{cp:.6f}", str(np_),
                f"{mr:.6f}", f"{sr:.6f}", f"{cr:.6f}", str(nr),
                f"{mfg:.6f}", f"{sfg:.6f}", f"{cfg:.6f}", str(nfg),
                f"{mmi:.6f}", f"{smi:.6f}", f"{cmi:.6f}", str(nmi),
                f"{ma:.6f}", f"{sa:.6f}", f"{ca:.6f}", str(na),
                f"{mif:.6f}", f"{sif:.6f}", f"{cif:.6f}", str(nif),
                f"{mia:.6f}", f"{sia:.6f}", f"{cia:.6f}", str(nia),
                f"{mim:.6f}", f"{sim:.6f}", f"{cim:.6f}", str(nim),
                f"{md:.6f}", f"{sd:.6f}", f"{cd:.6f}", str(nd)
            ])
            if best_local > best_global:
                best_global = best_local
                best_global_alpha = float(a)
                best_global_pred = best_pred.copy()
                best_global_seed = int(best_seed)
        per_f.close()


    if 'first_pred' in locals() and first_pred is not None:
        fig_dir = OUT / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)

        _save_three_planes(
            y_true, first_pred, fig_dir,
            method_name=run_name, seed=int(args.seed),
            classes=int(args.classes), fs=9, alpha=alphas[0]
        )

    if best_global_pred is not None and args.save_best:
        best_out = OUT / "Out" / f"gkac_k{args.classes}_{run_name}_best_alpha_{best_global_alpha:.3f}_seed{best_global_seed}.nii.gz"
        save_nifti(str(best_out), best_global_pred.astype(np.int16), affine)
        Kcm = args.classes + 1
        cm_best = confusion_flat(y_true.astype(np.int32), best_global_pred.astype(np.int32), n_classes=Kcm)
        prec_fg = precision_from_cm(cm_best, ignore_background=True, background_class=0, mode="macro")
        rec_fg = recall_from_cm(cm_best, ignore_background=True, background_class=0, mode="macro")
        f1_fg = f1_from_cm(cm_best, ignore_background=True, background_class=0, mode="macro")
        f1_mi = f1_from_cm(cm_best, ignore_background=False, background_class=0, mode="micro")
        acc = accuracy_from_cm(cm_best)
        iou_fg = iou_from_cm(cm_best, ignore_background=True, background_class=0, mode="macro")
        iou_ma = iou_from_cm(cm_best, ignore_background=False, background_class=0, mode="macro")
        iou_mi = iou_from_cm(cm_best, ignore_background=False, background_class=0, mode="micro")
        dice_fg = dice_from_cm(cm_best, ignore_background=True, background_class=0, mode="macro")
        with open(csv_path, "a", newline="") as f2:
            wr2 = csv.writer(f2)
            wr2.writerow([
                f"{best_global_alpha:.6f}",
                f"{iou_fg:.6f}", f"{dice_fg:.6f}", f"{prec_fg:.6f}", f"{rec_fg:.6f}", f"{acc:.6f}",
                f"{0.0000:.4f}", str(best_global_seed), str(best_out)
            ])
        _save_three_planes(
            y_true, best_global_pred, fig_dir,
            method_name=f"{run_name}_best", seed=int(best_global_seed), classes=int(args.classes), fs=9, alpha=best_global_alpha
        )
    agg_alpha_csv = OUT / f"result_k{args.classes}_{run_name}_aggregate_alpha.csv"
    if 'agg_alpha_rows' in locals() and agg_alpha_rows:
        with open(agg_alpha_csv, "w", newline="") as fa:
            w = csv.writer(fa)
            w.writerow([
                "alpha",
                "precision_macro_fg_mean", "precision_macro_fg_std", "precision_macro_fg_ci95", "precision_macro_fg_n",
                "recall_macro_fg_mean", "recall_macro_fg_std", "recall_macro_fg_ci95", "recall_macro_fg_n",
                "f1_macro_fg_mean", "f1_macro_fg_std", "f1_macro_fg_ci95", "f1_macro_fg_n",
                "f1_micro_all_mean", "f1_micro_all_std", "f1_micro_all_ci95", "f1_micro_all_n",
                "accuracy_mean", "accuracy_std", "accuracy_ci95", "accuracy_n",
                "iou_macro_fg_mean", "iou_macro_fg_std", "iou_macro_fg_ci95", "iou_macro_fg_n",
                "iou_macro_all_mean", "iou_macro_all_std", "iou_macro_all_ci95", "iou_macro_all_n",
                "iou_micro_all_mean", "iou_micro_all_std", "iou_micro_all_ci95", "iou_micro_all_n",
                "dice_macro_fg_mean", "dice_macro_fg_std", "dice_macro_fg_ci95", "dice_macro_fg_n"
            ])
            for row in agg_alpha_rows:
                w.writerow(row)
    _cleanup_alpha_seed_csv(OUT)

    if metric_label in ("riemannian", "airm"):
        method_type = "Riemannian"
    else:
        method_type = metric_label

    if summary_rows:
        best_row = max(summary_rows, key=lambda r: r[1])
        a_best, best_iou, best_dice, best_prec, best_rec, best_acc, best_time, best_iter = best_row
        best_iter = best_row[-1]
        print(f" IOU= {best_iou:.4f} | DICE= {best_dice:.4f} | PREC= {best_prec:.4f} | RECL= {best_rec:.4f} | ACC= {best_acc:.4f} | SEED= {args.seed} | MAXITER= {args.max_iter} | REST= {args.restarts} |  ITER= {best_iter}  |  TIME(s)= {best_time:.4f}")
        final_csv_path = OUT / f"result_k{args.classes}_{run_name}_final_summary.csv"
        with open(final_csv_path, "w", newline="") as fsum:
            wsum = csv.writer(fsum)
            wsum.writerow(["IoU", "Dice", "Precision", "Recall", "Accuracy", "Time(s)", "Seed", "Iter", "MaxIter", "Restarts"])
            wsum.writerow([
                f"{best_iou:.4f}", f"{best_dice:.4f}", f"{best_prec:.4f}", f"{best_rec:.4f}",
                f"{best_acc:.4f}", f"{best_time:.4f}", str(args.seed),
                str(best_iter), str(args.max_iter), str(args.restarts)
            ])

if __name__ == "__main__":
    main()
