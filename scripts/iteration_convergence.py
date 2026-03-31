import csv
import numpy as np
from segment_dti import segmentation
from metrics import confusion_flat, iou_from_cm

run_name_map = {
    "riemannian": "airm",
    "airm": "airm",
}

def add_iteration_and_convergence_to_csv(OUT, alphas, dti, mask, y_true, args, relabel_function):
    run_tag = run_name_map.get(args.metric, args.metric) 

    iter_csv_path = OUT / f"result_k{args.classes}_{run_tag}_iteration_convergence.csv"
    iter_exists = iter_csv_path.exists()

    with open(iter_csv_path, "a", newline="") as iter_f:
        iter_writer = csv.writer(iter_f)
        if not iter_exists:
            iter_writer.writerow(["alpha", "iteration", "convergence"])

        for a in alphas:
            for r in range(args.restarts):
                np.random.seed(args.seed + r)

                (y_pred_raw, iter_real) = segmentation(
                    dti,
                    n_claster=args.classes,
                    mask=mask,
                    metric_type=args.metric,
                    expoent=float(a),
                    max_iterations=args.max_iter,
                    dim_point=3
                )

                y_pred = relabel_function(y_true, y_pred_raw, args.classes)

                cm = confusion_flat(y_true.astype(np.int32), y_pred.astype(np.int32), n_classes=args.classes + 1)

                iou_fg = iou_from_cm(cm, ignore_background=True, background_class=0, mode="macro")
                iou_ma = iou_from_cm(cm, ignore_background=False, background_class=0, mode="macro")
                iou_mi = iou_from_cm(cm, ignore_background=False, background_class=0, mode="micro")

                score = {"macro-fg": iou_fg, "macro-all": iou_ma, "micro-all-legacy": iou_mi}[args.iou_scheme]

                iteration = int(iter_real)
                convergence = score             
                iter_writer.writerow([f"{a:.6f}", iteration, convergence])

