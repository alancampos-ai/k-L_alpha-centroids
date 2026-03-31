import numpy as np  

def confusion_flat(y_true, y_pred, n_classes):
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < n_classes and 0 <= p < n_classes:
            cm[t, p] += 1
    return cm

def accuracy_from_cm(cm):
    tot = cm.sum()
    return float(np.trace(cm) / tot) if tot > 0 else 0.0

def iou_from_cm(cm, *, ignore_background=True, background_class=0, mode="macro"):
    K = cm.shape[0]
    if mode == "micro":
        tp = np.trace(cm)
        fp = cm.sum(0).sum() - tp
        fn = cm.sum(1).sum() - tp
        den = tp + fp + fn
        return float(tp / den) if den > 0 else 0.0
    classes = list(range(K))
    if ignore_background and background_class in classes:
        classes.remove(background_class)
    vals = []
    for k in classes:
        tp = cm[k, k]
        fp = cm[:, k].sum() - tp
        fn = cm[k, :].sum() - tp
        den = tp + fp + fn
        if den > 0:
            vals.append(tp / den)
    return float(np.mean(vals)) if vals else 0.0

def _per_class_prf(cm, ignore_background=True, background_class=0):
    K = cm.shape[0]
    classes = list(range(K))
    if ignore_background and background_class in classes:
        classes.remove(background_class)
    prec = []; rec = []; f1 = []
    for k in classes:
        tp = cm[k, k]
        fp = cm[:, k].sum() - tp
        fn = cm[k, :].sum() - tp
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        pr = p + r
        f = (2.0 * p * r / pr) if pr > 0 else 0.0
        prec.append(p); rec.append(r); f1.append(f)
    return np.array(prec, float), np.array(rec, float), np.array(f1, float), np.array(classes, int)

def precision_from_cm(cm, *, ignore_background=True, background_class=0, mode="macro"):
    if mode == "micro":
        tp = np.trace(cm); fp = cm.sum(0).sum() - tp
        den = tp + fp
        return float(tp / den) if den > 0 else 0.0
    prec, _, _, _ = _per_class_prf(cm, ignore_background, background_class)
    return float(prec.mean()) if prec.size else 0.0

def recall_from_cm(cm, *, ignore_background=True, background_class=0, mode="macro"):
    if mode == "micro":
        tp = np.trace(cm); fn = cm.sum(1).sum() - tp
        den = tp + fn
        return float(tp / den) if den > 0 else 0.0
    _, rec, _, _ = _per_class_prf(cm, ignore_background, background_class)
    return float(rec.mean()) if rec.size else 0.0

def f1_from_cm(cm, *, ignore_background=True, background_class=0, mode="macro"):
    if mode == "micro":
        p = precision_from_cm(cm, ignore_background=False, background_class=background_class, mode="micro")
        r = recall_from_cm(cm,    ignore_background=False, background_class=background_class, mode="micro")
        pr = p + r
        return float(2.0 * p * r / pr) if pr > 0 else 0.0
    _, _, f1, _ = _per_class_prf(cm, ignore_background, background_class)
    return float(f1.mean()) if f1.size else 0.0

def dice_from_cm(cm, *, ignore_background=False, background_class=0, mode="macro"):
    if mode == "micro":
        return f1_from_cm(cm, ignore_background=False, background_class=background_class, mode="micro")
    K = cm.shape[0]
    classes = list(range(K))
    if ignore_background and background_class in classes:
        classes.remove(background_class)
    vals = []
    for k in classes:
        tp = cm[k, k]
        fn = cm[k, :].sum() - tp
        fp = cm[:, k].sum() - tp
        den = 2*tp + fp + fn
        if den > 0:
            vals.append((2*tp) / den)
    return float(np.mean(vals)) if vals else 0.0

def per_class_metrics_from_cm(cm, *, ignore_background=True, background_class=0):
    K = cm.shape[0]
    classes = list(range(K))
    if ignore_background and background_class in classes:
        classes.remove(background_class)
    P = []; R = []; F = []; D = []; J = []
    for k in classes:
        tp = cm[k, k]
        fp = cm[:, k].sum() - tp
        fn = cm[k, :].sum() - tp
        den_iou  = tp + fp + fn
        den_dice = 2*tp + fp + fn
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        pr = p + r
        f = (2.0 * p * r / pr) if pr > 0 else 0.0
        j = tp / den_iou if den_iou > 0 else 0.0
        d = (2*tp) / den_dice if den_dice > 0 else 0.0
        P.append(p); R.append(r); F.append(f); D.append(d); J.append(j)
    return {
        "classes": np.array(classes, int),
        "precision": np.array(P, float),
        "recall":    np.array(R, float),
        "f1":        np.array(F, float),
        "dice":      np.array(D, float),
        "iou":       np.array(J, float),
    }

