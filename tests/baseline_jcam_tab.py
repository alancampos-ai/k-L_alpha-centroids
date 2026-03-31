import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import rankdata, wilcoxon


METRICS = [
    ("IoU", "IOU"),
    ("Dice", "DICE"),
    ("Precision", "PREC"),
    ("Recall", "RECL"),
    ("Accuracy", "ACC"),
]

KEY_MAP: Dict[str, str] = {
    "EUC": "euclidean_no_spd_alpha_1.08",
    "LOG": "log_euclidean_spd_le_alpha_1.06",
    "AIRM": "riemannian_airm_alpha_1.0",
    "AIRM-GMM": "gmm",
}


def wilcoxon_pvalue(
    a: np.ndarray,
    b: np.ndarray,
    atol: float = 1e-4,
    rtol: float = 0.0,
    exact_max_n: int = 25,
) -> Optional[float]:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if a.size == 0 or b.size == 0 or a.size != b.size:
        return None
    diffs = a - b
    diffs = np.where(np.isclose(diffs, 0.0, atol=atol, rtol=rtol), 0.0, diffs)
    diffs_nz = diffs[diffs != 0.0]
    n = diffs_nz.size
    if n == 0:
        return 1.0
    absd = np.abs(diffs_nz)
    ranks = rankdata(absd, method="average")
    if n <= exact_max_n:
        r2 = np.rint(ranks * 2.0).astype(int)
        total2 = int(np.sum(r2))
        wplus_obs2 = int(np.sum(r2[diffs_nz > 0.0]))
        w_obs2 = min(wplus_obs2, total2 - wplus_obs2)
        counts: Dict[int, int] = {0: 1}
        for rr in r2:
            new_counts = dict(counts)
            for s, c in counts.items():
                new_counts[s + rr] = new_counts.get(s + rr, 0) + c
            counts = new_counts
        total_assign = 2**n
        extreme = 0
        for s, c in counts.items():
            w = min(s, total2 - s)
            if w <= w_obs2:
                extreme += c
        return float(extreme / total_assign)
    try:
        stat = wilcoxon(
            diffs_nz,
            zero_method="wilcox",
            alternative="two-sided",
            mode="auto",
        )
        return float(stat.pvalue)
    except Exception:
        return None


def bootstrap_ci_mean_of_diffs(
    diffs: np.ndarray,
    n_boot: int = 20000,
    alpha: float = 0.05,
    rng_seed: int = 123,
) -> Optional[Tuple[float, float]]:
    diffs = np.asarray(diffs, float)
    if diffs.size == 0:
        return None
    rng = np.random.default_rng(rng_seed)
    n = diffs.size
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = diffs[idx].mean(axis=1)
    lo = float(np.quantile(boot_means, alpha / 2.0))
    hi = float(np.quantile(boot_means, 1.0 - alpha / 2.0))
    return lo, hi


def hodges_lehmann(diffs: np.ndarray) -> Optional[float]:
    diffs = np.asarray(diffs, float)
    n = diffs.size
    if n == 0:
        return None
    walsh: List[float] = []
    for i in range(n):
        for j in range(i, n):
            walsh.append(0.5 * (diffs[i] + diffs[j]))
    return float(np.median(np.asarray(walsh, float)))


def bootstrap_ci_hodges_lehmann(
    diffs: np.ndarray,
    n_boot: int = 20000,
    alpha: float = 0.05,
    rng_seed: int = 123,
) -> Optional[Tuple[float, float]]:
    diffs = np.asarray(diffs, float)
    n = diffs.size
    if n == 0:
        return None
    rng = np.random.default_rng(rng_seed)
    boot = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        sample = diffs[rng.integers(0, n, size=n)]
        walsh: List[float] = []
        for i in range(n):
            for j in range(i, n):
                walsh.append(0.5 * (sample[i] + sample[j]))
        boot[b] = float(np.median(np.asarray(walsh, float)))
    lo = float(np.quantile(boot, alpha / 2.0))
    hi = float(np.quantile(boot, 1.0 - alpha / 2.0))
    return lo, hi


def rank_biserial_effect(
    a: np.ndarray,
    b: np.ndarray,
    atol: float = 1e-4,
    rtol: float = 0.0,
) -> Optional[float]:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if a.size == 0 or b.size == 0 or a.size != b.size:
        return None
    diffs = a - b
    diffs = np.where(np.isclose(diffs, 0.0, atol=atol, rtol=rtol), 0.0, diffs)
    diffs_nz = diffs[diffs != 0.0]
    if diffs_nz.size == 0:
        return 0.0
    absd = np.abs(diffs_nz)
    ranks = rankdata(absd, method="average")
    R_plus = float(np.sum(ranks[diffs_nz > 0.0]))
    R_minus = float(np.sum(ranks[diffs_nz < 0.0]))
    n = float(diffs_nz.size)
    denom = n * (n + 1.0) / 2.0
    if denom == 0.0:
        return 0.0
    r_rb = (R_plus - R_minus) / denom
    return float(r_rb)


def p_adjust_holm(pvals: List[Optional[float]]) -> List[Optional[float]]:
    n = len(pvals)
    valid = [(i, p) for i, p in enumerate(pvals) if p is not None]
    if not valid:
        return pvals[:]
    valid_sorted = sorted(valid, key=lambda t: t[1])
    m = len(valid_sorted)
    adj: List[Optional[float]] = [None] * n
    max_prev = 0.0
    for j, (idx, p) in enumerate(valid_sorted):
        k = m - j
        raw_adj = p * k
        if j == 0:
            max_prev = raw_adj
        else:
            max_prev = max(max_prev, raw_adj)
        adj_val = min(max_prev, 1.0)
        adj[idx] = float(adj_val)
    return adj


def build_metrics_from_results(raw: Dict[str, Any]) -> Dict[str, Any]:
    if "results" not in raw:
        raise KeyError("JSON does not contain 'results' key in the root.")
    results = raw["results"]
    metrics: Dict[str, Any] = {}
    for method_label, result_key in KEY_MAP.items():
        if result_key not in results:
            raise KeyError(f"Key '{result_key}' not found under 'results'.")
        runs = results[result_key]["runs"]
        aggregate: Dict[str, Any] = {}
        for metric_name, field in METRICS:
            values = np.array([r[field] for r in runs], dtype=float)
            mean = float(values.mean())
            std = float(values.std(ddof=1)) if values.size > 1 else 0.0
            aggregate[field] = {"mean": mean, "std": std}
        metrics[method_label] = {
            "aggregate": aggregate,
            "runs": runs,
        }
    out: Dict[str, Any] = {
        "K": raw.get("K", None),
        "metrics": metrics,
    }
    return out


def mean_std_str_from_agg(
    data: Dict[str, Any],
    method_key: str,
    metric_field: str,
    decimals: int = 3,
) -> str:
    agg = data["metrics"][method_key]["aggregate"][metric_field]
    return f"{agg['mean']:.{decimals}f} $\\pm$ {agg['std']:.{decimals}f}"


def format_ci(ci: Optional[Tuple[float, float]], decimals: int = 3) -> str:
    if ci is None:
        return "--"
    lo, hi = ci
    return f"[{lo:.{decimals}f}, {hi:.{decimals}f}]"


def build_table2(data: Dict[str, Any]) -> str:
    pairs = [
        ("AIRM", "EUC", "AIRM vs EUC"),
        ("LOG", "EUC", "LOG vs EUC"),
        ("AIRM", "LOG", "AIRM vs LOG"),
        ("AIRM", "AIRM-GMM", "AIRM vs AIRM--GMM"),
    ]
    lines: List[str] = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\footnotesize")
    lines.append(r"\setlength{\tabcolsep}{2pt}")
    lines.append(
        r"\caption{Comparação estatística das métricas no $\mathrm{SPD}(3)$ para $k=3$ "
        r"com o melhor $\alpha$ de cada método.}"
    )
    lines.append(r"\label{tab:stat_comparison_spd3_all_pairs_k3}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{ccccccc}")
    lines.append(r"\toprule")
    lines.append(
        r"{\scriptsize\textbf{Pair}} &"
        r"{\scriptsize\textbf{Metric}} &"
        r"\shortstack{{\scriptsize\textbf{(mean $\pm$ std)}}\\{\scriptsize\textbf{Method 1}}} &"
        r"\shortstack{{\scriptsize\textbf{(mean $\pm$ std)}}\\{\scriptsize\textbf{Method 2}}} &"
        r"\shortstack{{\scriptsize\textbf{Wilcoxon}}\\{\scriptsize\textbf{$p$-value}}} &"
        r"\shortstack{{\scriptsize\textbf{Bootstrap 95\% CI}}\\{\scriptsize\textbf{(Method 1 -- Method 2)}}} &"
        r"\shortstack{{\scriptsize\textbf{Hodges--Lehmann}}\\{\scriptsize\textbf{95\% CI (Method 1 -- Method 2)}}} \\"
    )
    lines.append(r"\midrule")
    for m1, m2, label in pairs:
        for metric, field in METRICS:
            m1_ms = mean_std_str_from_agg(data, m1, field, decimals=3)
            m2_ms = mean_std_str_from_agg(data, m2, field, decimals=3)
            runs1 = data["metrics"][m1]["runs"]
            runs2 = data["metrics"][m2]["runs"]
            if runs1 is not None and runs2 is not None:
                a = np.array([r[field] for r in runs1], float)
                b = np.array([r[field] for r in runs2], float)
                p = wilcoxon_pvalue(a, b)
                diffs = a - b
                ci = bootstrap_ci_mean_of_diffs(diffs)
                hl_ci = bootstrap_ci_hodges_lehmann(diffs)
                p_str = f"{p:.3f}" if p is not None else "--"
                ci_str = format_ci(ci, decimals=3)
                hl_ci_str = format_ci(hl_ci, decimals=3)
            else:
                p_str = "--"
                ci_str = "--"
                hl_ci_str = "--"
            line = (
                f"{label} & {metric} & {m1_ms} & {m2_ms} & "
                f"{p_str} & {ci_str} & {hl_ci_str} \\\\"
            )
            lines.append(line)
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def build_table3(data: Dict[str, Any]) -> str:
    pairs = [
        ("AIRM", "EUC", "AIRM vs EUC"),
        ("LOG", "EUC", "LOG vs EUC"),
        ("AIRM", "LOG", "AIRM vs LOG"),
        ("AIRM", "AIRM-GMM", "AIRM vs AIRM--GMM"),
    ]
    rows_num: List[Dict[str, Any]] = []
    p_list: List[Optional[float]] = []
    for m1, m2, label in pairs:
        runs1 = data["metrics"][m1]["runs"]
        runs2 = data["metrics"][m2]["runs"]
        if runs1 is not None and runs2 is not None:
            a = np.array([r["DICE"] for r in runs1], float)
            b = np.array([r["DICE"] for r in runs2], float)
            diffs = a - b
            hl = hodges_lehmann(diffs)
            hl_ci = bootstrap_ci_hodges_lehmann(diffs)
            r_rb = rank_biserial_effect(a, b)
            p_raw = wilcoxon_pvalue(a, b)
            rows_num.append(
                {
                    "pair": label,
                    "hl": hl,
                    "hl_ci": hl_ci,
                    "r_rb": r_rb,
                    "p_raw": p_raw,
                }
            )
            p_list.append(p_raw)
        else:
            rows_num.append(
                {
                    "pair": label,
                    "hl": None,
                    "hl_ci": None,
                    "r_rb": None,
                    "p_raw": None,
                }
            )
            p_list.append(None)
    p_adj_list = p_adjust_holm(p_list)
    lines: List[str] = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\footnotesize")
    lines.append(r"\setlength{\tabcolsep}{2pt}")
    lines.append(
        r"\caption{Diferenças pareadas de Dice entre métodos no $\mathrm{SPD}(3)$ para $k=3$ com o melhor $\alpha$ de cada método.}"
    )
    lines.append(r"\label{tab:Dice_pairwise_comparison}")
    lines.append(r"\begin{tabular}{lcccccc}")
    lines.append(r"\midrule")
    lines.append(
        r"Pair of methods     & \(\Delta_{\text{HL}}\) & 95\% CI              & \(r_{\text{rb}}\) & \(p\) & \(p_{\text{adj}}\) \\"
    )
    lines.append(r"\midrule")
    for row, p_adj in zip(rows_num, p_adj_list):
        if (
            row["hl"] is None
            or row["hl_ci"] is None
            or row["r_rb"] is None
            or row["p_raw"] is None
            or p_adj is None
        ):
            hl_str = "--"
            ci_str = "--"
            rrb_str = "--"
            p_str = "--"
            padj_str = "--"
        else:
            hl_str = f"{row['hl']:.3f}"
            lo, hi = row["hl_ci"]
            ci_str = f"[{lo:.3f}, {hi:.3f}]"
            rrb_str = f"{row['r_rb']:.3f}"
            p_str = f"{row['p_raw']:.3f}"
            padj_str = f"{p_adj:.3f}"
        line = f"{row['pair']} & {hl_str} & {ci_str} & {rrb_str} & {p_str} & {padj_str} \\\\"
        lines.append(line)
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def main() -> None:
    json_path_candidates = [Path("tab_jcam.json"), Path("tab.json")]
    json_path: Optional[Path] = None
    for cand in json_path_candidates:
        if cand.exists():
            json_path = cand
            break
    if json_path is None:
        raise FileNotFoundError(
            "No JSON file found. Expected 'tab_jcam.json' or 'tab.json' in the current directory."
        )
    with json_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if "metrics" in raw:
        data = raw
    else:
        data = build_metrics_from_results(raw)
    table2 = build_table2(data)
    table3 = build_table3(data)
    print("% TABLE 2: COMPARISON")
    print(table2)
    print()
    print("% TABLE 3: Dice Pairwise Comparison")
    print(table3)


if __name__ == "__main__":
    main()
