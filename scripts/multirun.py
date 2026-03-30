import sys, subprocess

def run_multi_seed(script_path, metric, classes, a_min, a_max, a_step, max_iter, restarts,
                   data_dir, dti_file, mask_file, gt_pattern, seed_base, seeds,
                   iou_scheme, save_best):
    base = [sys.executable, str(script_path), 
            "--metric", metric,
            "--classes", str(classes),
            "--a-min", str(a_min),
            "--a-max", str(a_max),
            "--a-step", str(a_step),
            "--max-iter", str(max_iter),
            "--restarts", str(restarts),
            "--data-dir", str(data_dir),
            "--dti-file", str(dti_file),
            "--mask-file", str(mask_file),
            "--gt-pattern", str(gt_pattern),
            "--iou-scheme", iou_scheme,
            "--multi-seed", "1"]
    if save_best:
        base += ["--save-best"]
    for off in range(seeds):
        sd = seed_base + off
        cmd = base + ["--seed", str(sd)]
        subprocess.run(cmd, check=True)
