"""Supplementary H1 ablation of the proposed system (full+B5(ext)): the same system without session state
(memoryless) and with a reference but no history (ref_only), on the same sessions as the final run (seeds 20-24).
No code of the system changes: the variants are refitted exactly like the memoryless / ref_only variants of H1.
Only B4 (FT-C alone) and B5 (ext) are built; no public guard model is run.  See PREREG_ft.md, note of 2026-11-16.
usage: python followup/run_ablation.py --seeds 20 21 22 23 24 --ours-fpr-scale 0.4 --out results/final_ft/ablation"""
import argparse, json, sys
from pathlib import Path
import yaml
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_ft as R

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=[20, 21, 22, 23, 24])
    ap.add_argument("--out", default="results/final_ft/ablation")
    ap.add_argument("--ours-fpr-scale", type=float, default=0.4)
    a, rest = ap.parse_known_args()
    out = Path(a.out); out = out if out.is_absolute() else R.ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    R.OURS_FPR_SCALE = a.ours_fpr_scale
    R.MODES_BY["B5"] = ["ext"]
    cfg_path = R.make_config(a.seeds)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    h = cfg["h1"]
    h["baselines"] = ["B4"]; h["ext_content"] = ["B5"]; h["ext_modes_by"] = {"B5": ["ext"]}
    h["ext_ablate"] = ["memoryless", "ref_only"]
    cfg_path = HERE / "config_ablation.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    h1 = R.load_h1()
    ft, taus = R.install(h1, h["target_session_fpr"])
    sys.argv = ["h1", "--config", str(cfg_path), "--out", str(out)] + rest
    rc = h1.main()
    (out / "ft_train_info.json").write_text(json.dumps(ft.train_info, indent=1))
    (out / "content_path_thresholds.json").write_text(json.dumps(taus, indent=1))
    print(f"\nresults in {out}")
    sys.path.insert(0, str(R.ROOT))
    from pidetect.roc_plot import auto_figures
    auto_figures(out, all_methods=True)
    return rc

if __name__ == "__main__":
    sys.exit(main() or 0)
