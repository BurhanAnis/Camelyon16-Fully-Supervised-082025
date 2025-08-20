#!/usr/bin/env python3
"""
Generate ground-truth + prediction heatmap plots
for all slides in a slide_index dictionary.

Usage:
    python save_heatmaps.py \
        --index_file slide_index.pkl \
        --slides_dir /path/to/slides /other/path \
        --out_dir heatmap_plots
"""

import os, glob, argparse, pickle
import numpy as np
import openslide
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


# --- Auto-size figure based on slide aspect ratio ---
def _autosize_for_slide(W_plot, H_plot, ncols=2,
                        base_panel_height_in=8.0,
                        min_panel_in=4.0, max_panel_in=12.0, gap_in=0.6):
    aspect = max(W_plot, 1) / max(H_plot, 1)
    panel_h = base_panel_height_in
    panel_w = panel_h * aspect
    panel_w = min(max(panel_w, min_panel_in), max_panel_in)
    panel_h = max(panel_w / max(aspect, 1e-6), min_panel_in)
    panel_h = min(panel_h, max_panel_in)
    panel_w = max(panel_h * aspect, min_panel_in)
    fig_w = panel_w * ncols + gap_in * (ncols - 1)
    fig_h = panel_h
    return fig_w, fig_h


# --- Robust path resolver ---
def resolve_slide_path(index_path: str, slide_dirs: list[str]) -> str | None:
    """Find a slide by basename/stem recursively under slide_dirs."""
    base = os.path.basename(index_path) if index_path else None
    if not base:
        return None

    stem, ext = os.path.splitext(base)

    # 1) Exact basename anywhere
    for root in slide_dirs:
        matches = glob.glob(os.path.join(root, "**", base), recursive=True)
        if matches:
            return os.path.abspath(matches[0])

    # 2) Any extension with same stem
    patterns = [f"{stem}.*", f"{stem}.*".lower(), f"{stem}.*".upper()]
    for root in slide_dirs:
        for pat in patterns:
            matches = glob.glob(os.path.join(root, "**", pat), recursive=True)
            if matches:
                preferred = [m for m in matches if os.path.splitext(m)[1].lower()
                             in (".svs", ".tif", ".tiff", ".ndpi", ".scn", ".mrxs", ".bif")]
                pick = preferred[0] if preferred else matches[0]
                return os.path.abspath(pick)
    return None


def is_openslide_readable(path: str) -> bool:
    try:
        s = openslide.OpenSlide(path)
        s.close()
        return True
    except Exception:
        return False


# --- Plot function ---
def plot_slide_heatmap(
    slide_path: str,
    slide_id: str,
    work_lv: int,
    slide_index: dict,
    patch_size: int,
    plot_level: int = 4,
    prob_threshold: float = 0.05,
    top_k: int | None = None,
    alpha: float = 0.5,
    out_path: str | None = None
):
    info    = slide_index[slide_id]
    patches = info['patches']
    probs   = np.array(info.get('probs', []), dtype=float)

    slide   = openslide.OpenSlide(slide_path)
    ds_work = slide.level_downsamples[work_lv]
    ds_plot = slide.level_downsamples[plot_level]
    scale   = ds_work / ds_plot

    W_plot, H_plot = slide.level_dimensions[plot_level]
    img     = slide.read_region((0, 0), plot_level, (W_plot, H_plot)).convert("RGB")

    gt_heat   = np.full((H_plot, W_plot), np.nan, dtype=np.float32)
    pred_heat = np.full((H_plot, W_plot), np.nan, dtype=np.float32)
    w_p_plot = max(1, int(round(patch_size * scale)))

    # Ground truth
    for (y_work, x_work, is_tumor) in patches:
        if is_tumor:
            x_plot = int(x_work * scale)
            y_plot = int(y_work * scale)
            gt_heat[y_plot:y_plot + w_p_plot, x_plot:x_plot + w_p_plot] = 1.0

    # Predictions
    if probs.size > 0:
        if top_k is not None and top_k < len(probs):
            idxs = np.argsort(probs)[-top_k:]
        else:
            idxs = np.nonzero(probs >= prob_threshold)[0]

        for i in idxs:
            y_work, x_work, _ = patches[i]
            p = probs[i]
            if p >= prob_threshold:
                x_plot = int(x_work * scale)
                y_plot = int(y_work * scale)
                pred_heat[y_plot:y_plot + w_p_plot, x_plot:x_plot + w_p_plot] = p

    fig_w, fig_h = _autosize_for_slide(W_plot, H_plot)

    fig, (ax_gt, ax_hm) = plt.subplots(
        1, 2,
        figsize=(fig_w, fig_h),
        sharex=True, sharey=True,
        layout='constrained'
    )

    cmap = plt.get_cmap('jet')
    norm_prob = Normalize(vmin=0.0, vmax=1.0)

    ax_gt.imshow(img)
    ax_gt.imshow(gt_heat, cmap=cmap, norm=norm_prob, interpolation="nearest", alpha=alpha)
    ax_gt.set_title(f"{slide_id} — Ground Truth Heatmap")
    ax_gt.axis("off")

    ax_hm.imshow(img)
    ax_hm.imshow(pred_heat, cmap=cmap, norm=norm_prob, interpolation="nearest", alpha=alpha)
    ax_hm.set_title(f"{slide_id} — CNN Probability Heatmap")
    ax_hm.axis("off")

    cbar = fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap, norm=norm_prob),
        ax=[ax_gt, ax_hm],
        fraction=0.046, pad=0.04
    )
    cbar.set_label('Tumour probability (0–1)')

    for ax in (ax_gt, ax_hm):
        ax.set_aspect('equal')

    slide.close()

    if out_path:
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()


# --- Main driver ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_file", type=str, required=True,
                        help="Pickle with slide_index dict.")
    parser.add_argument("--slides_dir", type=str, nargs="+", required=True,
                        help="One or more directories to search for slide files.")
    parser.add_argument("--out_dir", type=str, default="heatmap_plots",
                        help="Output directory for plots.")
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--plot_level", type=int, default=6)
    parser.add_argument("--prob_threshold", type=float, default=0.05)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--strict", action="store_true",
                        help="Stop on missing/unsupported slides instead of skipping.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.index_file, "rb") as f:
        slide_index = pickle.load(f)

    total, done, skipped = len(slide_index), 0, 0

    for slide_id, info in slide_index.items():
        slide_path = resolve_slide_path(info.get("slide_path", ""), args.slides_dir)
        if slide_path is None:
            msg = f"[MISS] {slide_id}: {os.path.basename(info.get('slide_path',''))} not found."
            if args.strict: raise FileNotFoundError(msg)
            print(msg); skipped += 1; continue

        if not is_openslide_readable(slide_path):
            msg = f"[MISS] {slide_id}: unsupported by OpenSlide -> {slide_path}"
            if args.strict: raise RuntimeError(msg)
            print(msg); skipped += 1; continue

        out_path = os.path.join(args.out_dir, f"{slide_id}_heatmap.png")
        print(f"[INFO] Saving heatmap for {slide_id} -> {out_path}")

        try:
            # use per-slide work_level if present
            work_lv = 2
            plot_slide_heatmap(
                slide_path=slide_path,
                slide_id=slide_id,
                work_lv=work_lv,
                slide_index=slide_index,
                patch_size=args.patch_size,
                plot_level=args.plot_level,
                prob_threshold=args.prob_threshold,
                top_k=args.top_k,
                alpha=args.alpha,
                out_path=out_path
            )
            done += 1
        except Exception as e:
            msg = f"[ERROR] {slide_id}: plotting failed with {type(e).__name__}: {e}"
            if args.strict: raise
            print(msg); skipped += 1

    print(f"[SUMMARY] total={total}, saved={done}, skipped={skipped}")


if __name__ == "__main__":
    main()









