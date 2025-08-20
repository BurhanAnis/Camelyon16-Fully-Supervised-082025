#!/usr/bin/env python3
import argparse
import io
import os
import pickle
from collections import defaultdict
from typing import Dict, List, Tuple
import umap.umap_ as umap

import lmdb
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Optional but recommended: pip install hdbscan
import hdbscan
import matplotlib.pyplot as plt


def read_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def build_keys_and_labels(slide_index: Dict) -> Tuple[List[str], np.ndarray, Dict[str, List[str]]]:
    """
    Recreate the *exact* LMDB key order used in training:
      key = f"{slide_id}_{count:06d}" with a single global counter.
    Also build a per-slide list of keys in the same order as info['patches'].
    Returns:
      - keys (global order)
      - labels (global 0/1 array)
      - slide_to_keys mapping with per-slide key lists (patch order preserved)
    """
    keys, labels = [], []
    slide_to_keys = defaultdict(list)

    count = 0
    # Important: rely on the same iteration order that created the index
    for slide_id, info in slide_index.items():
        for _patch in info["patches"]:
            key = f"{slide_id}_{count:06d}"
            is_tumor = 1 if _patch[2] else 0
            keys.append(key)
            labels.append(is_tumor)
            slide_to_keys[slide_id].append(key)
            count += 1
    return keys, np.array(labels, dtype=np.int64), slide_to_keys


def split_test_indices(keys: List[str]) -> np.ndarray:
    """
    Matches the training script's rule:
      test set = any slide_id containing 'test'
    """
    test_idx = []
    for i, key in enumerate(keys):
        slide_id = key.rsplit("_", 1)[0]
        if "test" in slide_id:
            test_idx.append(i)
    return np.array(test_idx, dtype=np.int64)


class LMDBReader:
    def __init__(self, lmdb_path: str):
        self.lmdb_path = lmdb_path
        self.env = lmdb.open(lmdb_path, readonly=True, lock=False,
                             readahead=False, max_readers=32)
        self.txn = self.env.begin(buffers=False)

    def get_image(self, key_ascii: str) -> Image.Image:
        val = self.txn.get(key_ascii.encode("ascii"))
        if val is None:
            raise KeyError(f"LMDB key missing: {key_ascii}")
        return Image.open(io.BytesIO(val)).convert("RGB")

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None
            self.txn = None


class PenultimateEmbedding(nn.Module):
    """
    Wrap ResNet50 (with final fc -> 2) to expose penultimate 2048-D features.
    """
    def __init__(self, resnet: nn.Module):
        super().__init__()
        # everything except the final fc
        self.backbone = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu,
            resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4,
            resnet.avgpool  # outputs [B, 2048, 1, 1]
        )

    def forward(self, x):
        feats = self.backbone(x)             # [B, 2048, 1, 1]
        feats = torch.flatten(feats, 1)      # [B, 2048]
        return feats


def load_model(best_model_path: str, device: torch.device) -> PenultimateEmbedding:
    """
    Recreate the ResNet50 head (fc->2), load weights, then expose penultimate.
    Handles DataParallel checkpoints.
    """
    # Match training: models.resnet50(weights=False)
    base = models.resnet50(weights=None)
    base.fc = nn.Linear(base.fc.in_features, 2)

    ckpt = torch.load(best_model_path, map_location="cpu")
    sd = ckpt.get("model_state", ckpt)  # tolerate raw state_dict

    # Some checkpoints may have 'module.' prefixes (DataParallel)
    new_sd = {}
    for k, v in sd.items():
        new_k = k.replace("module.", "") if k.startswith("module.") else k
        new_sd[new_k] = v
    base.load_state_dict(new_sd, strict=True)
    base.to(device)
    base.eval()

    return PenultimateEmbedding(base).to(device).eval()


def collect_false_positive_keys(
    slide_index: Dict,
    slide_index_test: Dict,
    slide_to_keys: Dict[str, List[str]]
) -> List[Tuple[str, str, int, float]]:
    """
    Determine FP patches using slide_index_test (preds/probs) and original labels in slide_index.
    Returns a list of tuples:
      (slide_id, lmdb_key, patch_idx_within_slide, predicted_prob)
    """
    fp_records = []
    for slide_id, info in slide_index.items():
        if slide_id not in slide_index_test:
            continue
        preds = slide_index_test[slide_id]["preds"]
        probs = slide_index_test[slide_id]["probs"]
        patches = info["patches"]  # list of tuples; index aligns with preds/probs

        # Defensive checks
        if not (len(patches) == len(preds) == len(probs)):
            # If this triggers, the original assumption about ordering didn't hold.
            # You can bail or try to reconcile; here we bail with a clear message.
            raise ValueError(
                f"Length mismatch for {slide_id}: "
                f"patches={len(patches)} preds={len(preds)} probs={len(probs)}"
            )

        per_slide_keys = slide_to_keys[slide_id]
        if len(per_slide_keys) != len(patches):
            raise ValueError(
                f"Key count mismatch for {slide_id}: "
                f"keys={len(per_slide_keys)} patches={len(patches)}"
            )

        for i, ((_, _, is_tumor), pred, prob) in enumerate(zip(patches, preds, probs)):
            if (not is_tumor) and (pred == 1):  # false positive
                fp_records.append((slide_id, per_slide_keys[i], i, float(prob)))
    return fp_records


def extract_embeddings_for_keys(
    fp_records: List[Tuple[str, str, int, float]],
    lmdb_reader: LMDBReader,
    device: torch.device,
    batch_size: int = 128,
) -> Tuple[np.ndarray, List[Tuple[str, str, int, float]]]:
    """
    Read images by LMDB key, push through embedding model.
    Returns:
      - embeddings [N, 2048]
      - filtered fp_records (if any images failed to load, they are dropped)
    """
    # Keep transform identical to "base_tf" from training (no aug)
    norm = transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])
    tf = transforms.Compose([transforms.ToTensor(), norm])

    # We'll batch images for GPU efficiency
    imgs_batch, meta_batch = [], []
    all_embeds, all_meta = [], []

    def flush():
        if not imgs_batch:
            return
        x = torch.stack(imgs_batch, dim=0).to(device)
        with torch.no_grad():
            feats = model(x)  # [B, 2048]
        all_embeds.append(feats.cpu().numpy())
        all_meta.extend(meta_batch)
        imgs_batch.clear()
        meta_batch.clear()

    for rec in fp_records:
        slide_id, key, idx_within_slide, prob = rec
        try:
            img = lmdb_reader.get_image(key)
        except KeyError:
            # Skip missing keys, but keep going
            continue
        tensor_img = tf(img)
        imgs_batch.append(tensor_img)
        meta_batch.append(rec)
        if len(imgs_batch) >= batch_size:
            flush()
    flush()

    if len(all_embeds) == 0:
        return np.zeros((0, 2048), dtype=np.float32), []

    return np.concatenate(all_embeds, axis=0), all_meta


def run_pca_and_hdbscan(
    embeds: np.ndarray,
    pca_dims_for_cluster: int = 50,
    pca_dims_for_plot: int = 2,   # kept but unused for viz now
    min_cluster_size: int = 25,
    min_samples: int = None,
    umap_neighbors: int = 15,
    umap_min_dist: float = 0.1,
    umap_metric: str = "euclidean",
    umap_seed: int = 42,
):
    """
    Standardize -> PCA (for clustering) -> HDBSCAN.
    Also compute a 2D UMAP projection for plotting.
    """
    scaler = StandardScaler(with_mean=True, with_std=True)
    Z = scaler.fit_transform(embeds)

    # PCA for clustering (stable, fast)
    pca_cluster = PCA(n_components=min(pca_dims_for_cluster, Z.shape[1]))
    Zc = pca_cluster.fit_transform(Z)

    # HDBSCAN clustering on PCA-reduced features
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom"
    )
    labels = clusterer.fit_predict(Zc)
    probs = clusterer.probabilities_
    outlier_scores = getattr(clusterer, "outlier_scores_", np.zeros_like(probs))

    # UMAP(2) for visualization on standardized features
    umap_2d = umap.UMAP(
        n_neighbors=umap_neighbors,
        min_dist=umap_min_dist,
        metric=umap_metric,
        random_state=umap_seed,
        n_components=2,
        verbose=False,
    )
    Z2 = umap_2d.fit_transform(Z)

    return {
        "Zc": Zc,
        "labels": labels,
        "probs": probs,
        "outlier_scores": outlier_scores,
        "Z2": Z2,                   # UMAP(2) now
        "scaler": scaler,
        "pca_cluster": pca_cluster,
        "umap_2d": umap_2d,         # save the UMAP model
        "clusterer": clusterer,
    }


def save_cluster_csv(out_csv, fp_meta, labels, probs, outlier_scores):
    rows = []
    for (slide_id, key, idx_within_slide, pred_prob), cl, pr, oscore in zip(fp_meta, labels, probs, outlier_scores):
        rows.append({
            "slide_id": slide_id,
            "lmdb_key": key,
            "patch_idx_within_slide": idx_within_slide,
            "pred_prob_positive": pred_prob,
            "cluster_label": int(cl),
            "cluster_membership_prob": float(pr),
            "outlier_score": float(oscore),
        })
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    return df


def plot_umap_scatter(Z2, labels, out_png):
    plt.figure(figsize=(7, 6))
    unique = np.unique(labels)
    for lab in unique:
        mask = labels == lab
        plt.scatter(Z2[mask, 0], Z2[mask, 1], s=10, alpha=0.7, label=f"Cluster {lab}")
    plt.legend(markerscale=2, fontsize=8)
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.title("False Positives — UMAP(2) with HDBSCAN labels")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def save_cluster_thumbnails(
    out_dir: str,
    df: pd.DataFrame,
    lmdb_reader: LMDBReader,
    max_per_cluster: int = 64,
    thumb_size: int = 128
):
    """
    Saves small thumbnails for each cluster to help rapid triage with pathologists.
    One PNG per cluster with up to max_per_cluster tiles arranged in a grid.
    """
    os.makedirs(out_dir, exist_ok=True)
    for lab in sorted(df["cluster_label"].unique(), key=lambda x: (x == -1, x)):
        sub = df[df["cluster_label"] == lab].head(max_per_cluster)
        if sub.empty:
            continue
        # determine grid
        n = len(sub)
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))
        canvas = Image.new("RGB", (cols * thumb_size, rows * thumb_size), (255, 255, 255))

        for i, row in enumerate(sub.itertuples(index=False)):
            try:
                img = lmdb_reader.get_image(row.lmdb_key)
            except KeyError:
                continue
            img = img.resize((thumb_size, thumb_size), Image.BILINEAR)
            r, c = divmod(i, cols)
            canvas.paste(img, (c * thumb_size, r * thumb_size))
        out_path = os.path.join(out_dir, f"cluster_{lab}.png")
        canvas.save(out_path)


def main():
    parser = argparse.ArgumentParser(description="Cluster false-positive patches via PCA + HDBSCAN on penultimate embeddings.")
    parser.add_argument("--index_path", type=str, required=True, help="Path to original slide_index (pickle) used to build LMDB keys.")
    parser.add_argument("--lmdb_path", type=str, required=True, help="LMDB containing PNG patch bytes.")
    parser.add_argument("--best_model", type=str, default="checkpoints/best_model.pt", help="Path to trained best model checkpoint.")
    parser.add_argument("--slide_index_test", type=str, default="slide_index_test.pkl", help="slide_index_test.pkl produced by training script.")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--pca_dims", type=int, default=50, help="PCA dims for clustering.")
    parser.add_argument("--min_cluster_size", type=int, default=100)
    parser.add_argument("--min_samples", type=int, default=None)
    parser.add_argument("--out_dir", type=str, default="fp_cluster_analysis")
    parser.add_argument("--save_thumbnails", action="store_true", help="Save per-cluster contact sheets.")
    parser.add_argument("--umap_neighbors", type=int, default=15, help="UMAP n_neighbors for 2D viz.")
    parser.add_argument("--umap_min_dist", type=float, default=0.1, help="UMAP min_dist for 2D viz.")
    parser.add_argument("--umap_metric", type=str, default="euclidean", help="UMAP metric for 2D viz.")
    parser.add_argument("--umap_seed", type=int, default=42, help="UMAP random_state for determinism.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Device
    device = (
        torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cuda") if torch.cuda.is_available()
        else torch.device("cpu")
    )

    # --- Load indexes and reconstruct LMDB keys exactly like training ---
    slide_index = read_pickle(args.index_path)
    keys, labels, slide_to_keys = build_keys_and_labels(slide_index)
    test_idx = split_test_indices(keys)

    # --- Load test results (preds/probs) produced by your training script ---
    slide_index_test = read_pickle(args.slide_index_test)

    # --- Determine false positives and collect their LMDB keys ---
    fp_records = collect_false_positive_keys(slide_index, slide_index_test, slide_to_keys)
    if len(fp_records) == 0:
        print("No false positives found in slide_index_test. Exiting.")
        return
    print(f"Found {len(fp_records)} false-positive patches.")

    # --- Load model and embedding wrapper ---
    global model
    model = load_model(args.best_model, device)

    # --- Read FP images and extract penultimate embeddings ---
    lmdb_reader = LMDBReader(args.lmdb_path)
    embeds, fp_meta = extract_embeddings_for_keys(fp_records, lmdb_reader, device, batch_size=args.batch_size)
    print(f"Extracted embeddings for {embeds.shape[0]} FP patches.")
    if embeds.shape[0] == 0:
        print("No embeddings extracted (keys missing?). Exiting.")
        lmdb_reader.close()
        return

    # --- PCA + HDBSCAN ---
    results = run_pca_and_hdbscan(
        embeds,
        pca_dims_for_cluster=args.pca_dims,
        pca_dims_for_plot=2,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        umap_neighbors=args.umap_neighbors,
        umap_min_dist=args.umap_min_dist,
        umap_metric=args.umap_metric,
        umap_seed=args.umap_seed,
    )

    # --- Save outputs ---
    df = save_cluster_csv(
        out_csv=os.path.join(args.out_dir, "false_positive_clusters.csv"),
        fp_meta=fp_meta,
        labels=results["labels"],
        probs=results["probs"],
        outlier_scores=results["outlier_scores"],
    )
    print(f"Saved cluster assignments to {os.path.join(args.out_dir, 'false_positive_clusters.csv')}")

    # 2D PCA scatter
    plot_umap_scatter(results["Z2"], results["labels"], os.path.join(args.out_dir, "umap_hdbscan.png"))
    print(f"Saved UMAP(2) scatter with HDBSCAN labels to {os.path.join(args.out_dir, 'umap_hdbscan.png')}")

    # Optional thumbnails per cluster
    if args.save_thumbnails:
        thumbs_dir = os.path.join(args.out_dir, "cluster_thumbnails")
        save_cluster_thumbnails(thumbs_dir, df, lmdb_reader, max_per_cluster=64, thumb_size=128)
        print(f"Saved per-cluster thumbnails in {thumbs_dir}")

    # Persist PCA/HDBSCAN objects for reproducibility
    with open(os.path.join(args.out_dir, "dimred_hdbscan_artifacts.pkl"), "wb") as f:
        pickle.dump({
            "scaler": results["scaler"],
            "pca_cluster": results["pca_cluster"],
            "umap_2d": results["umap_2d"],
            "clusterer": results["clusterer"]
        }, f)

    lmdb_reader.close()
    print("Done.")


if __name__ == "__main__":
    main()
