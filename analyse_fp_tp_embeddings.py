#!/usr/bin/env python3
import argparse, os, io, pickle, json
from collections import defaultdict
from tqdm import tqdm

import numpy as np
import pandas as pd
from PIL import Image

import lmdb
import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt

# --------------------- data ---------------------
class LMDBDatasetForInference(Dataset):
    """Returns (image_tensor, label, key) for given keys."""
    def __init__(self, lmdb_path, keys, labels, tf):
        self.lmdb_path = lmdb_path
        self.keys_b = [k.encode("ascii") for k in keys]
        self.keys   = keys
        self.labels = labels
        self.tf = tf
        self.env = None
        self.txn = None

    def _init_env(self):
        self.env = lmdb.open(self.lmdb_path, readonly=True, lock=False,
                             readahead=False, max_readers=32)
        self.txn = self.env.begin(buffers=False)

    def __len__(self): return len(self.keys_b)

    def __getitem__(self, idx):
        if self.env is None: self._init_env()
        data = self.txn.get(self.keys_b[idx])
        img  = Image.open(io.BytesIO(data)).convert("RGB")
        return self.tf(img), int(self.labels[idx]), self.keys[idx]

    def __del__(self):
        if self.env is not None:
            self.env.close()

# --------------------- utils ---------------------
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def load_state_dict_forgiving(model, ckpt_path, map_location="cpu"):
    ckpt = torch.load(ckpt_path, map_location=map_location)
    if isinstance(ckpt, dict) and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        state = ckpt
    elif isinstance(ckpt, dict):
        for key in ["model_state", "state_dict", "model_state_dict", "model", "net", "weights"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                state = ckpt[key]; break
        else:
            raise RuntimeError(f"Could not find state_dict in checkpoint; keys: {list(ckpt.keys())}")
    else:
        state = ckpt
    cleaned = { (k[7:] if k.startswith("module.") else k): v for k, v in state.items() }
    model.load_state_dict(cleaned, strict=True)

def build_test_keys(index_path, seed=42, val_size=0.1):
    with open(index_path, "rb") as f:
        slide_index = pickle.load(f)
    keys, labels = [], []
    count = 0
    for slide_id, info in slide_index.items():
        for (_, _, is_tumor) in info["patches"]:
            keys.append(f"{slide_id}_{count:06d}")
            labels.append(1 if is_tumor else 0)
            count += 1
    labels = np.array(labels, dtype=np.int64)
    all_idx = np.arange(len(keys))
    test_idx = np.array([i for i, k in enumerate(keys) if 'test' in k.split('_')[0]], dtype=int)
    test_keys   = [keys[i] for i in test_idx]
    test_labels = labels[test_idx]
    return slide_index, test_keys, test_labels

# --------------------- feature extractor ---------------------
# --------------------- feature extractor ---------------------
class ResNet50WithFeat(nn.Module):
    """Outputs (embeddings, logits); embeddings are 2048-d pooled features."""
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet50(weights=None)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 2)

    def forward(self, x):
        m = self.backbone
        x = m.conv1(x); x = m.bn1(x); x = m.relu(x); x = m.maxpool(x)
        x = m.layer1(x); x = m.layer2(x); x = m.layer3(x); x = m.layer4(x)
        feats = torch.flatten(m.avgpool(x), 1)   # [B, 2048]
        logits = m.fc(feats)                     # [B, 2]
        return feats, logits


# --------------------- main ---------------------
def main():
    ap = argparse.ArgumentParser("Analyze separability of FP vs TP embeddings (test set)")
    ap.add_argument("--index_path", type=str, required=True)
    ap.add_argument("--lmdb_path",  type=str, required=True)
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed",       type=int, default=42)
    ap.add_argument("--pred_thresh",type=float, default=0.5, help="prob threshold for predicted-positive")
    ap.add_argument("--out_dir",    type=str, default="fp_tp_embedding_analysis")
    ap.add_argument("--tsne",       action="store_true", help="also compute t-SNE (slower)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)

    # device
    device = (torch.device("mps") if torch.backends.mps.is_available()
              else torch.device("cuda") if torch.cuda.is_available()
              else torch.device("cpu"))

    # data
    slide_index, test_keys, test_labels = build_test_keys(args.index_path, seed=args.seed)
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    test_ds = LMDBDatasetForInference(args.lmdb_path, test_keys, test_labels, tf)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # model
    model = ResNet50WithFeat()
    load_state_dict_forgiving(model.backbone, args.checkpoint, map_location="cpu")
    model.to(device).eval()

    # inference: collect embeddings, probs, labels, keys
    all_feats, all_probs, all_lbls, all_keys = [], [], [], []
    with torch.no_grad():
        for imgs, lbls, keys in tqdm(test_loader, desc="Extract feats on test"):
            imgs = imgs.to(device)
            feats, logits = model(imgs)
            probs = torch.softmax(logits, dim=1)[:,1].cpu().numpy()
            all_probs.append(probs)
            all_feats.append(feats.cpu().numpy())
            all_lbls.append(lbls.numpy())
            all_keys.extend(list(keys))

    feats = np.vstack(all_feats)             # [N, 2048]
    probs = np.hstack(all_probs)             # [N]
    lbls  = np.hstack(all_lbls).astype(int)  # [N]

    # predicted positives at chosen threshold
    pred_pos = (probs >= args.pred_thresh)
    # among predicted positives, mark FP vs TP using GT labels
    mask_fp = pred_pos & (lbls == 0)
    mask_tp = pred_pos & (lbls == 1)

    # save a tidy CSV for just predicted positives
    df = pd.DataFrame({
        "key": all_keys,
        "label": lbls,
        "prob": probs,
        "pred_pos": pred_pos.astype(int),
        "is_fp": mask_fp.astype(int),
        "is_tp": mask_tp.astype(int),
    })
    df_predpos = df[df["pred_pos"] == 1].copy()
    df_predpos.to_csv(os.path.join(args.out_dir, "predpos_patch_table.csv"), index=False)

    # basic counts
    n_predpos = int(pred_pos.sum())
    n_fp = int(mask_fp.sum())
    n_tp = int(mask_tp.sum())
    print(f"Predicted-positive patches: {n_predpos} (TP={n_tp}, FP={n_fp})")

    # Scale features for analysis
    scaler = StandardScaler().fit(feats[pred_pos])
    Z = scaler.transform(feats[pred_pos])  # only predicted positives
    y = (lbls[pred_pos] == 1).astype(int)  # 1=TP, 0=FP

    # PCA to 2D for plotting + also to 50D for linear separability
    pca2 = PCA(n_components=2, random_state=args.seed).fit(Z)
    Z2 = pca2.transform(Z)

    # Save PCA 2D scatter (TP vs FP)
    fig, ax = plt.subplots(1,1, figsize=(7,6))
    ax.scatter(Z2[y==1,0], Z2[y==1,1], s=6, alpha=0.5, label="TP", marker='o')
    ax.scatter(Z2[y==0,0], Z2[y==0,1], s=6, alpha=0.5, label="FP", marker='x')
    ax.set_title("Predicted-positive patches: PCA(2) of embeddings")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.legend()
    plt.tight_layout()
    p_pca = os.path.join(args.out_dir, "pca2_predpos_tp_vs_fp.png")
    plt.savefig(p_pca, dpi=200)
    plt.close()
    print(f"Saved {p_pca}")

    # Optional t-SNE
    if args.tsne:
        tsne = TSNE(n_components=2, perplexity=30, learning_rate="auto", init="pca", random_state=args.seed)
        Zt = tsne.fit_transform(Z)
        fig, ax = plt.subplots(1,1, figsize=(7,6))
        ax.scatter(Zt[y==1,0], Zt[y==1,1], s=6, alpha=0.5, label="TP", marker='o')
        ax.scatter(Zt[y==0,0], Zt[y==0,1], s=6, alpha=0.5, label="FP", marker='x')
        ax.set_title("Predicted-positive patches: t-SNE(2) of embeddings")
        ax.set_xlabel("tSNE-1"); ax.set_ylabel("tSNE-2"); ax.legend()
        plt.tight_layout()
        p_tsne = os.path.join(args.out_dir, "tsne2_predpos_tp_vs_fp.png")
        plt.savefig(p_tsne, dpi=200)
        plt.close()
        print(f"Saved {p_tsne}")

    # Quantitative separability
    scores = {}
    # 1) Silhouette (higher is better separation); requires both classes present
    if (y==0).any() and (y==1).any() and len(np.unique(y))==2:
        try:
            sil = silhouette_score(Z, y, metric="euclidean")
            scores["silhouette"] = float(sil)
        except Exception as e:
            print("Silhouette failed:", e)

        # 2) Linear separability AUC with CV on predicted-positives
        X_lin = PCA(n_components=min(50, Z.shape[1]), random_state=args.seed).fit_transform(Z)
        skf = StratifiedKFold(n_splits=min(5, np.bincount(y).min()), shuffle=True, random_state=args.seed)
        aucs = []
        for tr, va in skf.split(X_lin, y):
            clf = LogisticRegression(max_iter=200, class_weight="balanced")
            clf.fit(X_lin[tr], y[tr])
            pr = clf.predict_proba(X_lin[va])[:,1]
            aucs.append(roc_auc_score(y[va], pr))
        scores["lin_auc_mean"] = float(np.mean(aucs))
        scores["lin_auc_std"]  = float(np.std(aucs))
    else:
        print("Not enough TP/FP in predicted positives to compute separability metrics.")

    # Save metrics + small preview counts
    out = {
        "n_predpos": n_predpos,
        "n_tp": n_tp,
        "n_fp": n_fp,
        "pred_thresh": args.pred_thresh,
        "scores": scores
    }
    with open(os.path.join(args.out_dir, "fp_tp_separability.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("Saved metrics:", out)

    # Also save raw arrays for future work (npz)
    np.savez_compressed(
        os.path.join(args.out_dir, "predpos_arrays.npz"),
        Z=Z, y=y, probs=probs[pred_pos], keys=np.array(df_predpos["key"].values, dtype=object)
    )
    print("Saved predpos_arrays.npz")

if __name__ == "__main__":
    main()





  