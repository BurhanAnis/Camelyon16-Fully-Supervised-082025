#!/usr/bin/env python3
# slide_ocsvm_aggregate.py
import argparse, os, io, pickle, json
from collections import defaultdict

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
from sklearn.svm import OneClassSVM

# --------------------- utils ---------------------
def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class LMDBDatasetForInference(Dataset):
    """Minimal LMDB reader that returns (tensor, key)."""
    def __init__(self, lmdb_path, keys, tf):
        self.lmdb_path = lmdb_path
        self.keys = [k.encode("ascii") for k in keys]
        self.tf = tf
        self.env = None
        self.txn = None

    def _init_env(self):
        self.env = lmdb.open(self.lmdb_path, readonly=True, lock=False,
                             readahead=False, max_readers=32)
        self.txn = self.env.begin(buffers=False)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        if self.env is None:
            self._init_env()
        data = self.txn.get(self.keys[idx])
        img = Image.open(io.BytesIO(data)).convert("RGB")
        return self.tf(img), self.keys[idx].decode("ascii")

    def __del__(self):
        if self.env is not None:
            self.env.close()

def load_state_dict_forgiving(model, ckpt_path, map_location="cpu", strict=True):
    ckpt = torch.load(ckpt_path, map_location=map_location)

    # If the file is already a pure state_dict (tensor values), use it
    if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        state = ckpt
    else:
        # Try common keys
        for key in ["state_dict", "model_state", "model_state_dict", "model", "net", "weights"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                state = ckpt[key]
                break
        else:
            raise RuntimeError(
                f"Couldn't find a state_dict in checkpoint. Top-level keys: {list(ckpt.keys())}"
            )

    # Remove DataParallel "module." prefixes if present
    cleaned = {}
    for k, v in state.items():
        new_k = k[7:] if k.startswith("module.") else k
        cleaned[new_k] = v

    missing, unexpected = model.load_state_dict(cleaned, strict=strict)
    if not strict:
        # In non-strict mode, you can log what didn't match
        print(f"Loaded with non-strict: missing={missing}, unexpected={unexpected}")

def slide_label_from_index(entry):
    # entry['patches'] is list of tuples (..., is_tumor)
    return int(any(is_tumor for *_, is_tumor in entry["patches"]))

def build_splits(keys, labels, seed=42, val_size=0.1):
    # test slides = those whose slide_id (part before first "_") contains "test"
    all_idx = np.arange(len(keys))
    test_idx = [i for i, k in enumerate(keys) if 'test' in k.split('_')[0]]
    trainval_idx = np.setdiff1d(all_idx, test_idx)
    # stratified split for val
    from sklearn.model_selection import train_test_split
    labels_arr = np.array(labels, dtype=np.int64)
    train_idx, val_idx = train_test_split(
        trainval_idx, test_size=val_size,
        stratify=labels_arr[trainval_idx], random_state=seed
    )
    return train_idx, val_idx, np.array(test_idx, dtype=int)

def slide_features(probs, ks=(1,5,10,20), thr=(0.3,0.5,0.7,0.9), nbins=10):
    from scipy.stats import skew, kurtosis
    p = np.asarray(probs, dtype=np.float32)
    if p.size == 0:
        return np.zeros(5 + 4 + 2*len(ks) + 1 + len(thr) + nbins, dtype=np.float32)
    p.sort()
    feats = []
    feats += [p.mean(), p.std(), p.min(), p.max(), np.median(p)]
    feats += list(np.quantile(p, [0.8, 0.9, 0.95, 0.99]))
    feats += [skew(p), kurtosis(p, fisher=True)]
    for k in ks:
        k = min(k, len(p))
        topk = p[-k:]
        feats += [topk.mean(), topk.max()]
    k = min(5, len(p))
    feats += [p[-1] - p[-k:].mean()]
    for t in thr:
        feats += [(p >= t).mean()]
    hist, _ = np.histogram(p, bins=nbins, range=(0,1), density=False)
    hist = hist / max(1, hist.sum())
    feats += list(hist.astype(np.float32))
    return np.array(feats, dtype=np.float32)

def safe_pca_fit_transform(Xtr, Xva, Xte, seed, target=32):
    X_combo = np.vstack([Xtr, Xva])
    n_samples, n_features = X_combo.shape
    max_valid = max(0, min(n_samples, n_features) - 1)
    n_comp = min(target, max_valid)
    if n_comp >= 1:
        pca = PCA(n_components=n_comp, whiten=False, random_state=seed)
        pca.fit(X_combo)
        return pca.transform(Xtr), pca.transform(Xva), pca.transform(Xte), pca
    return Xtr, Xva, Xte, None

# --------------------- main ---------------------
def main():
    ap = argparse.ArgumentParser("Slide-level OC-SVM aggregation (inference-only)")
    ap.add_argument("--index_path", type=str, required=True)
    ap.add_argument("--lmdb_path", type=str, required=True)
    ap.add_argument("--checkpoint", type=str, required=True, help="best.pt from training")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--val_size", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--nu", type=float, default=0.05, help="OC-SVM nu (fraction of allowed outliers)")
    ap.add_argument("--target_spec", type=float, default=0.90)
    ap.add_argument("--use_pca", action="store_true")
    ap.add_argument("--pca_target", type=int, default=32)
    ap.add_argument("--out_dir", type=str, default="ocsvm_outputs")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)

    # device
    device = (torch.device("mps") if torch.backends.mps.is_available()
              else torch.device("cuda") if torch.cuda.is_available()
              else torch.device("cpu"))

    # ---------- load index & build key list (must match your LMDB keying) ----------
    with open(args.index_path, "rb") as f:
        slide_index = pickle.load(f)

    keys, labels = [], []
    count = 0
    for slide_id, info in slide_index.items():
        for (_, _, is_tumor) in info["patches"]:
            keys.append(f"{slide_id}_{count:06d}")
            labels.append(1 if is_tumor else 0)
            count += 1
    labels = np.array(labels, dtype=np.int64)

    train_idx, val_idx, test_idx = build_splits(keys, labels, seed=args.seed, val_size=args.val_size)
    train_keys = [keys[i] for i in train_idx]
    val_keys   = [keys[i] for i in val_idx]
    test_keys  = [keys[i] for i in test_idx]

    # ---------- model (must mirror the training architecture) ----------
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    load_state_dict_forgiving(model, args.checkpoint)
    model.to(device)
    model.eval()

    # ---------- data loaders (inference only) ----------
    def make_loader(klist):
        ds = LMDBDatasetForInference(args.lmdb_path, klist, tf)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    train_loader = make_loader(train_keys)
    val_loader   = make_loader(val_keys)
    test_loader  = make_loader(test_keys)

    # ---------- collect patch probabilities per slide ----------
    @torch.no_grad()
    def collect_probs(loader):
        out_probs, out_keys = [], []
        for imgs, batch_keys in loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = torch.softmax(logits, dim=1)[:,1].cpu().numpy()
            out_probs.extend(probs.tolist())
            out_keys.extend(batch_keys)
        by_slide = defaultdict(list)
        for k, p in zip(out_keys, out_probs):
            sid = k.rsplit("_", 1)[0]
            by_slide[sid].append(p)
        return by_slide

    tr_slide_probs = collect_probs(train_loader)
    va_slide_probs = collect_probs(val_loader)
    te_slide_probs = collect_probs(test_loader)

    # ---------- build feature matrices & labels ----------
    def build_Xy(slide_probs):
        X, y, sids = [], [], []
        for sid, probs in slide_probs.items():
            X.append(slide_features(probs))
            y.append(slide_label_from_index(slide_index[sid]))
            sids.append(sid)
        return np.vstack(X).astype(np.float32), np.array(y, dtype=np.int64), sids

    Xtr, ytr, sid_tr = build_Xy(tr_slide_probs)
    Xva, yva, sid_va = build_Xy(va_slide_probs)
    Xte, yte, sid_te = build_Xy(te_slide_probs)

    # ---------- scale (+ optional PCA) ----------
    scaler = StandardScaler().fit(np.vstack([Xtr, Xva]))
    Xtr_s, Xva_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xva), scaler.transform(Xte)

    if args.use_pca:
        Xtr_s, Xva_s, Xte_s, pca = safe_pca_fit_transform(Xtr_s, Xva_s, Xte_s, seed=args.seed, target=args.pca_target)
    else:
        pca = None

    # ---------- OC-SVM on NORMAL slides (train+val normals) ----------
    normal_mask_tr = (ytr == 0)
    normal_mask_va = (yva == 0)
    X_oc_train = np.vstack([Xtr_s[normal_mask_tr], Xva_s[normal_mask_va]])
    ocsvm = OneClassSVM(kernel="rbf", gamma="scale", nu=args.nu)
    ocsvm.fit(X_oc_train)

    df_tr = ocsvm.decision_function(Xtr_s).ravel()
    df_va = ocsvm.decision_function(Xva_s).ravel()
    df_te = ocsvm.decision_function(Xte_s).ravel()

    # ---------- choose threshold on validation to meet target specificity ----------
    thr_candidates = np.unique(df_va)
    best_thr, best_spec_gap, best_sens_at_spec = thr_candidates[0], 1e9, 0.0
    for thr in thr_candidates:
        yhat_va_normal = (df_va >= thr).astype(int)  # 1=normal
        tn = ((yva == 0) & (yhat_va_normal == 1)).sum()
        fp = ((yva == 0) & (yhat_va_normal == 0)).sum()
        tp = ((yva == 1) & (yhat_va_normal == 0)).sum()
        fn = ((yva == 1) & (yhat_va_normal == 1)).sum()
        spec = tn / max(1, tn + fp)
        sen  = tp / max(1, tp + fn)
        if spec >= args.target_spec:
            gap = spec - args.target_spec
            if gap < best_spec_gap or (abs(gap - best_spec_gap) < 1e-9 and sen > best_sens_at_spec):
                best_thr, best_spec_gap, best_sens_at_spec = thr, gap, sen

    # fallback: Youden's J if target specificity cannot be reached
    if best_spec_gap == 1e9:
        best_thr, best_J = thr_candidates[0], -1
        for thr in thr_candidates:
            yhat_va_normal = (df_va >= thr).astype(int)
            tn = ((yva == 0) & (yhat_va_normal == 1)).sum()
            fp = ((yva == 0) & (yhat_va_normal == 0)).sum()
            tp = ((yva == 1) & (yhat_va_normal == 0)).sum()
            fn = ((yva == 1) & (yhat_va_normal == 1)).sum()
            spec = tn / max(1, tn + fp)
            sen  = tp / max(1, tp + fn)
            J = sen + spec - 1
            if J > best_J:
                best_J, best_thr = J, thr

    # ---------- test predictions & metrics ----------
    yhat_te_normal = (df_te >= best_thr).astype(int)
    yhat_te_tumor  = 1 - yhat_te_normal
    tn = ((yte == 0) & (yhat_te_tumor == 0)).sum()
    fp = ((yte == 0) & (yhat_te_tumor == 1)).sum()
    tp = ((yte == 1) & (yhat_te_tumor == 1)).sum()
    fn = ((yte == 1) & (yhat_te_tumor == 0)).sum()
    test_spec = tn / max(1, tn + fp)
    test_sen  = tp / max(1, tp + fn)
    test_acc  = (tp + tn) / max(1, tp + tn + fp + fn)

    print(f"[OCSVM] nu={args.nu}, target_spec={args.target_spec:.2f}, "
          f"Test Acc={test_acc:.3f}, Sen={test_sen:.3f}, Spec={test_spec:.3f}")

    # ---------- save outputs ----------
    results = {
        "nu": args.nu,
        "target_spec": args.target_spec,
        "threshold": float(best_thr),
        "test_metrics": {"acc": float(test_acc), "sen": float(test_sen), "spec": float(test_spec)},
        "val_size": int(len(yva)),
        "test_size": int(len(yte)),
        "use_pca": bool(args.use_pca),
        "pca_components": int(pca.n_components_) if pca is not None else 0,
    }
    with open(os.path.join(args.out_dir, "slide_level_ocsvm_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    df_slides = pd.DataFrame({
        "slide_id": sid_te,
        "label": yte,
        "ocsvm_decision_fn": df_te,  # higher = more normal
        "pred_tumor": yhat_te_tumor, # 1 = tumor
    })
    df_slides.to_csv(os.path.join(args.out_dir, "slide_level_ocsvm_test.csv"), index=False)

    ocsvm_slide_pred = {sid: {"gt": int(gt), "df": float(df), "pred_tumor": int(pred)}
                        for sid, gt, df, pred in zip(sid_te, yte, df_te, yhat_te_tumor)}
    with open(os.path.join(args.out_dir, "slide_level_ocsvm_test.pkl"), "wb") as f:
        pickle.dump(ocsvm_slide_pred, f)

    print(f"Saved: {args.out_dir}/slide_level_ocsvm_results.json, "
          f"{args.out_dir}/slide_level_ocsvm_test.csv, "
          f"{args.out_dir}/slide_level_ocsvm_test.pkl")

if __name__ == "__main__":
    main()





