import argparse
import os
import pickle
import io

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from collections import defaultdict

import lmdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class LMDBAugDataset(Dataset):
    def __init__(self, lmdb_path, keys, labels, flags, base_tf=None, aug_tf=None):
        self.lmdb_path = lmdb_path
        self.keys       = [k.encode('ascii') for k in keys]
        self.labels     = labels
        self.flags      = flags
        self.base_tf    = base_tf or transforms.ToTensor()
        self.aug_tf     = aug_tf  or self.base_tf
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
        img  = Image.open(io.BytesIO(data)).convert('RGB')
        tf   = self.aug_tf if self.flags[idx] else self.base_tf
        return tf(img), self.labels[idx]

    def __del__(self):
        if self.env is not None:
            self.env.close()


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma: float = 2.0, reduction='mean'):
        super().__init__()
        self.weight    = weight
        self.gamma     = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce)
        loss = (1 - pt) ** self.gamma * ce
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


def main():
    parser = argparse.ArgumentParser(description='Train DL model on WSI patches with OHEM + Snapshot Ensembling')
    parser.add_argument('--index_path', type=str, required=True)
    parser.add_argument('--lmdb_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--patience', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--loss_csv', type=str, default='epoch_losses.csv')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--save_interval', type=int, default=5)
    parser.add_argument('--aug_factor', type=int, default=5)
    parser.add_argument('--val_size', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--ohem_ratio', type=float, default=3.0)
    parser.add_argument('--T_0', type=int, default=10)
    parser.add_argument('--T_mult', type=int, default=2)
    parser.add_argument('--eta_min', type=float, default=1e-5)
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    with open(args.index_path, 'rb') as f:
        slide_index = pickle.load(f)

    keys, labels = [], []
    count = 0
    for slide_id, info in slide_index.items():
        for (_, _, is_tumor) in info['patches']:
            keys.append(f"{slide_id}_{count:06d}")
            labels.append(1 if is_tumor else 0)
            count += 1
    labels = np.array(labels, dtype=np.int64)

    all_idx = np.arange(len(keys))
    test_idx = [i for i, k in enumerate(keys) if 'test' in k.split('_')[0]]
    trainval_idx = np.setdiff1d(all_idx, test_idx)
    train_idx, val_idx = train_test_split(trainval_idx, test_size=args.val_size,
                                          stratify=labels[trainval_idx], random_state=args.seed)

    def make_entries(indices, augment=False):
        out_k, out_l, out_f = [], [], []
        for i in indices:
            out_k.append(keys[i]); out_l.append(int(labels[i])); out_f.append(False)
            if augment and labels[i] == 1:
                for _ in range(args.aug_factor):
                    out_k.append(keys[i]); out_l.append(1); out_f.append(True)
        return out_k, out_l, out_f

    train_keys, train_labels, train_flags = make_entries(train_idx, augment=True)
    val_keys, val_labels, val_flags = make_entries(val_idx, augment=False)
    test_keys, test_labels, test_flags = make_entries(test_idx, augment=False)

    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    base_tf = transforms.Compose([transforms.ToTensor(), norm])
    aug_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(90),
        transforms.ToTensor(), norm
    ])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_ds = LMDBAugDataset(args.lmdb_path, train_keys, train_labels, train_flags, base_tf, aug_tf)
    val_ds = LMDBAugDataset(args.lmdb_path, val_keys, val_labels, val_flags, base_tf, base_tf)
    test_ds = LMDBAugDataset(args.lmdb_path, test_keys, test_labels, test_flags, base_tf, base_tf)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    class_weights = torch.tensor(np.bincount(labels[train_idx], minlength=2).sum() / np.bincount(labels[train_idx], minlength=2),
                                 device=device, dtype=torch.float)

    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    criterion = FocalLoss(weight=class_weights, gamma=2.0, reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=args.T_0, T_mult=args.T_mult, eta_min=args.eta_min
    )

    metrics = []
    best_val_loss = float('inf')
    no_improve = 0
    snapshot_paths = []

    for epoch in range(1, args.max_epochs + 1):
        model.train()
        epoch_loss, epoch_acc = 0.0, 0

        for imgs, lbls in tqdm(train_loader, desc=f"Epoch {epoch} [Train]"):
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            ce = F.cross_entropy(logits, lbls, weight=class_weights, reduction='none')
            pt = torch.exp(-ce)
            losses = criterion(logits, lbls)

            pos_mask = lbls == 1
            neg_mask = lbls == 0
            pos_losses = losses[pos_mask]
            neg_losses = losses[neg_mask]

            num_pos = pos_losses.size(0)
            if num_pos > 0:
                k = min(neg_losses.size(0), int(args.ohem_ratio * num_pos))
                hard_neg, _ = torch.topk(neg_losses, k)
                batch_loss = torch.cat([pos_losses, hard_neg]).mean()
            else:
                k = min(64, neg_losses.size(0))
                hard_neg, _ = torch.topk(neg_losses, k)
                batch_loss = hard_neg.mean()

            batch_loss.backward()
            optimizer.step()

            epoch_loss += batch_loss.item() * imgs.size(0)
            preds = logits.argmax(dim=1)
            epoch_acc += (preds == lbls).sum().item()

        epoch_loss /= len(train_loader.dataset)
        epoch_acc /= len(train_loader.dataset)

        model.eval()
        val_loss, val_acc = 0.0, 0
        with torch.no_grad():
            for imgs, lbls in tqdm(val_loader, desc=f"Epoch {epoch} [Val]"):
                imgs, lbls = imgs.to(device), lbls.to(device)
                logits = model(imgs)
                ce = F.cross_entropy(logits, lbls, weight=class_weights, reduction='none')
                pt = torch.exp(-ce)
                losses = (1 - pt) ** criterion.gamma * ce
                val_loss += losses.mean().item() * imgs.size(0)
                preds = logits.argmax(dim=1)
                val_acc += (preds == lbls).sum().item()

        val_loss /= len(val_loader.dataset)
        val_acc /= len(val_loader.dataset)

        print(f"Epoch {epoch:02d}  Train Loss: {epoch_loss:.4f}  Train Acc: {epoch_acc:.4f}  Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.4f}")

        metrics.append({'epoch': epoch, 'train_loss': epoch_loss, 'train_acc': epoch_acc, 'val_loss': val_loss, 'val_acc': val_acc})
        pd.DataFrame(metrics).to_csv(args.loss_csv, index=False)
        scheduler.step(epoch + 1)

        if (epoch + 1) % args.save_interval == 0:
            snapshot_path = os.path.join(args.checkpoint_dir, f"snapshot_epoch_{epoch}.pt")
            torch.save(model.state_dict(), snapshot_path)
            snapshot_paths.append(snapshot_path)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, 'best.pt'))
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"Early stopping after {args.patience} no-improve epochs.")
                break

    print("Starting test phase with snapshot ensembling...")

    ensemble_logits = torch.zeros((len(test_ds), 2), device=device)
    all_lbls = []

    # Collect ensemble logits
    with torch.no_grad():
        for snapshot_path in snapshot_paths:
            print(f"Loading snapshot: {snapshot_path}")
            snapshot_model = models.resnet50(weights=None)
            snapshot_model.fc = nn.Linear(snapshot_model.fc.in_features, 2)
            snapshot_model.load_state_dict(torch.load(snapshot_path, map_location=device))
            snapshot_model.to(device)
            snapshot_model.eval()
            if torch.cuda.device_count() > 1:
                snapshot_model = nn.DataParallel(snapshot_model)

            start = 0
            for imgs, lbls in tqdm(test_loader, desc=f"Inference [{os.path.basename(snapshot_path)}]"):
                imgs = imgs.to(device)
                logits = snapshot_model(imgs)
                bsz = imgs.size(0)
                ensemble_logits[start:start+bsz] += F.softmax(logits, dim=1)
                if len(all_lbls) < len(test_ds):
                    all_lbls.extend(lbls.cpu().tolist())
                start += bsz

    # Average over ensemble
    ensemble_logits /= len(snapshot_paths)
    probs_class1 = ensemble_logits[:, 1].cpu().numpy()
    preds_class1 = (probs_class1 > 0.5).astype(int)
    true_labels = np.array(all_lbls)

    # Compute patch-level loss and accuracy using FocalLoss manually
    test_loss, test_acc = 0.0, 0
    with torch.no_grad():
        batch_idx = 0
        for imgs, lbls in tqdm(test_loader, desc="Test Loss Eval"):
            imgs, lbls = imgs.to(device), lbls.to(device)
            logits = ensemble_logits[batch_idx:batch_idx + imgs.size(0)].to(device)
            ce = F.cross_entropy(logits, lbls, weight=class_weights, reduction='none')
            pt = torch.exp(-ce)
            losses = (1 - pt) ** criterion.gamma * ce
            test_loss += losses.mean().item() * imgs.size(0)
            preds = logits.argmax(dim=1)
            test_acc += (preds == lbls).sum().item()
            batch_idx += imgs.size(0)

    test_loss /= len(test_loader.dataset)
    test_acc /= len(test_loader.dataset)
    print(f"Snapshot Ensemble Test Loss: {test_loss:.4f}  Test Acc: {test_acc:.4f}")

        # Per-slide aggregation
    slide_results = defaultdict(lambda: {'probs': [], 'preds': []})
    for k, prob, pred in zip(test_keys, probs_class1, preds_class1):
        sid = k.rsplit('_', 1)[0]
        slide_results[sid]['probs'].append(float(prob))
        slide_results[sid]['preds'].append(int(pred))

    with open('slide_index_test.pkl', 'wb') as f:
        pickle.dump({sid: {**slide_index[sid], **res} for sid, res in slide_results.items()}, f)
    print(f"Saved per-slide results for {len(slide_results)} slides.")



if __name__ == '__main__':
    main()
