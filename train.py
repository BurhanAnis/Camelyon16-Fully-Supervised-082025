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
    """
    Reads patches (PNG bytes) + labels from an LMDB store.
    keys:     list of string keys (e.g. "slide123_000042")
    labels:   parallel list of 0/1 ints
    flags:    parallel list of bool: whether to apply aug_tf (True) or base_tf (False)
    """
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
    """
    Focal Loss, as in https://arxiv.org/abs/1708.02002
      - weight: 1D tensor of shape [num_classes], class-weights
      - gamma: focusing parameter (>=0), higher => focus on hard examples
    """
    def __init__(self, weight=None, gamma: float = 2.0, reduction='mean'):
        super().__init__()
        self.weight    = weight
        self.gamma     = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        # standard CE per-sample
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction='none')
        # pt = probability of true class
        pt = torch.exp(-ce)
        # focal term
        loss = (1 - pt) ** self.gamma * ce
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

    



def main():
    parser = argparse.ArgumentParser(
        description='Train DL model on WSI patches with train/val/test splits'
    )
    parser.add_argument('--index_path',      type=str,   required=True)
    parser.add_argument('--lmdb_path',       type=str,   required=True)
    parser.add_argument('--batch_size',      type=int,   default=64)
    parser.add_argument('--num_workers',     type=int,   default=4)
    parser.add_argument('--max_epochs',      type=int,   default=50)
    parser.add_argument('--patience',        type=int, default=2)
    parser.add_argument('--lr',              type=float, default=1e-3)
    parser.add_argument('--loss_csv',        type=str,   default='epoch_losses.csv')
    parser.add_argument('--checkpoint_dir',  type=str,   default='checkpoints')
    parser.add_argument('--save_interval',   type=int,   default=5)
    parser.add_argument('--aug_factor',      type=int,   default=5)
    parser.add_argument('--val_size',        type=float, default=0.1,
                        help='Fraction of training set used for validation')
    parser.add_argument('--seed',            type=int,   default=42,
                        help='Random seed')
    args = parser.parse_args()


    set_seed(args.seed)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # load index
    with open(args.index_path, 'rb') as f:
        slide_index = pickle.load(f)

    # build key list and labels in same order as LMDB
    keys, labels = [], []
    count = 0
    for slide_id, info in slide_index.items():
        for (_, _, is_tumor) in info['patches']:
            keys.append(f"{slide_id}_{count:06d}")
            labels.append(1 if is_tumor else 0)
            count += 1
    labels = np.array(labels, dtype=np.int64)

    # split by slide name: any slide_id containing "test" -> test set
    all_idx = np.arange(len(keys))
    test_idx = [
        i for i, key in enumerate(keys)
        if "test" in key.rsplit("_", 1)[0]
    ]
    trainval_idx = np.setdiff1d(all_idx, test_idx)
    # split trainval into train and val
    train_idx, val_idx = train_test_split(
        trainval_idx, test_size=args.val_size,
        stratify=labels[trainval_idx], random_state=42
    )

    # helper to expand positive samples
    def make_entries(indices, augment=False):
        out_keys, out_labels, out_flags = [], [], []
        for i in indices:
            out_keys.append(keys[i])
            out_labels.append(int(labels[i]))
            out_flags.append(False)
            if augment and labels[i] == 1:
                for _ in range(args.aug_factor):
                    out_keys.append(keys[i])
                    out_labels.append(1)
                    out_flags.append(True)
        return out_keys, out_labels, out_flags

    train_keys, train_labels, train_flags = make_entries(train_idx, augment=True)
    val_keys,   val_labels,   val_flags   = make_entries(val_idx,   augment=False)
    test_keys,  test_labels,  test_flags  = make_entries(test_idx,  augment=False)


    # transforms
    norm = transforms.Normalize([0.485,0.456,0.406],
                                [0.229,0.224,0.225])
    base_tf = transforms.Compose([transforms.ToTensor(), norm])
    aug_tf  = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(90),
        transforms.ToTensor(),
        norm
    ])

    # device
    device = (
    torch.device('mps') if torch.backends.mps.is_available()
    else torch.device('cuda') if torch.cuda.is_available()
    else torch.device('cpu')
    )

    pin = True if device.type == 'cuda' else False
    # datasets and loaders
    train_ds = LMDBAugDataset(args.lmdb_path, train_keys, train_labels, train_flags,
                               base_tf, aug_tf)
    val_ds   = LMDBAugDataset(args.lmdb_path, val_keys,   val_labels,   val_flags,
                               base_tf, base_tf)
    test_ds  = LMDBAugDataset(args.lmdb_path, test_keys,  test_labels,  test_flags,
                               base_tf, base_tf)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers,
                              pin_memory=pin)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers,
                              pin_memory=pin)

    train_labels_arr = np.array(train_labels)
    val_labels_arr   = np.array(val_labels)
    test_labels_arr  = np.array(test_labels)

    train_norm = int((train_labels_arr == 0).sum())
    train_tum  = int((train_labels_arr == 1).sum())
    val_norm   = int((val_labels_arr   == 0).sum())
    val_tum    = int((val_labels_arr   == 1).sum())
    test_norm  = int((test_labels_arr  == 0).sum())
    test_tum   = int((test_labels_arr  == 1).sum())

    print(f"Train: {len(train_ds)} patches "
        f"(normal={train_norm}, tumor={train_tum})")
    print(f"Val:   {len(val_ds)} patches "
        f"(normal={val_norm}, tumor={val_tum})")
    print(f"Test:  {len(test_ds)} patches "
        f"(normal={test_norm}, tumor={test_tum})")


    # model
    model = models.resnet50(weights=False)
    model.fc = nn.Linear(model.fc.in_features, 2)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)

    # compute inverse-frequency class weights
    class_counts  = torch.tensor([train_norm, train_tum], dtype=torch.float)
    class_weights = class_counts.sum() / class_counts
    class_weights = class_weights.to(device)

    # use focal loss 
    criterion = FocalLoss(weight=class_weights, gamma=2.0)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    # training
    metrics = []
    best_val_loss = float('inf')
    no_improve = 0

    for epoch in range(1, args.max_epochs + 1):
        # train epoch
        model.train()
        train_loss, train_correct = 0.0, 0
        for imgs, lbls in tqdm(train_loader, desc=f"Epoch {epoch} [Train]"):
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, lbls)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
            train_correct += (outputs.argmax(1) == lbls).sum().item()
        train_loss /= len(train_loader.dataset)
        train_acc = train_correct / len(train_loader.dataset)

        # validate epoch
        model.eval()
        val_loss, val_correct = 0.0, 0
        for imgs, lbls in tqdm(val_loader, desc=f"Epoch {epoch} [Val]"):
            imgs, lbls = imgs.to(device), lbls.to(device)
            with torch.no_grad():
                outputs = model(imgs)
                loss = criterion(outputs, lbls)
            val_loss += loss.item() * imgs.size(0)
            val_correct += (outputs.argmax(1) == lbls).sum().item()
        val_loss /= len(val_loader.dataset)
        val_acc = val_correct / len(val_loader.dataset)

        print(f"Epoch {epoch:02d}  "
              f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.4f}  "
              f"Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.4f}")

        # record metrics
        metrics.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc
        })
        pd.DataFrame(metrics).to_csv(args.loss_csv, index=False)

        # scheduler step
        scheduler.step(val_loss)

        # periodic checkpoint
        if epoch % args.save_interval == 0:
            ckpt_path = os.path.join(args.checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
            state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
            torch.save({
                'epoch': epoch,
                'model_state': state,
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'best_val_loss': best_val_loss
            }, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

        # best-model checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            best_path = os.path.join(args.checkpoint_dir, 'best_model.pt')
            state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
            torch.save({
                'epoch': epoch,
                'model_state': state,
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'best_val_loss': best_val_loss
            }, best_path)
            print(f"New best model (val loss {val_loss:.4f}) saved to {best_path}")
        else:
            no_improve += 1

        # early stopping
        if no_improve >= args.patience:
            print(f"No val-loss improvement for {args.patience} epochs; stopping early.")
            break

    # final test evaluation
    model.eval()
    test_loss, test_correct = 0.0, 0
    for imgs, lbls in tqdm(test_loader, desc="Final Evaluation [Test]"):
        imgs, lbls = imgs.to(device), lbls.to(device)
        with torch.no_grad():
            outputs = model(imgs)
            loss = criterion(outputs, lbls)
        test_loss += loss.item() * imgs.size(0)
        test_correct += (outputs.argmax(1) == lbls).sum().item()
    test_loss /= len(test_loader.dataset)
    test_acc = test_correct / len(test_loader.dataset)

    print(f"Test Loss: {test_loss:.4f}  Test Acc: {test_acc:.4f}")
    print(f"Training complete. Metrics in {args.loss_csv}, checkpoints in {args.checkpoint_dir}")

    model.eval()
    all_probs, all_preds = [], []
    with torch.no_grad():
        for imgs, _ in tqdm(test_loader, desc="Collecting patch‐level preds"):
            imgs = imgs.to(device)
            logits = model(imgs)
            probs  = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds  = (probs > 0.5).astype(int)
            all_probs.extend(probs.tolist())
            all_preds.extend(preds.tolist())

    # `test_keys` is in loader‐order because shuffle=False
    assert len(all_preds) == len(test_keys)

    # group by slide_id
    slide_probs, slide_preds = defaultdict(list), defaultdict(list)

    for key, prob, pred in zip(test_keys, all_probs, all_preds):
        slide_id = key.rsplit("_", 1)[0]
        slide_probs[slide_id].append(prob)
        slide_preds[slide_id].append(pred)

    # rebuild a test‐only slide_index
    slide_index_test = {}
    for slide_id, info in slide_index.items():
        if slide_id in slide_preds:
            slide_index_test[slide_id] = {
                **info,                          # original 'patches', etc.
                "preds": slide_preds[slide_id], # list[int], same order as info['patches']
                "probs": slide_probs[slide_id], # list[float]
            }

    # optional: save to disk for downstream use
    with open("slide_index_test.pkl", "wb") as f:
        pickle.dump(slide_index_test, f)

    print(f"Built slide_index_test for {len(slide_index_test)} slides,"
          f" with per‐patch preds & probs.")

if __name__ == '__main__':
    main()
