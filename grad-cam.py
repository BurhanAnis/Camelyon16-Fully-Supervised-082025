import argparse
import io
import pickle
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
import matplotlib.pyplot as plt
from tqdm import tqdm

import lmdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


# ============================= Dataset ============================= #

class LMDBTestDataset(Dataset):
    """
    LMDB dataset that pre-validates keys and skips missing/corrupt entries.
    Expects keys as bytes and labels as np.int64.
    """
    def __init__(self, lmdb_path, keys_bytes, labels, transform=None, validate=True):
        self.lmdb_path = lmdb_path
        self.transform = transform or transforms.ToTensor()
        self.env = None
        self.txn = None

        self._init_env()

        valid_keys = []
        valid_labels = []

        missing = 0
        corrupt = 0

        if validate:
            for kb, lbl in tqdm(zip(keys_bytes, labels), total=len(keys_bytes),
                                desc="Validating LMDB keys", leave=False):
                data = self.txn.get(kb)
                if data is None:
                    missing += 1
                    continue
                if not self._is_image_readable(data):
                    corrupt += 1
                    continue
                valid_keys.append(kb)
                valid_labels.append(int(lbl))
        else:
            valid_keys = list(keys_bytes)
            valid_labels = [int(x) for x in labels]

        if missing > 0 or corrupt > 0:
            warnings.warn(
                f"[LMDBTestDataset] Skipping {missing} missing and {corrupt} corrupt patches "
                f"(kept {len(valid_keys)}/{len(keys_bytes)})."
            )

        self.keys = valid_keys
        self.labels = np.array(valid_labels, dtype=np.int64)

    def _init_env(self):
        if self.env is None:
            self.env = lmdb.open(
                self.lmdb_path, readonly=True, lock=False,
                readahead=False, max_readers=32
            )
            self.txn = self.env.begin(buffers=False)

    @staticmethod
    def _is_image_readable(data_bytes):
        try:
            with Image.open(io.BytesIO(data_bytes)) as im:
                im.verify()
            return True
        except (UnidentifiedImageError, OSError):
            return False

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        if self.env is None:
            self._init_env()

        kb = self.keys[idx]
        data = self.txn.get(kb)
        if data is None:
            raise IndexError(f"Key missing at runtime: {kb!r}")

        # Original (numpy)
        try:
            with Image.open(io.BytesIO(data)) as img:
                img = img.convert('RGB')
                img_original = np.array(img)
        except (UnidentifiedImageError, OSError) as e:
            raise IndexError(f"Corrupt image for key {kb!r}: {e}")

        # Transformed tensor
        with Image.open(io.BytesIO(data)) as img2:
            img2 = img2.convert('RGB')
            img_tensor = self.transform(img2) if self.transform else transforms.ToTensor()(img2)

        key_str = kb.decode('ascii')
        label = int(self.labels[idx])
        return img_tensor, label, img_original, key_str

    def __del__(self):
        try:
            if self.env is not None:
                self.env.close()
        except Exception:
            pass


# ============================= Utils ============================= #

def collate_single(batch):
    """Keep batch_size=1 samples as a flat tuple instead of stacked structures."""
    assert len(batch) == 1
    return batch[0]


def build_test_keys_from_slide_index(slide_index, test_marker='test'):
    """Rebuild LMDB keys with the same global counter mechanics used for writing."""
    keys_str = []
    labels = []
    global_count = 0

    for slide_id, info in slide_index.items():
        patches = info.get('patches', [])
        for (y, x, is_tumor) in patches:
            lmdb_key_str = f"{slide_id}_{global_count:06d}"
            if test_marker in slide_id:
                keys_str.append(lmdb_key_str)
                labels.append(1 if is_tumor else 0)
            global_count += 1
    return keys_str, np.array(labels, dtype=np.int64)


# ============================= Main ============================= #

def main():
    parser = argparse.ArgumentParser(
        description='Grad-CAM (pytorch-grad-cam) for WSI patches.'
    )
    parser.add_argument('--index_path', type=str, required=True)
    parser.add_argument('--lmdb_path', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='gradcam_outputs')
    parser.add_argument('--prob_threshold', type=float, default=0.9)
    parser.add_argument('--num_correct_samples', type=int, default=100)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no_validate', action='store_true')
    parser.add_argument('--test_marker', type=str, default='test')
    parser.add_argument('--use_cached_preds', type=str, default='')
    parser.add_argument('--max_per_category', type=int, default=0)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Output dirs
    output_path = Path(args.output_dir)
    for sub in ['FP', 'FN', 'TP', 'TN']:
        (output_path / sub).mkdir(parents=True, exist_ok=True)

    # Load slide index
    with open(args.index_path, 'rb') as f:
        slide_index = pickle.load(f)

    # Rebuild keys
    keys_str, labels = build_test_keys_from_slide_index(slide_index, test_marker=args.test_marker)
    keys_bytes = [k.encode('ascii') for k in keys_str]

    # Device
    device = (
        torch.device('cuda') if torch.cuda.is_available()
        else torch.device('mps') if torch.backends.mps.is_available()
        else torch.device('cpu')
    )

    # Model
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)

    ckpt = torch.load(args.model_path, map_location=device)
    if isinstance(ckpt, dict) and 'state_dict' in ckpt and isinstance(ckpt['state_dict'], dict):
        state_dict = ckpt['state_dict']
    else:
        state_dict = ckpt
    if len(state_dict) > 0 and next(iter(state_dict.keys())).startswith('module.'):
        state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    # Transforms
    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
    transform = transforms.Compose([transforms.ToTensor(), normalize])

    # Dataset
    test_dataset = LMDBTestDataset(
        args.lmdb_path, keys_bytes, labels, transform,
        validate=(not args.no_validate)
    )
    if len(test_dataset) == 0:
        print("No valid test patches found. Exiting.")
        return

    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=0,
        pin_memory=torch.cuda.is_available(), collate_fn=collate_single
    )

    # Grad-CAM setup
    target_layers = [model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)

    # Cached preds
    cached_lookup = {}
    if args.use_cached_preds:
        try:
            with open(args.use_cached_preds, 'rb') as f:
                cached = pickle.load(f)
            cached_lookup = cached
            print(f"Loaded cached predictions for {len(cached_lookup)} slides.")
        except Exception as e:
            print(f"Warning: could not load cached preds: {e}")

    def slide_from_key(k: str):
        return k.rsplit('_', 1)[0]

    # Classify patches
    classifications = {'FP': [], 'FN': [], 'TP': [], 'TN': []}

    with torch.no_grad():
        for img_tensor, label, img_original, key in tqdm(test_loader, desc="Classifying"):
            slide_id = slide_from_key(key)
            prob1, pred = None, None

            if cached_lookup and slide_id in cached_lookup:
                try:
                    patch_idx = int(key.split('_')[-1])
                    if patch_idx < len(cached_lookup[slide_id].get('probs', [])):
                        prob1 = float(cached_lookup[slide_id]['probs'][patch_idx])
                        pred = int(cached_lookup[slide_id]['preds'][patch_idx])
                except Exception:
                    pass

            if prob1 is None:
                logits = model(img_tensor.unsqueeze(0).to(device))
                prob1 = F.softmax(logits, dim=1)[0, 1].item()
                pred = 1 if prob1 > args.prob_threshold else 0

            true_label = int(label)
            if true_label == 1 and pred == 1:
                cat = 'TP'
            elif true_label == 0 and pred == 0:
                cat = 'TN'
            elif true_label == 0 and pred == 1:
                cat = 'FP'
            else:
                cat = 'FN'

            classifications[cat].append({
                'key': key,
                'img_tensor': img_tensor.detach().cpu(),
                'img_original': img_original,
                'prob': prob1,
                'true_label': true_label,
                'pred': pred
            })

    # Downsample TP/TN
    rng = np.random.default_rng(args.seed)
    for cat in ['TP', 'TN']:
        samples = classifications[cat]
        if len(samples) > args.num_correct_samples:
            idxs = rng.choice(len(samples), args.num_correct_samples, replace=False)
            classifications[cat] = [samples[i] for i in idxs]

    # Global cap
    if args.max_per_category > 0:
        for cat in ['FP', 'FN', 'TP', 'TN']:
            samples = classifications[cat]
            if len(samples) > args.max_per_category:
                idxs = rng.choice(len(samples), args.max_per_category, replace=False)
                classifications[cat] = [samples[i] for i in idxs]

    # Visualization
    manifest_rows = []
    for category in ['FP', 'FN', 'TP', 'TN']:
        samples = classifications[category]
        print(f"Processing {len(samples)} {category} samples...")
        for sample in tqdm(samples, desc=f"Grad-CAM {category}", leave=False):
            key = sample['key']
            img_tensor = sample['img_tensor']
            img_original = sample['img_original']
            prob = sample['prob']
            true_label = sample['true_label']
            pred = sample['pred']

            input_tensor = img_tensor.unsqueeze(0).to(device)
            targets = [ClassifierOutputTarget(pred)]
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

            overlay = show_cam_on_image(
                img_original.astype(np.float32) / 255.0,
                grayscale_cam,
                use_rgb=True,
                image_weight=1 - args.alpha
            )

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(img_original)
            axes[0].set_title('Original')
            axes[0].axis('off')

            axes[1].imshow(grayscale_cam, cmap='jet')
            axes[1].set_title('Grad-CAM Heatmap')
            axes[1].axis('off')

            axes[2].imshow(overlay)
            axes[2].set_title(f"Overlay\nTrue: {true_label}, Pred: {pred}\nProb: {prob:.3f}")
            axes[2].axis('off')

            out_file = output_path / category / f"{key}_gradcam.png"
            plt.savefig(out_file, dpi=100, bbox_inches='tight')
            plt.close(fig)

            manifest_rows.append({
                'key': key,
                'category': category,
                'true_label': true_label,
                'pred': pred,
                'prob': prob,
                'path': str(out_file)
            })

    # Save outputs
    summary = {
        'threshold': args.prob_threshold,
        'FP_count': len(classifications['FP']),
        'FN_count': len(classifications['FN']),
        'TP_count': len(classifications['TP']),
        'TN_count': len(classifications['TN']),
    }
    pd.DataFrame([summary]).to_csv(output_path / 'summary.csv', index=False)
    pd.DataFrame(manifest_rows).to_csv(output_path / 'manifest.csv', index=False)

    print(f"\nGrad-CAM visualizations saved to {output_path}/")
    print(f"Summary: {output_path / 'summary.csv'} | Manifest: {output_path / 'manifest.csv'}")


if __name__ == '__main__':
    main()




