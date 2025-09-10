import os
import io
import random
import pickle
import argparse
from tqdm import tqdm
from collections import defaultdict

import numpy as np
from PIL import Image

import torch
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


class LMDBTestDataset(Dataset):
    def __init__(self, lmdb_path, keys, transform=None):
        import lmdb  # locally scoped so script doesn't crash if missing early
        self.keys = [k.encode('ascii') for k in keys]
        self.env = lmdb.open(lmdb_path, readonly=True, lock=False)
        self.txn = self.env.begin()
        self.transform = transform or transforms.ToTensor()

    def __getitem__(self, idx):
        data = self.txn.get(self.keys[idx])
        img = Image.open(io.BytesIO(data)).convert('RGB')
        return self.transform(img), self.keys[idx].decode('ascii')

    def __len__(self):
        return len(self.keys)


def load_model(checkpoint_path, device):
    model = models.resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)

    # Load checkpoint and strip "module." prefix if needed
    state_dict = torch.load(checkpoint_path, map_location=device)
    if any(k.startswith("module.") for k in state_dict.keys()):
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_key = k.replace("module.", "")
            new_state_dict[new_key] = v
        state_dict = new_state_dict

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def categorize_keys(slide_index, threshold):
    false_pos, false_neg, true_pos, true_neg = [], [], [], []
    for sid, info in slide_index.items():
        if 'probs' not in info or 'preds' not in info:
            continue
        for i, (patch_info, prob, pred) in enumerate(zip(info['patches'], info['probs'], info['preds'])):
            patch_name = f"{sid}_{i:06d}"
            true_label = 1 if patch_info[2] else 0

            if pred == 1 and true_label == 0 and prob > threshold:
                false_pos.append((patch_name, true_label))
            elif pred == 0 and true_label == 1 and prob < 1 - threshold:
                false_neg.append((patch_name, true_label))
            elif pred == true_label and true_label == 1 and prob > threshold:
                true_pos.append((patch_name, true_label))
            elif pred == true_label and true_label == 0 and prob < 1 - threshold:
                true_neg.append((patch_name, true_label))

    return false_pos, false_neg, true_pos, true_neg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lmdb_path', type=str, required=True)
    parser.add_argument('--index_pkl', type=str, default='slide_index_test.pkl')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best.pt')
    parser.add_argument('--threshold', type=float, default=0.9)
    parser.add_argument('--out_dir', type=str, default='gradcam_output')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load index
    with open(args.index_pkl, 'rb') as f:
        slide_index = pickle.load(f)

    # Categorize patch keys
    fp, fn, tp, tn = categorize_keys(slide_index, args.threshold)
    print(f"False Positives: {len(fp)} | False Negatives: {len(fn)}")
    print(f"True Positives:  {len(tp)} | True Negatives:  {len(tn)}")

    # Subsample correctly predicted for GradCAM
    tp = random.sample(tp, min(100, len(tp)))
    tn = random.sample(tn, min(100, len(tn)))

    all_samples = fp + fn + tp + tn
    categories = (
        ['false_pos'] * len(fp) +
        ['false_neg'] * len(fn) +
        ['true_pos'] * len(tp) +
        ['true_neg'] * len(tn)
    )
    cat_map = dict(zip([k for k, _ in all_samples], categories))
    label_map = dict(all_samples)

    # Image transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # Load model
    device = torch.device(args.device)
    model = load_model(args.checkpoint, device)
    target_layer = model.layer4 if not isinstance(model, torch.nn.DataParallel) else model.module.layer4
    cam = GradCAM(model=model, target_layers=[target_layer], use_cuda=(device.type == 'cuda'))

    # Dataset
    dataset = LMDBTestDataset(args.lmdb_path, [k for k, _ in all_samples], transform)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Grad-CAM generation
    for img_tensor, key in tqdm(loader, desc="Generating Grad-CAMs"):
        img_tensor = img_tensor[0].to(device)
        key_str = key[0]
        label = label_map[key_str]
        category = cat_map[key_str]

        # Run Grad-CAM
        grayscale_cam = cam(input_tensor=img_tensor.unsqueeze(0),
                            targets=[ClassifierOutputTarget(label)])[0]

        # Prepare image for overlay
        rgb_img = img_tensor.permute(1, 2, 0).detach().cpu().numpy()
        rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min() + 1e-8)
        cam_img = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        # Save image
        out_path = os.path.join(args.out_dir, category)
        os.makedirs(out_path, exist_ok=True)
        out_file = os.path.join(out_path, f"{key_str}.png")
        Image.fromarray(cam_img).save(out_file)

    print("✅ Grad-CAM visualizations saved.")


if __name__ == '__main__':
    main()
