import os
import glob
import numpy as np
from typing import List, Tuple, Optional
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import AutoImageProcessor
from transformers.models.dinov2 import Dinov2Backbone
from transformers import AutoImageProcessor, AutoModel
import argparse

# -----------------------------
# Dataset returning tensor + basename
# -----------------------------
class ImageClassDataset(Dataset):
    def __init__(self, img_paths: List[str], processor: AutoImageProcessor):
        self.paths = img_paths
        self.proc = processor

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        with Image.open(p) as im:
            img = im.convert("RGB")
        batch = self.proc(images=img, return_tensors="pt")
        x = batch["pixel_values"].squeeze(0)  # [3,H,W]
        name = os.path.basename(p)            # only filename
        return x, name

def collate_with_names(batch):
    xs, names = zip(*batch)
    xs = torch.stack(xs, dim=0)  # [B,3,H,W]
    return xs, list(names)

# -----------------------------
# Load DINOv2-Base
# -----------------------------
def load_dinov2(device: Optional[str] = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained('facebook/dinov2-base').to(device)
    model.eval()
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    return model, processor, device

# -----------------------------
# Extract features for ONE class dir and save npz
# -----------------------------
@torch.no_grad()
def extract_one_class_dir(
    class_dir: str,
    out_dir: str,
    model: Dinov2Backbone,
    processor: AutoImageProcessor,
    device: str,
    batch_size: int = 256,
    num_workers: int = 8,
    amp: bool = True,
):
    # collect images for this class
    image_names = os.listdir(class_dir)
    paths = [os.path.join(class_dir, n) for n in image_names if n.endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp", '.JPEG'))]

    paths = sorted(paths)
    if not paths:
        print(f"[WARN] No images in {class_dir}, skip.")
        return

    ds = ImageClassDataset(paths, processor)
    dl = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device=="cuda"),
        persistent_workers=(num_workers>0), drop_last=False,
        collate_fn=collate_with_names, 
        # multiprocessing_context="spawn" # more robust on HPC
    )
    feats_list = []
    names = []
    image_feature_maps = {}
    ctx = (torch.autocast(device_type="cuda", dtype=torch.float16)
           if (amp and device=="cuda") else torch.cuda.amp.autocast(enabled=False))
    with ctx:
        for xb, nb in dl:
            xb = xb.to(device, non_blocking=True)          # [B,3,H,W]
            out = model(pixel_values=xb)
            last_hidden_states = out.last_hidden_state
            representation = last_hidden_states[:, 0, :]  #CLS token [B, 768]          

            feats_list.append(representation.cpu())
            names.extend(nb)
    # print("the shape of features:", feats_list[0].shape)
    # print("the length of names:", len(feats_list))
    feats = torch.cat(feats_list, 0).numpy()               # (N,768)
    # print("the shape of feats:", feats.shape)
    os.makedirs(out_dir, exist_ok=True)
    class_name = os.path.basename(os.path.normpath(class_dir))
    out_path = os.path.join(out_dir, f"{class_name}.npz")
    for feat, name in zip(feats, names):
        image_feature_maps[name] = feat

    np.savez_compressed(out_path, **image_feature_maps)

def extract_all_classes(
    root_dir: str,
    out_dir: str,
    batch_size: int = 256,
    num_workers: int = 8,
    amp: bool = True,
    split_by_num: Optional[str] = None,  # e.g. "1/10" means process 1/10 of classes
):
    # class dirs = immediate subfolders of root
    class_dirs = [os.path.join(root_dir, d) for d in sorted(os.listdir(root_dir))
                  if os.path.isdir(os.path.join(root_dir, d))]
    if not class_dirs:
        raise FileNotFoundError(f"No class subdirectories under {root_dir}")
    if split_by_num is not None:
        part, total = map(int, split_by_num.split("/"))
        assert 1 <= part <= total, f"Invalid split_by_num {split_by_num}"
        num_classes = len(class_dirs)
        classes_per_part = (num_classes + total - 1) // total
        begin = (part - 1) * classes_per_part
        end = min(part * classes_per_part, num_classes)
        class_dirs = class_dirs[begin:end]
        print(f"✅ Split classes by {split_by_num}, processing {len(class_dirs)}/{num_classes} classes from index {begin} to {end-1}.")
    model, processor, device = load_dinov2()
    for cdir in tqdm(class_dirs):

        class_name = os.path.basename(os.path.normpath(cdir))
        if os.path.exists(os.path.join(out_dir, f"{class_name}.npz")):
            print(f"[SKIP] {class_name} already done.")
            continue
        extract_one_class_dir(
            cdir, out_dir, model, processor, device,
            batch_size=batch_size, num_workers=num_workers, amp=amp
        )
if __name__ == "__main__":
  
  args = argparse.ArgumentParser()
  args.add_argument("--split", type=str, default=None, help="e.g. '1/10' to process 1/10 of classes")
  args.add_argument("--root", type=str, default="",
                    help="Root directory containing class subfolders")
  args.add_argument("--out", type=str, default="",
                    help="Output directory to save npz files")
  args = args.parse_args()

  ROOT = args.root
  OUT  = args.out
  extract_all_classes(ROOT, OUT, batch_size=512, num_workers=8, amp=True, split_by_num = args.split)