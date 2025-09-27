import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import timm
import numpy as np
import subprocess
import argparse 
from torch.utils.data import Dataset


def get_pretrained_Moco_vision_encoder():
    current_storage = os.environ.get('PFSDIR', None)
    url = "https://dl.fbaipublicfiles.com/moco-v3/vit-b-300ep/vit-b-300ep.pth.tar"

    # wget https://dl.fbaipublicfiles.com/moco-v3/vit-b-300ep/vit-b-300ep.pth.tar # download checkpoint to current directory
    if current_storage is not None:
        os.makedirs(current_storage, exist_ok=True)
        out_path = os.path.join(current_storage, os.path.basename(url))
        print(f"Downloading model to {out_path} ...")
        subprocess.run(["wget", "-O", out_path, url], check=True)
    else:
        print("PFSDIR not set, downloading to current directory ...")
        subprocess.run(["wget", url], check=True)
    checkpoint_path = out_path if current_storage is not None else os.path.basename(url)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    vit_model = timm.create_model('vit_base_patch16_224', pretrained=False)

    state_dict = checkpoint['state_dict']
    base_encoder_state_dict = {
        k.replace('module.base_encoder.', ''): v
        for k, v in state_dict.items()
        if 'module.base_encoder' in k
    }

    vit_model.load_state_dict(base_encoder_state_dict, strict=False)
    vit_model.head = torch.nn.Identity()
    vit_model.eval()
    vit_model.to("cuda" if torch.cuda.is_available() else "cpu")
    # remove model pth
    os.remove(checkpoint_path)
    return vit_model

class syn_cifar10_dataset(Dataset):
  def __init__(self, syn_path, transform=None):
    if syn_path.endswith(".npz"):
      self.data = np.load(syn_path)
      self.images = self.data["image"]
      self.labels = self.data["label"]
    else:
      raise ValueError("Invalid synthetic dataset path")
    self.transform = transform
    self.classes = np.unique(self.labels).tolist()
    
    self.class_to_idx = {}
    for current_class in self.classes:
      indices = []
      for index, current_label in enumerate(self.labels):
        if current_label == current_class:
            indices.append(index)
      self.class_to_idx[current_class] = indices

  def __getitem__(self, index):
      img = self.images[index]
      target = self.labels[index]
      if self.transform:
          img = self.transform(img)
      return index, img, target

  def __len__(self):
      return len(self.images)

def main(
  IMAGE_ROOT = None,
  OUTPUT_ROOT = None,
  BATCH_SIZE = 1024,
  NUM_WORKERS = 10,
  dataset = None
  ):
  os.makedirs(OUTPUT_ROOT, exist_ok=True)
  DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  model = get_pretrained_Moco_vision_encoder()
  if torch.cuda.device_count() > 1:
      print(f"🚀 Using {torch.cuda.device_count()} GPUs!")
      model = torch.nn.DataParallel(model)
  model = model.to(DEVICE)

  syn_name = os.path.basename(IMAGE_ROOT)
  syn_name = syn_name.split(".")[0]

  # 🚩 2️⃣ ImageNet 预处理

  if dataset == "CIFAR10":
    cifar10_trans = transforms.Compose([
      transforms.ToPILImage(),
      transforms.Resize((224, 224)),  # ResNet50 expects 224x224 input
      transforms.ToTensor(),
      transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                          std=(0.2470, 0.2435, 0.2616)),
      ])
    transform = cifar10_trans
  else:
    raise ImportError(f"Unknown dataset, {dataset}")

  dataset = syn_cifar10_dataset(IMAGE_ROOT, transform=transform)
  print(f"✅ Found {len(dataset.classes)} classes in {IMAGE_ROOT}, with length {len(dataset)}")

  all_classes = dataset.classes


  dataloader = DataLoader(
      dataset,
      batch_size=BATCH_SIZE,
      shuffle=False,
      num_workers=NUM_WORKERS,
      pin_memory=True,
      drop_last=False
  )

  
  temp_indices = []
  temp_features = []
  temp_labels = []
  with torch.no_grad():
      for indices, images, labels in tqdm(dataloader, desc=f"Extracting {syn_name}"):
          images = images.to(DEVICE, non_blocking=True)
          with torch.cuda.amp.autocast():  # AMP for speed on A100/H100
              batch_features = model(images).cpu().numpy()
          temp_indices.extend(indices)
          temp_features.extend(batch_features)
          temp_labels.extend(labels)

  for each_class in all_classes:
      features_dict = {}
      class_indices = [i for i, label in zip(temp_indices, temp_labels) if label == each_class]
      class_features = [temp_features[i] for i in class_indices]
      for i, feature in zip(class_indices, class_features):
          features_dict[str(int(i))] = feature

      output_npz = os.path.join(OUTPUT_ROOT, f"{syn_name}_{each_class}_features.npz")
      os.makedirs(OUTPUT_ROOT, exist_ok=True)
      np.savez(output_npz, **features_dict)
      print(f"✅ Saved {len(features_dict)} features to {output_npz}")

  print("\n🎉 All classes finished!")

if __name__ == "__main__":
  args = argparse.ArgumentParser()
  args.add_argument("--data_path", 
                    default="./AAAI/ablation/generative_model_VS_alpha/synthetic_data_from_diff_models/EDM_synthetic/DDO_EDM_1M.npz")
  args.add_argument("--output_dir", 
                    default="./AAAI/ablation/generative_model_VS_alpha/synthetic_data_from_diff_models/synthetic_features")
  args.add_argument("--batch_size", default=1024)
  args.add_argument("--num_workers", default=10)
  args.add_argument("--dataset", default="CIFAR10")
  args = args.parse_args()
  image_roots = [
    "./AAAI/ablation/generative_model_VS_alpha/synthetic_data_from_diff_models/EDM_synthetic/DDO_EDM_1M.npz",
    # "./AAAI/ablation/generative_model_VS_alpha/synthetic_data_from_diff_models/EDM_synthetic/EDM_1m.npz",
    #              "./AAAI/ablation/generative_model_VS_alpha/synthetic_data_from_diff_models/EDM_synthetic/stylegan2_ada.npz"
    ]
  for image_root in image_roots:
      main(
          IMAGE_ROOT=image_root,
          OUTPUT_ROOT=args.output_dir,
          BATCH_SIZE=args.batch_size,
          NUM_WORKERS=args.num_workers,
          dataset=args.dataset
          )