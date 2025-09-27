import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import timm
import numpy as np
import subprocess
from tqdm import tqdm

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
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

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

def main(
  IMAGENET_ROOT = "./imagenet1K_dataset/Data/CLS-LOC/train",
  OUTPUT_ROOT = "./imagenet1K_dataset/MocoV3_Real_feature_map_debug/npz_per_class",
  BATCH_SIZE = 1024,
  NUM_WORKERS = 10,
  begin = None,
  end = None
  ):
  IMAGENET_ROOT = IMAGENET_ROOT
  OUTPUT_ROOT = OUTPUT_ROOT
  os.makedirs(OUTPUT_ROOT, exist_ok=True)
  BATCH_SIZE = BATCH_SIZE
  NUM_WORKERS = NUM_WORKERS
  DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  model = get_pretrained_Moco_vision_encoder()
  if torch.cuda.device_count() > 1:
      print(f"🚀 Using {torch.cuda.device_count()} GPUs!")
      model = torch.nn.DataParallel(model)
  model = model.to(DEVICE)
  
  transform = transforms.Compose([
      transforms.Resize(256),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      transforms.Normalize(
          mean=(0.485, 0.456, 0.406),
          std=(0.229, 0.224, 0.225)
      ),
  ])

  dataset = datasets.ImageFolder(IMAGENET_ROOT, transform=transform)
  print(f"✅ Found {len(dataset.classes)} classes in {IMAGENET_ROOT}")


  if begin is not None and end is not None:
      subset_indices = [i for i, (_, label) in enumerate(dataset.samples) if begin <= label < end]
      dataset = Subset(dataset, subset_indices)
      print(f"✅ Selected {len(dataset)} samples in range {begin, end}.")

  all_classes = dataset.dataset.classes if isinstance(dataset, Subset) else dataset.classes
  class_to_idx = dataset.dataset.class_to_idx if isinstance(dataset, Subset) else dataset.class_to_idx
  all_samples = dataset.dataset.samples if isinstance(dataset, Subset) else dataset.samples
  subset_indices = dataset.indices if isinstance(dataset, Subset) else list(range(len(dataset)))

  for class_name in all_classes:
      output_npz = os.path.join(OUTPUT_ROOT, f"{class_name}.npz")
      if os.path.exists(output_npz):
          print(f"✅ {class_name} already done, skip.")
          continue

      class_idx = class_to_idx[class_name]
      class_sample_indices = [i for i in subset_indices if all_samples[i][1] == class_idx]

      if not class_sample_indices:
          print(f"⚠️ No images found for {class_name}")
          continue

      print(f"\n🚩 Processing class: {class_name} | Images: {len(class_sample_indices)}")

      class_subset = Subset(dataset.dataset, class_sample_indices)
      dataloader = DataLoader(
          class_subset,
          batch_size=BATCH_SIZE,
          shuffle=False,
          num_workers=NUM_WORKERS,
          pin_memory=True,
          drop_last=False
      )

      features_dict = {}
      with torch.no_grad():
          for batch_idx, (images, _) in enumerate(tqdm(dataloader, desc=f"Extracting {class_name}")):
              images = images.to(DEVICE, non_blocking=True)
              with torch.cuda.amp.autocast():  # AMP for speed on A100/H100
                  batch_features = model(images).cpu().numpy()

              for i, feat in enumerate(batch_features):
                  global_idx = class_sample_indices[batch_idx * BATCH_SIZE + i]
                  filename = os.path.basename(all_samples[global_idx][0])
                  features_dict[filename] = feat

      os.makedirs(OUTPUT_ROOT, exist_ok=True)
      np.savez(output_npz, **features_dict)
      print(f"✅ Saved {len(features_dict)} features to {output_npz}")

  print("\n🎉 All classes finished!")

if __name__ == "__main__":
  synthetic_dir = "./Generation_EMD2/edm2_synthetics/"
  feature_dir = "./imagenet1K_dataset"
  step = 1
  x = 4
  for i in tqdm(range(x, x + step), desc="Processing synthetic datasets",):
    IMAGENET_ROOT = os.path.join(synthetic_dir, f"imagenet_synthetic_{i}/")
    OUTPUT_ROOT = os.path.join(feature_dir, f"MocoV3_EDM2_{i}/npz_per_class")
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    
    BATCH_SIZE = 1024
    NUM_WORKERS = 10
    begin = 0
    end =  1000
    main(
        IMAGENET_ROOT=IMAGENET_ROOT,
        OUTPUT_ROOT=OUTPUT_ROOT,
        BATCH_SIZE=BATCH_SIZE,
        NUM_WORKERS=NUM_WORKERS,
        begin=begin,
        end=end
    )