import os
import json
import numpy as np
from tqdm import tqdm
import torch
import gc
import torch.nn.functional as F
import os
import json
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision.models import resnet50
from torch.utils.data import DataLoader, Subset
from PIL import Image
from tqdm import tqdm

base_dir = ".//"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_real_subset_features_matrix(real_np, keys):
  feats = [real_np[k] for k in keys]
  return torch.from_numpy(np.stack(feats, axis=0)).to(device)  # [M, D]

def compute_synthetic_common_alignment_scores(
    syn_normed,               # [N, D]  归一化synthetic特征
    real_common_normed        # [M_1, D] 归一化real-common特征
    ):
    # cosine similarity (fidelity)
    synthetic_common_fidelity = syn_normed @ real_common_normed.T  # [N, M_1]

    # diversity score
    centroid_common = F.normalize(real_common_normed.mean(dim=0, keepdim=True), dim=1)
    A = syn_normed.unsqueeze(1) - real_common_normed.unsqueeze(0)  # [N, M_1, D]
    B = centroid_common - real_common_normed  # [M_1, D]
    
    numerator = (A * B.unsqueeze(0)).sum(dim=-1)      # [N, M_1]
    A_norm = A.norm(dim=-1)                           # [N, M_1]
    B_norm = B.norm(dim=-1).unsqueeze(0)              # [1, M_1]
    
    synthetic_common_diversity = numerator / (A_norm * B_norm + 1e-8)  # [N, M_1]
    
    # score = fidelity - diversity
    synthetic_common_score = synthetic_common_fidelity - synthetic_common_diversity  # [N, M_1]

    return synthetic_common_fidelity, synthetic_common_diversity, synthetic_common_score

def compute_synthetic_rare_alignment_scores(
  syn_normed = None,             # 归一化后的 synthetic
  real_rare_normed = None,      # 归一化后的 real rare 特征
  real_np = None,  # 真实特征矩阵
  class_rare_2_common_in_real_dict = None
  ):
  # cosine similarity (fidelity)
  synthetic_rare_fidelity = syn_normed @ real_rare_normed.T  # [N, M_2]
  
    # diversity score
  rare_2_common_in_real_dict = class_rare_2_common_in_real_dict

  common_counterpart_index = [i["common_image"] for i in rare_2_common_in_real_dict.values()]

  common_counterpart_matrix = get_real_subset_features_matrix(real_np, common_counterpart_index)  # [M_2, D]
  normalized_common_counterpart_matrix = F.normalize(common_counterpart_matrix, dim=1)  # [M_2, D]
  
  assert normalized_common_counterpart_matrix.shape[0] == real_rare_normed.shape[0], "Mismatched dimensions between common and rare subsets."
  
  rare_2_common_matrix = normalized_common_counterpart_matrix - real_rare_normed  # shape: [M2, D]
  norm_rare_2_common_matrix = rare_2_common_matrix / rare_2_common_matrix.norm(dim=-1, keepdim=True)  # Normalize to unit vectors

  rare_2_synthetic_matrix = syn_normed.unsqueeze(1) - real_rare_normed.unsqueeze(0)  # [N, M2, D]
  norm_rare_2_synthetic_matrix = rare_2_synthetic_matrix / rare_2_synthetic_matrix.norm(dim=-1, keepdim=True)  # [N, M2, D]

  # [N, M2, D] ⋅ [M2, D] → [N, M2]
  cos_sim_matrix = (norm_rare_2_synthetic_matrix * norm_rare_2_common_matrix.unsqueeze(0)).sum(dim=-1)
  synthetic_rare_diversity = cos_sim_matrix  # [N, M2]
  
  synthetic_rare_score = synthetic_rare_fidelity - synthetic_rare_diversity  # [N, M2]
  
  
  return synthetic_rare_fidelity, synthetic_rare_diversity, synthetic_rare_score

def get_score(
  real_feature_path = ".//imagenet1K_dataset/MocoV3_Real_feature_map/npz_per_class",
  synthetic_feature_path = ".//imagenet1K_dataset/MocoV3_self_SD1.5_map/npz_per_class",
  common_rare_dict_path = ".//Imagenet1K_common_rare.json",
  rare_2_common_in_real_dict_path = ".//Imagenet1K_rare_2_common_pair.json",
  synthetic_data_name = "EMD2",
  extractor = "MocoV3"  # or "DinoV2", "MocoV2", etc. based on your feature extraction method
  ):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  real_feature_path = real_feature_path
  synthetic_feature_path = synthetic_feature_path

  sub_data_index = synthetic_feature_path.split("/")[-2][-1]
  print("current_sub_data_index: ", sub_data_index)

  if os.path.exists(common_rare_dict_path):
      with open(common_rare_dict_path, "r") as f:
          common_rare_dict = json.load(f)  
  if os.path.exists(rare_2_common_in_real_dict_path):
      with open(rare_2_common_in_real_dict_path, "r") as f:
          rare_2_common_in_real_dict = json.load(f)
  
  synthetic_common_score_dict = {}
  synthetic_common_fidelity_dict = {}
  synthetic_common_diversity_dict = {}

  synthetic_rare_score_dict = {}
  synthetic_rare_fidelity_dict = {}
  synthetic_rare_diversity_dict = {}
  

  for class_name in tqdm(common_rare_dict.keys(), desc="Computing scores"):
      common_keys = common_rare_dict[class_name]["common"]
      rare_keys   = common_rare_dict[class_name]["rare"]
      real_path = os.path.join(real_feature_path, class_name)
      syn_path  = os.path.join(synthetic_feature_path, class_name)  
      real_np   = np.load(real_path)
      syn_np    = np.load(syn_path)

      class_rare_2_common = rare_2_common_in_real_dict[class_name]
      class_rare_2_common_in_real_dict = class_rare_2_common
      real_common_features = get_real_subset_features_matrix(real_np, common_keys) # tensor [M_1, D]
      real_rare_features   = get_real_subset_features_matrix(real_np, rare_keys)   # tensor [M_2, D]
      real_common_normed = F.normalize(real_common_features, dim=1)        # [M_1, D]
      real_rare_normed   = F.normalize(real_rare_features, dim=1)          # [M_2, D]

      syn_feats = [syn_np[k] for k in syn_np.files]
      synthetic_mat = torch.from_numpy(np.stack(syn_feats, axis=0)).to(device)  # [N, D]
      syn_normed  = F.normalize(synthetic_mat, dim=1)    # [N, D]

      # common subset
      synthetic_common_fidelity, synthetic_common_diversity, synthetic_common_score = compute_synthetic_common_alignment_scores(
          syn_normed = syn_normed,             # 归一化后的 synthetic
          real_common_normed = real_common_normed      # 归一化后的 real common 特征
      )
      
      synthetic_common_score_dict[class_name] = synthetic_common_score.cpu()
      synthetic_common_fidelity_dict[class_name] = synthetic_common_fidelity.cpu()
      synthetic_common_diversity_dict[class_name] = synthetic_common_diversity.cpu()

      synthetic_rare_fidelity, synthetic_rare_diversity, synthetic_rare_score = compute_synthetic_rare_alignment_scores(
        syn_normed = syn_normed,             # 归一化后的 synthetic
        real_rare_normed = real_rare_normed,      # 归一化后的 real rare 特征
        real_np = real_np,  # 真实特征矩阵
        class_rare_2_common_in_real_dict = class_rare_2_common_in_real_dict  # rare subset对应的common subset映射关系
      )
      
      synthetic_rare_score_dict[class_name] = synthetic_rare_score.cpu()
      synthetic_rare_fidelity_dict[class_name] = synthetic_rare_fidelity.cpu()
      synthetic_rare_diversity_dict[class_name] = synthetic_rare_diversity.cpu()
      
      # clear memory, cuda memory
      del syn_feats, synthetic_mat, syn_normed
      torch.cuda.empty_cache()
      del real_path, syn_path, real_np, syn_np
      del real_common_features, real_rare_features, real_common_normed, real_rare_normed
      del synthetic_common_fidelity, synthetic_common_diversity, synthetic_common_score, synthetic_rare_fidelity, synthetic_rare_diversity, synthetic_rare_score
      gc.collect()
      
  os.makedirs(f"{base_dir}/EDM2_{sub_data_index}_metric", exist_ok=True)
  torch.save(synthetic_common_score_dict, 
             f"{base_dir}/EDM2_{sub_data_index}_metric/all_{synthetic_data_name}_synthetic_common_scores_{extractor}.pt")
  torch.save(synthetic_common_fidelity_dict, 
             f"{base_dir}/EDM2_{sub_data_index}_metric/all_{synthetic_data_name}_synthetic_common_fidelity_{extractor}.pt")
  torch.save(synthetic_common_diversity_dict, 
             f"{base_dir}/EDM2_{sub_data_index}_metric/all_{synthetic_data_name}_synthetic_common_diversity_{extractor}.pt")
  torch.save(synthetic_rare_score_dict, 
             f"{base_dir}/EDM2_{sub_data_index}_metric/all_{synthetic_data_name}_synthetic_rare_scores_{extractor}.pt")
  torch.save(synthetic_rare_fidelity_dict, 
             f"{base_dir}/EDM2_{sub_data_index}_metric/all_{synthetic_data_name}_synthetic_rare_fidelity_{extractor}.pt")
  torch.save(synthetic_rare_diversity_dict, 
             f"{base_dir}/EDM2_{sub_data_index}_metric/all_{synthetic_data_name}_synthetic_rare_diversity_{extractor}.pt")
  print("Scores computed and saved successfully.")
  


if __name__ == "__main__":
  for i in range(4,5):
    # if i == 4:
    #   print("pass i = 4")
    #   continue
    real_feature_path = ".//imagenet1K_dataset/MocoV3_Real_feature_map/npz_per_class"
    synthetic_feature_path = f".//imagenet1K_dataset/MocoV3_EDM2_{i}/npz_per_class"
    common_rare_dict_path = ".//Imagenet1K_common_rare_moco_selection.json"
    rare_2_common_in_real_dict_path = ".//Imagenet1K_rare_2_common_pair_moco.json"
    synthetic_data_name = "EMD2"
    
    print("configuration:")
    print(f"Real Feature Path: {real_feature_path}")
    print(f"Synthetic Feature Path: {synthetic_feature_path}")
    print(f"Common/Rare Dict Path: {common_rare_dict_path}")
    print(f"Rare to Common Dict Path: {rare_2_common_in_real_dict_path}")
    print(f"Synthetic Data Name: {synthetic_data_name}")

    get_score(
      real_feature_path=real_feature_path,
      common_rare_dict_path=common_rare_dict_path,
      synthetic_feature_path=synthetic_feature_path,
      rare_2_common_in_real_dict_path=rare_2_common_in_real_dict_path,
      synthetic_data_name=synthetic_data_name,
      extractor="MocoV3"  # or "DinoV2", "MocoV2", etc. based on your feature extraction method
    )