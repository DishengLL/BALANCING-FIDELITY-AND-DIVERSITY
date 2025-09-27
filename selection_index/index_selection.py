import torch
import json
from tqdm import tqdm
import os
import numpy as np
import torch
import random
from collections import Counter

def get_the_highest_score_index(matrix, syntheic_name_list, top_k=250, top_n=2, reverse=False):
    if matrix.shape[0] > matrix.shape[1]:
        matrix = matrix.T
        number_of_real = matrix.shape[0]
        number_of_synthetic = matrix.shape[1]

    if reverse:
        matrix = -matrix
    
    # for each row return the top-n largest column indices and values
    topk_values, topk_indices = torch.topk(matrix, top_n, dim=1)  # shape: (rows, top_n)
    # print("the shape of matrix:", matrix.shape) 
    assert len(syntheic_name_list) == matrix.shape[1], \
        "Synthetic image names length must match the number of synthetic features."
    
    retrieval_syn_image_and_scores = {}

    # iterate through each row
    for indices, values in zip(topk_indices, topk_values):
        for idx, val in zip(indices, values):
            current_image_name = syntheic_name_list[idx.item()]
            current_value = retrieval_syn_image_and_scores.get(current_image_name, -9999)
            if val.item() > current_value:
                retrieval_syn_image_and_scores[current_image_name] = val.item()

    # check unique count
    assert len(set(retrieval_syn_image_and_scores.keys())) >= top_k, \
        f"The number of unique top-{top_n} column indices is less than top_k."

    return list(retrieval_syn_image_and_scores.keys()), retrieval_syn_image_and_scores

def weighted_unique_sample(pool, k):
    # 统计每个元素出现的次数（作为权重）
    counter = Counter(pool)  
    elements = list(counter.keys())
    weights = list(counter.values())  # 出现次数 = 权重

    if k > len(elements):
        raise ValueError("k cannot be larger than the number of unique elements")

    # 用一个临时列表来做加权不放回采样
    selected = []
    available_elements = elements.copy()
    available_weights = weights.copy()

    for _ in range(k):
        # 按当前权重采一个
        choice = random.choices(available_elements, weights=available_weights, k=1)[0]
        selected.append(choice)

        # 移除已选元素（不放回）
        idx = available_elements.index(choice)
        del available_elements[idx]
        del available_weights[idx]

    return selected


def top_k_selection_v2(
  synthetic_feature_path = "./imagenet1K_dataset/MocoV3_self_EDM2_map/npz_per_class",
  common_rare_dict_path = "./Imagenet1K_common_rare.json",
  EDM2_synthetic_common_metric_pt = "./all_EMD2_synthetic_common_score.pt",
  EDM2_synthetic_rare_metric_pt = "./all_EMD2_synthetic_rare_score.pt",
  top_k=500,
  storage_base_path = "./selection_index/EDM2/top_k_selection_2.0",
  extractor="MocoV3"
  ):
    
  

  reverse = False
  if "diversity" in EDM2_synthetic_common_metric_pt:
    current_metric = "diversity"
    reverse = True
  elif "fidelity" in EDM2_synthetic_common_metric_pt:
    current_metric = "fidelity"
  elif "scores" in EDM2_synthetic_common_metric_pt:
    current_metric = "scores"
  else:
    raise ValueError("The metric type is not recognized. Please check the file names.")
  print(f"Current metric is: {current_metric}")

  storage_base_path = storage_base_path
  
  common_metric_matrix = torch.load(EDM2_synthetic_common_metric_pt, weights_only=False)
  rare_metric_matrix = torch.load(EDM2_synthetic_rare_metric_pt, weights_only=False)

  with open(common_rare_dict_path, "r") as f:
    common_rare_dict = json.load(f)

  synthetic_common_selection = {}
  synthetic_rare_selection = {}
  synthetic_common_rare_score_selection = {}

  top_k_for_common = top_k // 2
  top_k_for_rare = top_k - top_k_for_common

  for class_name in tqdm(common_rare_dict.keys(), desc="Selecting top-k unique synthetic indices"):
    synthetic_npz_path = os.path.join(synthetic_feature_path, class_name)
    synthetic_npz = np.load(synthetic_npz_path)
    synthetic_image_name = list(synthetic_npz.keys())

    current_score_matrix_in_common = common_metric_matrix[class_name]#.cpu().numpy()  # [M_1, N]
    current_score_matrix_in_rare = rare_metric_matrix[class_name]#.cpu().numpy()  # [M_2, N]
    original_top_n = 0
    repeat = True
    
    while repeat:
      original_top_n += 1

      common_result, common_synthetic_score_dict = get_the_highest_score_index(current_score_matrix_in_common, 
                                                                            syntheic_name_list=synthetic_image_name,
                                                                            top_k=1, top_n=original_top_n, reverse=reverse)

      rare_result, rare_synthetic_score_dict = get_the_highest_score_index(current_score_matrix_in_rare, 
                                                                      syntheic_name_list=synthetic_image_name,
                                                                      top_k=1, top_n=original_top_n, reverse = reverse) 
      common_rare_selection = list(common_result + rare_result)
      
      if len(set(common_rare_selection)) >= top_k:
        repeat = False
    selected_k = weighted_unique_sample(common_rare_selection, top_k)
    assert len(set(selected_k)) == top_k, f"The number of unique selected indices is not equal to top_k: {len(set(selected_k))},{top_k}. "
    synthetic_common_selection[class_name] = selected_k
  os.makedirs(storage_base_path, exist_ok=True)
  metric_json_path = os.path.join(storage_base_path, f"{current_metric}_{extractor}_{top_k}.json")

  with open(metric_json_path, "w") as f:
      json.dump(synthetic_common_selection, f, separators=(",", ":"))


if __name__ == "__main__":
  base_dir = ".//"
  base_storage_path = ".//selection_index/"
  for sub_data_index  in range(4,5):
    synthetic_feature_path = f"./imagenet1K_dataset/MocoV3_EDM2_{sub_data_index}/npz_per_class"
    common_rare_dict_path = "./Imagenet1K_common_rare_moco_selection.json"
    top_k=300

    EDM2_synthetic_common_metric_pt = f"{base_dir}/EDM2_{sub_data_index}_metric/all_EMD2_synthetic_common_scores_MocoV3.pt"
    EDM2_synthetic_rare_metric_pt = f"{base_dir}/EDM2_{sub_data_index}_metric/all_EMD2_synthetic_rare_scores_MocoV3.pt"
    top_k_selection_v2(
      synthetic_feature_path=synthetic_feature_path,
      common_rare_dict_path=common_rare_dict_path,
      EDM2_synthetic_common_metric_pt=EDM2_synthetic_common_metric_pt,
      EDM2_synthetic_rare_metric_pt=EDM2_synthetic_rare_metric_pt,
      top_k=top_k,
      extractor = "MocoV3",
      storage_base_path = f"{base_storage_path}/EDM2_{sub_data_index}/top_k_selection_2.0"
    )
    

    EDM2_synthetic_common_metric_pt = f"{base_dir}/EDM2_{sub_data_index}_metric/all_EMD2_synthetic_common_fidelity_MocoV3.pt"
    EDM2_synthetic_rare_metric_pt = f"{base_dir}/EDM2_{sub_data_index}_metric/all_EMD2_synthetic_rare_fidelity_MocoV3.pt"
    top_k_selection_v2(
      synthetic_feature_path=synthetic_feature_path,
      common_rare_dict_path=common_rare_dict_path,
      EDM2_synthetic_common_metric_pt=EDM2_synthetic_common_metric_pt,
      EDM2_synthetic_rare_metric_pt=EDM2_synthetic_rare_metric_pt,
      top_k=top_k,
      extractor = "MocoV3",
      storage_base_path = f"{base_storage_path}/EDM2_{sub_data_index}/top_k_selection_2.0"
    )
    
    
    EDM2_synthetic_common_metric_pt = f"{base_dir}/EDM2_{sub_data_index}_metric/all_EMD2_synthetic_common_diversity_MocoV3.pt"
    EDM2_synthetic_rare_metric_pt = f"{base_dir}/EDM2_{sub_data_index}_metric/all_EMD2_synthetic_rare_diversity_MocoV3.pt"
    top_k_selection_v2(
      synthetic_feature_path=synthetic_feature_path,
      common_rare_dict_path=common_rare_dict_path,
      EDM2_synthetic_common_metric_pt=EDM2_synthetic_common_metric_pt,
      EDM2_synthetic_rare_metric_pt=EDM2_synthetic_rare_metric_pt,
      top_k=top_k,
      extractor = "MocoV3",
      storage_base_path = f"{base_storage_path}/EDM2_{sub_data_index}/top_k_selection_2.0"
    )

    print(f">>>>>>{sub_data_index} complete<<<<<<<")