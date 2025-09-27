amount = 100
for i in range(2,10):
  print(i)
  path = f".//Generation_EMD2/edm2_synthetics/imagenet_synthetic_{i}"
  import os
  import random
  import json
  data = {}
  total = 0
  for dir_name in os.listdir(path):
    dir_path = os.path.join(path, dir_name)
    if os.path.isdir(dir_path):
      # random selection 500 images within the dir_path contained in the list
      images = os.listdir(dir_path)

      selected_images = random.sample(images, amount)
      total += len(selected_images)
      data[dir_name] = selected_images

  save_path = f".//AAAI/selection_index/random_index/EDM2_{i}_{amount}.json"

  with open(save_path, "w") as f:
    json.dump(data, f)