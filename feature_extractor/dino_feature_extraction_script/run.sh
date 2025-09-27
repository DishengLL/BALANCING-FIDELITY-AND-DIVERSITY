#!/bin/bash
#SBATCH --job-name=1extraction_dino
#SBATCH --output=logs/1extraction_dino_%j.out  
#SBATCH --error=logs/1extraction_dino_%j.err
#SBATCH -A aiscii
#SBATCH -p aiscii                  
#SBATCH --gres=gpu:1                          
#SBATCH --cpus-per-task=20
#SBATCH --mem=50G                               
#SBATCH --time=48:00:00                         
#SBATCH --mail-type=END,FAIL                    

cd xxx


nvidia-smi
conda activate tiny 


for i in {1..5}
do
  ~/.conda/envs/tiny/bin/python ./run_feature_extractor_dino.py \
    --split $i/20 \
    --root ./Generation_EMD2/edm2/imagenet_synthetic_1/ \
    --out ./imagenet1K_dataset/DinoV2_EDM2_1/npz_per_class
done

echo "✅ Job finished!"