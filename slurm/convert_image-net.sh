#!/bin/bash

output_dir="cluster/scratch/patelm/imagenet-1k-depth-anything-v2"
# input_files=("train_images_0" "train_images_1" "train_images_3" "train_images_4" "val_images" "test_images")
inputt_files=("val_images" "test_images")

for input_file in "${input_files[@]}"; do
    # Run the sbatch job
    sbatch slurm/depth_image-net.sh "$input_file" "$output_dir"
done