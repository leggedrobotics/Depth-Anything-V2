#!/bin/bash

#SBATCH -n 8
#SBATCH --time=36:00:00
#SBATCH --mem-per-cpu=8000
#SBATCH --tmp=100G
#SBATCH --gpus=1
#SBATCH --gres=gpumem:16G
#SBATCH --job-name=imagenet-depthanything
#SBATCH --output=/cluster/work/rsl/patelm/result/slurm_output/%x_%j.out
#SBATCH --error=/cluster/work/rsl/patelm/result/slurm_output/%x_%j.err

# Check if the correct number of arguments is provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <src_file> <outdir>"
    exit 1
fi

source ~/.bashrc
conda activate depthv2
echo "Preparing the dataset"
echo $TMPDIR

src_dir=/cluster/work/rsl/patelm/imagenet-1k
src_file=${src_dir}/$1.tar.gz
outdir=$2/$1

echo "Source file: ${src_file}"
echo "Output directory: ${outdir}"

local_dir=$TMPDIR/${src_file}

echo "Local directory: ${local_dir}"

mkdir -p $local_dir
tar -xzf $src_file -C $local_dir

echo "Finished extracting the dataset"

cd /cluster/home/patelm/ws/rsl/Depth-Anything-V2

echo "Cuda visible devices ${CUDA_VISIBLE_DEVICES}"

python run.py --img-path $local_dir --outdir $outdir --pred-only --grayscale --save-npz

echo "Finished running the model"
echo "Compressing the output"

#Zip the output
tar -czf $outdir.tar.gz $outdir