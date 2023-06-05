#!/bin/bash
#SBATCH --job-name="Semantic Kitti Preprocessing"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:1,VRAM:12G
#SBATCH --mem=32G
#SBATCH --time=0:05:00
#SBATCH --mail-type=ALL
#SBATCH --output=/storage/slurm/logs/slurm-%j.out
#SBATCH --error=/storage/slurm/logs/slurm-%j.out
srun python datasets/kitti_semantic/preprocess_semantic_kitti.py

# Comment (Lukas K.)
# For a normal DL task, it is useful to use multiple CPUs per GPU for example --cpus-per-task=3. This depends on how many CPUs your dataloader (pytorch num_workers) can use effectively.
# For a normal DL task, one should use about 32G of memory. Of course, this depends on the task. Try not to take too much, but if the memory is insufficient the task will get killed.
