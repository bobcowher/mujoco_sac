#!/bin/bash
mkdir -p logs
export MUJOCO_GL=egl
source ~/anaconda3/etc/profile.d/conda.sh
conda activate mujoco_sac
nohup python ./train.py >> logs/train.log 2>&1 & 
nohup tensorboard --logdir runs --port 4444 --bind_all >> logs/tensorboard.log 2>&1 &
