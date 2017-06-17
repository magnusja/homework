#!/bin/bash
set -eux
for e in Hopper-v1 Ant-v1 HalfCheetah-v1 Humanoid-v1 Reacher-v1 Walker2d-v1
do
    python train.py data/${e}_observation_data.npy data/${e}_action_data.npy $e
done
