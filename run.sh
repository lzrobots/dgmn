#!/usr/bin/env bash

#export NGPUS=8
#python -m torch.distributed.launch --nproc_per_node=$NGPUS ./tools/train_net.py --config-file "./configs/dw/e2e_mask_rcnn_R_50_FPN_2x.yaml"


export NGPUS=8
python -m torch.distributed.launch --nproc_per_node=$NGPUS ./tools/train_net.py --config-file "./configs/dw/e2e_mask_rcnn_X_101_32x8d_FPN_1x.yaml"
