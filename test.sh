#!/bin/bash
# Usage:
# ./test_single_model_1gpu_totalbs1.sh <path/to/config> <path/to/model.pth> <extra_args>
export NGPUS=4
for i in 0002500 0005000 0007500 0010000 0012500 0015000 0017500 0020000 0022500 0025000 0027500 0030000 0032500 0035000 0037500 0040000 0042500 0045000 0047500 0050000 0052500 0055000 0057500 0060000 0062500 0065000 0067500 0070000 0072500 0075000 0077500 0080000 0082500 0085000 0087500 0090000; do \
python -m torch.distributed.launch --nproc_per_node=$NGPUS \
    tools/test_net.py \
    --config-file "configs/e2e_mask_rcnn_R_50_FPN_1x.yaml" \
    OUTPUT_DIR "models/e2e_mask_rcnn_R_50_FPN_1x" \
    MODEL.WEIGHT "models/e2e_mask_rcnn_R_50_FPN_1x/model_${i}.pth" \
    TEST.IMS_PER_BATCH 16; \
    done

#for i in 0090000 0002500 0005000 0007500 0010000 0012500 0015000 0017500 0020000 0022500 0025000 0027500 0030000 0032500 0035000 0037500 0040000 0042500 0045000 0047500 0050000 0052500 0055000 0057500 0060000 0062500 0065000 0067500 0070000 0072500 0075000 0077500 0080000 0082500 0085000 0087500 0090000; do \
#python -m torch.distributed.launch --nproc_per_node=$NGPUS \
#    tools/test_net.py \
#    --config-file "configs/dw/e2e_mask_rcnn_R_50_FPN_2x.yaml" \
#    OUTPUT_DIR "models/e2e_mask_rcnn_R_50_FPN_001da_33_re_2x" \
#    MODEL.WEIGHT "models/e2e_mask_rcnn_R_50_FPN_001da_33_re_2x/model_${i}.pth" \
#    TEST.IMS_PER_BATCH 16; \
#    done
#    MODEL.WEIGHT "/scratch_new/engs1870/maskrcnn/e2e_mask_rcnn_R_50_FPN_001da_33_re_2x/model_${i}.pth" \



#for i in 0002500 0005000 0007500 0010000 0012500 0015000 0017500 0020000 0022500 0025000 0027500 0030000 0032500 0035000 0037500 0040000 0042500 0045000 0047500 0050000 0052500 0055000 0057500 0060000 0062500 0065000 0067500 0070000 0072500 0075000 0077500 0080000 0082500 0085000 0087500 0090000; do \
#python -m torch.distributed.launch --nproc_per_node=$NGPUS \
#    tools/test_net.py \
#    --config-file "configs/e2e_mask_rcnn_R_50_FPN_1x.yaml" \
#    OUTPUT_DIR "models/Non-local" \
#    MODEL.WEIGHT "models/Non-local/model_${i}.pth" \
#    TEST.IMS_PER_BATCH 16; \
#    done