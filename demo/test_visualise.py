import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import os
import cv2

from PIL import Image
import numpy as np

# this makes our figures bigger
pylab.rcParams['figure.figsize'] = 20, 12

from maskrcnn_benchmark.config import cfg
from predictor import COCODemo

config_file = "configs/e2e_mask_rcnn_X_101_32x8d_FPN_1x.yaml"

# update the config options with the config file
cfg.merge_from_file(config_file)
cfg.MODEL.WEIGHT = 'models/e2e_mask_rcnn_X_101_32x8d_FPN_1x/e2e_mask_rcnn_X_101_32x8d_FPN_1x.pth'
# manual override some options
# cfg.merge_from_list(["MODEL.DEVICE", "cpu"])

coco_demo = COCODemo(
    cfg,
    min_image_size=800,
    confidence_threshold=0.7,
)

# val_path = 'datasets/coco/val2017/'  # this is the validation image data
val_path = '/home/lz/Downloads/sample/'
# output_path = 'models/e2e_mask_rcnn_R_50_FPN_011d_33_g4_14812_sum_re_1x/output/'
output_path = '/home/lz/Downloads/output/'
imglistval = os.listdir(val_path)
for name in imglistval:
    print(name)
    imgfile = val_path + name
    pil_image = Image.open(imgfile).convert("RGB")
    image = np.array(pil_image)[:, :, [2, 1, 0]]

    predictions = coco_demo.run_on_opencv_image(image)  # forward predict

    savefile = output_path + name

    cv2.imwrite(savefile, predictions)
