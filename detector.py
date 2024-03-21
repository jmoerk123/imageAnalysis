from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.video_visualizer import ColorMode, Visualizer
from detectron2 import model_zoo

import cv2 
import numpy as np

class Detector:
    def __init__(self, model_type) -> None:
        self.cfg = get_cfg()

        # Load model config and pretrained model
        if model_type == "KS":
            model_name = "COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml"
        elif model_type == "OD":
            model_name == "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"
        elif model_type == "IS":
            model_name = "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
        self.cfg.merge_from_file(model_zoo.get_config_file(model_name))
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_name)

        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
        self.cfg.MODEL.DEVICE = "cuda"

        self.predictor = DefaultPredictor(self.cfg)

    def onImage(self, imagePath):
        image = cv2.imread(imagePath)
        predictions = self.predictor(image)

        viz = Visualizer(image[:,:,::-1], metadata = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), 
                         instance_mode=ColorMode.IMAGE_BW)
        
        output = viz.draw_instance_predictions(predictions["instances"].to("cpu"))

        cv2.imshow("Results", output.get_image()[:,:,::-1])
        cv2.waitKey(0)

    # def onVideo