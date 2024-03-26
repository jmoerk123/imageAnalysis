from detectron2.engine import DefaultPredictor # type: ignore
from detectron2.config import get_cfg # type: ignore
from detectron2.data import MetadataCatalog # type: ignore
from detectron2.utils.video_visualizer import ColorMode, Visualizer # type: ignore
from detectron2 import model_zoo # type: ignore
from detectron2.structures.instances import Instances

import cv2 
import numpy as np
import matplotlib.pyplot as plt # type: ignore

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

    def onImage(self, imagePath, save_img = False):
        image = cv2.imread(imagePath)
        predictions = self.predictor(image)

        viz = Visualizer(image[:,:,::-1], metadata = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), 
                         instance_mode=ColorMode.IMAGE_BW)
        
        output = viz.draw_instance_predictions(predictions["instances"].to("cpu"))

        # cv2.imshow("Results", output.get_image()[:,:,::-1])
        # cv2.waitKey(0)

        plt.imshow(output.get_image())
        if save_img:
            plt.savefig("test.png")
        
        return predictions

    def onVideo(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if (cap.isOpened() == False):
            print("Error opening the file...")
            return
        (sucess, image) = cap.read()
        while sucess:
            predictions = self.predictor(image)

            viz = Visualizer(image[:,:,::-1], metadata = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), 
                            instance_mode=ColorMode.IMAGE_BW)
            
            output = viz.draw_instance_predictions(predictions["instances"].to("cpu"))

            cv2.imshow("Results", output.get_image()[:,:,::-1])
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            (sucess, image) = cap.read()


class Compare:
    def __init__(self, image_path, correct_path) -> None:
        self.image = cv2.imread(image_path)
        self.correct_img = cv2.imread(correct_path)
        self.detector = Detector("KS")

    def compare(self, save_img):
        predictions_comp = self.detector.predictor(self.image)
        predictions_correct = self.detector.predictor(self.correct_img)

        comp_fields = predictions_comp["instances"].get_fields()

        predictions_correct["instances"]._fields = self.scaleKeyPoints(
            field_to_scale=predictions_correct["instances"].get_fields(),
            scale_to_field=comp_fields
        )
        predictions_correct["instances"]._fields = self.moveKeyPoints(
            field_to_move=predictions_correct["instances"].get_fields(),
            move_to_field=comp_fields
        )

        predictions_correct["instances"]._fields = self.change_keypoint_heatmap_colors(
            predictions_correct["instances"].get_fields(),
            5
        )

        correct_keypoints = predictions_correct["instances"].get_fields()["pred_keypoints"][0]

        viz = Visualizer(self.image[:,:,::-1], metadata = MetadataCatalog.get(self.detector.cfg.DATASETS.TRAIN[0]), 
                         instance_mode=ColorMode.IMAGE_BW)
        
        viz.draw_instance_predictions(predictions_comp["instances"].to("cpu"))
        output = viz.draw_and_connect_keypoints(correct_keypoints.to("cpu"))
        # output = viz.draw_instance_predictions(predictions_correct["instances"].to("cpu"))

        # cv2.imshow("Results", output.get_image()[:,:,::-1])
        # cv2.waitKey(0)

        plt.imshow(output.get_image())
        if save_img:
            plt.savefig("test2.png")
        
        # return predictions
            
    def scaleKeyPoints(self, field_to_scale, scale_to_field):
        x_val_to = scale_to_field["pred_boxes"].tensor[0][2] - scale_to_field["pred_boxes"].tensor[0][0]
        y_val_to = scale_to_field["pred_boxes"].tensor[0][3] - scale_to_field["pred_boxes"].tensor[0][1]

        x_val_from = field_to_scale["pred_boxes"].tensor[0][2] - field_to_scale["pred_boxes"].tensor[0][0]
        y_val_from = field_to_scale["pred_boxes"].tensor[0][3] - field_to_scale["pred_boxes"].tensor[0][1]
        x_scale = self._cal_scale(x_val_to, x_val_from)
        y_scale = self._cal_scale(y_val_to, y_val_from)

        for indx, key_point in enumerate(field_to_scale["pred_keypoints"][0]):
            field_to_scale["pred_keypoints"][0][indx][0] = key_point[0]*x_scale
            field_to_scale["pred_keypoints"][0][indx][1] = key_point[1]*y_scale

        return field_to_scale

    def _cal_scale(self, val1, val2):
        return val1/val2

    def moveKeyPoints(self, field_to_move, move_to_field):
        nose_point_to = move_to_field["pred_keypoints"][0][0]
        nose_point_from = field_to_move["pred_keypoints"][0][0]

        x_move = nose_point_to[0] - nose_point_from[0]
        y_move = nose_point_to[1] - nose_point_from[1]

        for indx, key_point in enumerate(field_to_move["pred_keypoints"][0]):
            field_to_move["pred_keypoints"][0][indx][0] = key_point[0] + x_move
            field_to_move["pred_keypoints"][0][indx][1] = key_point[1] + y_move

        return field_to_move


    def change_keypoint_heatmap_colors(self, fields, color):
        for indx, keypoint in enumerate(fields["pred_keypoints"][0]):
            fields["pred_keypoints"][0][indx][2] = color
        return fields
"""
I can not find the ids for the keypoint! should I assume that the ids are as COCO?
"name": "person", # (specific category)
"supercategory": "person", #
"id": 1, # class id
"keypoints": [
"nose",
"left_eye",
"right_eye",
"left_ear",
"right_ear",
"left_shoulder",
"right_shoulder",
"left_elbow",
"right_elbow",
"left_wrist",
"right_wrist",
"left_hip",
"right_hip",
"left_knee",
"right_knee",
"left_ankle",
"right_ankle"
]
}
"""