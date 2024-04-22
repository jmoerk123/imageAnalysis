from detectron2.engine import DefaultPredictor  # type: ignore
from detectron2.config import get_cfg  # type: ignore
from detectron2.data import MetadataCatalog  # type: ignore
from detectron2.utils.video_visualizer import ColorMode, Visualizer  # type: ignore
from detectron2 import model_zoo  # type: ignore
from detectron2.structures.instances import Instances

import cv2
import numpy as np
import matplotlib.pyplot as plt  # type: ignore
from pathlib import Path
from typing import Union, List


class Detector:
    def __init__(self, model_type) -> None:
        self.cfg = get_cfg()

        # Load model config and pretrained model
        if model_type == "KS":
            model_name = "COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml"
        elif model_type == "OD":
            model_name = "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"
        elif model_type == "IS":
            model_name = "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
        self.cfg.merge_from_file(model_zoo.get_config_file(model_name))
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_name)

        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
        self.cfg.MODEL.DEVICE = "cuda"

        self.predictor = DefaultPredictor(self.cfg)

    def get_prediction(self, image: Union[str, np.ndarray]):
        if type(image) == str:
            image = cv2.imread(image)
        predictions = self.predictor(image)
        return predictions


class Viz:
    def __init__(self, detector: Detector, image: Union[str, np.ndarray]) -> None:
        self.detector = detector
        self.image_with_pred = None

        if type(image) == str:
            self.image = cv2.imread(image)
        else:
            self.image = image

        self.viz = Visualizer(
            self.image[:, :, ::-1],  # type: ignore
            metadata=MetadataCatalog.get(self.detector.cfg.DATASETS.TRAIN[0]),
            instance_mode=ColorMode.IMAGE,  # For BW use IMAGE_BW
        )

        self.output = self.viz.output
        self.predictions = self.detector.get_prediction(image=self.image)

    def add_keypoints(self, keypoints=None, color=None):
        if not keypoints:
            keypoints = self.predictions["instances"].get_fields()["pred_keypoints"][0]

        if color:
            self.change_keypoint_colors(color)

        self.output = self.viz.draw_and_connect_keypoints(keypoints.to("cpu"))

    def add_predictions(self, predictions=None):
        if not predictions:
            self.output = self.viz.draw_instance_predictions(
                self.predictions["instances"].to("cpu")
            )
        else:
            self.output = self.viz.draw_instance_predictions(
                predictions["instances"].to("cpu")
            )

    def change_keypoint_colors(self, color: list):
        for i in range(len(self.viz.metadata.keypoint_connection_rules)):
            self.viz.metadata.keypoint_connection_rules[i] = (
                self.viz.metadata.keypoint_connection_rules[i][0],
                self.viz.metadata.keypoint_connection_rules[i][1],
                color,
            )

    def add_keypoints_from_other_img(self, image: Union[str, np.ndarray]):
        if type(image) == str:
            image = cv2.imread(image)

        new_predictions = self.detector.predictor(image)
        fields = self.predictions["instances"].get_fields()

        new_predictions["instances"]._fields = self.scaleKeyPoints(
            new_field=new_predictions["instances"].get_fields(), org_field=fields
        )
        new_predictions["instances"]._fields = self.moveKeyPoints(
            new_field=new_predictions["instances"].get_fields(), org_field=fields
        )

        new_keypoints = new_predictions["instances"].get_fields()["pred_keypoints"][0]
        self.output = self.viz.draw_and_connect_keypoints(new_keypoints.to("cpu"))

    def scaleKeyPoints(self, new_field, org_field):
        x_val_to = (
            org_field["pred_boxes"].tensor[0][2] - org_field["pred_boxes"].tensor[0][0]
        )
        y_val_to = (
            org_field["pred_boxes"].tensor[0][3] - org_field["pred_boxes"].tensor[0][1]
        )

        x_val_from = (
            new_field["pred_boxes"].tensor[0][2] - new_field["pred_boxes"].tensor[0][0]
        )
        y_val_from = (
            new_field["pred_boxes"].tensor[0][3] - new_field["pred_boxes"].tensor[0][1]
        )
        x_scale = self._cal_scale(x_val_to, x_val_from)
        y_scale = self._cal_scale(y_val_to, y_val_from)

        for indx, key_point in enumerate(new_field["pred_keypoints"][0]):
            new_field["pred_keypoints"][0][indx][0] = key_point[0] * x_scale
            new_field["pred_keypoints"][0][indx][1] = key_point[1] * y_scale

        return new_field

    def moveKeyPoints(self, new_field, org_field):
        nose_point_to = org_field["pred_keypoints"][0][0]
        nose_point_from = new_field["pred_keypoints"][0][0]

        x_move = nose_point_to[0] - nose_point_from[0]
        y_move = nose_point_to[1] - nose_point_from[1]

        for indx, key_point in enumerate(new_field["pred_keypoints"][0]):
            new_field["pred_keypoints"][0][indx][0] = key_point[0] + x_move
            new_field["pred_keypoints"][0][indx][1] = key_point[1] + y_move

        return new_field

    def _cal_scale(self, val1, val2):
        return val1 / val2

    def get_image(self):
        return self.output.get_image()

    def save_image(self, image_path: str):
        cv2.imwrite(image_path, self.output.get_image()[:, :, ::-1])  # type: ignore

    def show_image(self, title):
        cv2.imshow(title, self.output.get_image()[:, :, ::-1])  # type: ignore
        while True:
            k = cv2.waitKey(
                100
            )  # change the value from the original 0 (wait forever) to something appropriate
            if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                break
            elif k == 27:  # Close on ESC
                cv2.destroyAllWindows()
                break

        cv2.destroyAllWindows()
