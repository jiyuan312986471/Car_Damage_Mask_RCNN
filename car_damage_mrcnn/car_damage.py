# -*- coding: utf-8 -*-
"""
Mask R-CNN
Configurations and data loading code for car damage dataset.

Written by Yuan Ji
"""

import json
import os
import sys
import numpy as np
import skimage
from typing import Tuple

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils


class CarDamageConfig(Config):
    """Configuration for training on the car damage dataset.
    Derives from the base Config class and overrides values specific
    to the car damage dataset.
    """
    # config name
    NAME = "car_damage"

    # Set to 1 when using CPU
    GPU_COUNT = 1

    # Number of images to train with on each GPU. A 12GB GPU can typically
    # handle 2 images of 1024x1024px.
    # Adjust based on your GPU memory and image sizes. Use the highest
    # number that your GPU can handle for best performance.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 3  # background + 3 damages

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9


class CarDamageDataset(utils.Dataset):

    def load_car_damage(self, dataset_dir: str, anno_path: str) -> None:
        """Load car damage dataset.

        Args:
            dataset_dir: directory of the dataset
            anno_path: path of annotation file
        """
        # Add classes
        self.add_class("damage", 1, "rayure")
        self.add_class("damage", 2, "roue")
        self.add_class("damage", 3, "retro")

        # Load annotations
        # VGG Image Annotator saves each image in the form:
        # { 'filename': 'xxx.jpg',
        #   'regions': {
        #       '0': {
        #           'region_attributes': {},
        #           'shape_attributes': {
        #               'all_points_x': [...],
        #               'all_points_y': [...],
        #               'name': 'polygon'},
        #           'region_attributes': {'cls': '...'}},
        #       ... more regions ...
        #   },
        #   'size': 100202
        # }
        # We mostly care about the x and y coordinates of each region
        with open(anno_path) as f:
            annotations = json.load(f)

        # Add images
        for a in annotations:
            # Get the x, y coordinates of points of the polygons that make up
            # the outline of each object instance. There are stores in the
            # shape_attributes (see json format above)
            polygons = [(r['shape_attributes'], r['region_attributes'])
                        for r in a['regions'].values()]

            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            img_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(img_path)
            height, width = image.shape[:2]

            self.add_image("damage", image_id=a['filename'], path=img_path,
                           width=width, height=height, polygons=polygons)
        return None

    def load_mask(self, image_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate instance masks for an image.

        Args:
            image_id: image id

        Returns:
            masks: A bool array of shape [height, width, instance count] with
                one mask per instance.
            class_ids: a 1D array of class IDs of the instance masks.

        Notes:
            To use this method, ``self.prepare()`` must be called before. Then
            use elements from ``self.image_ids`` as parameter of this method.
        """
        # If not a car damage dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "damage":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)

        # 1D array of instances' class_ids
        cls_ids = np.zeros([mask.shape[-1]], dtype=np.int32)

        for i, p in enumerate(info["polygons"]):
            # get mask and class
            m, cls = p[0], p[1]['cls']

            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(m['all_points_y'], m['all_points_x'])
            mask[rr, cc, i] = 1

            # set cls_ids by the classes of polygons
            cls_id = [c_in for c_in in self.class_info if c_in['name'] == cls]
            if len(cls_id) > 0:
                cls_ids[i] = cls_id[0]['id']

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), cls_ids

    def image_reference(self, image_id: str) -> str:
        """Return the path of the image"""
        info = self.image_info[image_id]
        if info["source"] == "damage":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

