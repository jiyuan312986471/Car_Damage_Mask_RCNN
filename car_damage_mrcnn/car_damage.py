# -*- coding: utf-8 -*-
"""
Mask R-CNN
Configurations and data loading code for car damage dataset.

Written by Yuan Ji

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 car_damage.py train --dataset=/path/to/car_damage/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 car_damage.py train --dataset=/path/to/car_damage/dataset --weights=last

    # Apply color splash to an image
    python3 car_damage.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python3 car_damage.py splash --weights=last --video=<URL or path to file>
"""

import datetime
import json
import os
import sys
import numpy as np
import skimage
from random import sample
from typing import List
from typing import Tuple

# Directories
ROOT_DIR = os.path.abspath("../")
DATASET_DIR = os.path.join(ROOT_DIR, "dataset\\img")
ANNO_PATH = os.path.join(ROOT_DIR, "dataset\\annotation\\all.json")

# File prefixes
PREFIXES = ['car', 'rayure', 'retro']

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "models\\mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")


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

    def load_car_damage(self, dataset_dir: str, anno_path: str,
                        fnames: List[str]=None) -> None:
        """Load car damage dataset.

        Args:
            dataset_dir: directory of the dataset
            anno_path: path of annotation file
            fnames: file names of images to load
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
        if fnames is not None:
            annotations = [d for d in annotations if d['filename'] in fnames]

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


def train_validation_split_subset(
        dataset_imgs: List[str],
        train_size: int or float=0.8) -> Tuple[List[str], List[str]]:
    """Split a subset of images files into train and validation set

    Args:
        dataset_imgs: list of image file names
        train_size: size or proportion of training set. If ``float``, should be
            between 0.0(exclude) and 1.0(include). If ``int``, should be between
            0(exclude) and ``len(dataset_imgs)``(include).

    Returns:
        train: training set
        val: validation set
    """
    n_imgs = len(dataset_imgs)

    # check train size
    if train_size % 1 > 0:
        assert 0 < train_size <= 1
    else:
        assert 0 < train_size <= n_imgs

    # compute train size
    if train_size < 1:
        train_size *= n_imgs
        if train_size < n_imgs - 1:
            train_size = int(round(train_size))
        else:
            train_size = int(train_size)  # keep at least 1 instance in val set

    # sample train and val
    idx = set(range(n_imgs))
    train_idx = set(sample(idx, train_size))
    val_idx = idx - train_idx

    # get train and val
    train = [dataset_imgs[i] for i in train_idx]
    val = [dataset_imgs[i] for i in val_idx]
    return train, val


def train_validation_split_all(
        dataset_imgs: List[str], train_size: int or float=0.8,
        prefixes: List[str]=None) -> Tuple[List[str], List[str]]:
    """Split image files into train and validation set

    Args:
        dataset_imgs: list of image file names
        train_size: size or proportion of training set. If ``float``, should be
            between 0.0(exclude) and 1.0(include). If ``int``, should be between
            0(exclude) and ``len(dataset_imgs)``(include).
        prefixes: image file name prefixes, each prefix represents a subset. A
            split will be performed on each prefix.

    Returns:
        train: training set
        val: validation set
    """
    if prefixes is None:
        return train_validation_split_subset(dataset_imgs,
                                             train_size=train_size)

    train, val = [], []
    for pre in prefixes:
        subset = [f for f in dataset_imgs if f.startswith(pre)]
        if len(subset) > 0:
            sub_train, sub_val = \
                train_validation_split_subset(subset, train_size=train_size)
            train.extend(sub_train)
            val.extend(sub_val)
    return train, val


def train_model(model: modellib.MaskRCNN, training_set: List[str],
                validation_set: List[str]) -> None:
    """Train Mask-RCNN model

    Args:
        model: Mask RCNN model to train
        training_set: image file names in training set
        validation_set: image file names in validation set
    """
    # Training dataset.
    dataset_train = CarDamageDataset()
    dataset_train.load_car_damage(args.dataset, args.annotation,
                                  fnames=training_set)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = CarDamageDataset()
    dataset_val.load_car_damage(args.dataset, args.annotation,
                                fnames=validation_set)
    dataset_val.prepare()

    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=10,
                layers='heads')
    return None


def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]
    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # We're treating all instances as one, so collapse the mask into one layer
    mask = (np.sum(mask, -1, keepdims=True) >= 1)
    # Copy color pixels from the original color image where mask is set
    if mask.shape[0] > 0:
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray
    return splash


def detect_and_color_splash(model, image_path=None, video_path=None):
    assert image_path or video_path

    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(args.image))
        # Read image
        image = skimage.io.imread(args.image)
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        # Color splash
        splash = color_splash(image, r['masks'])
        # Save output
        file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        skimage.io.imsave(file_name, splash)
    elif video_path:
        import cv2
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))

        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                # Color splash
                splash = color_splash(image, r['masks'])
                # RGB -> BGR to save image to video
                splash = splash[..., ::-1]
                # Add image to video writer
                vwriter.write(splash)
                count += 1
        vwriter.release()
    print("Saved to ", file_name)


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect custom class.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        default=DATASET_DIR,
                        metavar="/path/to/custom/dataset/",
                        help='Directory of the custom dataset')
    parser.add_argument('--annotation', required=False,
                        default=ANNO_PATH,
                        metavar="/path/to/annotaion/file.json",
                        help='Path of the annotation file')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file, or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "splash":
        assert args.image or args.video,\
               "Provide --image or --video to apply color splash"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = CarDamageConfig()
    else:
        class InferenceConfig(CarDamageConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()[1]
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train validation split
    imgs = os.listdir(DATASET_DIR)
    train, val = train_validation_split_all(imgs, train_size=0.8,
                                            prefixes=PREFIXES)

    # Train or evaluate
    if args.command == "train":
        train_model(model, train, val)
    elif args.command == "splash":
        detect_and_color_splash(model, image_path=args.image,
                                video_path=args.video)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))
