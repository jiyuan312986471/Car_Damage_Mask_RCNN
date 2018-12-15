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

import json
import os
import sys
import cv2 as cv
import numpy as np
import skimage
import tensorflow as tf
from random import sample
from typing import List
from typing import Tuple
from mrcnn import model as modellib
from mrcnn import utils
from mrcnn.config import Config
from mrcnn.visualize import random_colors
from car_damage_mrcnn.utils import color_splash

# Directories
ROOT_DIR = os.path.abspath("../")

# File prefixes
PREFIXES = ['rayure', 'car', 'retro']

# Classes
CLS = ["rayure", "roue", "retro"]

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library

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
        for i, cls in enumerate(CLS):
            self.add_class("damage", i+1, cls)

        # Load annotations
        with open(anno_path) as f:
            annotations = json.load(f)
        if type(annotations) == dict:
            annotations = [anno for anno in annotations.values()]

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
        dataset_imgs: image file names
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


def train_model(dataset_dir: str, anno_path: str, model: modellib.MaskRCNN,
                training_set: List[str], validation_set: List[str]) -> None:
    """Train Mask-RCNN model

    Args:
        dataset_dir: data set directory
        anno_path: path of annotation file
        model: Mask RCNN model to train
        training_set: image file names in training set
        validation_set: image file names in validation set
    """
    # Training dataset.
    dataset_train = CarDamageDataset()
    dataset_train.load_car_damage(dataset_dir, anno_path, fnames=training_set)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = CarDamageDataset()
    dataset_val.load_car_damage(dataset_dir, anno_path, fnames=validation_set)
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


def detect_and_color_splash(model: modellib.MaskRCNN, image_path: str=None,
                            video_path: str=None, out_dir: str=None) -> None:
    """Detect objects in image/video and highlight them

    Args:
        model: trained model
        image_path: path of image to be detected
        video_path: path of video to be detected
        out_dir: output directory
    """
    assert image_path or video_path

    # Get original file name
    if image_path:
        fname = os.path.basename(image_path)
        fname = os.path.splitext(fname)[0]
    elif video_path:
        fname = os.path.basename(video_path)
        fname = os.path.splitext(fname)[0]
    else:
        fname = ''
    fname += '_splash'

    # Generate output path
    path = os.path.join(out_dir, fname)

    # Generate colors for classes
    colors = random_colors(len(CLS))

    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(image_path))
        # Read image
        image = skimage.io.imread(image_path)
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        # Color splash
        splash = color_splash(image, r['rois'], r['masks'], r['class_ids'], CLS,
                              r['scores'], colors=colors)
        # Save output
        path += ".png"
        skimage.io.imsave(path, splash)
        print("Saved to", path)
    elif video_path:
        print('Splash video:', video_path)
        # Video capture
        vcapture = cv.VideoCapture(video_path)
        width = int(vcapture.get(cv.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv.CAP_PROP_FPS)
        print('Video size: ({}, {})'.format(width, height))
        print('FPS:', fps)

        # Define codec and create video writer
        path += ".avi"
        vwriter = cv.VideoWriter(path, cv.VideoWriter_fourcc(*'MJPG'), fps,
                                 (width, height))

        count = 0
        success = True
        while success:
            print("frame:", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                # Color splash
                splash = color_splash(image, r['rois'], r['masks'],
                                      r['class_ids'], CLS, r['scores'],
                                      colors=colors)
                # RGB -> BGR to save image to video
                splash = splash[..., ::-1]
                # Add image to video writer
                vwriter.write(splash)
                count += 1
        vwriter.release()
        print("Saved to", path)
    return None


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect custom class.')
    parser.add_argument("command", metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/custom/dataset/",
                        help='Directory of the dataset')
    parser.add_argument('--annotation', required=False,
                        metavar="/path/to/annotaion/file.json",
                        help='Path of the annotation file')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file, or 'coco'")
    parser.add_argument('--logs', required=False, default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    parser.add_argument('--dir-out', required=False,
                        metavar="/path/to/output/dir/",
                        help='Directory to save splashed image/video')
    parser.add_argument('--gpu', required=False, default=False,
                        action='store_true', help='Whether to use GPU')
    args = parser.parse_args()

    # Validate arguments
    assert args.command in ["train", "splash"], \
        "'{}' is not recognized. Use 'train' or 'splash'".format(args.command)
    if args.command == "train":
        assert args.dataset and args.annotation, \
            "Argument --dataset and --annotation is required for training"
    else:  # splash
        assert args.image or args.video, \
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
        if args.gpu:
            with tf.device("/gpu:0"):
                model = modellib.MaskRCNN(mode="training", config=config,
                                          model_dir=args.logs)
        else:
            model = modellib.MaskRCNN(mode="training", config=config,
                                      model_dir=args.logs)
    else:
        if args.gpu:
            with tf.device("/gpu:0"):
                model = modellib.MaskRCNN(mode="inference", config=config,
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
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        if os.path.isabs(args.weights):
            weights_path = args.weights
        else:
            weights_path = os.path.join(ROOT_DIR, args.weights)

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

    # Assume absolute paths
    if args.dataset is not None:
        dataset_dir = args.dataset if os.path.isabs(args.dataset or '') \
            else os.path.join(ROOT_DIR, args.dataset)
    else:
        dataset_dir = None

    if args.annotation is not None:
        anno_path = args.annotation if os.path.isabs(args.annotation or '') \
            else os.path.join(ROOT_DIR, args.annotation)
    else:
        anno_path = None

    if args.image is not None:
        img_path = args.image if os.path.isabs(args.image or '') \
            else os.path.join(ROOT_DIR, args.image)
    else:
        img_path = None

    if args.video is not None:
        video_path = args.video if os.path.isabs(args.video or '') \
            else os.path.join(ROOT_DIR, args.video)
    else:
        video_path = None

    if args.dir_out is not None:
        dir_out = args.dir_out if os.path.isabs(args.dir_out or '') \
            else os.path.join(ROOT_DIR, args.dir_out)
    else:
        dir_out = None

    # Train or evaluate
    if args.command == "train":
        # train val split
        imgs = os.listdir(dataset_dir)
        train, val = train_validation_split_all(imgs, train_size=0.8,
                                                prefixes=PREFIXES)
        # train model
        train_model(dataset_dir, anno_path, model, train, val)
    else:  # splash
        detect_and_color_splash(model, image_path=img_path,
                                video_path=video_path, out_dir=dir_out)
