# -*- coding: utf-8 -*-
"""
Mask R-CNN package utils

Written by Yuan Ji
"""

import cv2 as cv
import numpy as np
from typing import List
from typing import Tuple
from mrcnn.visualize import random_colors, apply_mask
from skimage.measure import find_contours


def color_splash(image: np.ndarray, boxes: np.ndarray, masks: np.ndarray,
                 class_ids: np.ndarray, class_names: List[str],
                 scores: np.ndarray, show_mask: bool=True, show_bbox: bool=True,
                 colors: List[Tuple[float]]=None) -> np.ndarray:
    """Paint color for detected objects

    Args:
        image: image to paint color on
        boxes: matrix of bounding boxes
        masks: matrix of boolean to indicate if the pixel is a part of detected
            objects
        class_ids: class id
        class_names: class names excluding background, in the same order of
            ``class_ids``
        scores: matrix of detection scores
        show_mask: whether to paint masks on image
        show_bbox: whether to paint bounding boxes on image
        colors: colors to be used for painting

    Returns:
        masked_image: image with painting
    """
    # Number of instances
    N = boxes.shape[0]
    if not N:
        return image
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    # Generate random colors
    if colors is None:
        colors = random_colors(len(class_names))

    masked_image = image.astype(np.uint8).copy()
    for i in range(N):
        # class id
        class_id = class_ids[i]

        # get color of class
        colors_rgb = colors[class_id-1]
        color_bgr = colors_rgb[::-1]
        color_bgr_255 = tuple(round(255 * x) for x in color_bgr)

        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        y1, x1, y2, x2 = boxes[i]
        if show_bbox:
            masked_image = cv.rectangle(masked_image, (x1, y1), (x2, y2),
                                        color_bgr_255, 2)

        # Mask
        mask = masks[:, :, i]
        if show_mask:
            masked_image = apply_mask(masked_image, mask, color_bgr)

        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        padded_mask = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2),
                               dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)
        for verts in contours:
            # Subtract the padding and flip (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            masked_image = cv.polylines(masked_image, np.int32([verts]), True,
                                        color_bgr_255, 3)

        # Label
        score = scores[i] if scores is not None else None
        label = class_names[class_id-1]  # ids include bg but not in names
        caption = "{} {:.3f}".format(label, score) if score else label
        cv.putText(masked_image, caption, (x1, y2), cv.FONT_HERSHEY_PLAIN, 1.5,
                   (255, 255, 255), 2, cv.LINE_4)
    return masked_image

