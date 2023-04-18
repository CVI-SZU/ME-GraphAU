import numpy as np

from .SCRFD_ONNX import SCRFD_ONNX


class ExpandBbox:
    def __init__(self, size_expand=1.55):
        self.size_expand = size_expand

    def _check_bbox(self, bbox, img):
        """
        Make sure all coordinates are valid
        """
        h, w = img.shape[:2]
        bbox[0] = 0 if bbox[0] < 0 else bbox[0]
        bbox[1] = 0 if bbox[1] < 0 else bbox[1]
        bbox[2] = w if bbox[2] > w else bbox[2]
        bbox[3] = h if bbox[3] > h else bbox[3]
        return bbox

    def _transform_to_square_bbox(self, bbox, img):
        left, top, right, bottom = bbox[:4]
        old_size = (right - left + bottom - top) / 2
        center_x = right - (right - left) / 2.0
        center_y = bottom - (bottom - top) / 2.0 + old_size * -0.15
        size = int(old_size * self.size_expand)
        roi_box = [0] * 4
        roi_box[0] = center_x - size / 2
        roi_box[1] = center_y - size / 2
        roi_box[2] = roi_box[0] + size
        roi_box[3] = roi_box[1] + size
        roi_box = self._check_bbox(roi_box, img)
        return np.uint32(roi_box)

    def __call__(self, img, bboxes):
        bboxes_sizes = [(bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) for bbox in bboxes]
        biggest_bbox = bboxes[np.argmax(bboxes_sizes)][:4]
        square_bbox = self._transform_to_square_bbox(biggest_bbox, img)
        return square_bbox
