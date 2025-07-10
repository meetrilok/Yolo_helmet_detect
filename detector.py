"""
YOLO-ONNX helmet detector (single class) – tuned thresholds.

• 0.45 confidence cutoff
• Drops boxes smaller than 0.2 % of the frame
• Letter-box preprocessing (640×640) and CHW float32 tensor
"""

import math
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

# ─── Tuned thresholds ────────────────────────────────────────────
CONF_THRES   = 0.50       # score ≥ 0.45
MIN_BOX_FRAC = 0.002      # area ≥ 0.2 % of frame
IOU_THRES    = 0.45
IMG_SIZE     = 640

# ─── Letter-box helper ───────────────────────────────────────────
def letterbox(img, new_shape=IMG_SIZE, color=(114, 114, 114)):
    h, w = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / h, new_shape[1] / w)
    new_unpad = (int(round(w * r)), int(round(h * r)))

    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw, dh = dw / 2, dh / 2

    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(math.floor(dh)), int(math.ceil(dh))
    left, right = int(math.floor(dw)), int(math.ceil(dw))
    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=color)
    return img, r, left, top


# ─── Detector class ──────────────────────────────────────────────
class YOLOOnnxDetector:
    def __init__(self,
                 onnx_path: str = "model/best.onnx",
                 providers: tuple = ("CPUExecutionProvider",)):
        self.session  = ort.InferenceSession(str(Path(onnx_path)),
                                             providers=list(providers))
        self.inp_name = self.session.get_inputs()[0].name
        self.out_name = self.session.get_outputs()[0].name

    # ── inference ────────────────────────────────────────────────
    def __call__(self, img_bgr):
        blob, r, pad_x, pad_y = self._preprocess(img_bgr)
        preds = self.session.run(
            [self.out_name], {self.inp_name: blob})[0][0]

        boxes, scores = preds[:, :4], preds[:, 4]

        # 1) confidence filter
        m = scores >= CONF_THRES
        boxes, scores = boxes[m], scores[m]
        if boxes.size == 0:
            return np.empty((0, 4), dtype=int), np.array([])

        # 2) undo letter-box & clip
        boxes[:, [0, 2]] -= pad_x
        boxes[:, [1, 3]] -= pad_y
        boxes /= r
        h, w = img_bgr.shape[:2]
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)

        # 3) min-area filter
        area_thr = MIN_BOX_FRAC * h * w
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        m = areas >= area_thr
        boxes, scores = boxes[m], scores[m]
        if boxes.size == 0:
            return np.empty((0, 4), dtype=int), np.array([])

        # 4) NMS
        keep = self._nms(boxes, scores, IOU_THRES)
        return boxes[keep].astype(int), scores[keep]

    # ── helpers ──────────────────────────────────────────────────
    @staticmethod
    def _preprocess(img_bgr):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_ltr, r, pad_x, pad_y = letterbox(img_rgb, IMG_SIZE)

        tensor = img_ltr.transpose(2, 0, 1).astype(np.float32)
        tensor /= 255.0
        tensor = tensor[None]  # 1×3×640×640
        return tensor, r, pad_x, pad_y

    @staticmethod
    def _iou(box, boxes):
        x1 = np.maximum(box[0], boxes[:, 0])
        y1 = np.maximum(box[1], boxes[:, 1])
        x2 = np.minimum(box[2], boxes[:, 2])
        y2 = np.minimum(box[3], boxes[:, 3])
        inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        area1 = (box[2] - box[0]) * (box[3] - box[1])
        area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        return inter / (area1 + area2 - inter + 1e-7)

    def _nms(self, boxes, scores, thr):
        idxs = scores.argsort()[::-1]
        keep = []
        while idxs.size:
            i = idxs[0]
            keep.append(i)
            if idxs.size == 1:
                break
            ious = self._iou(boxes[i], boxes[idxs[1:]])
            idxs = idxs[1:][ious < thr]
        return keep
