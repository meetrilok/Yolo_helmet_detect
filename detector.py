
from pathlib import Path
import math, cv2, numpy as np, onnxruntime as ort


# ── better letter-box (exact 640×640) ─────────────────────
def letterbox(img, new_shape=640, color=(114, 114, 114)):
    h, w = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / h, new_shape[1] / w)
    new_unpad = (int(round(w * r)), int(round(h * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw, dh = dw / 2, dh / 2                       # float halves

    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(math.floor(dh)), int(math.ceil(dh))
    left, right = int(math.floor(dw)), int(math.ceil(dw))
    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=color)
    return img, r, left, top                      # pad x, pad y


class YOLOOnnxDetector:
    def __init__(self, onnx_path: str,
                 conf_thres: float = 0.50,        # ← tighter threshold
                 iou_thres:  float = 0.45,
                 providers: tuple = ("CPUExecutionProvider",)):
        self.session  = ort.InferenceSession(str(Path(onnx_path)),
                                             providers=list(providers))
        self.conf_th  = conf_thres
        self.iou_th   = iou_thres
        self.inp_name = self.session.get_inputs()[0].name
        self.out_name = self.session.get_outputs()[0].name

    def __call__(self, img_bgr):
        blob, r, pad_x, pad_y = self._pre(blob_in=img_bgr)
        preds = self.session.run([self.out_name], {self.inp_name: blob})[0][0]

        boxes, scores = preds[:, :4], preds[:, 4]
        m = scores >= self.conf_th
        boxes, scores = boxes[m], scores[m]

        if boxes.size == 0:
            return np.empty((0, 4), dtype=int), np.array([])

        # de-scale & clip
        boxes[:, [0, 2]] -= pad_x
        boxes[:, [1, 3]] -= pad_y
        boxes /= r
        h, w = img_bgr.shape[:2]
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)

        keep = self._nms(boxes, scores, self.iou_th)
        return boxes[keep].astype(int), scores[keep]

    # internal helpers -------------------------------------------------
    def _pre(self, blob_in):
        img_rgb = cv2.cvtColor(blob_in, cv2.COLOR_BGR2RGB)
        return letterbox(img_rgb, 640)[:]

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
