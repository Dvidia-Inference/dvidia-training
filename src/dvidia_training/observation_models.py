# SPDX-License-Identifier: Apache-2.0
"""Offline CPU image observations with pinned OpenCV Zoo ONNX models.

Pre/postprocessing follows the reviewed OpenCV Zoo YOLOX and MediaPipe palm
references at ZOO_REVISION (Apache-2.0; YOLOX copyright Megvii Inc. 2021–2022).
Adaptations copyright DVIDIA contributors 2026. This implementation adds local hash verification, normalized clipped boxes,
bounded deterministic suppression, and explicit palm-only hand semantics.
No downloaded Python code is imported and no network access occurs here.
"""
from __future__ import annotations

from hashlib import sha256
import math
from pathlib import Path


ZOO_REVISION = '47534e27c9851bb1128ccc0102f1145e27f23f98'
ZOO_URL = 'https://github.com/opencv/opencv_zoo'
MODEL_ASSETS = (
    {'name': 'object_detection_yolox_2022nov.onnx',
     'directory': 'object_detection_yolox', 'role': 'objects',
     'bytes': 35858002,
     'license_bytes': 11371,
     'license_sha256': '0ec3668d3274bcf29e8a29e9576d5a2cd96fc78d3c5bec4387355a796e5d9088',
     'sha256': 'c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063'},
    {'name': 'palm_detection_mediapipe_2023feb.onnx',
     'directory': 'palm_detection_mediapipe', 'role': 'palms',
     'bytes': 3905734,
     'license_bytes': 11358,
     'license_sha256': 'cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30',
     'sha256': '78ff51c38496b7fc8b8ebdb6cc8c1abb02fa6c38427c6848254cdaba57fcce7c'},
)
CLASSES = (
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
    'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
    'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush',
)


def model_metadata(asset):
    relative = f"models/{asset['directory']}/{asset['name']}"
    return {**asset, 'revision': ZOO_REVISION, 'license': 'Apache-2.0',
            'source': f'{ZOO_URL}/blob/{ZOO_REVISION}/{relative}',
            'download_url': f'https://media.githubusercontent.com/media/opencv/opencv_zoo/{ZOO_REVISION}/{relative}',
            'license_url': f"https://raw.githubusercontent.com/opencv/opencv_zoo/{ZOO_REVISION}/models/{asset['directory']}/LICENSE"}


def _fraction(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value < 1:
        raise ValueError(f'{name} must be a finite number between zero and one.')
    return float(value)


def _iou(left, right):
    overlap = max(0., min(left[2], right[2]) - max(left[0], right[0])) * max(0., min(left[3], right[3]) - max(left[1], right[1]))
    union = (left[2] - left[0]) * (left[3] - left[1]) + (right[2] - right[0]) * (right[3] - right[1]) - overlap
    return overlap / union if union > 0 else 0.


def _suppress(rows, threshold, limit):
    kept = []
    for row in sorted(rows, key=lambda r: (-r['score'], r['label'], r['xyxy']))[:1000]:
        if any(old['label'] == row['label'] and _iou(old['xyxy'], row['xyxy']) > threshold for old in kept):
            continue
        kept.append(row)
        if len(kept) >= limit:
            break
    return kept


class OpenCVDetector:
    """A frame detector; scores are uncalibrated and hand boxes cover palms.

    Both exact model files must be provisioned separately, unless include_hands
    is false. Construction rejects changed bytes before loading any network.
    Instances are intended for serial use in one worker, with at most four CPU
    threads. detect accepts a local image path and returns normalized xyxy boxes.
    """

    def __init__(self, weights_dir, *, threshold=0.35, hand_threshold=0.5,
                 max_detections=30, include_hands=True):
        self.threshold = _fraction(threshold, 'threshold')
        self.hand_threshold = _fraction(hand_threshold, 'hand_threshold')
        if type(max_detections) is not int or not 1 <= max_detections <= 100:
            raise ValueError('max_detections must be an integer from 1 to 100.')
        if type(include_hands) is not bool:
            raise ValueError('include_hands must be boolean.')
        self.max_detections = max_detections
        root = Path(weights_dir).expanduser().resolve(strict=True)
        assets = MODEL_ASSETS if include_hands else MODEL_ASSETS[:1]
        paths = []
        for asset in assets:
            path = root / asset['name']
            if path.is_symlink() or not path.is_file() or path.stat().st_size != asset['bytes']:
                raise ValueError(f"Missing or invalid pinned local model: {asset['name']}")
            with path.open('rb') as stream:
                digest = sha256()
                for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(chunk)
            if digest.hexdigest() != asset['sha256']:
                raise ValueError(f"Model SHA-256 mismatch: {asset['name']}")
            paths.append(path)
        try:
            import cv2
            import numpy as np
        except ImportError as exc:
            raise RuntimeError('Observation detection requires a separately provisioned OpenCV and NumPy environment.') from exc
        self.cv, self.np = cv2, np
        cv2.setNumThreads(4)
        self.networks = []
        for path in paths:
            net = cv2.dnn.readNetFromONNX(str(path))
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            self.networks.append(net)
        grids, strides = [], []
        for stride in (8, 16, 32):
            size = 640 // stride
            y, x = np.mgrid[:size, :size]
            grid = np.stack((x, y), axis=-1).reshape(-1, 2)
            grids.append(grid)
            strides.append(np.full((len(grid), 1), stride))
        self.grid = np.concatenate(grids).astype(np.float32)
        self.strides = np.concatenate(strides).astype(np.float32)
        # The released palm model has 2 anchors/cell at 24x24 and 6 at 12x12.
        self.palm_anchors = np.asarray([
            ((x + .5) / size, (y + .5) / size)
            for size, repeats in ((24, 2), (12, 6))
            for y in range(size) for x in range(size) for _ in range(repeats)
        ], dtype=np.float32)
        self.metadata = {
            'kind': 'opencv-zoo-yolox-palm', 'implementation_version': 1,
            'implementation_sha256': sha256(Path(__file__).read_bytes()).hexdigest(),
            'models': [model_metadata(a) for a in assets],
            'device': 'cpu', 'backend': 'opencv', 'cpu_threads': cv2.getNumThreads(),
            'runtime': {'opencv': cv2.__version__, 'numpy': np.__version__},
            'threshold': self.threshold, 'hand_threshold': self.hand_threshold,
            'nms_iou': {'objects': .5, 'palms': .3},
            'max_detections': max_detections, 'max_nms_candidates': 1000,
            'hand_detector_enabled': include_hands,
            'score_kind': 'uncalibrated_model_evidence',
            'coordinates': 'normalized_image_xyxy',
            'hand_box_semantics': 'Palm detector rectangle; not full hand segmentation or observed contact.',
            'limitations': [
                'COCO labels can be wrong or absent, especially for small and occluded objects.',
                'Palm boxes do not establish finger contact, grasp, force, depth, task success or robot actions.',
                'Object identity and temporal relationships require a separate tracking stage.',
            ],
        }

    def _row(self, label, score, box, width, height):
        np = self.np
        if not math.isfinite(float(score)) or not 0 <= score <= 1 or not np.isfinite(box).all():
            raise ValueError('Detector produced nonfinite or invalid evidence.')
        normalized = np.clip(np.asarray(box, dtype=np.float64) / [width, height, width, height], 0., 1.).tolist()
        if normalized[2] <= normalized[0] or normalized[3] <= normalized[1]:
            return None
        return {'label': label, 'score': float(score), 'xyxy': normalized}

    def _objects(self, image):
        cv, np = self.cv, self.np
        h, w = image.shape[:2]
        ratio = min(640 / h, 640 / w)
        resized = cv.resize(cv.cvtColor(image, cv.COLOR_BGR2RGB), (max(1, int(w * ratio)), max(1, int(h * ratio))))
        padded = np.full((640, 640, 3), 114., dtype=np.float32)
        padded[:resized.shape[0], :resized.shape[1]] = resized
        net = self.networks[0]
        net.setInput(np.transpose(padded, (2, 0, 1))[None])
        outputs = net.forward(net.getUnconnectedOutLayersNames())
        if len(outputs) != 1 or outputs[0].shape != (1, 8400, 85) or not np.isfinite(outputs[0]).all():
            raise ValueError('Unexpected YOLOX output.')
        data = outputs[0][0]
        scores = data[:, 4:5] * data[:, 5:]
        classes = scores.argmax(axis=1)
        best = scores[np.arange(len(scores)), classes]
        selected = np.flatnonzero(best >= self.threshold)
        rows = []
        with np.errstate(over='raise', invalid='raise'):
            centers = (data[selected, :2] + self.grid[selected]) * self.strides[selected]
            sizes = np.exp(data[selected, 2:4]) * self.strides[selected]
            boxes = np.concatenate((centers - sizes / 2, centers + sizes / 2), axis=1) / ratio
        for index, box in zip(selected, boxes, strict=True):
            row = self._row(CLASSES[int(classes[index])], best[index], box, w, h)
            if row is not None:
                rows.append(row)
        return _suppress(rows, .5, self.max_detections)

    def _palms(self, image):
        cv, np = self.cv, self.np
        h, w = image.shape[:2]
        ratio = min(192 / h, 192 / w)
        nw, nh = max(1, int(w * ratio)), max(1, int(h * ratio))
        left, top = (192 - nw) // 2, (192 - nh) // 2
        resized = cv.resize(cv.cvtColor(image, cv.COLOR_BGR2RGB), (nw, nh))
        padded = np.zeros((192, 192, 3), dtype=np.float32)
        padded[top:top + nh, left:left + nw] = resized / 255.
        net = self.networks[1]
        net.setInput(padded[None])
        outputs = net.forward(net.getUnconnectedOutLayersNames())
        by_shape = {array.shape: array for array in outputs}
        if set(by_shape) != {(1, 2016, 18), (1, 2016, 1)} or any(not np.isfinite(a).all() for a in outputs):
            raise ValueError('Unexpected palm detector output.')
        delta = by_shape[(1, 2016, 18)][0, :, :4]
        logits = by_shape[(1, 2016, 1)][0, :, 0].astype(np.float64)
        scores = 1. / (1. + np.exp(-np.clip(logits, -100., 100.)))
        selected = np.flatnonzero(scores >= self.hand_threshold)
        centers = delta[selected, :2] + self.palm_anchors[selected] * 192
        sizes = delta[selected, 2:4]
        boxes = (np.concatenate((centers - sizes / 2, centers + sizes / 2), axis=1) - [left, top, left, top]) / ratio
        rows = []
        for index, box in zip(selected, boxes, strict=True):
            row = self._row('hand', scores[index], box, w, h)
            if row is not None:
                rows.append(row)
        return _suppress(rows, .3, self.max_detections)

    def detect(self, image_path):
        path = Path(image_path)
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 50 * 1024 * 1024:
            raise ValueError('Expected a bounded local image file.')
        image = self.cv.imread(str(path), self.cv.IMREAD_COLOR)
        if image is None or image.size == 0 or image.shape[0] * image.shape[1] > 16_000_000:
            raise ValueError('Image is undecodable or exceeds 16 million pixels.')
        rows = self._objects(image)
        if len(self.networks) == 2:
            rows.extend(self._palms(image))
        return sorted(rows, key=lambda r: (-r['score'], r['label'], r['xyxy']))[:self.max_detections]
