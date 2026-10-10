# Observation model provenance

Weights are downloaded explicitly into ignored local storage, never committed.
The observation adapter's pre/postprocessing is adapted from OpenCV Zoo; that
file is Apache-2.0 and retains upstream attribution. The rest of this repository
retains its existing license. Original licenses are included in
[`third_party/observation-models`](../third_party/observation-models/).

Pinned upstream revision: `47534e27c9851bb1128ccc0102f1145e27f23f98`.

| Role | Exact artifact | Bytes | SHA-256 |
| --- | --- | ---: | --- |
| COCO object detection | `object_detection_yolox_2022nov.onnx` | 35,858,002 | `c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063` |
| Palm detection | `palm_detection_mediapipe_2023feb.onnx` | 3,905,734 | `78ff51c38496b7fc8b8ebdb6cc8c1abb02fa6c38427c6848254cdaba57fcce7c` |

The official [YOLOX directory](https://github.com/opencv/opencv_zoo/tree/47534e27c9851bb1128ccc0102f1145e27f23f98/models/object_detection_yolox)
and [MediaPipe palm directory](https://github.com/opencv/opencv_zoo/tree/47534e27c9851bb1128ccc0102f1145e27f23f98/models/palm_detection_mediapipe)
each state that all files in the directory are licensed under Apache 2.0.
The downloaded license text is also pinned and hash-checked by the downloader.
YOLOX's included license attributes Megvii Inc., 2021–2022.

DVIDIA modifications include local weight verification, bounded deterministic
suppression, clipped normalized image boxes, explicit palm-only semantics,
metadata, runtime limits and validation of model output. No remote Python code
is downloaded or executed. Source data licensing remains independent from model
licensing. These license records do not establish rights to any user's footage.

No SpatialLM, MANO, depth model or language model weights are part of this pilot.
The exact detector recipes and runtime versions are recorded in each run. The
first reports preserve the adapter implementation hash as it existed at run time;
later source or notice changes do not retroactively change those receipts.
