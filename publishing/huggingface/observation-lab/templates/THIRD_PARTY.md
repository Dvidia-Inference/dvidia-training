# License and model boundaries

This static Space includes DVIDIA's MIT review interface, its client-only demo
adapter and DVIDIA-authored schematic videos/annotations. It contains **no
pretrained weights or upstream detector implementation**. No new DVIDIA-trained
weights were created for the observation pilot.

The separate installable local tool adapts preprocessing/postprocessing from
[OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/47534e27c9851bb1128ccc0102f1145e27f23f98).
That adapter and the named YOLOX/MediaPipe-palm artifacts retain Apache 2.0 terms.
The local source retains its upstream attribution; this Space preserves the
original license texts under `licenses/` as references rather than relicensing
weights. The YOLOX license attributes Megvii Inc., 2021–2022.

Model provisioning, exact byte/hash pins and upstream source links are documented
in the public training repository's `docs/observation-models.md` and downloader.
Model licenses do not grant rights to any person's footage. This synthetic demo
contains no human footage, faces, private source recordings or private metadata.
