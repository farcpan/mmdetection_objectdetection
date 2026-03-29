# ONNX変換手順

```bash
pip install mmdeploy
pip install onnx onnxruntime onnxsim
```

[デプロイ用のスクリプト](https://github.com/open-mmlab/mmdeploy/blob/main/tools/deploy.py)を入手して保存する。

---

```bash
python deploy.py \
  mmdeploy/configs/mmdet/detection/detection_onnxruntime_static.py \
  rtmdet_tiny_8xb32-300e_coco.py \
  rtmdet_tiny_8xb32-300e_coco.pth \
  demo/demo.jpg \
  --work-dir ./work_dir \
  --device cpu \
  --dump-info
```