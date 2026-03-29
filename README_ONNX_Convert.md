# ONNX変換手順

```bash
pip install mmdeploy
pip install onnx onnxruntime onnxsim
```

[デプロイ用のスクリプト](https://github.com/open-mmlab/mmdeploy/blob/main/tools/deploy.py)を入手して保存する。

---

```bash
python deploy.py \
  mmdeploy/configs/mmdet/detection/detection_onnxruntime_dynamic.py \
  rtmdet_tiny_8xb32-300e_coco.py \
  rtmdet_tiny_8xb32-300e_coco.pth \
  demo/demo.jpg \
  --work-dir ./work_dir \
  --device cpu \
  --dump-info
```

---

## 設定ファイル修正

`det_to_onnx.py`を用意して実行。

python deploy.py \
  det_to_onnx.py \
  rtmdet_tiny_8xb32-300e_coco.py \
  rtmdet_tiny_8xb32-300e_coco.pth \
  demo/demo.jpg \
  --work-dir ./work_dir \
  --device cpu \
  --dump-info

上記では、ONNXファイルの出力は成功する。ただし、ONNX Runtimeによる可視化処理に失敗する。これはONNX変換そのものには影響しないので無視する。
`work_dir`フォルダに生成された `.onnx`ファイルを利用して推論すればよい。

---
