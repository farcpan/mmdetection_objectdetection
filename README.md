# MMDetection + Object Detection

## 導入

### 前提

`PythonDeepLearning/.venv`仮想環境をWSL2/Ubuntu上で利用する。
ここには既にPyTorch+CUDA 11.8が導入済み。ここに依存関係を追加する。

### 依存関係導入

* `OpenMMLab基盤`導入: 公式が推奨する`mim`を経由する
    ```bash
    pip install openmim
    ```
* mmengine / mmcv / mmdetection
    ```bash
    mim install mmengine
    mim install "mmcv>=2.0.0"
    mim install mmdet
    ```
* 動作確認
    ```bash
    python -c "import mmdet; print(mmdet.__version__)"

    3.3.0
    ```

---

## 設定ファイルの確認

```bash
mim download mmdet --config rtmdet_tiny_8xb32-300e_coco --dest .
mv rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth rtmdet_tiny_8xb32-300e_coco.pth # ファイルをリネーム
```

---

## MMDetectionからdemoデータ入手

[MMDetection Githubリポジトリ](https://github.com/open-mmlab/mmdetection.git)にアクセスし、`demo`フォルダ内一式を取得。
あるいは `git clone https://github.com/open-mmlab/mmdetection.git`して `demo` フォルダのみを本プロジェクトルートに移動させる。

---

## 推論確認

```bash
python demo/image_demo.py demo/demo.jpg rtmdet_tiny_8xb32-300e_coco.py --weights rtmdet_tiny_8xb32-300e_coco.pth --device cuda
python demo/image_demo.py demo/large_image.jpg rtmdet_tiny_8xb32-300e_coco.py --weights rtmdet_tiny_8xb32-300e_coco.pth --device cuda
```

---

## 簡易スクリプト

`demo/image_demo.py`を流用し、簡易的な推論・時間計測を実装（`detect.py`）。CPUとGPUで推論時間比較。

```bash
python detect.py demo/demo.jpg rtmdet_tiny_8xb32-300e_coco.py --weights rtmdet_tiny_8xb32-300e_coco.pth --device cuda
Elapsed Time: 8.457868099212646 [sec]

python detect.py demo/demo.jpg rtmdet_tiny_8xb32-300e_coco.py --weights rtmdet_tiny_8xb32-300e_coco.pth --device cpu
Elapsed Time: 7.4528727531433105 [sec]
```

---
