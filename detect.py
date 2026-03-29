"""
MMDetectionが提供するdemo/image_demo.pyを参考に、画像から物体を検出する簡単なスクリプトを作成。
"""

import ast
import time
from argparse import ArgumentParser
from mmengine.logging import print_log

from mmdet.apis import DetInferencer
from mmdet.evaluation import get_classes


def parse_args():
    parser = ArgumentParser()
    # 必須パラメータ1: 入力画像ファイルパス
    parser.add_argument('inputs', type=str, help='Input image file or folder path.')
    # 必須パラメータ2: MMDet設定ファイル。事前に`mim download mmdet`で入手したファイル（.pyファイル）
    parser.add_argument(
        'model',
        type=str,
        help='Config or checkpoint .pth file or the model name '
        'and alias defined in metafile. The model configuration '
        'file will try to read from .pth if the parameter is '
        'a .pth weights file.')
    # モデルチェックポイントファイル（.pth）
    parser.add_argument('--weights', default=None, help='Checkpoint file')
    # cpu or cuda
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')

    parser.add_argument(
        '--palette',
        default='none',
        choices=['coco', 'voc', 'citys', 'random', 'none'],
        help='Color palette used for visualization')
    parser.add_argument(
        '--chunked-size',
        '-s',
        type=int,
        default=-1,
        help='If the number of categories is very large, '
        'you can specify this parameter to truncate multiple predictions.')
    
    call_args = vars(parser.parse_args())
    init_kws = ['model', 'weights', 'device', 'palette']
    init_args = {}
    for init_kw in init_kws:
        init_args[init_kw] = call_args.pop(init_kw)

    return init_args, call_args


def main():
    """
    python detect.py demo/demo.jpg rtmdet_tiny_8xb32-300e_coco.py --weights rtmdet_tiny_8xb32-300e_coco.pth --device cuda
    """
    init_args, call_args = parse_args()
    start = time.time()
    inferencer = DetInferencer(**init_args)
    chunked_size = call_args.pop('chunked_size')
    inferencer.model.test_cfg.chunked_size = chunked_size

    inferencer(**call_args)
    print(f"Elapsed Time: {time.time() - start} [sec]")


if __name__ == '__main__':
    main()