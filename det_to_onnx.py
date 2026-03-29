onnx_config = dict(
    type='onnx',
    export_params=True,
    opset_version=17,
    save_file='rtmdet.onnx',

    # 👇 これ必須
    input_names=['input'],

    # 👇 これも必要（安全）
    output_names=['output']
)

codebase_config = dict(
    type='mmdet',
    task='ObjectDetection',
    post_processing=dict(
        score_threshold=0.05,
        iou_threshold=0.5,
        max_output_boxes_per_class=200,
        pre_top_k=5000,
        keep_top_k=100,
        background_label_id=-1,
    )
)

backend_config = dict(
    type='onnxruntime'
)