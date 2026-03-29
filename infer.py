import cv2
import numpy as np
import onnxruntime as ort


# ① 画像読み込み
img = cv2.imread("demo/demo.jpg")
orig_img = img.copy()

# ② リサイズ（重要）
img = cv2.resize(img, (640, 640))

# ③ BGR → RGB（重要）
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# ④ float32化 & 正規化
img = img.astype(np.float32) / 255.0

# ⑤ CHW変換
img = np.transpose(img, (2, 0, 1))

# ⑥ バッチ次元追加
input_tensor = np.expand_dims(img, axis=0)

print(input_tensor.shape)  # (1, 3, 640, 640)

sess = ort.InferenceSession("work_dir/rtmdet.onnx")
print(sess.get_outputs())

outputs = sess.run(None, {"input": input_tensor})
print(len(outputs))

# 🎯 描画用（リサイズ後画像に描く）
draw_img = cv2.resize(orig_img, (640, 640))

dets, labels = outputs
for box, label in zip(dets[0], labels[0]):
    x1, y1, x2, y2, score = box

    if score < 0.5:
        continue
    #print(box)
    #print(label)

    print(f"class={label}, score={score}, box={box}")
    cv2.rectangle(draw_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)


# 表示
cv2.imshow("RTMDet Result", draw_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
