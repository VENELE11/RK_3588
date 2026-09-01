import cv2
import numpy as np
from rknn.api import RKNN

IMG_SIZE = 640
OBJ_THRESH = 0.25
NMS_THRESH = 0.45

CLASSES = ("person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
           "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
           "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
           "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
           "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
           "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
           "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
           "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop",
           "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
           "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush")

anchors = [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45], [59, 119], [116, 90], [156, 198], [373, 326]]
anchor_masks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def filter_boxes(boxes, confidences, class_probs, obj_thresh):
    scores = confidences * np.max(class_probs, axis=1)
    mask = scores >= obj_thresh
    return boxes[mask], scores[mask], np.argmax(class_probs[mask], axis=1)

def nms(boxes, scores, iou_thresh):
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= iou_thresh)[0]
        order = order[inds + 1]
    return keep

def postprocess(outputs, img_shape):
    # 每个输出形如 (1, N, 85): [cx, cy, w, h, obj, class0...class79]
    pred = np.concatenate(
        [out.reshape(-1, out.shape[-1]) for out in outputs], axis=0
    )

    if pred.shape[1] != 85:
        raise ValueError(f"Expected output (..., 85), got {pred.shape}")

    # 防御性过滤，避免异常数值传递到 NMS/绘图
    pred = pred[np.isfinite(pred).all(axis=1)]
    if len(pred) == 0:
        return [], [], []

    boxes_xywh = pred[:, :4]
    obj = pred[:, 4]
    class_probs = pred[:, 5:85]

    classes = np.argmax(class_probs, axis=1)
    scores = obj * class_probs[np.arange(len(pred)), classes]
    keep = scores >= OBJ_THRESH

    boxes_xywh = boxes_xywh[keep]
    scores = scores[keep]
    classes = classes[keep]
    if len(boxes_xywh) == 0:
        return [], [], []

    # cx, cy, w, h -> x1, y1, x2, y2（坐标仍为 640×640 输入尺度）
    boxes = np.empty_like(boxes_xywh)
    boxes[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
    boxes[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
    boxes[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
    boxes[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2

    keep = nms(boxes, scores, NMS_THRESH)
    boxes, scores, classes = boxes[keep], scores[keep], classes[keep]

    h, w = img_shape
    boxes[:, [0, 2]] *= w / IMG_SIZE
    boxes[:, [1, 3]] *= h / IMG_SIZE
    return boxes, scores, classes

if __name__ == "__main__":
    rknn = RKNN()
    rknn.load_rknn('model/yolov5s.rknn')
    rknn.init_runtime(target='rk3588')
    
    img_path = 'model/data/images/bus.jpg'
    img_src = cv2.imread(img_path)
    if img_src is None:
        print("Failed to load image")
        exit(1)
    h, w = img_src.shape[:2]
    
    # 预处理
    img_resized = cv2.resize(img_src, (IMG_SIZE, IMG_SIZE))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_input = np.expand_dims(img_rgb, 0).astype(np.float32)
    
    outputs = rknn.inference(inputs=[img_input], data_format='nhwc')
    
    boxes, scores, classes = postprocess(outputs, (h, w))
    
    for box, score, cls_id in zip(boxes, scores, classes):
        x1, y1, x2, y2 = [int(b) for b in box]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        label = f"{CLASSES[cls_id]} {score:.2f}"
        cv2.rectangle(img_src, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_src, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        print(f"Detected: {label} at ({x1},{y1},{x2},{y2})")
    
    cv2.imwrite('result_rknn.jpg', img_src)
    print("✅ 结果已保存为 result_rknn.jpg")
    rknn.release()