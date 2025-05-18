import os
import time
import json
import io
import cv2
import numpy as np
import redis
import traceback
from minio import Minio

# Env config
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
BUCKET_NAME = "images"
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
QUEUE_NAME = "objectdetect-queue"
NEXT_QUEUE = "tag-queue"

# For eliminating results with low confidence in obectdetect functions
CONFIDENCE_MIN = 0.4
# Initialize the list of class labels MobileNet SSD was trained to detect
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle",
           "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
           "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

# Load DNN
print("[INFO] Loading model...")
try:
    net = cv2.dnn.readNetFromCaffe(
        "./MobileNetSSD_deploy.prototxt",
        "./MobileNetSSD_deploy.caffemodel"
    )
    print("[INFO] Model loaded successfully.")
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")
    traceback.print_exc()
    exit(1)

# Clients
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
minio_client = Minio(
    MINIO_ENDPOINT,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=False
)

def detect_objects(data, origin_w, origin_h):
    arr = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("[ERROR] Failed to decode image data from MinIO")

    (h, w) = image.shape[:2]

    scale_x = origin_w / w
    scale_y = origin_h / h

    blob = cv2.dnn.blobFromImage(image, 0.007843, (image.shape[1], image.shape[0]), 127.5)

    net.setInput(blob)
    detections = net.forward()

    labels_and_coords = []
    for i in range(detections.shape[2]):
        confidence = float(detections[0, 0, i, 2])
        if confidence > CONFIDENCE_MIN:
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Scale to original dimensions
            startX = int(startX * scale_x)
            endX = int(endX * scale_x)
            startY = int(startY * scale_y)
            endY = int(endY * scale_y)

            labels_and_coords.append({
                "startX": startX,
                "startY": startY,
                "endX": endX,
                "endY": endY,
                "label": {
                    "name": CLASSES[idx],
                    "index": idx
                },
                "confidence": confidence
            })

    return labels_and_coords

print("[INFO] ObjectDetect service started. Listening for jobs...")

while True:
    task = redis_client.rpop(QUEUE_NAME)
    if not task:
        time.sleep(1)
        continue

    try:
        job = json.loads(task)
        print(f"[INFO] Received job: {job}")

        filename = job.get("filename")
        origin_h = job.get("origin_h")
        origin_w = job.get("origin_w")

        if not filename or origin_h is None or origin_w is None:
            print("[WARNING] Job missing required fields. Skipping:", job)
            continue

        print(f"[INFO] Detecting objects in: {filename}")
        data = minio_client.get_object(BUCKET_NAME, filename).read()
        detections = detect_objects(data, origin_w, origin_h)

        def sanitize(obj):
            if isinstance(obj, dict):
                return {k: sanitize(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [sanitize(v) for v in obj]
            elif isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            else:
                return obj

        job["detections"] = sanitize(detections)
        job["origin_filename"] = job.get("origin_filename")
        redis_client.lpush(NEXT_QUEUE, json.dumps(job))
        print(f"[INFO] Detected {len(detections)} objects, pushed to {NEXT_QUEUE}")

    except Exception as e:
        print(f"[ERROR] Exception during object detection: {e}")
        traceback.print_exc()
        continue
