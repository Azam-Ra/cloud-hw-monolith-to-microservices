#!/usr/bin/env python3
import os
import time
import json
import io
import cv2
import numpy as np
import redis
from minio import Minio

# Environment Variables
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
BUCKET_NAME = "images"
QUEUE_NAME = "tag-queue"

# Init clients
print("[INIT] Connecting to Redis...")
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
print("[INIT] Connecting to MinIO...")
minio_client = Minio(
    MINIO_ENDPOINT,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=False
)
print("[INIT] Initialized clients")

# Label colors
COLORS = np.random.uniform(0, 255, size=(21, 3))


def draw_tags(image_data, detections, original_name):
    print(f"[DRAW] Decoding image for tagging: {original_name}")
    arr = np.frombuffer(image_data, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("[ERROR] Failed to decode image data for tagging")

    orig_h, orig_w = image.shape[:2]
    print(f"[DRAW] Image shape (h,w): ({orig_h},{orig_w})")
    print(f"[DRAW] Number of detections to draw: {len(detections)}")

    for i, det in enumerate(detections):
        sx, sy = det.get("startX", 0), det.get("startY", 0)
        ex, ey = det.get("endX", 0), det.get("endY", 0)
        label = det.get("label", {}).get("name", "?")
        idx = det.get("label", {}).get("index", 0)
        conf = det.get("confidence", 0.0)

        # Clamp coordinates to image bounds
        orig = (sx, sy, ex, ey)
        sx = max(0, min(sx, orig_w - 1))
        sy = max(0, min(sy, orig_h - 1))
        ex = max(0, min(ex, orig_w - 1))
        ey = max(0, min(ey, orig_h - 1))
        clamped = (sx, sy, ex, ey)
        if orig != clamped:
            print(f"[WARN] Clamped coords from {orig} to {clamped}")

        print(f"[DETECT] {i}: {label} ({conf:.2f}) @ {clamped}")

        color = tuple(int(c) for c in COLORS[idx % len(COLORS)])
        cv2.rectangle(image, (sx, sy), (ex, ey), color, 2)
        text_y = sy - 10 if sy - 10 > 10 else sy + 10
        text = f"{label} ({conf:.2f})"
        cv2.putText(image, text, (sx, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    print("[DRAW] Encoding final tagged image to JPEG buffer")
    success, buffer = cv2.imencode('.jpg', image)
    if not success:
        raise RuntimeError("[ERROR] Failed to encode tagged image")
    return io.BytesIO(buffer)

print("[INFO] Tag service started. Listening for jobs...")

while True:
    task = redis_client.rpop(QUEUE_NAME)
    if not task:
        time.sleep(1)
        continue

    try:
        print(f"[INFO] Received task payload: {task}")
        job = json.loads(task)

        filename = job.get("filename")
        origin_filename = job.get("origin_filename")
        origin_h = job.get("origin_h")
        origin_w = job.get("origin_w")
        detections = job.get("detections", [])

        print(f"[TASK] filename={filename}")
        if origin_filename:
            print(f"[TASK] origin_filename provided: {origin_filename}")
        else:
            if filename.startswith("grayscale-resized-"):
                origin_filename = filename.replace("grayscale-resized-", "")
                print(f"[WARN] No origin_filename; derived: {origin_filename}")
            else:
                origin_filename = filename
                print(f"[WARN] No origin_filename; defaulted to: {origin_filename}")

        print(f"[TASK] origin_h={origin_h}, origin_w={origin_w}")
        print(f"[TASK] Total detections received: {len(detections)}")

        print(f"[INFO] Fetching image '{origin_filename}' from bucket '{BUCKET_NAME}'")
        obj = minio_client.get_object(BUCKET_NAME, origin_filename)
        image_data = obj.read()
        obj.close()
        print(f"[INFO] Fetched {len(image_data)} bytes")

        tagged_io = draw_tags(image_data, detections, origin_filename)
        output_filename = f"tagged-{filename}"

        print(f"[INFO] Uploading tagged image as '{output_filename}' to bucket '{BUCKET_NAME}'")
        minio_client.put_object(
            BUCKET_NAME,
            output_filename,
            data=tagged_io,
            length=tagged_io.getbuffer().nbytes,
            content_type="image/jpeg"
        )
        print(f"[INFO] Tagged image saved to MinIO: {output_filename}")

    except Exception as e:
        print(f"[ERROR] Error processing tag task: {e}")
        import traceback; traceback.print_exc()
        continue
