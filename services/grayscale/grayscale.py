import os
import time
import json
import io
import traceback
import cv2
import numpy as np
import redis
from minio import Minio

# Environment config
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
BUCKET_NAME = "images"
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
QUEUE_NAME = "grayscale-queue"
NEXT_QUEUE = "objectdetect-queue"

# Redis and MinIO clients
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
minio_client = Minio(
    MINIO_ENDPOINT,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=False
)

def grayscale(data):
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("[ERROR] Could not decode image from data buffer")

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Reconvert to BGR (3 channels) for next stages like object detection
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    _, buffer = cv2.imencode('.jpg', gray_bgr)
    return io.BytesIO(buffer)

print("[INFO] Grayscale service started. Listening for jobs...")

while True:
    try:
        task = redis_client.rpop(QUEUE_NAME)
        if not task:
            time.sleep(1)
            continue

        job = json.loads(task)
        print(f"[INFO] Received job: {job}")

        filename = job.get("filename")
        image_id = job.get("image_id")
        origin_h = job.get("origin_h")
        origin_w = job.get("origin_w")

        if not filename:
            print("[WARNING] Job missing filename. Skipping...")
            continue

        print(f"[INFO] Processing grayscale for: {filename}")
        data = minio_client.get_object(BUCKET_NAME, filename).read()
        gray_io = grayscale(data)

        output_filename = f"grayscale-{filename}"
        minio_client.put_object(
            BUCKET_NAME,
            output_filename,
            data=gray_io,
            length=gray_io.getbuffer().nbytes,
            content_type="image/jpeg"
        )
        print(f"[INFO] Saved grayscale image: {output_filename}")

        next_msg = {
            "image_id": image_id,
            "filename": output_filename,
            "origin_filename": job.get("origin_filename"),
            "origin_h": origin_h,
            "origin_w": origin_w
        }

        print(f"[INFO] Forwarding to {NEXT_QUEUE}: {next_msg}")
        redis_client.lpush(NEXT_QUEUE, json.dumps(next_msg))

    except Exception as e:
        print(f"[ERROR] Error in grayscale service: {e}")
        traceback.print_exc()
        continue
