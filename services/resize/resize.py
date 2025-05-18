import os
import time
import json
import io
import traceback
import cv2
import numpy as np
import redis
from minio import Minio

# Env config
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
BUCKET_NAME = "images"
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
QUEUE_NAME = "resize-queue"
NEXT_QUEUE = "grayscale-queue"

# For changing resolution in the `resize' function
SCALE_PERCENT = int(os.getenv("SCALE_PERCENT", 25))

# Clients
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
minio_client = Minio(
    MINIO_ENDPOINT,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=False
)

def resize(data):
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("[ERROR] Could not decode image from buffer")

    origin_h, origin_w = img.shape[:2]
    width = int(origin_w * SCALE_PERCENT / 100)
    height = int(origin_h * SCALE_PERCENT / 100)

    resized = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    _, buffer = cv2.imencode('.jpg', resized)
    return io.BytesIO(buffer), origin_h, origin_w

print("[INFO] Resizer service started. Listening for jobs...", flush=True)

while True:
    try:
        task = redis_client.rpop(QUEUE_NAME)
        if not task:
            time.sleep(1)
            continue

        print(f"[INFO] Pulled task: {task}")
        job = json.loads(task)

        filename = job.get("filename")
        image_id = job.get("image_id")

        if not filename or not image_id:
            print("[WARNING] Job missing required fields. Skipping...")
            continue

        print(f"[INFO] Resizing image: {filename}")
        data = minio_client.get_object(BUCKET_NAME, filename).read()
        resized_io, origin_h, origin_w = resize(data)

        output_filename = f"resized-{filename}"
        minio_client.put_object(
            BUCKET_NAME,
            output_filename,
            data=resized_io,
            length=resized_io.getbuffer().nbytes,
            content_type="image/jpeg"
        )
        print(f"[INFO] Saved resized image: {output_filename}")

        next_msg = {
            "image_id": image_id,
            "filename": output_filename,
            "origin_h": origin_h,
            "origin_w": origin_w
        }

        print(f"[INFO] Forwarding to {NEXT_QUEUE}: {next_msg}")
        redis_client.lpush(NEXT_QUEUE, json.dumps(next_msg))

    except Exception as e:
        print(f"[ERROR] Error in resizer service: {e}")
        traceback.print_exc()
        time.sleep(1)
        continue
