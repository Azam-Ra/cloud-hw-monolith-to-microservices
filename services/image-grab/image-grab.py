import uuid
import os
import json
import redis
import cv2
import numpy as np
from io import BytesIO
from fastapi import FastAPI, UploadFile, File
from minio import Minio
import traceback

app = FastAPI()

# Config from environment
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY")
BUCKET_NAME = os.getenv("BUCKET_NAME")
REDIS_HOST = os.getenv("REDIS_HOST")
REDIS_PORT = int(os.getenv("REDIS_PORT"))
QUEUE_NAME = os.getenv("QUEUE_NAME")

try:
    minio_client = Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False
    )
    print("[INFO] MinIO client initialized.")
    if not minio_client.bucket_exists(BUCKET_NAME):
        minio_client.make_bucket(BUCKET_NAME)
        print(f"[INFO] Created bucket: {BUCKET_NAME}")
    else:
        print(f"[INFO] Using existing bucket: {BUCKET_NAME}")
except Exception as e:
    print(f"[ERROR] Failed to connect to MinIO: {e}")
    traceback.print_exc()
    raise

try:
    redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT)
    redis_client.ping()
    print("[INFO] Redis client connected.")
except Exception as e:
    print(f"[ERROR] Failed to connect to Redis: {e}")
    traceback.print_exc()
    raise

@app.post("/grab")
async def grab_image(file: UploadFile = File(...)):
    try:
        image_id = str(uuid.uuid4())
        file_name = f"image_{image_id}.jpg"
        contents = await file.read()

        print(f"[INFO] Grabbed an image: {file.filename} -> {file_name} ({len(contents)} bytes)")

        img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            print("[ERROR] Failed to decode image with OpenCV")
            return {"error": "Invalid image format"}

        origin_h, origin_w = img.shape[:2]
        print(f"[INFO] Original image size: {origin_w}x{origin_h}")

        minio_client.put_object(
            BUCKET_NAME,
            file_name,
            data=BytesIO(contents),
            length=len(contents),
            content_type="image/jpeg"
        )
        print(f"[INFO] Uploaded to MinIO: {file_name}")

        msg = {
            "image_id": image_id,
            "filename": file_name,
            "original_filename": file_name,
            "origin_h": origin_h,
            "origin_w": origin_w
        }
        redis_client.lpush(QUEUE_NAME, json.dumps(msg))
        print(f"[INFO] Pushed to Redis queue ({QUEUE_NAME}): {msg}")

        return {"status": "uploaded", "filename": file_name}
    except Exception as e:
        print(f"[ERROR] Failed to handle upload: {e}")
        traceback.print_exc()
        return {"error": str(e)}
