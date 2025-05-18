FROM python:3.12-slim

WORKDIR /app

COPY . .

# Install system packages and OpenCV dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1
RUN pip install --no-cache-dir -r requirements.txt
RUN apt-get clean && rm -rf /var/lib/apt/lists/*


CMD ["python", "grayscale.py"]
