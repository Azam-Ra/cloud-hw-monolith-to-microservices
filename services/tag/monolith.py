#!/usr/bin/env python3

import copy
import cv2
import numpy as np

# Initialize the list of class labels MobileNet SSD was trained to detect
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# Load serialized model
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe("./MobileNetSSD_deploy.prototxt",
                               "./MobileNetSSD_deploy.caffemodel")

# For drawing bounding boxes of various colors in the `tag' function
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
# For changing resolution in the `resize' function
SCALE_PERCENT = 25
# For eliminating results with low confidence in obectdetect functions
CONFIDENCE_MIN = 0.4

def imagegrab(picture):
    cap = cv2.VideoCapture(picture)
    ret, image = cap.read()
    if not np.any(np.equal(image, None)):
        return image
    else:
        print("[WARN] No image found")
        return False

def resize(origin_image):
    # Get image height and width
    (origin_h, origin_w) = origin_image.shape[:2]
    # Scale image
    scale_percent = SCALE_PERCENT  # percent of original size
    width = int(origin_w * scale_percent / 100)
    height = int(origin_h * scale_percent / 100)
    image = cv2.resize(origin_image, (width, height),
                       interpolation=cv2.INTER_AREA)
    return image, origin_h, origin_w

def grayscale(image):
    # Turn image to black and white: convert to one color channel
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image

def objectdetect(image, origin_h, origin_w):
    # Expects an image with only one color channel

    # "Revert" grayscale: the model can work only with (multiples of)
    # three color channels
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    (height, width) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 0.007843, (height, width), 127.5)

    net.setInput(blob)
    detections = net.forward()

    labels_and_coords = []
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > CONFIDENCE_MIN:
            idx = int(detections[0, 0, i, 1])
            # Mark area on the original-sized picture not the resized
            box = detections[0, 0, i, 3:7] * np.array([origin_w,
                                                       origin_h,
                                                       origin_w,
                                                       origin_h])
            (startX, startY, endX, endY) = box.astype("int")
            labels_and_coords.append(
                {"startX": int(startX),
                 "startY": int(startY),
                 "endX": int(endX),
                 "endY": int(endY),
                 "label": {"name": CLASSES[idx],
                           "index": int(idx)},
                 "confidence": float(confidence)})
    return labels_and_coords

def tag(labels_and_coords, origin_image):
    image = copy.deepcopy(origin_image)
    for label_and_coord in labels_and_coords:
        label = label_and_coord["label"]["name"]
        index = label_and_coord["label"]["index"]
        startY = label_and_coord["startY"]
        cv2.rectangle(image,
                      (label_and_coord["startX"],
                       startY),
                      (label_and_coord["endX"],
                       label_and_coord["endY"]),
	    	              COLORS[index], 2)
        y = startY
        cv2.putText(image, label, (label_and_coord["startX"], y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    COLORS[index], 2)
    out_file = f"result.jpg"
    print(f"[INFO] write: {out_file}")
    cv2.imwrite(out_file, image)


if __name__ == '__main__':
    picture_file = "test-1.jpg"
    origin_image = imagegrab(picture_file)
    # If we read an image
    if hasattr(origin_image, "__len__"):
        image, origin_height, origin_width = resize(origin_image)
        image = grayscale(image)
        labels_and_coords = objectdetect(image, origin_height, origin_width)
        # Log object count
        print("[INFO] Object count:", len(labels_and_coords))
        tag(labels_and_coords, origin_image)