# Task Description: Decomposing a Monolith into Microservices

## Overview

Decomposing a monolith application is not trivial, so let's start small. Your task is to decompose a monolithic image processing application into a set of microservices and deploy them to Kubernetes. 
The original application is defined in [`monolith.py`](./monolith.py) and performs a series of image processing steps to detect objects in images.

## Original Application

The monolithic application performs the following steps in sequence:
1. **Image Grab**: Reads an image from a file
2. **Resize**: Scales down the image to reduce processing time
3. **Grayscale**: Converts the image to black and white
4. **Object Detect**: Runs object detection on the smaller, black and white image
5. **Tag**: Draws bounding boxes and labels on the original image based on detection results

### Object detection config files
Download the following files, and copy them to the same folder as monolith.py.

https://github.com/PINTO0309/MobileNet-SSD-RealSense/blob/master/caffemodel/MobileNetSSD/MobileNetSSD_deploy.caffemodel

https://github.com/PINTO0309/MobileNet-SSD-RealSense/blob/master/caffemodel/MobileNetSSD/MobileNetSSD_deploy.prototxt

### Run the application
Start a python virtual env
```sh
sudo apt install python3-pip
sudo apt install python3.12-venv
python3 -m venv venv
. venv/bin/activate
```

Install the dependencies
```sh
pip install -r requirements.txt
```

Run the application
```sh
python monolith.py
```

The image with the detected objects is generated to the result.jpg file.

## Your Task

Your task is to decompose this monolithic application into the following microservices:

1. **ImageGrab Service**: Accepts images via an HTTP interface and sends it for processing
2. **Resize Service**: Resizes images
3. **Grayscale Service**: Grayscales images
4. **ObjectDetect Service**: Detects objects on grayscale images
5. **Tag Service**: Based on the detected objects, draws bounding boxes on the original image

And the end, the image with the detected objects should be available in [MinIO](https://min.io/)

## Implementation Guidelines

1. **Service Communication**:
    - Services should communicate with each other using a messaging service (this can be Redis, RabbitMQ or other service of you choice)
    - After processing, each service should push a message to the next service's queue

2. **Image Storage**:
   - As images are usually large files, try to avoid sending them via the messaging service
     - All images should be stored in MinIO with appropriate naming conventions (the Minio config you can use is provided in the repo)
     - You can send messages via the messaging service and include the path to the file in MinIO which the image processing service can query and use

3. **HTTP Interface**:
   - The ImageGrab service should expose an HTTP endpoint to accept image uploads

4. **Deployment**:
   - Each service should be containerized
   - Services should be deployed to Kubernetes using appropriate deployment configurations

5. **Config values**:
   - DO NOT hard code configuration values (service host names, usernames, passwords), use configmaps, or configure the environment variables in the deployments

## Hints

- Start by understanding the flow of data in the original monolithic application
- Design the message format carefully
- Think about how to scale each service independently
- Remember to include appropriate logging for debugging

## [MinIO](https://min.io/)
MinIO is an open-source object storage server that is compatible with Amazon S3's API. 
It's designed to be lightweight, high-performance, and easy to deploy. 
MinIO provides a simple HTTP API for storing and retrieving objects (like files, images, or any binary data) and supports features such as versioning, encryption, and access control. 
In this lab, MinIO serves as the central storage for images at various stages of processing, allowing each microservice to retrieve and store images without having to pass large binary data through message queues. 
This approach is more efficient than sending entire images through the messaging service, especially for large files, and provides a persistent storage solution that can survive service restarts or failures.

You can find the [`minio.yaml`](./minio.yaml) file in this repository which you can use to deploy your minio instance to the cluster.
The minio service's console is exposed to the 30003 Node Port.

MinIO has a python client too that you can use to connect to it.

## [Redis](https://redis.io/)
Redis is a key-value store with lots of different features.
Redis can act as a message queue, so we provide a [`redis.yaml`](./redis.yaml) config as well which you can use. If you are more familiar with other solutions that can be used for messaging, feel free to use it.

Redis has a python client too that you can use to connect to it.

## Start the Cluster

1. Open AWS Academy login page: https://awsacademy.instructure.com/
2. Log in.
3. Start the AWS Academy Learner Lab and open the AWS Management console.
4. Click on this (CloudFormation) link: https://us-east-1.console.aws.amazon.com/cloudformation/home?region=us-east-1#/stacks/create/review?templateURL=https://vitmac12-resources.s3.amazonaws.com/k3s-multinode.template&stackName=k3s-multinode


## Good luck with the implementation!