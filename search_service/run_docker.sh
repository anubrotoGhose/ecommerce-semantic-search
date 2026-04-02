#!/bin/bash

IMAGE_NAME="fastapi-ssl"
CONTAINER_NAME="fastapi-ssl-container"

echo "Building Docker image..."
docker build -t $IMAGE_NAME .

echo "Running Docker container..."
docker run -d \
  --name $CONTAINER_NAME \
  -p 8008:8008 \
  $IMAGE_NAME

echo "FastAPI running at https://localhost:8008"
echo "Press CTRL+C to stop the container"

# Trap CTRL+C (SIGINT)
trap cleanup SIGINT

cleanup() {
  echo ""
  echo "Stopping and removing Docker container..."
  docker stop $CONTAINER_NAME >/dev/null 2>&1
  docker rm $CONTAINER_NAME >/dev/null 2>&1
  echo "Container stopped and removed"
  exit 0
}

# Keep script running
while true; do
  sleep 1
done
