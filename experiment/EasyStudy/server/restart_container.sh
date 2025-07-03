#!/bin/bash

CONTAINER_NAME="llm_diversity"
IMAGE_NAME="flask_app"
PORT_MAPPING="5000:5000"

echo "Stopping and removing existing container..."
docker stop $CONTAINER_NAME && docker rm $CONTAINER_NAME

echo "Rebuilding the Docker image..."
docker build -t $IMAGE_NAME .

docker run -d -p $PORT_MAPPING --name $CONTAINER_NAME -v ~/.ssh/:/root/.ssh:ro $IMAGE_NAME bash -c "./keep_ssh_tunnel.sh & exec gunicorn --bind 0.0.0.0:5000 -w 4 --threads 3 wsgi:app"

echo "Done. Container '$CONTAINER_NAME' restarted."
