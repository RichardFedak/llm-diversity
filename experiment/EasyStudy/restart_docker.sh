#!/bin/bash

# Usage:
#   ./restart-docker.sh           -> restart docker, remove containers/images
#   ./restart-docker.sh --volumes -> also remove volumes

set -e

REMOVE_VOLUMES=false
if [[ "$1" == "--volumes" ]]; then
  REMOVE_VOLUMES=true
fi

echo "Stopping running containers..."
CONTAINERS=$(docker ps -q)
[ -n "$CONTAINERS" ] && docker stop $CONTAINERS

echo "Removing all containers..."
ALL_CONTAINERS=$(docker ps -a -q)
[ -n "$ALL_CONTAINERS" ] && docker rm $ALL_CONTAINERS

echo "Removing all images..."
IMAGES=$(docker images -q)
[ -n "$IMAGES" ] && docker rmi -f $IMAGES

if $REMOVE_VOLUMES; then
  echo "Removing all volumes..."
  VOLUMES=$(docker volume ls -q)
  [ -n "$VOLUMES" ] && docker volume rm $VOLUMES
fi

echo "Restarting Docker service..."
sudo systemctl restart docker

docker compose up -d

