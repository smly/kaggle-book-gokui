#!/bin/sh
set -eux

docker build .
DOCKER_IMAGE=$(docker build -q .)

docker run -v ${PWD}:/workspace -it ${DOCKER_IMAGE}
