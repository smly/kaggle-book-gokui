#!/bin/sh
set -eux

NUM_PROCESSES=4

docker build .
DOCKER_IMAGE=$(docker build -q .)

docker run --gpus all --rm \
    -v ${PWD}:/workspace -v ${DATA_DIR}:/data -v ${HOME}/.kaggle:/root/.kaggle \
    -e DATA_DIR=${DATA_DIR} -e NUM_PROCESSES=${NUM_PROCESSES} --shm-size=2gb \
    -it ${DOCKER_IMAGE} "$@"
