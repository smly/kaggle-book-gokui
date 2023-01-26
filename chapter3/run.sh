#!/bin/sh
set -eux

docker build .
DOCKER_IMAGE=$(docker build -q .)

docker run --gpus all --rm -v ${PWD}:/workspace -v ${DATA_DIR}:/data -v ${HOME}/.kaggle:/root/.kaggle -e TORCH_HOME=/workspace/.cache/torch/ --shm-size=2gb -it ${DOCKER_IMAGE} "$@"