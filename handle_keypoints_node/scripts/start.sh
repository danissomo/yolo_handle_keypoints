#!/bin/bash
DOCKER_PATH="$( dirname -- "$( readlink -f -- "$0"; )"; )/../docker"
PROG_PATH="$( dirname -- "$( readlink -f -- "$0"; )"; )/../../../../"
docker build $DOCKER_PATH --tag keypoints_image
docker create --gpus all -it -p 1001:1001 --net=host  --ipc="host"  -v $PROG_PATH:/home/keypoint_regression --name=keypoints_container keypoints_image /bin/bash
docker start keypoints_container
docker exec -it keypoints_container /bin/bash -c "echo \"$HUSKY_IP cpr-fssl01\" >> /etc/hosts"
